use super::*;
use crate::{ErrorKind, images::ImageData};
use serde_json::{Value, json};
use std::time::Duration;
#[path = "video_http.rs"]
mod http_fixture;
use http_fixture::serve;

const MODEL: &str = "veo-3.1-fast-generate-preview";
const ID: &str = "models/veo-3.1-fast-generate-preview/operations/job_123";
const KEY: &str = "gemini-video-test-key";

fn url(base: &str, model: &str) -> ModelUrl {
    let mut url = ModelUrl::parse(&format!("gemini:///{model}")).unwrap();
    url.api_key = Some(KEY.into());
    url.base_url = Some(base.into());
    url
}

fn client(base: &str, options: VideoOptions) -> VideoClient {
    crate::providers::create_video_client(&url(base, MODEL), options).unwrap()
}

fn options() -> VideoOptions {
    VideoOptions::default().with_poll_interval(Duration::from_millis(1))
}

fn success(id: &str) -> Value {
    json!({"name":id, "done":true, "response":{"generateVideoResponse":{
        "generatedSamples":[{"video":{"uri":"https://generativelanguage.googleapis.com/v1beta/files/video:download?alt=media"}}]
    }}})
}

fn text() -> VideoRequest {
    VideoRequest::new(
        "A bluebird flies through a quiet garden.".into(),
        crate::video::VideoInput::TextToVideo,
    )
}

fn image() -> ImageData {
    ImageData::Base64 {
        mime_type: "image/png".into(),
        data: "iVBORw0KGgo=".into(),
    }
}

/// The public built-in factory submits once, polls running then success, and never downloads media.
#[tokio::test]
async fn generates_through_real_factory_and_long_running_operation() {
    let (base, requests) = serve(vec![
        (200, json!({"name":ID})),
        (200, json!({"name":ID})),
        (200, success(ID)),
    ]);
    let response = client(&base, options())
        .generate_video(&text())
        .await
        .unwrap();
    assert_eq!(response.videos.len(), 1);
    let calls: Vec<_> = requests.try_iter().collect();
    assert_eq!(calls.len(), 3);
    assert!(calls[0].starts_with(&format!("POST /models/{MODEL}:predictLongRunning ")));
    for call in &calls[1..] {
        assert!(call.starts_with(&format!("GET /{ID} ")));
    }
    for call in &calls {
        assert!(
            call.to_lowercase()
                .contains(&format!("x-goog-api-key: {KEY}"))
        );
    }
    let payload: Value = serde_json::from_str(calls[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["instances"][0]["prompt"], text().prompt);
    assert_eq!(payload["parameters"], json!({}));
    assert!(!calls[0].contains("webhook_signing_secret"));
}

/// Every documented Veo 3.1 variant returns a model-bound, caller-owned receipt.
#[tokio::test]
async fn routes_models_and_retains_operation_receipts() {
    for model in [
        "veo-3.1-generate-preview",
        MODEL,
        "veo-3.1-lite-generate-preview",
    ] {
        let id = format!("models/{model}/operations/abc-123");
        let raw = json!({"name":id, "metadata":{"private":"caller-owned"}});
        let (base, requests) = serve(vec![(200, raw.clone())]);
        let client = crate::providers::create_video_client(&url(&base, model), options()).unwrap();
        let job = client.submit_video(&text()).await.unwrap();
        assert_eq!(job.id, id);
        assert_eq!(job.provider, Provider::Gemini);
        assert_eq!(job.provider_model.as_deref(), Some(model));
        assert_eq!(
            job.status_url.as_deref(),
            Some(format!("{base}/{id}").as_str())
        );
        assert_eq!(job.raw_metadata, Some(raw));
        assert!(job.cancel_url.is_none());
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Inline first/last frames use the SDK's exact REST shape and request parameters override defaults.
#[tokio::test]
async fn maps_inline_frames_and_excludes_local_verification_settings() {
    let (base, requests) = serve(vec![(200, json!({"name":ID}))]);
    let options = options().with_provider_config(
        json!({"parameters":{"resolution":"720p"}, "webhook_signing_secret_env":"TEST_SECRET"}),
    );
    let request = VideoRequest::new(
        "Animate this bird.".into(),
        crate::video::VideoInput::ImageToVideo {
            start_image: image(),
            end_image: Some(image()),
        },
    )
    .with_provider_config(json!({"parameters":{"resolution":"1080p", "durationSeconds":8}}));
    let original = request.clone();
    client(&base, options).submit_video(&request).await.unwrap();
    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    let wire = requests.try_recv().unwrap();
    let raw: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    let frame = json!({"bytesBase64Encoded":"iVBORw0KGgo=", "mimeType":"image/png"});
    assert_eq!(raw["instances"][0]["image"], frame);
    assert_eq!(raw["instances"][0]["lastFrame"], frame);
    assert_eq!(
        raw["parameters"],
        json!({"resolution":"1080p", "durationSeconds":8})
    );
    assert!(raw.get("webhook_signing_secret_env").is_none());
}

/// Native controls enforce documented type/model limits before any paid request can be sent.
#[test]
fn rejects_invalid_generation_controls_locally() {
    let cases = [
        json!({"resolution":"4k"}),
        json!({"resolution":"1080p","durationSeconds":4}),
        json!({"durationSeconds":"8"}),
        json!({"durationSeconds":5}),
        json!({"sampleCount":2}),
        json!({"aspectRatio":"1:1"}),
        json!({"seed":1}),
        json!({"generateAudio":false}),
    ];
    for params in cases {
        let request = text().with_provider_config(json!({"parameters":params}));
        assert!(
            request::payload(
                &url("https://example.test", "veo-3.1-lite-generate-preview"),
                &options(),
                &request
            )
            .is_err()
        );
    }
    for config in [
        json!([]),
        json!({"parameters":[]} ),
        json!({"webhook_signing_secret_env":"attacker"}),
        json!({"model":"other"}),
    ] {
        assert!(
            request::payload(
                &url("https://example.test", MODEL),
                &options(),
                &text().with_provider_config(config)
            )
            .is_err()
        );
    }
    assert!(
        new_video_client(
            &url("https://example.test", MODEL),
            VideoOptions::default().with_poll_interval(Duration::ZERO)
        )
        .is_err()
    );
    assert!(new_video_client(&url("https://example.test", "gemini-3.8-flash"), options()).is_err());
}

/// URL inputs, dynamic callbacks and unsupported operations fail before contacting any endpoint.
#[tokio::test]
async fn unsupported_inputs_never_send_requests() {
    let (base, requests) = serve(vec![]);
    let client = client(&base, options());
    let requests_to_reject = [
        text().with_webhook_url("https://example.test/hooks".into()),
        VideoRequest::new(
            "Animate".into(),
            crate::video::VideoInput::ImageToVideo {
                start_image: ImageData::Url {
                    url: "https://example.test/image.png".into(),
                },
                end_image: None,
            },
        ),
        VideoRequest::new(
            "Native".into(),
            crate::video::VideoInput::Native { payload: json!({}) },
        ),
    ];
    for request in requests_to_reject {
        assert_eq!(
            client.submit_video(&request).await.unwrap_err().kind(),
            ErrorKind::UnsupportedCapability
        );
    }
    assert!(client.submit_video(&VideoRequest::default()).await.is_err());
    assert_eq!(requests.try_iter().count(), 0);
}

/// Operation IDs cannot change the authenticated origin, model, path or query.
#[tokio::test]
async fn rejects_operation_routing_injection_without_http() {
    let (base, requests) = serve(vec![]);
    let client = client(&base, options());
    for id in [
        "https://attacker.test/job",
        "operations/job",
        "models/other/operations/job",
        "models/veo-3.1-fast-generate-preview/operations/../job",
        "models/veo-3.1-fast-generate-preview/operations/job?key=foo",
        "models/veo-3.1-fast-generate-preview/operations/job%2Fother",
    ] {
        assert_eq!(
            client.get_video(id).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
    assert_eq!(requests.try_iter().count(), 0);
}

/// Provider failures remain remote failures, while waiting surfaces redacted structured evidence.
#[tokio::test]
async fn maps_remote_failures_and_filtered_output() {
    for raw in [
        json!({"name":ID,"done":true,"error":{"code":13,"message":format!("{KEY} rejected")}}),
        json!({"name":ID,"done":true,"response":{"generateVideoResponse":{"raiMediaFilteredCount":1,"raiMediaFilteredReasons":["blocked"]}}}),
    ] {
        let (base, requests) = serve(vec![(200, raw.clone()), (200, raw)]);
        let client = client(&base, options());
        let status = client.get_video(ID).await.unwrap();
        assert!(matches!(status, VideoJobStatus::Failed { .. }));
        assert!(!serde_json::to_string(&status).unwrap().contains(KEY));
        let error = client.wait_video(ID).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Provider);
        assert!(!format!("{error:?}").contains(KEY));
        assert_eq!(requests.try_iter().count(), 2);
    }
}

/// A malformed completion never becomes success or a fabricated provider failure.
#[tokio::test]
async fn rejects_malformed_and_contradictory_operations() {
    let cases = [
        json!({"name":"other", "done":true}),
        json!({"name":ID,"done":"true"}),
        json!({"name":ID,"done":false,"response":{}}),
        json!({"name":ID,"done":true,"error":{},"response":{}}),
        json!({"name":ID,"done":true,"error":{"code":13}}),
        json!({"name":ID,"done":true}),
        json!({"name":ID,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[]}}}),
        json!({"name":ID,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"http://attacker.test"}}]}}}),
    ];
    for raw in cases {
        let (base, requests) = serve(vec![(200, raw)]);
        let error = client(&base, options()).get_video(ID).await.unwrap_err();
        assert_eq!(error.http_status(), Some(200));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// HTTP failures retain status without exposing credentials, redirects or automatic resubmission.
#[tokio::test]
async fn normalizes_transport_failures_without_retrying() {
    for status in [302, 401, 429, 500] {
        let (base, requests) = serve(vec![(
            status,
            json!({"error":{"message":format!("{KEY} rejected")}}),
        )]);
        let error = client(&base, options())
            .submit_video(&text())
            .await
            .unwrap_err();
        assert_eq!(error.http_status(), Some(status));
        assert!(!format!("{error:?}").contains(KEY));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Applications without webhook verification configured retain the trait's explicit unsupported behavior.
#[tokio::test]
async fn webhook_parser_is_optional() {
    let (base, requests) = serve(vec![]);
    let error = client(&base, options())
        .parse_webhook(&::http::HeaderMap::new(), b"{}")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(requests.try_iter().count(), 0);
}
