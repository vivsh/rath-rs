use super::super::tests::http::{json as reply, serve};
use super::*;
use crate::images::ImageData;
use crate::video::VideoInput;
use serde_json::json;

/// Exercises the public URL factory through submission, queue states, and final media.
#[tokio::test]
async fn image_to_video_end_to_end() {
    let (base, requests) = serve(vec![
        reply(
            json!({"request_id":"job-1","status_url":"BASE/status","response_url":"BASE/result"}),
        ),
        reply(json!({"status":"IN_QUEUE","queue_position":2})),
        reply(json!({"status":"IN_PROGRESS"})),
        reply(json!({"status":"COMPLETED","response_url":"BASE/result"})),
        reply(json!({"video":{"url":"https://assets.example/out.mp4"}})),
    ]);
    let client = VideoOptions {
        poll_interval: Duration::ZERO,
        ..Default::default()
    }
    .create(&format!(
        "fal:///{}?base_url={base}&api_key_env=PATH",
        video_input::KLING_I2V
    ))
    .await
    .unwrap();
    let result = client
        .generate_video(&VideoRequest {
            prompt: "Walk forward".into(),
            input: VideoInput::ImageToVideo {
                start_image: ImageData::Url {
                    url: "https://assets.example/frame.png".into(),
                },
                end_image: None,
            },
            webhook_url: Some("https://app.example/hooks?job=1&token=private".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.videos.len(), 1);
    let calls: Vec<_> = requests.try_iter().collect();
    assert_eq!(calls.len(), 5);
    assert!(calls[0].contains("fal_webhook=https%3A%2F%2Fapp.example"));
    let input: Value = serde_json::from_str(calls[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(input["start_image_url"], "https://assets.example/frame.png");
    assert!(input.get("webhook_url").is_none());
    assert_eq!(
        calls
            .iter()
            .filter(|call| call.starts_with("POST "))
            .count(),
        1
    );
}

fn local_client(base: &str, model: &str) -> VideoClient {
    let mut url = ModelUrl::parse(&format!("fal:///{model}")).unwrap();
    url.base_url = Some(base.into());
    url.api_key = Some("FAKE-SECRET".into());
    new_client(
        &url,
        VideoOptions {
            poll_interval: Duration::ZERO,
            ..Default::default()
        },
    )
    .unwrap()
}

fn native() -> VideoRequest {
    VideoRequest {
        input: VideoInput::Native {
            payload: json!({"prompt":"native"}),
        },
        ..Default::default()
    }
}

/// Unknown endpoints remain usable through the explicit native path and shared video decoder.
#[tokio::test]
async fn native_endpoint_end_to_end() {
    let (base, calls) = serve(vec![
        reply(json!({"request_id":"custom-job"})),
        reply(json!({"status":"COMPLETED","response_url":"BASE/result"})),
        reply(json!({"video_url":"https://assets.example/native.mp4"})),
    ]);
    let result = local_client(&base, "owner/custom/video")
        .generate_video(&native())
        .await
        .unwrap();
    assert_eq!(result.videos.len(), 1);
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 3);
    assert!(calls[0].starts_with("POST /owner/custom/video "));
    assert!(calls[0].contains(r#"{"prompt":"native"}"#));
}

/// Endpoint mismatch and invalid callback/input fail before any network request.
#[tokio::test]
async fn invalid_submissions_fail_before_network() {
    let client = local_client("http://127.0.0.1:1", "owner/custom");
    let req = VideoRequest {
        prompt: "typed".into(),
        ..Default::default()
    };
    assert_eq!(
        client.submit_video(&req).await.unwrap_err().kind(),
        ErrorKind::UnsupportedCapability
    );
    for callback in [
        "http://app.example/hook",
        "https://user:secret@app.example/hook",
        "https://app.example/hook#fragment",
        "bad",
    ] {
        let mut req = native();
        req.webhook_url = Some(callback.into());
        assert_eq!(
            client.submit_video(&req).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
    let mut req = native();
    req.provider_config = Some(json!([]));
    assert_eq!(
        client.submit_video(&req).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        client
            .get_video("../evil?token=x")
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

/// Non-Fal video schemes fail instead of silently selecting an available adapter.
#[tokio::test]
async fn unsupported_providers_and_config_fail_at_construction() {
    let err = VideoOptions::default()
        .create("elevenlabs:///video")
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind(), ErrorKind::UnsupportedCapability);
    let mut url = ModelUrl::parse("fal:///owner/custom").unwrap();
    url.api_key = Some("FAKE-SECRET".into());
    let options = VideoOptions {
        provider_config: Some(json!([])),
        ..Default::default()
    };
    assert_eq!(
        new_client(&url, options).err().unwrap().kind(),
        ErrorKind::Validation
    );
}

/// Provider-returned URLs on another origin are rejected before any credential-bearing request.
#[tokio::test]
async fn malicious_queue_urls_are_not_followed() {
    for field in ["status_url", "response_url", "cancel_url"] {
        let (base, calls) = serve(vec![reply(
            json!({"request_id":"job",field:"http://127.0.0.1:1/private"}),
        )]);
        let err = local_client(&base, "owner/custom")
            .submit_video(&native())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidResponse);
        assert_eq!(calls.try_iter().count(), 1);
    }
    let (base, calls) = serve(vec![reply(
        json!({"status":"COMPLETED","response_url":"http://127.0.0.1:1/private"}),
    )]);
    let err = local_client(&base, "owner/custom")
        .get_video("job")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidResponse);
    assert_eq!(calls.try_iter().count(), 1);
}

/// Authenticated video transport never follows HTTP redirects or resubmits.
#[tokio::test]
async fn video_redirects_are_not_followed() {
    let (sink, sink_calls) = serve(vec![reply(json!({"request_id":"unexpected"}))]);
    let mime = format!("text/plain\r\nLocation: {sink}");
    let (base, calls) = serve(vec![(302, &mime, "redirect".into())]);
    let err = local_client(&base, "owner/custom")
        .submit_video(&native())
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Http);
    assert_eq!(err.http_status(), Some(302));
    assert_eq!(calls.try_iter().count(), 1);
    assert_eq!(sink_calls.try_iter().count(), 0);
}

/// Failed generation and HTTP errors do not create another paid job.
#[tokio::test]
async fn failures_never_resubmit() {
    let (base, calls) = serve(vec![
        reply(json!({"request_id":"job"})),
        reply(
            json!({"status":"COMPLETED","error":"bad generation FAKE-SECRET","private":"evidence"}),
        ),
    ]);
    let err = local_client(&base, "owner/custom")
        .generate_video(&native())
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Provider);
    assert!(!format!("{err} {err:?}").contains("FAKE-SECRET"));
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls
            .iter()
            .filter(|call| call.starts_with("POST "))
            .count(),
        1
    );
}

/// Invalid native outputs and malformed encoded videos cannot become empty successes.
#[tokio::test]
async fn invalid_outputs_fail_after_one_submission() {
    for output in [
        json!({}),
        json!({"images":[{"url":"https://assets.example/a.png"}]}),
        json!({"video":{"url":""}}),
        json!({"video":{"base64":"!","content_type":"video/mp4"}}),
    ] {
        let (base, calls) = serve(vec![
            reply(json!({"request_id":"job"})),
            reply(json!({"status":"COMPLETED","response_url":"BASE/result"})),
            reply(output),
        ]);
        let err = local_client(&base, "owner/custom")
            .generate_video(&native())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidResponse);
        assert_eq!(calls.try_iter().count(), 3);
    }
}

/// Dropping a wait future cancels only local polling and never issues cancellation or submission.
#[tokio::test]
async fn waiting_is_locally_cancellable() {
    let (base, calls) = serve(vec![reply(json!({"status":"IN_PROGRESS"}))]);
    let mut url = ModelUrl::parse("fal:///owner/custom").unwrap();
    url.api_key = Some("FAKE-SECRET".into());
    url.base_url = Some(base);
    let client = new_client(&url, VideoOptions::default()).unwrap();
    let result = tokio::time::timeout(Duration::from_millis(100), client.wait_video("job")).await;
    assert!(result.is_err());
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].starts_with("GET "));
}

/// A serialized lip-sync job resumes in a fresh client using documented queue paths.
#[tokio::test]
async fn lip_sync_resumes_after_restart() {
    use crate::audio::generation::AudioData;
    use crate::video::{LipSyncMode, LipSyncModel};
    let (base, calls) = serve(vec![
        reply(
            json!({"request_id":"job-1", "status_url":"BASE/fal-ai/sync-lipsync/requests/job-1/status", "response_url":"BASE/fal-ai/sync-lipsync/requests/job-1"}),
        ),
        reply(json!({"status":"IN_QUEUE"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"video":{"url":"https://assets.example/result.mp4"}})),
    ]);
    let request = VideoRequest::new(
        String::new(),
        VideoInput::LipSync {
            video: VideoData::Url {
                url: "https://assets.example/input.mp4".into(),
            },
            audio: AudioData::Url {
                url: "https://assets.example/speech.mp3".into(),
            },
            model: LipSyncModel::Standard,
            sync_mode: LipSyncMode::Loop,
        },
    );
    let job = local_client(&base, video_input::LIP_SYNC)
        .submit_video(&request)
        .await
        .unwrap();
    let receipt: VideoJob = serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
    let fresh = local_client(&base, video_input::LIP_SYNC);
    assert!(matches!(
        fresh.get_video(&receipt.id).await.unwrap(),
        VideoJobStatus::Queued { .. }
    ));
    assert!(matches!(
        fresh.get_video(&receipt.id).await.unwrap(),
        VideoJobStatus::Succeeded { .. }
    ));
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 4);
    assert!(calls[0].starts_with("POST /fal-ai/sync-lipsync/v2 "));
    assert!(calls[1].starts_with("GET /fal-ai/sync-lipsync/requests/job-1/status "));
    assert!(calls[2].starts_with("GET /fal-ai/sync-lipsync/requests/job-1/status "));
    assert!(calls[3].starts_with("GET /fal-ai/sync-lipsync/requests/job-1 "));
    assert_lip_payload(&calls[0]);
}

/// Verifies the typed request reached the Fal submit endpoint.
fn assert_lip_payload(call: &str) {
    let payload: Value = serde_json::from_str(call.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["video_url"], "https://assets.example/input.mp4");
    assert_eq!(payload["audio_url"], "https://assets.example/speech.mp3");
    assert_eq!(payload["model"], "lipsync-2");
    assert_eq!(payload["sync_mode"], "loop");
}

/// Builds an explicit lip-sync request for transport tests.
fn lip_request() -> VideoRequest {
    use crate::audio::generation::AudioData;
    use crate::video::{LipSyncMode, LipSyncModel};
    VideoRequest::new(
        String::new(),
        VideoInput::LipSync {
            video: VideoData::Url {
                url: "https://assets.example/video.mp4".into(),
            },
            audio: AudioData::Url {
                url: "https://assets.example/audio.mp3".into(),
            },
            model: LipSyncModel::Standard,
            sync_mode: LipSyncMode::CutOff,
        },
    )
}

/// Completed lip-sync polling honors a same-origin provider result URL.
#[tokio::test]
async fn lip_sync_uses_provider_result_url() {
    let (base, calls) = serve(vec![
        reply(json!({"status":"COMPLETED","response_url":"BASE/provider/result"})),
        reply(json!({"video":{"url":"https://assets.example/result.mp4"}})),
    ]);
    let state = local_client(&base, video_input::LIP_SYNC)
        .get_video("job")
        .await
        .unwrap();
    assert!(matches!(state, VideoJobStatus::Succeeded { .. }));
    let calls: Vec<_> = calls.try_iter().collect();
    assert!(calls[0].starts_with("GET /fal-ai/sync-lipsync/requests/job/status "));
    assert!(calls[1].starts_with("GET /provider/result "));
}

/// Remote failures stay distinct from polling transport errors and do not resubmit.
#[tokio::test]
async fn lip_sync_failures_and_transport() {
    let (base, calls) = serve(vec![reply(
        json!({"status":"COMPLETED", "error":"failed FAKE-SECRET"}),
    )]);
    let status = local_client(&base, video_input::LIP_SYNC)
        .get_video("job")
        .await
        .unwrap();
    assert!(matches!(status, VideoJobStatus::Failed { .. }));
    assert!(!format!("{status:?}").contains("FAKE-SECRET"));
    assert_eq!(calls.try_iter().count(), 1);
    let error = local_client("http://127.0.0.1:1", video_input::LIP_SYNC)
        .get_video("job")
        .await
        .unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::Transport | ErrorKind::Timeout
    ));
}

/// Invalid lip-sync requests fail locally before the adapter contacts the queue.
#[tokio::test]
async fn lip_sync_invalid_before_http() {
    use crate::audio::generation::AudioData;
    let (base, calls) = serve(vec![reply(json!({"request_id":"unexpected"}))]);
    let client = local_client(&base, video_input::LIP_SYNC);
    let mut request = lip_request();
    request.prompt = "unexpected".into();
    assert_eq!(
        client.submit_video(&request).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    request.prompt.clear();
    if let VideoInput::LipSync { audio, .. } = &mut request.input {
        *audio = AudioData::Base64 {
            mime_type: "audio/wav".into(),
            data: "!".into(),
        };
    }
    assert_eq!(
        client.submit_video(&request).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    let wrong = local_client(&base, video_input::KLING_MOTION);
    assert_eq!(
        wrong.submit_video(&lip_request()).await.unwrap_err().kind(),
        ErrorKind::UnsupportedCapability
    );
    assert_eq!(calls.try_iter().count(), 0);
}

/// Lip-sync polling rejects foreign result URLs before sending credentials.
#[tokio::test]
async fn lip_sync_rejects_foreign_result() {
    let (sink, sink_calls) = serve(vec![reply(
        json!({"video":{"url":"https://example.com/out"}}),
    )]);
    let (base, calls) = serve(vec![reply(
        json!({"status":"COMPLETED", "response_url":format!("{sink}/result")}),
    )]);
    let err = local_client(&base, video_input::LIP_SYNC)
        .get_video("job")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidResponse);
    assert_eq!(calls.try_iter().count(), 1);
    assert_eq!(sink_calls.try_iter().count(), 0);
}

/// Existing motion-transfer polling drops endpoint suffixes from queue status paths.
#[tokio::test]
async fn motion_transfer_queue_route() {
    let (base, calls) = serve(vec![reply(json!({"status":"IN_PROGRESS"}))]);
    let status = local_client(&base, video_input::KLING_MOTION)
        .get_video("job")
        .await
        .unwrap();
    assert!(matches!(status, VideoJobStatus::Running { .. }));
    let call = calls.recv().unwrap();
    assert!(call.starts_with("GET /fal-ai/kling-video/requests/job/status "));
}
