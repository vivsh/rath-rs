use super::*;
use crate::{ErrorKind, providers::tests::http::serve};
use serde_json::json;

const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9N8AAAAASUVORK5CYII=";

/// Routes image generation through the actual factory with a synthetic credential and loopback endpoint.
fn client(base: &str, model: &str, options: ImageOptions) -> ImageClient {
    let mut url = ModelUrl::parse(&format!("gemini:///{model}")).unwrap();
    url.api_key = Some("gemini-image-test-key".into());
    url.base_url = Some(base.into());
    crate::providers::create_image_client(&url, options).unwrap()
}

fn reply(parts: Value) -> Value {
    json!({"candidates":[{"finishReason":"STOP", "content":{"parts":parts}}]})
}

/// All documented native image models send text-to-image requests and preserve ordered image bytes/MIME.
#[tokio::test]
async fn routes_native_image_models_and_excludes_thought_images() {
    for model in [
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
        "gemini-nano-banana-2.1",
    ] {
        let raw = reply(json!([
            {"inlineData":{"mimeType":"image/png", "data":"private thought"}, "thought":true},
            {"text":"Here is your image"},
            {"inlineData":{"mimeType":"image/png", "data":PNG}},
            {"inlineData":{"mimeType":"image/png", "data":PNG}}
        ]));
        let (base, requests) = serve(200, raw.clone());
        let response = client(&base, model, ImageOptions::default())
            .generate_image(&ImageRequest::new("A blue bird".into()))
            .await
            .unwrap();
        assert_eq!(response.images.len(), 2);
        for image in response.images {
            assert!(
                matches!(image, ImageData::Base64 {mime_type, data} if mime_type == "image/png" && data == PNG)
            );
        }
        assert_eq!(response.raw_metadata, Some(raw));
        let wire = requests.try_recv().unwrap();
        assert!(wire.starts_with(&format!("POST /models/{model}:generateContent ")));
        assert!(wire.contains("x-goog-api-key: gemini-image-test-key"));
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(payload["contents"][0]["parts"][0]["text"], "A blue bird");
        assert_eq!(
            payload["generationConfig"]["responseModalities"],
            json!(["TEXT", "IMAGE"])
        );
    }
}

/// Shallow request defaults and typed size preserve native aspect ratio without mutating input.
#[tokio::test]
async fn typed_size_overrides_native_size() {
    let (base, requests) = serve(
        200,
        reply(json!([{"inlineData":{"mimeType":"image/png", "data":PNG}}])),
    );
    let options = ImageOptions::default().with_provider_config(
        json!({"generationConfig":{"imageConfig":{"aspectRatio":"1:1", "imageSize":"1K"}}}),
    );
    let request = ImageRequest::new("A blue bird".into())
        .with_size("4K".into())
        .with_provider_config(
            json!({"generationConfig":{"imageConfig":{"aspectRatio":"16:9", "imageSize":"2K"}}}),
        );
    let original = request.clone();
    client(&base, "gemini-3-pro-image", options)
        .generate_image(&request)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    let wire = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        payload["generationConfig"]["imageConfig"],
        json!({"aspectRatio":"16:9", "imageSize":"4K"})
    );
}

/// Native size/aspect constraints and routing reject unsupported settings before any provider request.
#[tokio::test]
async fn validates_image_model_options_and_prompt() {
    let (base, requests) = serve(200, json!({}));
    for (model, request) in [
        (
            "gemini-2.5-flash-image",
            ImageRequest::new("bird".into()).with_size("1K".into()),
        ),
        (
            "gemini-3.1-flash-lite-image",
            ImageRequest::new("bird".into()).with_size("2K".into()),
        ),
        (
            "gemini-nano-banana-2.1",
            ImageRequest::new("bird".into()).with_size("512".into()),
        ),
        (
            "gemini-3-pro-image",
            ImageRequest::new("bird".into()).with_size("1024x1024".into()),
        ),
        ("gemini-3-pro-image", ImageRequest::new(" ".into())),
        (
            "gemini-3-pro-image",
            ImageRequest::new("bird".into()).with_provider_config(
                json!({"generationConfig":{"imageConfig":{"aspectRatio":"1:8"}}}),
            ),
        ),
        (
            "gemini-3-pro-image",
            ImageRequest::new("bird".into())
                .with_provider_config(json!({"reference_image":"https://example.com/private"})),
        ),
        (
            "gemini-3-pro-image",
            ImageRequest::new("bird".into())
                .with_provider_config(json!({"generationConfig":{"responseModalities":["TEXT"]}})),
        ),
    ] {
        assert_eq!(
            client(&base, model, ImageOptions::default())
                .generate_image(&request)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
    assert_eq!(requests.try_iter().count(), 0);
}

/// Ordinary chat models cannot be constructed as image generators.
#[test]
fn rejects_chat_models_for_image_generation() {
    let mut url = ModelUrl::parse("gemini:///gemini-3.8-flash").unwrap();
    url.api_key = Some("synthetic".into());
    assert_eq!(
        new_image_client(&url, ImageOptions::default())
            .err()
            .unwrap()
            .kind(),
        ErrorKind::UnsupportedCapability
    );
}

/// Invalid/empty encoded data, missing images and MIME mismatches never produce successful media.
#[tokio::test]
async fn rejects_malformed_image_output_with_evidence() {
    for parts in [
        json!([]),
        json!([{"text":"No image"}]),
        json!([{"inlineData":{}}]),
        json!([{"inlineData":{"mimeType":"image/png", "data":""}}]),
        json!([{"inlineData":{"mimeType":"image/png", "data":"not-base64!"}}]),
        json!([{"inlineData":{"mimeType":"image/jpeg", "data":PNG}}]),
        json!([{"inlineData":{"mimeType":"text/plain", "data":PNG}}]),
    ] {
        let raw = reply(parts);
        let (base, requests) = serve(200, raw.clone());
        let error = client(&base, "gemini-3-pro-image", ImageOptions::default())
            .generate_image(&ImageRequest::new("bird".into()))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert_eq!(error.operation(), Some("image generation"));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            raw
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Provider failures preserve HTTP/provider distinctions and redact configured credentials.
#[tokio::test]
async fn normalizes_image_provider_failures() {
    for status in [200, 403, 429, 500] {
        let (base, requests) = serve(
            status,
            json!({"error":{"code":status, "message":"gemini-image-test-key rejected"}}),
        );
        let error = client(&base, "gemini-3-pro-image", ImageOptions::default())
            .generate_image(&ImageRequest::new("bird".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            if status == 200 {
                ErrorKind::Provider
            } else {
                ErrorKind::Http
            }
        );
        assert!(!format!("{error} {error:?}").contains("gemini-image-test-key"));
        assert!(
            !String::from_utf8_lossy(error.response_body().unwrap().bytes())
                .contains("gemini-image-test-key")
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}
