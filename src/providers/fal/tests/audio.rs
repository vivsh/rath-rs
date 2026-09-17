use serde_json::json;

use super::http::{json as reply, serve};
use crate::audio::tts::{TtsOptions, TtsRequest};
use crate::core::ModelUrl;

/// Exercises construction, queue completion, payload mapping and unauthenticated audio download.
#[tokio::test]
async fn kokoro_complete_path() {
    let (base, requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"audio":{"url":"BASE/audio"}})),
        (200, "audio/wav", "RIFF-test-audio".into()),
    ]);
    let mut url = ModelUrl::parse("fal:///fal-ai/kokoro/american-english").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("test-secret".into());
    let client = crate::providers::create_tts_client(&url, TtsOptions::default()).unwrap();
    let result = client
        .synthesize_speech(&TtsRequest {
            input: "Hello 世界".into(),
            voice: Some("af_heart".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.mime_type, "audio/wav");
    assert_eq!(result.data, b"RIFF-test-audio");
    assert!(result.raw_metadata.unwrap().get("audio").is_some());
    let wires: Vec<_> = requests.try_iter().collect();
    assert_eq!(wires.len(), 4);
    assert!(wires[0].starts_with("POST /fal-ai/kokoro/american-english "));
    assert!(wires[0].contains("\"prompt\":\"Hello 世界\""));
    for request in &wires[..3] {
        assert!(request.contains("Key test-secret"));
    }
    assert!(!wires[3].to_lowercase().contains("authorization"));
}
