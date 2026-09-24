use super::http::serve;
use rath::audio::{
    tts::{TtsOptions, TtsRequest},
    voice::Voice,
};
use rath::registry::ProviderRegistry;
use serde_json::{Value, json};

/// Built-in registry construction preserves the legacy payload, endpoint and explicit credential.
#[tokio::test]
async fn builtin_speech_wire_matches_legacy() {
    let mut payloads = Vec::new();
    for registered in [false, true] {
        let (base, requests) = serve(200, "synthetic-audio");
        let raw = format!("openai:///tts-1?api_key_env=PATH&base_url={base}");
        let options = {
            let mut value = TtsOptions::default();
            value.provider_config = Some(json!({"speed":0.8}));
            value
        };
        let client = if registered {
            ProviderRegistry::with_builtins()
                .tts(&raw, options)
                .await
                .unwrap()
        } else {
            options.create(&raw).await.unwrap()
        };
        assert!(requests.try_recv().is_err());
        let request = {
            let mut value = TtsRequest::default();
            value.input = "Hello".into();
            value.voice = Some(Voice::id("openai", "alloy"));
            value.provider_config = Some(json!({"speed":1.0}));
            value
        };
        let response = client.synthesize_speech(&request).await.unwrap();
        assert_eq!(response.data, b"synthetic-audio");
        let wire = requests.try_recv().unwrap();
        assert!(wire.starts_with("POST /audio/speech "));
        assert!(wire.contains(&format!("Bearer {}", std::env::var("PATH").unwrap())));
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        payloads.push(payload);
        assert!(requests.try_recv().is_err());
    }
    assert_eq!(payloads[0], payloads[1]);
    assert_eq!(payloads[0]["speed"], 1.0);
    assert_eq!(payloads[0]["voice"], "alloy");
}

/// Registry routing preserves useful built-in provider HTTP failures and original response bytes.
#[tokio::test]
async fn builtin_speech_errors_match_legacy() {
    let mut errors = Vec::new();
    for registered in [false, true] {
        let (base, requests) = serve(
            400,
            r#"{"error":{"message":"bad voice","code":"invalid_voice"}}"#,
        );
        let raw = format!("openai:///tts-1?api_key_env=PATH&base_url={base}");
        let client = if registered {
            ProviderRegistry::with_builtins()
                .tts(&raw, TtsOptions::default())
                .await
                .unwrap()
        } else {
            TtsOptions::default().create(&raw).await.unwrap()
        };
        let error = client
            .synthesize_speech(&{
                let mut value = TtsRequest::default();
                value.input = "hello".into();
                value
            })
            .await
            .unwrap_err();
        assert_eq!(error.http_status(), Some(400));
        assert_eq!(error.provider_code(), Some("invalid_voice"));
        assert_eq!(requests.try_iter().count(), 1);
        errors.push(error);
    }
    assert_eq!(errors[0].to_string(), errors[1].to_string());
    assert_eq!(errors[0].response_body(), errors[1].response_body());
}

/// Multipart settings are shallow-merged before encoding; request settings never produce duplicates.
#[tokio::test]
async fn transcription_native_fields_are_unique() {
    use rath::audio::stt::{SttOptions, SttRequest};
    let (base, wire) = super::http::serve(200, r#"{"text":"ok"}"#);
    let client = SttOptions::default()
        .with_provider_config(serde_json::json!({"language":"en"}))
        .create(&format!("openai:///bound?api_key_env=USER&base_url={base}"))
        .await
        .unwrap();
    let request = SttRequest::new("audio/wav".into(), vec![1, 2])
        .with_provider_config(serde_json::json!({"language":"hi"}));
    assert_eq!(client.transcribe_audio(&request).await.unwrap().text, "ok");
    let wire = wire.recv().unwrap();
    assert_eq!(wire.matches("name=\"language\"").count(), 1);
    assert_eq!(wire.matches("name=\"model\"").count(), 1);
    assert!(wire.contains("\r\n\r\nhi\r\n"));
}
