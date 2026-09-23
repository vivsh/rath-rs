use serde_json::json;

use super::http::{json as reply, serve};
use crate::audio::tts::{TtsOptions, TtsRequest};
use crate::core::ModelUrl;
fn clone_request(data: Vec<u8>, mime: &str, text: &str) -> VoiceCloneRequest {
    VoiceCloneRequest {
        samples: vec![VoiceSample {
            data,
            mime_type: mime.into(),
            transcript: Some(text.into()),
        }],
        ..Default::default()
    }
}
use crate::audio::{
    voice::{Voice, VoiceData, VoiceSample},
    voice_clone::{VoiceCloneOptions, VoiceCloneRequest},
    voice_design::{VoiceDesignOptions, VoiceDesignRequest},
};

/// Qwen design and synthesis preserve native settings through the existing queued audio path.
#[tokio::test]
async fn qwen_design_and_speech_complete_paths() {
    for (endpoint, config) in [
        (
            "fal-ai/qwen-3-tts/voice-design/1.7b",
            json!({"prompt":"A warm voice", "max_new_tokens":2048}),
        ),
        (
            "fal-ai/qwen-3-tts/text-to-speech/1.7b",
            json!({"speaker_voice_embedding_file_url":"data:application/octet-stream;base64,AQID", "reference_text":"Hello"}),
        ),
    ] {
        let (base, requests) = serve(vec![
            reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
            reply(json!({"status":"COMPLETED"})),
            reply(json!({"audio":{"url":"BASE/audio"}})),
            (200, "audio/mpeg", "speech".into()),
        ]);
        let mut url = ModelUrl::parse(&format!("fal:///{endpoint}")).unwrap();
        url.base_url = Some(base);
        url.api_key = Some("test-secret".into());
        let data = qwen_audio(&url, &config).await;
        assert_eq!(data, b"speech");
        let wires: Vec<_> = requests.try_iter().collect();
        let (_, body) = wires[0].split_once("\r\n\r\n").unwrap();
        let mut expected = config;
        expected["text"] = json!("Hello");
        if endpoint.contains("text-to-speech") {
            expected["max_new_tokens"] = json!(4096);
        }
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(body).unwrap(),
            expected
        );
        assert!(!wires[3].to_lowercase().contains("authorization"));
    }
}

/// Calls the independent design or speech factory and checks reference audio metadata.
async fn qwen_audio(url: &ModelUrl, config: &serde_json::Value) -> Vec<u8> {
    let endpoint = &url.model;
    if endpoint.contains("voice-design") {
        let client =
            crate::providers::create_voice_design_client(url, VoiceDesignOptions::default())
                .unwrap();
        let result = client
            .design_voice(&VoiceDesignRequest {
                text: "Hello".into(),
                description: "A warm voice".into(),
                provider_config: Some(config.clone()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            result.previews[0].sample.transcript.as_deref(),
            Some("Hello")
        );
        assert!(result.previews[0].registration_token.is_none());
        result.previews[0].sample.data.clone()
    } else {
        let client = crate::providers::create_tts_client(url, TtsOptions::default()).unwrap();
        client
            .synthesize_speech(&TtsRequest {
                input: "Hello".into(),
                voice: Some(Voice {
                    scope: "fal/qwen3-tts-1.7b".into(),
                    data: VoiceData::Embedding {
                        format: "qwen3-tts-1.7b/safetensors".into(),
                        data: vec![1, 2, 3],
                        reference_text: Some("Hello".into()),
                    },
                }),
                ..Default::default()
            })
            .await
            .unwrap()
            .data
    }
}

/// Voice cloning downloads the embedding without credentials and sends exact audio/transcript inputs.
#[tokio::test]
async fn qwen_clone_complete_path() {
    let (base, requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"speaker_embedding":{"url":"BASE/embedding"}})),
        (200, "application/octet-stream", "embedding".into()),
    ]);
    let mut url = ModelUrl::parse("fal:///fal-ai/qwen-3-tts/clone-voice/1.7b").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("test-secret".into());
    let client =
        crate::providers::create_voice_clone_client(&url, VoiceCloneOptions::default()).unwrap();
    let voice = client
        .clone_voice(&clone_request(vec![1, 2, 3], "audio/mpeg", "Hello"))
        .await
        .unwrap();
    assert_eq!(voice.scope, "fal/qwen3-tts-1.7b");
    assert!(matches!(voice.data, VoiceData::Embedding {data, ..} if data == b"embedding"));
    let wires: Vec<_> = requests.try_iter().collect();
    assert_eq!(wires.len(), 4);
    let (_, body) = wires[0].split_once("\r\n\r\n").unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        json!({
            "audio_url":"data:audio/mpeg;base64,AQID", "reference_text":"Hello"
        })
    );
    assert!(!wires[3].to_lowercase().contains("authorization"));
}

/// Cloning rejects invalid inputs and speech endpoints before any HTTP request is made.
#[tokio::test]
async fn qwen_clone_validates_inputs_and_capability() {
    for endpoint in [
        "fal-ai/qwen-3-tts/clone-voice/1.7b",
        "fal-ai/kokoro/american-english",
    ] {
        let mut url = ModelUrl::parse(&format!("fal:///{endpoint}")).unwrap();
        url.base_url = Some("http://127.0.0.1:1".into());
        url.api_key = Some("test-secret".into());
        let result =
            crate::providers::create_voice_clone_client(&url, VoiceCloneOptions::default());
        if endpoint.contains("kokoro") {
            assert!(result.is_err());
            continue;
        }
        let client = result.unwrap();
        for (data, mime, text) in [
            (vec![], "audio/mpeg", "Hello"),
            (vec![1], "text/plain", "Hello"),
            (vec![1], "audio/mpeg", " "),
        ] {
            let error = client
                .clone_voice(&clone_request(data, mime, text))
                .await
                .unwrap_err();
            assert!(matches!(
                error.kind(),
                crate::core::ErrorKind::Validation | crate::core::ErrorKind::UnsupportedCapability
            ));
        }
    }
}

/// A provider result missing the embedding URL retains its typed response evidence.
#[tokio::test]
async fn qwen_clone_rejects_missing_embedding() {
    let (base, _requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"speaker_embedding":{}})),
    ]);
    let mut url = ModelUrl::parse("fal:///fal-ai/qwen-3-tts/clone-voice/1.7b").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("test-secret".into());
    let client =
        crate::providers::create_voice_clone_client(&url, VoiceCloneOptions::default()).unwrap();
    let error = client
        .clone_voice(&clone_request(vec![1], "audio/mpeg", "Hello"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::core::ErrorKind::InvalidResponse);
    assert!(error.response_body().is_some());
}

/// Eleven v3 uses the public factory and existing queue/download path without altering audio tags.
#[tokio::test]
async fn eleven_v3_complete_path() {
    let (base, requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"audio":{"url":"BASE/audio"}})),
        (200, "audio/mpeg", "test-audio".into()),
    ]);
    let mut url = ModelUrl::parse("fal:///fal-ai/elevenlabs/tts/eleven-v3").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("test-secret".into());
    let client = crate::providers::create_tts_client(&url, TtsOptions::default()).unwrap();
    let result = client
        .synthesize_speech(&TtsRequest {
            input: "[whispers] Hello 世界".into(),
            voice: Some(Voice::id("fal/elevenlabs", "Rachel")),
            provider_config: Some(json!({"stability":0.5})),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.mime_type, "audio/mpeg");
    assert_eq!(result.data, b"test-audio");
    assert!(result.raw_metadata.unwrap().get("audio").is_some());
    let wires: Vec<_> = requests.try_iter().collect();
    assert_eq!(wires.len(), 4);
    assert!(wires[0].starts_with("POST /fal-ai/elevenlabs/tts/eleven-v3 "));
    let (_, body) = wires[0].split_once("\r\n\r\n").unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        json!({
            "text":"[whispers] Hello 世界", "voice":"Rachel", "stability":0.5
        })
    );
    for request in &wires[..3] {
        assert!(request.contains("Key test-secret"));
    }
    assert!(!wires[3].to_lowercase().contains("authorization"));
}

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
            voice: Some(Voice::id("fal/kokoro/american-english", "af_heart")),
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
