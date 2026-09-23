use super::super::elevenlabs::*;
use crate::audio::voice::VoiceSample;
use crate::providers::tests::http::serve;

fn url(base: String, model: &str) -> ModelUrl {
    let mut url = ModelUrl::parse(&format!("elevenlabs:///{model}")).unwrap();
    url.api_key = Some("test-secret".into());
    url.base_url = Some(base);
    url
}

fn request() -> VoiceCloneRequest {
    VoiceCloneRequest {
        samples: vec![VoiceSample {
            mime_type: "audio/wav".into(),
            data: b"reference-audio".to_vec(),
            transcript: None,
        }],
        name: Some("Nadia".into()),
        provider_config: Some(json!({"remove_background_noise":false})),
    }
}

/// Public cloning factory returns a scoped remote ID and submits audio as multipart once.
#[tokio::test]
async fn clones_reference_audio_to_provider_id() {
    let (base, wires) = serve(
        200,
        json!({"voice_id":"voice123", "requires_verification":false}),
    );
    let client = crate::providers::create_voice_clone_client(
        &url(base, "ivc"),
        VoiceCloneOptions::default(),
    )
    .unwrap();
    let voice = client.clone_voice(&request()).await.unwrap();
    assert_eq!(voice, Voice::id("elevenlabs", "voice123"));
    let wire = wires.recv().unwrap();
    assert!(wire.starts_with("POST /voices/add "));
    assert!(wire.contains("name=\"files\""));
    assert!(wire.contains("reference-audio"));
    assert!(wire.contains("Nadia"));
    assert!(wire.contains("xi-api-key: test-secret"));
    assert!(wires.try_recv().is_err());
    synthesize_id(voice).await;
}

/// Exercises a returned remote ID through the public synthesis factory, including typed precedence.
async fn synthesize_id(voice: Voice) {
    let (base, wires) =
        crate::providers::tests::http::serve_content(200, "audio/mpeg", "speech".into());
    let client =
        crate::providers::create_tts_client(&url(base, "eleven_v3"), TtsOptions::default())
            .unwrap();
    let audio = client
        .synthesize_speech(&TtsRequest {
            input: "Hello".into(),
            voice: Some(voice),
            provider_config: Some(
                json!({"text":"ignored", "model_id":"ignored", "voice_settings":{"stability":0.5}}),
            ),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(audio.data, b"speech");
    let wire = wires.recv().unwrap();
    assert!(wire.starts_with("POST /text-to-speech/voice123?output_format=mp3_44100_128 "));
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["text"], "Hello");
    assert_eq!(payload["model_id"], "eleven_v3");
    assert_eq!(payload["voice_settings"]["stability"], 0.5);
}

/// Verification-required results retain remote identity evidence but never return a ready voice.
#[tokio::test]
async fn pending_verification_is_not_success() {
    let (base, wires) = serve(
        200,
        json!({"voice_id":"voice123", "requires_verification":true}),
    );
    let client = new_clone(&url(base, "ivc"), VoiceCloneOptions::default()).unwrap();
    let error = client.clone_voice(&request()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Provider);
    assert!(String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("voice123"));
    assert_eq!(wires.try_iter().count(), 1);
}

/// Design produces a preview, and registration is a separate explicit provider mutation.
#[tokio::test]
async fn designs_then_explicitly_registers() {
    let text = "Hello. ".repeat(20);
    let (base, wires) = serve(
        200,
        json!({"text":text,"previews":[{
            "audio_base_64":"AQID", "media_type":"audio/mpeg", "generated_voice_id":"preview123"
        }]}),
    );
    let client = crate::providers::create_voice_design_client(
        &url(base, "eleven_ttv_v3"),
        VoiceDesignOptions::default(),
    )
    .unwrap();
    let result = client
        .design_voice(&VoiceDesignRequest {
            description: "A warm and softly spoken adult voice".into(),
            text: text.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.previews[0].sample.data, [1, 2, 3]);
    assert_eq!(
        result.previews[0].sample.transcript.as_deref(),
        Some(text.as_str())
    );
    let wire = wires.recv().unwrap();
    assert!(wire.starts_with("POST /text-to-voice/design "));
    assert!(wires.try_recv().is_err());
    let (base, wires) = serve(200, json!({"voice_id":"voice123"}));
    let client = new_design(&url(base, "eleven_ttv_v3"), VoiceDesignOptions::default()).unwrap();
    let voice = client
        .register_voice(
            &result.previews[0],
            "Nadia",
            Some(&json!({
                "voice_description":"A warm and softly spoken adult voice"
            })),
        )
        .await
        .unwrap();
    assert_eq!(voice, Voice::id("elevenlabs", "voice123"));
    let wire = wires.recv().unwrap();
    assert!(wire.starts_with("POST /text-to-voice "));
    assert!(wire.contains("\"generated_voice_id\":\"preview123\""));
}

/// Cross-provider identities, non-ID voices, missing names and native config errors fail locally.
#[tokio::test]
async fn rejects_incompatible_inputs_without_network() {
    let configured = url("http://127.0.0.1:1".into(), "eleven_v3");
    let tts = new_tts(&configured, TtsOptions::default()).unwrap();
    for voice in [
        Voice::id("fal/elevenlabs", "voice123"),
        Voice::id("elevenlabs", "../bad"),
    ] {
        let error = tts
            .synthesize_speech(&TtsRequest {
                input: "Hello".into(),
                voice: Some(voice),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    let clone = new_clone(
        &url("http://127.0.0.1:1".into(), "ivc"),
        VoiceCloneOptions::default(),
    )
    .unwrap();
    let mut request = request();
    request.name = None;
    assert_eq!(
        clone.clone_voice(&request).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
}

/// Malformed previews retain wire evidence without leaking audio or secrets through diagnostics.
#[tokio::test]
async fn malformed_preview_preserves_diagnostics() {
    let (base, wires) = serve(
        200,
        json!({"text":"Hello","previews":[{
            "audio_base_64":"invalid!", "media_type":"audio/mpeg", "generated_voice_id":"preview123", "private":"PRIVATE_AUDIO_MARKER"
        }]}),
    );
    let client = new_design(&url(base, "eleven_ttv_v3"), VoiceDesignOptions::default()).unwrap();
    let error = client
        .design_voice(&VoiceDesignRequest {
            description: "A warm and softly spoken adult voice".into(),
            text: "Hello. ".repeat(20),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert!(error.response_body().is_some());
    assert!(!format!("{error} {error:?}").contains("PRIVATE_AUDIO_MARKER"));
    assert_eq!(wires.try_iter().count(), 1);
}

/// Provider failures preserve status and sanitized evidence, with no implicit retry.
#[tokio::test]
async fn clone_http_error_is_sanitized_and_not_retried() {
    let (base, wires) = serve(
        429,
        json!({"detail":{"status":"quota_exceeded","message":"quota for test-secret"}}),
    );
    let client = new_clone(&url(base, "ivc"), VoiceCloneOptions::default()).unwrap();
    let error = client.clone_voice(&request()).await.unwrap_err();
    assert_eq!(error.http_status(), Some(429));
    assert!(!format!("{error} {error:?}").contains("test-secret"));
    assert!(
        !String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("test-secret")
    );
    assert_eq!(wires.try_iter().count(), 1);
}

/// Unsupported delivery controls and bad previews fail without contacting ElevenLabs.
#[tokio::test]
async fn controls_and_preview_registration_are_validated() {
    let tts = new_tts(
        &url("http://127.0.0.1:1".into(), "eleven_v3"),
        TtsOptions::default(),
    )
    .unwrap();
    let error = tts
        .synthesize_speech(&TtsRequest {
            input: "Hello".into(),
            voice: Some(Voice::id("elevenlabs", "voice123")),
            instructions: Some("whisper".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    let design = new_design(
        &url("http://127.0.0.1:1".into(), "eleven_ttv_v3"),
        VoiceDesignOptions::default(),
    )
    .unwrap();
    let preview = VoicePreview {
        sample: request().samples.remove(0),
        scope: "fal/qwen3-tts-1.7b".into(),
        registration_token: Some("preview123".into()),
    };
    assert_eq!(
        design
            .register_voice(&preview, "Nadia", None)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}
