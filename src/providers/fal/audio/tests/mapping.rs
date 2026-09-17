use serde_json::json;

use super::super::*;

/// Keeps typed values authoritative while request configuration overrides client defaults.
#[test]
fn tts_configuration_precedence() {
    let config = Some(json!({"speed":0.9,"stability":0.2,"text":"old", "voice":"old"}));
    let request = TtsRequest {
        input: "Hello".into(),
        voice: Some("Aria".into()),
        provider_config: Some(json!({"speed":1.1,"text":"override","voice":"override"})),
        ..Default::default()
    };
    let payload = tts_payload(ELEVENLABS, &config, &request).unwrap();
    assert_eq!(payload["text"], "Hello");
    assert_eq!(payload["voice"], "Aria");
    assert_eq!(payload["speed"], 1.1);
    assert_eq!(payload["stability"], 0.2);
    assert!(!payload.contains_key("model"));
    let payload = tts_payload(
        KOKORO,
        &None,
        &TtsRequest {
            input: "Hi".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(payload["prompt"], "Hi");
    assert!(!payload.contains_key("voice"));
}

/// Refuses unsupported models, empty input, explicit format and malformed configuration locally.
#[test]
fn rejects_invalid_tts_requests() {
    let valid = TtsRequest {
        input: "Hi".into(),
        ..Default::default()
    };
    assert!(tts_payload("unknown", &None, &valid).is_err());
    for request in [
        TtsRequest::default(),
        TtsRequest {
            format: Some("mp3".into()),
            ..valid.clone()
        },
        TtsRequest {
            provider_config: Some(json!([])),
            ..valid.clone()
        },
    ] {
        assert!(tts_payload(KOKORO, &None, &request).is_err());
    }
    assert!(tts_payload(KOKORO, &Some(json!(null)), &valid).is_err());
}

/// Encodes binary bytes faithfully and keeps language settings in the selected native schema.
#[test]
fn stt_data_uri_and_precedence() {
    let request = SttRequest {
        mime_type: "audio/wav".into(),
        data: vec![0, 255, 1, 128],
        model: None,
        provider_config: Some(json!({"audio_url":"untrusted", "language":null})),
    };
    for model in [WIZPER, SCRIBE] {
        let payload = stt_payload(model, &Some(json!({"language":"en"})), &request).unwrap();
        assert_eq!(payload["audio_url"], "data:audio/wav;base64,AP8BgA==");
        assert!(payload["language"].is_null());
    }
}

/// Rejects invalid MIME syntax, empty audio, unsupported models and non-object settings.
#[test]
fn rejects_invalid_stt_requests() {
    let mut request = SttRequest {
        mime_type: "audio/wav".into(),
        data: vec![1],
        model: None,
        provider_config: None,
    };
    assert!(stt_payload("unknown", &None, &request).is_err());
    assert!(stt_payload(WIZPER, &Some(json!(1)), &request).is_err());
    for mime in [
        "text/plain",
        "audio/",
        "audio/wav\r\nsecret",
        "audio/wav;secret=foo",
    ] {
        request.mime_type = mime.into();
        assert!(stt_payload(WIZPER, &None, &request).is_err());
    }
    request.mime_type = "audio/wav".into();
    request.data.clear();
    assert!(stt_payload(WIZPER, &None, &request).is_err());
}

/// Accepts audio headers and generic downloads with metadata, while rejecting HTML responses.
#[test]
fn resolves_download_mime_without_hiding_errors() {
    assert_eq!(
        download_mime(Some("audio/WAV; charset=binary"), None).unwrap(),
        "audio/wav"
    );
    assert_eq!(
        download_mime(Some("application/octet-stream"), Some("audio/mpeg")).unwrap(),
        "audio/mpeg"
    );
    assert_eq!(download_mime(None, Some("audio/wav")).unwrap(), "audio/wav");
    assert!(download_mime(Some("text/html"), Some("audio/wav")).is_err());
    assert!(download_mime(None, None).is_err());
}
