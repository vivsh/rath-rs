use super::*;
use crate::{
    ErrorKind,
    providers::tests::http::{serve, serve_content},
};

/// Creates the actual factory adapter with a synthetic key and isolated loopback endpoint.
fn client(base: &str) -> SttClient {
    let mut url = ModelUrl::parse("gemini:///models/gemini-3.8-flash").unwrap();
    url.api_key = Some("gemini-audio-test-key".into());
    url.base_url = Some(base.into());
    crate::providers::create_stt_client(&url, SttOptions::default()).unwrap()
}

/// Factory routing sends inline browser audio and returns only completed transcript parts.
#[tokio::test]
async fn transcribes_inline_audio_without_thoughts_or_translation() {
    let raw = json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[
        {"text":"private reasoning", "thought":true}, {"text":"नमस्ते "}, {"text":"दुनिया"}
    ]}}], "usageMetadata":{"promptTokenCount":7}});
    let (base, requests) = serve(200, raw.clone());
    let response = client(&base)
        .transcribe_audio(&SttRequest::new(
            "audio/webm;codecs=opus".into(),
            vec![0, 255, 1],
        ))
        .await
        .unwrap();
    assert_eq!(response.text, "नमस्ते दुनिया");
    assert_eq!(response.raw_metadata, Some(raw));
    let wire = requests.try_recv().unwrap();
    assert!(wire.starts_with("POST /models/gemini-3.8-flash:generateContent "));
    assert!(wire.contains("x-goog-api-key: gemini-audio-test-key"));
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        payload["contents"][0]["parts"][1]["inlineData"],
        json!({"mimeType":"audio/webm", "data":"AP8B"})
    );
    assert_eq!(
        payload["generationConfig"]["responseModalities"],
        json!(["TEXT"])
    );
    assert!(
        payload["contents"][0]["parts"][0]["text"]
            .as_str()
            .unwrap()
            .contains("only the transcript")
    );
    assert_eq!(requests.try_iter().count(), 0);
}

/// Encoded-file aliases and browser codec metadata map to documented Gemini MIME tokens.
#[test]
fn validates_input_mime_and_parameters() {
    for (source, target) in [
        ("audio/x-wav", "audio/wav"),
        ("Audio/MP3", "audio/mpeg"),
        ("audio/mp4", "audio/m4a"),
        ("audio/x-aiff", "audio/aiff"),
        ("audio/ogg; codecs=\"opus\"", "audio/ogg"),
        ("audio/flac", "audio/flac"),
    ] {
        assert_eq!(input_mime(source).unwrap(), target);
    }
    for source in [
        "text/plain",
        "audio/",
        "audio/unknown",
        "audio/l16;rate=16000",
        "audio/webm;codecs=",
        "audio/webm;bad",
        "audio/webm;codecs=opus\r\n",
        "audio/webm;codecs=\"opus",
        "audio/webm;codecs=\" \"",
        "audio/webm;codecs=opus,",
        "audio/webm;codecs=opus;codecs=vorbis",
    ] {
        assert!(input_mime(source).is_err(), "accepted {source}");
    }
}

/// Invalid audio and unhandled configuration fail before any authenticated request.
#[tokio::test]
async fn rejects_invalid_requests_before_transport() {
    let (base, requests) = serve(200, json!({}));
    let client = client(&base);
    for request in [
        SttRequest::new("audio/wav".into(), vec![]),
        SttRequest::new("text/plain".into(), vec![1]),
        SttRequest::new("audio/wav".into(), vec![1]).with_provider_config(json!({"language":"en"})),
        SttRequest::new("audio/wav".into(), vec![1]).with_provider_config(json!([])),
        SttRequest::new("audio/wav".into(), vec![0; 14_000_001]),
    ] {
        let error = client.transcribe_audio(&request).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.operation(), Some("speech-to-text"));
    }
    assert_eq!(requests.try_iter().count(), 0);
}

/// Missing, empty, mixed, blocked and unfinished provider results never become transcripts.
#[tokio::test]
async fn rejects_malformed_or_partial_transcripts_with_evidence() {
    for raw in [
        json!({}),
        json!({"candidates":[]}),
        json!({"promptFeedback":{"blockReason":"SAFETY"}}),
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":" "}]}}]}),
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":42}]}}]}),
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"inlineData":{}}]}}]}),
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":"reason", "thought":true}]}}]}),
        json!({"candidates":[{"finishReason":"MAX_TOKENS", "content":{"parts":[{"text":"partial"}]}}]}),
        json!({"candidates":[{"finishReason":"SAFETY", "content":{"parts":[{"text":"partial"}]}}]}),
        json!({"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}),
    ] {
        let (base, requests) = serve(200, raw.clone());
        let error = client(&base)
            .transcribe_audio(&SttRequest::new("audio/wav".into(), vec![1]))
            .await
            .unwrap_err();
        assert_eq!(error.provider(), Some(&Provider::Gemini));
        assert_eq!(error.http_status(), Some(200));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            raw
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// HTTP errors, success-status provider failures and malformed JSON retain normalized redacted diagnostics.
#[tokio::test]
async fn normalizes_provider_and_decode_failures() {
    for status in [200, 401, 429, 500] {
        let (base, requests) = serve(
            status,
            json!({"error":{"code":429, "status":"RESOURCE_EXHAUSTED", "message":"quota gemini-audio-test-key"}}),
        );
        let error = client(&base)
            .transcribe_audio(&SttRequest::new("audio/wav".into(), vec![1]))
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
        assert_eq!(error.http_status(), Some(status));
        assert!(!format!("{error} {error:?}").contains("gemini-audio-test-key"));
        assert!(
            !String::from_utf8_lossy(error.response_body().unwrap().bytes())
                .contains("gemini-audio-test-key")
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
    let (base, _requests) = serve_content(200, "application/json", "{invalid json".into());
    let error = client(&base)
        .transcribe_audio(&SttRequest::new("audio/wav".into(), vec![1]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Deserialize);
    assert!(error.source().is_some());
}
