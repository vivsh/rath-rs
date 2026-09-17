use std::time::Duration;

use serde_json::json;

use super::super::*;
use crate::providers::fal::tests::http::{json as reply, serve};

/// Constructs an audio client with a short poll interval for isolated local HTTP tests.
fn client(base: String, model: &str) -> FalClient {
    let mut url = ModelUrl::parse(&format!("fal:///{model}")).unwrap();
    url.api_key = Some("secret-key".into());
    url.base_url = Some(base);
    let mut client = audio_client(&url, None).unwrap();
    client.poll_interval = Duration::from_millis(1);
    client
}

/// Exercises pending states, native endpoint override and empty-transcript success for both STT models.
#[tokio::test]
async fn transcribes_both_models_and_keeps_metadata() {
    for model in [WIZPER, SCRIBE] {
        let (base, requests) = serve(vec![
            reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
            reply(json!({"status":"IN_QUEUE"})),
            reply(json!({"status":"IN_PROGRESS"})),
            reply(json!({"status":"COMPLETED"})),
            reply(json!({"text":"", "words":[{"speaker_id":"speaker_0"}]})),
        ]);
        let client = client(base, WIZPER);
        let response = client
            .transcribe_audio(&SttRequest {
                mime_type: "audio/wav".into(),
                data: vec![0, 255],
                model: Some(model.into()),
                provider_config: Some(json!({"language_code":"hi"})),
            })
            .await
            .unwrap();
        assert_eq!(response.text, "");
        assert!(response.raw_metadata.unwrap().get("words").is_some());
        assert_eq!(client.model, WIZPER);
        let wires: Vec<_> = requests.try_iter().collect();
        assert_eq!(wires.len(), 5);
        assert!(wires[0].starts_with(&format!("POST /{model} ")));
        assert!(wires[0].contains("data:audio/wav;base64,AP8="));
        assert_eq!(wires.iter().filter(|r| r.starts_with("POST")).count(), 1);
    }
}

/// Confirms model override changes the endpoint and text mapping without mutating the client.
#[tokio::test]
async fn elevenlabs_override_downloads_audio() {
    let (base, requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"audio":{"url":"BASE/audio", "content_type":"audio/mpeg"}})),
        (200, "application/octet-stream", "MP3".into()),
    ]);
    let client = client(base, KOKORO);
    let result = client
        .synthesize_speech(&TtsRequest {
            input: "Hello".into(),
            model: Some(ELEVENLABS.into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.mime_type, "audio/mpeg");
    assert_eq!(result.data, b"MP3");
    assert_eq!(client.model, KOKORO);
    let wires: Vec<_> = requests.try_iter().collect();
    assert!(wires[0].starts_with(&format!("POST /{ELEVENLABS} ")));
    assert!(wires[0].contains("\"text\":\"Hello\""));
    assert!(!wires[0].contains("\"prompt\""));
}

/// Rejects malformed results without including response content in ordinary errors.
#[tokio::test]
async fn missing_transcript_is_redacted_error() {
    let (base, _requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"private":"DISTINCTIVE_SECRET"})),
    ]);
    let error = client(base, WIZPER)
        .transcribe_audio(&SttRequest {
            mime_type: "audio/wav".into(),
            data: vec![1],
            model: None,
            provider_config: None,
        })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("transcript decoding"));
    assert!(!format!("{error:?} {error}").contains("DISTINCTIVE_SECRET"));
}

/// A paused clock verifies the five-minute operation deadline without a long real wait.
#[tokio::test]
async fn audio_deadline_stops_polling_without_resubmitting() {
    let (base, requests) = serve(vec![
        reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
        reply(json!({"status":"IN_QUEUE"})),
    ]);
    let mut client = client(base, WIZPER);
    client.poll_interval = Duration::from_secs(600);
    let task = tokio::spawn(async move {
        client
            .transcribe_audio(&SttRequest {
                mime_type: "audio/wav".into(),
                data: vec![1],
                model: None,
                provider_config: None,
            })
            .await
    });
    let mut wires = Vec::new();
    tokio::time::timeout(Duration::from_secs(3), async {
        while wires.len() < 2 {
            wires.extend(requests.try_iter());
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::pause();
    tokio::time::advance(queue::AUDIO_TIMEOUT + Duration::from_secs(1)).await;
    let error = task.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("timeout"));
    assert_eq!(wires.iter().filter(|r| r.starts_with("POST")).count(), 1);
}

/// Rejects off-origin or credential-bearing queue URLs before making any authenticated follow-up.
#[tokio::test]
async fn queue_urls_cannot_redirect_credentials() {
    for url in [
        "http://127.0.0.1:1/SECRET",
        "https://user:SECRET@example.com/status",
        "file:///SECRET",
    ] {
        let (base, requests) = serve(vec![reply(json!({
            "status_url":url, "response_url":"BASE/result"
        }))]);
        let error = queue::run(&client(base, WIZPER), WIZPER, json!({}))
            .await
            .unwrap_err();
        assert!(!format!("{error} {error:?}").contains("SECRET"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Rejects redirects, authentication, validation, rate limits and malformed JSON without resubmission.
#[tokio::test]
async fn submission_failures_are_redacted_and_not_retried() {
    for status in [302, 401, 422, 429, 500, 200] {
        let (base, requests) = serve(vec![(
            status,
            "application/json",
            "DISTINCTIVE_SECRET".into(),
        )]);
        let error = queue::run(&client(base, WIZPER), WIZPER, json!({}))
            .await
            .unwrap_err();
        assert!(!format!("{error} {error:?}").contains("DISTINCTIVE_SECRET"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Failed or unknown queue status is never decoded as a successful audio response.
#[tokio::test]
async fn failed_queue_status_does_not_fetch_result() {
    for status in [
        json!({"status":"COMPLETED", "error":"DISTINCTIVE_SECRET"}),
        json!({"status":"ALIEN"}),
    ] {
        let (base, requests) = serve(vec![
            reply(json!({"status_url":"BASE/status", "response_url":"BASE/result"})),
            reply(status),
        ]);
        let error = queue::run(&client(base, WIZPER), WIZPER, json!({}))
            .await
            .unwrap_err();
        assert!(!format!("{error:?}").contains("DISTINCTIVE_SECRET"));
        assert_eq!(requests.try_iter().count(), 2);
    }
}

/// Output download failures and HTML bodies cannot masquerade as successful generated audio.
#[tokio::test]
async fn rejects_failed_or_non_audio_downloads() {
    for (status, mime, body) in [
        (403, "audio/wav", "SECRET"),
        (200, "text/html", "SECRET"),
        (200, "audio/wav", ""),
    ] {
        let (base, requests) = serve(vec![(status, mime, body.into())]);
        let error = download(
            &client(base.clone(), KOKORO),
            &json!({
                "audio":{"url":format!("{base}/audio?token=SECRET"), "content_type":"audio/wav"}
            }),
        )
        .await
        .unwrap_err();
        assert!(!format!("{error} {error:?}").contains("SECRET"));
        let wires: Vec<_> = requests.try_iter().collect();
        assert_eq!(wires.len(), 1);
        assert!(!wires[0].to_lowercase().contains("authorization"));
    }
}

/// Both factories validate capability and options before attempting network traffic.
#[test]
fn factory_validation() {
    let mut url = ModelUrl::parse(&format!("fal:///{SCRIBE}")).unwrap();
    url.api_key = Some("test".into());
    assert!(crate::providers::create_stt_client(&url, SttOptions::default()).is_ok());
    assert!(crate::providers::create_tts_client(&url, TtsOptions::default()).is_err());
    assert!(
        new_stt_client(
            &url,
            SttOptions {
                provider_config: Some(json!([]))
            }
        )
        .is_err()
    );
    url.model = ELEVENLABS.into();
    assert!(crate::providers::create_tts_client(&url, TtsOptions::default()).is_ok());
    assert!(crate::providers::create_stt_client(&url, SttOptions::default()).is_err());
    assert!(
        new_tts_client(
            &url,
            TtsOptions {
                provider_config: Some(json!(false))
            }
        )
        .is_err()
    );
}
