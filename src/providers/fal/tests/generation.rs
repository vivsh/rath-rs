use super::super::tests::http::{json as reply, serve};
use super::*;
use serde_json::json;

fn sound(base: &str) -> SoundClient {
    let mut url = ModelUrl::parse(&format!("fal:///{SOUND}")).unwrap();
    url.base_url = Some(base.into());
    url.api_key = Some("dummy-secret".into());
    new_sound(&url, SoundOptions::default()).unwrap()
}

/// A fresh client resumes a serialized receipt and never submits while polling.
#[tokio::test]
async fn sound_receipt_survives_restart() {
    let (base, calls) = serve(vec![
        reply(
            json!({"request_id":"job-1","status_url":"BASE/fal-ai/elevenlabs/requests/job-1/status","response_url":"BASE/fal-ai/elevenlabs/requests/job-1"}),
        ),
        reply(json!({"status":"IN_QUEUE"})),
        reply(json!({"status":"COMPLETED"})),
        reply(json!({"audio":{"url":"https://media.example/result.mp3"}})),
    ]);
    let job = sound(&base)
        .submit_sound(
            &SoundRequest::new("rain".into())
                .with_duration_seconds(2.0)
                .with_looped(true),
        )
        .await
        .unwrap();
    let job = serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
    let fresh = sound(&base);
    assert!(matches!(
        fresh.get_sound(&job).await.unwrap(),
        AudioJobStatus::Queued { .. }
    ));
    assert!(matches!(
        fresh.get_sound(&job).await.unwrap(),
        AudioJobStatus::Succeeded { .. }
    ));
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 4);
    assert_eq!(calls.iter().filter(|c| c.starts_with("POST ")).count(), 1);
    let payload: Value = serde_json::from_str(calls[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["text"], "rain");
    assert_eq!(payload["loop"], true);
    assert_eq!(payload["output_format"], "mp3_44100_128");
}

/// Invalid controls and incompatible receipts fail locally, before any request.
#[tokio::test]
async fn sound_rejects_invalid_input_and_receipts() {
    let client = sound("http://127.0.0.1:1");
    for duration in [f64::NAN, 0.49, 22.01, f64::INFINITY] {
        let error = client
            .submit_sound(&SoundRequest::new("rain".into()).with_duration_seconds(duration))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    let mut job = AudioJob::new("job".into(), Provider::Fal);
    job.provider_model = Some(SOUND.into());
    job.status_url = Some("http://evil.example/status".into());
    job.response_url = Some("http://127.0.0.1:1/fal-ai/elevenlabs/requests/job".into());
    assert_eq!(
        client.get_sound(&job).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
}

fn music(base: &str, options: MusicOptions) -> MusicClient {
    let mut url = ModelUrl::parse(&format!("fal:///{MUSIC}")).unwrap();
    url.base_url = Some(base.into());
    url.api_key = Some("dummy-secret".into());
    new_music(&url, options).unwrap()
}

/// Music applies typed settings after native defaults and survives client reconstruction.
#[tokio::test]
async fn music_mapping_and_restart() {
    let (base, calls) = serve(vec![
        reply(
            json!({"request_id":"music-1","status_url":"BASE/fal-ai/elevenlabs/requests/music-1/status","response_url":"BASE/fal-ai/elevenlabs/requests/music-1"}),
        ),
        reply(json!({"status":"IN_PROGRESS"})),
        reply(json!({"status":"COMPLETED"})),
        reply(
            json!({"audio":{"url":"https://media.example/music.mp3","content_type":"audio/mpeg"}}),
        ),
    ]);
    let options = MusicOptions::default().with_provider_config(
        json!({"music_length_ms": 4000, "force_instrumental": true, "output_format":"pcm_8000"}),
    );
    let request = MusicRequest::new("jazz".into())
        .with_duration_seconds(3.1236)
        .with_instrumental(false)
        .with_provider_config(json!({"music_length_ms": 5000}));
    let receipt = music(&base, options).submit_music(&request).await.unwrap();
    let receipt: AudioJob =
        serde_json::from_str(&serde_json::to_string(&receipt).unwrap()).unwrap();
    let fresh = music(&base, MusicOptions::default());
    assert!(matches!(
        fresh.get_music(&receipt).await.unwrap(),
        AudioJobStatus::Running { .. }
    ));
    assert!(matches!(
        fresh.get_music(&receipt).await.unwrap(),
        AudioJobStatus::Succeeded { .. }
    ));
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 4);
    assert_eq!(calls.iter().filter(|c| c.starts_with("POST ")).count(), 1);
    let payload: Value = serde_json::from_str(calls[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["music_length_ms"], 3124);
    assert_eq!(payload["force_instrumental"], false);
    assert_eq!(payload["output_format"], "mp3_44100_128");
}

/// Invalid typed and native music controls fail before any network activity.
#[tokio::test]
async fn music_validation_precedes_submission() {
    let client = music("http://127.0.0.1:1", MusicOptions::default());
    for seconds in [f64::NAN, f64::INFINITY, 2.999, 600.001] {
        assert_eq!(
            client
                .submit_music(&MusicRequest::new("jazz".into()).with_duration_seconds(seconds))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
    for config in [
        json!({"music_length_ms": 2999}),
        json!({"music_length_ms": 3000.2}),
        json!({"force_instrumental":"yes"}),
        json!({"composition_plan":{}}),
    ] {
        assert_eq!(
            client
                .submit_music(&MusicRequest::new("jazz".into()).with_provider_config(config))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
}

/// Provider-declared failure remains distinct from transport errors and sanitizes evidence.
#[tokio::test]
async fn remote_failure_and_transport_errors() {
    let (base, _calls) = serve(vec![reply(
        json!({"status":"COMPLETED", "error":"failure dummy-secret"}),
    )]);
    let mut job = AudioJob::new("job".into(), Provider::Fal);
    job.provider_model = Some(SOUND.into());
    job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job/status"));
    job.response_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job"));
    let state = sound(&base).get_sound(&job).await.unwrap();
    assert!(matches!(state, AudioJobStatus::Failed { .. }));
    assert!(!format!("{state:?}").contains("dummy-secret"));
    job.status_url = Some("http://127.0.0.1:1/fal-ai/elevenlabs/requests/job/status".into());
    job.response_url = Some("http://127.0.0.1:1/fal-ai/elevenlabs/requests/job".into());
    let error = sound("http://127.0.0.1:1")
        .get_sound(&job)
        .await
        .unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::Transport | ErrorKind::Timeout
    ));
}

/// Malformed output and non-audio MIME never become successful audio.
#[tokio::test]
async fn malformed_result_is_error() {
    for output in [
        json!({"audio":{}}),
        json!({"audio":{"url":"https://media.example/out", "content_type":"text/html"}}),
    ] {
        let (base, _calls) = serve(vec![reply(json!({"status":"COMPLETED"})), reply(output)]);
        let mut job = AudioJob::new("job".into(), Provider::Fal);
        job.provider_model = Some(SOUND.into());
        job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job/status"));
        job.response_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job"));
        assert_eq!(
            sound(&base).get_sound(&job).await.unwrap_err().kind(),
            ErrorKind::InvalidResponse
        );
    }
}

/// Redirects are not followed with provider credentials attached.
#[tokio::test]
async fn queue_redirect_is_not_followed() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let target_addr = target.local_addr().unwrap();
    let source = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", source.local_addr().unwrap());
    let worker = thread::spawn(move || {
        let (mut conn, _) = source.accept().unwrap();
        let mut buffer = [0; 4096];
        let _ = conn.read(&mut buffer).unwrap();
        write!(conn, "HTTP/1.1 302 Found\r\nLocation: http://{target_addr}/stolen\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        thread::sleep(Duration::from_millis(100));
        assert!(target.accept().is_err());
    });
    let mut job = AudioJob::new("job".into(), Provider::Fal);
    job.provider_model = Some(SOUND.into());
    job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job/status"));
    job.response_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job"));
    assert!(sound(&base).get_sound(&job).await.is_err());
    worker.join().unwrap();
}

/// Typed sound controls override client and request defaults without mutating either.
#[tokio::test]
async fn sound_precedence_and_bounds() {
    let (base, calls) = serve(vec![reply(
        json!({"request_id":"job", "status_url":"BASE/fal-ai/elevenlabs/requests/job/status", "response_url":"BASE/fal-ai/elevenlabs/requests/job"}),
    )]);
    let mut url = ModelUrl::parse(&format!("fal:///{SOUND}")).unwrap();
    url.base_url = Some(base);
    url.api_key = Some("dummy-secret".into());
    let client = new_sound(
        &url,
        SoundOptions::default().with_provider_config(
            json!({"loop": false,"duration_seconds": 1.0, "output_format":"pcm_8000"}),
        ),
    )
    .unwrap();
    let request = SoundRequest::new("bell".into())
        .with_provider_config(json!({"loop":false, "duration_seconds":2.0}))
        .with_duration_seconds(22.0)
        .with_looped(true);
    client.submit_sound(&request).await.unwrap();
    let call = calls.recv().unwrap();
    let body: Value = serde_json::from_str(call.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["duration_seconds"], 22.0);
    assert_eq!(body["loop"], true);
    assert_eq!(body["output_format"], "mp3_44100_128");
    assert_eq!(request.provider_config.unwrap()["duration_seconds"], 2.0);
}

/// Receipts reject foreign models, unsafe URL syntax and unknown provider statuses.
#[tokio::test]
async fn receipt_validation_and_unknown_status() {
    let (base, _calls) = serve(vec![reply(json!({"status":"FUTURE_STATE"}))]);
    let client = sound(&base);
    let mut job = AudioJob::new("job".into(), Provider::Fal);
    job.provider_model = Some(MUSIC.into());
    job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job/status"));
    job.response_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job"));
    assert_eq!(
        client.get_sound(&job).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    job.provider_model = Some(SOUND.into());
    job.status_url = Some(format!(
        "{base}/fal-ai/elevenlabs/requests/job/status?token=secret"
    ));
    assert_eq!(
        client.get_sound(&job).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/job/status"));
    assert_eq!(
        client.get_sound(&job).await.unwrap_err().kind(),
        ErrorKind::InvalidResponse
    );
}

/// A submitted receipt cannot retain a configured credential in a queue URL.
#[tokio::test]
async fn rejects_credential_in_receipt() {
    let (base, _calls) = serve(vec![reply(
        json!({"request_id":"job", "status_url":"BASE/dummy-secret", "response_url":"BASE/result"}),
    )]);
    let error = sound(&base)
        .submit_sound(&SoundRequest::new("rain".into()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert!(!format!("{error:?}").contains("dummy-secret"));
}

/// Builds a correctly correlated audio receipt for a queue fixture.
fn job_for(base: &str, id: &str) -> AudioJob {
    let mut job = AudioJob::new(id.into(), Provider::Fal);
    job.provider_model = Some(SOUND.into());
    job.status_url = Some(format!("{base}/fal-ai/elevenlabs/requests/{id}/status"));
    job.response_url = Some(format!("{base}/fal-ai/elevenlabs/requests/{id}"));
    job
}

/// Same-origin URLs from different jobs and apps never receive authenticated polling.
#[tokio::test]
async fn rejects_mixed_receipt_urls_before_http() {
    let (base, calls) = serve(vec![reply(json!({"status":"IN_PROGRESS"}))]);
    let client = sound(&base);
    for (status, result) in [("job-b", "job-a"), ("job-a", "job-b")] {
        let mut job = job_for(&base, "job-a");
        job.status_url = job_for(&base, status).status_url;
        job.response_url = job_for(&base, result).response_url;
        assert_eq!(
            client.get_sound(&job).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
    let mut job = job_for(&base, "job-a");
    job.status_url = Some(format!("{base}/fal-ai/other/requests/job-a/status"));
    assert_eq!(
        client.get_sound(&job).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(calls.try_iter().count(), 0);
}

/// A rejected provider acknowledgement retains its job ID and sanitized response.
#[tokio::test]
async fn invalid_acknowledgement_keeps_reconciliation_evidence() {
    let (base, _calls) = serve(vec![reply(json!({
        "request_id":"job-a", "status_url":"BASE/fal-ai/elevenlabs/requests/job-b/status?token=dummy-secret",
        "response_url":"BASE/fal-ai/elevenlabs/requests/job-a", "private":"dummy-secret"
    }))]);
    let error = sound(&base)
        .submit_sound(&SoundRequest::new("rain".into()))
        .await
        .unwrap_err();
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    assert!(body.contains("job-a"));
    assert!(!body.contains("dummy-secret"));
    assert!(!format!("{error:?} {error}").contains("job-a"));
}

/// A body-shaped 422 without job-specific failure evidence remains an HTTP error.
#[tokio::test]
async fn ambiguous_model_validation_is_not_failed() {
    let (base, calls) = serve(vec![reply(json!({"status":"COMPLETED"})),
        (422, "application/json", json!({"detail":[{"loc":["body","prompt"],"type":"content_policy_violation","msg":"rejected dummy-secret"}]}).to_string())]);
    let error = sound(&base)
        .get_sound(&job_for(&base, "job"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(422));
    assert!(
        !String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("dummy-secret")
    );
    assert_eq!(calls.try_iter().count(), 2);
}

/// Gateway, path validation and authentication failures stay HTTP errors.
#[tokio::test]
async fn result_gateway_errors_are_not_terminal_failures() {
    for (status, body) in [
        (401, json!({"detail":"unauthorized"})),
        (429, json!({"detail":"rate limited"})),
        (500, json!({"detail":"server error"})),
        (
            422,
            json!({"detail":[{"loc":["path","request_id"],"type":"invalid","msg":"invalid"}]}),
        ),
        (422, json!({"detail":"unknown validation"})),
        (
            422,
            json!({"request_id":"job", "detail":[{"loc":["body","prompt"],"type":"invalid","msg":"invalid"}]}),
        ),
        (
            422,
            json!({"request_id":"other", "detail":[{"loc":["body","prompt"],"type":"invalid","msg":"invalid"}]}),
        ),
    ] {
        let (base, _calls) = serve(vec![
            reply(json!({"status":"COMPLETED"})),
            (status, "application/json", body.to_string()),
        ]);
        let err = sound(&base)
            .get_sound(&job_for(&base, "job"))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Http);
        assert_eq!(err.http_status(), Some(status));
    }
}

/// An explicitly identified failed job result remains terminal even with an HTTP error.
#[tokio::test]
async fn identified_failure_result_is_terminal() {
    for status in [422, 500] {
        let (base, calls) = serve(vec![
            reply(json!({"status":"COMPLETED", "request_id":"job"})),
            (
                status,
                "application/json",
                json!({"status":"COMPLETED", "request_id":"job", "error":"failed dummy-secret"})
                    .to_string(),
            ),
        ]);
        let state = sound(&base)
            .get_sound(&job_for(&base, "job"))
            .await
            .unwrap();
        assert!(matches!(state, AudioJobStatus::Failed { .. }));
        assert!(!format!("{state:?}").contains("dummy-secret"));
        assert_eq!(calls.try_iter().count(), 2);
    }
}

/// Responses from a different job never become the current job's status or failure.
#[tokio::test]
async fn mismatched_response_job_id_is_rejected() {
    let (base, _calls) = serve(vec![reply(
        json!({"status":"COMPLETED", "request_id":"other", "error":"wrong job"}),
    )]);
    assert_eq!(
        sound(&base)
            .get_sound(&job_for(&base, "job"))
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidResponse
    );
    let (base, _calls) = serve(vec![
        reply(json!({"status":"COMPLETED"})),
        (
            500,
            "application/json",
            json!({"status":"COMPLETED", "request_id":"other", "error":"wrong job"}).to_string(),
        ),
    ]);
    assert_eq!(
        sound(&base)
            .get_sound(&job_for(&base, "job"))
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Http
    );
}

/// Acknowledgement errors are distinct from local validation and preserve HTTP evidence.
#[tokio::test]
async fn acknowledged_response_preserves_wire_and_headers() {
    let raw = "{\n  \"request_id\": \"job-a\",\n  \"status_url\": \"BASE/fal-ai/elevenlabs/requests/job-b/status\",\n  \"response_url\": \"BASE/fal-ai/elevenlabs/requests/job-a\",\n  \"private\": \"dummy-secret\"\n}";
    let (base, calls) = serve(vec![(
        200,
        "application/json\r\nx-fal-request-id: header-42\r\nretry-after: 7",
        raw.into(),
    )]);
    let client = sound(&base);
    let local_error = client
        .submit_sound(&SoundRequest::new("".into()))
        .await
        .unwrap_err();
    assert_eq!(local_error.kind(), ErrorKind::Validation);
    assert_eq!(calls.try_iter().count(), 0);
    let error = client
        .submit_sound(&SoundRequest::new("rain".into()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert_eq!(error.operation(), Some("audio submit"));
    assert_eq!(error.http_status(), Some(200));
    assert_eq!(error.request_id(), Some("header-42"));
    assert_eq!(error.retry_after(), Some("7"));
    let body = error.response_body().unwrap();
    assert!(body.is_complete());
    let expected = raw
        .replace("BASE", &base)
        .replace("dummy-secret", "[REDACTED]");
    assert_eq!(body.bytes(), expected.as_bytes());
    assert!(!format!("{error:?} {error}").contains("dummy-secret"));
    assert_eq!(calls.try_iter().count(), 1);
}

/// Invalid provider receipt variants remain invalid responses after a paid submit.
#[tokio::test]
async fn invalid_acknowledgement_classification() {
    for body in [
        json!({"request_id":"../bad", "status_url":"BASE/status", "response_url":"BASE/result"}),
        json!({"request_id":"job", "status_url":"http://127.0.0.1:1/status", "response_url":"BASE/result"}),
        json!({"request_id":"job", "status_url":"BASE/fal-ai/elevenlabs/requests/job/status", "response_url":null}),
    ] {
        let (base, calls) = serve(vec![reply(body)]);
        let error = music(&base, MusicOptions::default())
            .submit_music(&MusicRequest::new("jazz".into()))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert_eq!(error.operation(), Some("audio submit"));
        assert_eq!(error.http_status(), Some(200));
        assert!(error.response_body().is_some());
        assert_eq!(calls.try_iter().count(), 1);
    }
}
