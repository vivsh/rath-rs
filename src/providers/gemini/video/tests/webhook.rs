use super::*;
use serde_json::json;

const NOW: u64 = 1_800_000_000;
const SECRET: &str = "whsec_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
const BODY: &[u8] = br#"{"type":"video.generated","version":"v1","data":{"id":"models/veo-3.1-fast-generate-preview/operations/job_123","output_file_uri":"https://generativelanguage.googleapis.com/v1beta/files/video:download?alt=media"}}"#;

fn url() -> ModelUrl {
    let mut url = ModelUrl::parse("gemini:///veo-3.1-fast-generate-preview").unwrap();
    url.api_key = Some("synthetic-api-key".into());
    url
}

/// Uses an independently computed Python stdlib HMAC fixture for Google's signed message format.
fn headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("webhook-id", "msg_01".parse().unwrap());
    headers.insert("webhook-timestamp", NOW.to_string().parse().unwrap());
    headers.insert(
        "webhook-signature",
        "v1,h3BRFtggaBUKDlFSZRrFtXY2gmHqck4+siAwLs8/I+U="
            .parse()
            .unwrap(),
    );
    headers
}

/// Signs varied synthetic envelopes for adversarial decoding and clock tests only.
fn signed(body: &[u8], seconds: u64) -> HeaderMap {
    let mut headers = headers();
    headers.insert("webhook-timestamp", seconds.to_string().parse().unwrap());
    let key = STANDARD
        .decode(SECRET.strip_prefix("whsec_").unwrap())
        .unwrap();
    let mut mac = Hmac::<Sha256>::new_from_slice(&key).unwrap();
    mac.update(format!("msg_01.{seconds}.").as_bytes());
    mac.update(body);
    headers.insert(
        "webhook-signature",
        format!("v1,{}", STANDARD.encode(mac.finalize().into_bytes()))
            .parse()
            .unwrap(),
    );
    headers
}

/// A reference signature yields the same public status as polling and retains caller-owned delivery identity.
#[test]
fn authenticates_reference_event_without_retaining_replay_state() {
    for _ in 0..2 {
        let event = parse_at(&url(), &headers(), BODY, SECRET, NOW).unwrap();
        assert_eq!(
            event.job_id,
            "models/veo-3.1-fast-generate-preview/operations/job_123"
        );
        let VideoJobStatus::Succeeded { response } = event.status else {
            panic!("expected completed video");
        };
        assert_eq!(response.videos.len(), 1);
        let metadata = response.raw_metadata.unwrap();
        assert_eq!(metadata["webhook_id"], "msg_01");
        assert_eq!(
            metadata["event"],
            serde_json::from_slice::<Value>(BODY).unwrap()
        );
    }
}

/// Tampered bytes, wrong credentials and a different message ID cannot produce authenticated output.
#[test]
fn rejects_body_header_and_secret_tampering_before_decoding() {
    let mut altered = headers();
    altered.insert("webhook-id", "msg_02".parse().unwrap());
    for (headers, body, secret) in [
        (headers(), b"{not-json".as_slice(), SECRET),
        (altered, BODY, SECRET),
        (headers(), BODY, "whsec_d3Jvbmcta2V5"),
        (headers(), BODY, "not-base64"),
        (headers(), BODY, "whsec_"),
    ] {
        let error = parse_at(&url(), &headers, body, secret, NOW).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.response_body().is_none());
        assert!(!format!("{error:?}").contains(secret));
    }
}

/// Duplicate or missing signed headers are rejected instead of silently selecting one value.
#[test]
fn rejects_ambiguous_signed_headers() {
    for name in ["webhook-id", "webhook-timestamp", "webhook-signature"] {
        let mut missing = headers();
        missing.remove(name);
        assert!(parse_at(&url(), &missing, BODY, SECRET, NOW).is_err());
        let mut duplicate = headers();
        let value = duplicate[name].clone();
        duplicate.append(name, value);
        assert!(parse_at(&url(), &duplicate, BODY, SECRET, NOW).is_err());
    }
    for value in [
        "-1",
        "+1800000000",
        "1800000000 ",
        "999999999999999999999999",
    ] {
        let mut headers = headers();
        headers.insert("webhook-timestamp", value.parse().unwrap());
        assert!(parse_at(&url(), &headers, BODY, SECRET, NOW).is_err());
    }
}

/// Replay freshness accepts the boundary and rejects timestamps more than five minutes in either direction.
#[test]
fn enforces_timestamp_window_in_both_directions() {
    for offset in [0, 299, 300] {
        for seconds in [NOW - offset, NOW + offset] {
            assert!(parse_at(&url(), &signed(BODY, seconds), BODY, SECRET, NOW).is_ok());
        }
    }
    for seconds in [NOW - 301, NOW + 301] {
        let error = parse_at(&url(), &signed(BODY, seconds), BODY, SECRET, NOW).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.response_body().is_none());
    }
}

/// Standard Webhooks signature rotation may carry several versioned candidates in one header.
#[test]
fn accepts_a_valid_signature_among_rotation_candidates() {
    let mut headers = headers();
    let good = headers["webhook-signature"].to_str().unwrap().to_owned();
    headers.insert(
        "webhook-signature",
        format!("v2,ignored v1,AAAA {good}").parse().unwrap(),
    );
    assert!(parse_at(&url(), &headers, BODY, SECRET, NOW).is_ok());
    headers.insert("webhook-signature", "v2,ignored v1,AAAA".parse().unwrap());
    assert!(parse_at(&url(), &headers, BODY, SECRET, NOW).is_err());
}

/// Only a well-formed authenticated completion for the bound model can become a video event.
#[test]
fn rejects_unrelated_incomplete_or_rerouted_envelopes() {
    let valid: Value = serde_json::from_slice(BODY).unwrap();
    for (path, value) in [
        ("/type", json!("batch.succeeded")),
        ("/version", json!("v2")),
        ("/data/id", json!("models/other/operations/job_123")),
        ("/data/id", json!("../../attacker")),
        ("/data/output_file_uri", Value::Null),
        (
            "/data/output_file_uri",
            json!("https://user:secret@example.test/video"),
        ),
        (
            "/data/output_file_uri",
            json!("https://example.test/video#secret"),
        ),
    ] {
        let mut raw = valid.clone();
        *raw.pointer_mut(path).unwrap() = value;
        let body = serde_json::to_vec(&raw).unwrap();
        let error = parse_at(&url(), &signed(&body, NOW), &body, SECRET, NOW).unwrap_err();
        assert!(error.response_body().is_some());
    }
    let body = b"{not-json";
    let error = parse_at(&url(), &signed(body, NOW), body, SECRET, NOW).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Deserialize);
    assert!(error.response_body().is_some());
}

/// Authenticated diagnostics redact both the API key and prefixed/unprefixed signing credentials.
#[test]
fn redacts_credentials_from_authenticated_error_evidence() {
    let body = serde_json::to_vec(
        &json!({"type":"wrong", "api_key":"synthetic-api-key", "note":SECRET,
        "encoded":SECRET.strip_prefix("whsec_").unwrap()}),
    )
    .unwrap();
    let error = parse_at(&url(), &signed(&body, NOW), &body, SECRET, NOW).unwrap_err();
    let evidence = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    for secret in [
        SECRET,
        SECRET.strip_prefix("whsec_").unwrap(),
        "synthetic-api-key",
    ] {
        assert!(!evidence.contains(secret));
        assert!(!format!("{error:?}").contains(secret));
    }
}

/// Parser credential configuration is local, optional and rejects values that are not variable names.
#[test]
fn validates_optional_credential_selection() {
    assert!(configured_env(&VideoOptions::default()).unwrap().is_none());
    for value in [
        json!(""),
        json!("bad name"),
        json!("whsec_secret="),
        json!([]),
        Value::Null,
    ] {
        let options = VideoOptions::default().with_provider_config(json!({ENV_SETTING:value}));
        assert!(configured_env(&options).is_err());
    }
}

/// Real backend parsing resolves only the explicitly chosen variable in isolated child environments.
#[test]
fn resolves_parser_credentials_without_process_global_mutation() {
    const MODE: &str = "RATH_GEMINI_VIDEO_WEBHOOK_TEST_MODE";
    const KEY_ENV: &str = "RATH_GEMINI_VIDEO_WEBHOOK_TEST_SECRET";
    if let Ok(mode) = std::env::var(MODE) {
        let options = VideoOptions::default().with_provider_config(json!({ENV_SETTING:KEY_ENV}));
        let client = super::super::new_video_client(&url(), options).unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(client.parse_webhook(&signed(BODY, now), BODY));
        if mode == "valid" {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
        return;
    }
    for mode in ["valid", "missing", "invalid"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "providers::gemini::video::webhook::tests::resolves_parser_credentials_without_process_global_mutation"])
            .env(MODE,mode).env_remove(KEY_ENV);
        if mode == "valid" {
            command.env(KEY_ENV, SECRET);
        }
        if mode == "invalid" {
            command.env(KEY_ENV, "not-base64");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "isolated parser credential case {mode} failed"
        );
    }
}
