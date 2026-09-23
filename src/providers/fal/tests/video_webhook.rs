use super::super::tests::http::{json as reply, serve};
use super::*;
use crate::core::ModelUrl;
use ed25519_dalek::{Signer, SigningKey};
use serde_json::json;

const NOW: u64 = 1_800_000_000;

/// Uses explicit fake credentials and never opens a production service.
fn client() -> FalClient {
    let mut url = ModelUrl::parse("fal:///fal-ai/kling-video/v3/pro/image-to-video").unwrap();
    url.api_key = Some("FAKE-SECRET".into());
    FalClient::new(&url, None, "video").unwrap()
}

fn key() -> SigningKey {
    SigningKey::from_bytes(&[7; 32])
}

fn jwks() -> Value {
    json!({"keys":[{"kty":"OKP","crv":"Ed25519","use":"sig","x":URL_SAFE_NO_PAD.encode(key().verifying_key().as_bytes())}]})
}

/// Builds a deterministic signed notification over exactly the supplied raw bytes.
fn headers(body: &[u8], timestamp: u64) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-fal-webhook-request-id", "attempt-2".parse().unwrap());
    headers.insert("x-fal-webhook-user-id", "test-user".parse().unwrap());
    headers.insert(
        "x-fal-webhook-timestamp",
        timestamp.to_string().parse().unwrap(),
    );
    let hash = Sha256::digest(body);
    let message = format!("attempt-2\ntest-user\n{timestamp}\n{hash:x}");
    let signature = key().sign(message.as_bytes());
    let encoded: String = signature
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    headers.insert("x-fal-webhook-signature", encoded.parse().unwrap());
    headers
}

fn success_body() -> Vec<u8> {
    serde_json::to_vec(
        &json!({"request_id":"stable-job","gateway_request_id":"attempt-2",
        "status":"OK","payload":{"video":{"url":"https://assets.example/out.mp4"}}}),
    )
    .unwrap()
}

/// Verifies the full path and retains stable job identity rather than gateway retry identity.
#[tokio::test]
async fn signed_event_fetches_keys_without_credentials() {
    let body = success_body();
    let (base, calls) = serve(vec![reply(jwks())]);
    let event = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap();
    assert_eq!(event.job_id, "stable-job");
    let VideoJobStatus::Succeeded { response } = event.status else {
        panic!("expected success")
    };
    assert_eq!(response.videos.len(), 1);
    assert_eq!(
        response.raw_metadata.unwrap()["gateway_request_id"],
        "attempt-2"
    );
    let calls: Vec<_> = calls.try_iter().collect();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].to_lowercase().contains("authorization"));
}

/// Duplicate valid events yield equal values and fetch keys afresh rather than retaining state.
#[tokio::test]
async fn duplicate_delivery_is_stateless() {
    let body = success_body();
    let (base, calls) = serve(vec![reply(jwks()), reply(jwks())]);
    let first = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap();
    let second = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    assert_eq!(calls.try_iter().count(), 2);
}

/// Signed ERROR envelopes share polling's failure normalization and retain authenticated evidence.
#[tokio::test]
async fn signed_failure_matches_polling() {
    let raw = json!({"request_id":"job","status":"ERROR","error":"bad generation FAKE-SECRET","private":"evidence"});
    let body = serde_json::to_vec(&raw).unwrap();
    let (base, calls) = serve(vec![reply(jwks())]);
    let event = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap();
    let expected = video::video_failure(&raw, &client()).unwrap();
    assert_eq!(
        serde_json::to_value(event.status).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    assert_eq!(calls.try_iter().count(), 1);
}

/// Tampering with raw bytes or any signed identity fails even when the JSON meaning is unchanged.
#[test]
fn tampering_fails_signature_verification() {
    let body = success_body();
    let good = headers(&body, NOW);
    let keys = decode_keys(jwks()).unwrap();
    let mut changed_body = body.clone();
    changed_body.push(b' ');
    let (message, sig) = signed_message(&good, &changed_body, NOW).unwrap();
    assert_eq!(
        verify(&keys, &message, &sig).unwrap_err().kind(),
        ErrorKind::Validation
    );
    for field in [
        "x-fal-webhook-request-id",
        "x-fal-webhook-user-id",
        "x-fal-webhook-timestamp",
    ] {
        let mut altered = good.clone();
        altered.insert(
            field,
            if field.ends_with("timestamp") {
                "1800000001"
            } else {
                "other"
            }
            .parse()
            .unwrap(),
        );
        let (message, sig) = signed_message(&altered, &body, NOW).unwrap();
        assert!(verify(&keys, &message, &sig).is_err());
    }
}

/// All required headers reject omission and duplication before fetching keys.
#[test]
fn missing_and_duplicate_signed_headers_fail() {
    let body = success_body();
    for name in [
        "x-fal-webhook-request-id",
        "x-fal-webhook-user-id",
        "x-fal-webhook-timestamp",
        "x-fal-webhook-signature",
    ] {
        let mut h = headers(&body, NOW);
        h.remove(name);
        assert!(signed_message(&h, &body, NOW).is_err());
        let mut h = headers(&body, NOW);
        let value = h[name].clone();
        h.append(name, value);
        assert!(signed_message(&h, &body, NOW).is_err());
    }
}

/// Rejects malformed timestamp, signature and identity values without interpreting body text.
#[test]
fn malformed_headers_fail_safely() {
    let body = b"PRIVATE UNAUTHENTICATED BODY";
    for (name, value) in [
        ("x-fal-webhook-timestamp", "-1"),
        ("x-fal-webhook-timestamp", "+1800000000"),
        ("x-fal-webhook-timestamp", "999999999999999999999999999999"),
        ("x-fal-webhook-signature", "abcdef"),
        ("x-fal-webhook-signature", "not-hex"),
        ("x-fal-webhook-user-id", ""),
        ("x-fal-webhook-request-id", "two words"),
    ] {
        let mut h = headers(body, NOW);
        h.insert(name, value.parse().unwrap());
        let error = signed_message(&h, body, NOW).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.response_body().is_none());
        assert!(!format!("{error} {error:?}").contains("PRIVATE"));
    }
}

/// Accepts exact timestamp boundaries and rejects stale or future notifications.
#[test]
fn timestamp_window_is_symmetric() {
    let body = success_body();
    for time in [NOW - 300, NOW + 300] {
        assert!(signed_message(&headers(&body, time), &body, NOW).is_ok());
    }
    for time in [NOW - 301, NOW + 301] {
        assert!(signed_message(&headers(&body, time), &body, NOW).is_err());
    }
}

/// Rotation tries every valid key, ignoring unrelated key types and malformed entries.
#[test]
fn key_rotation_and_invalid_signatures() {
    let body = success_body();
    let (message, sig) = signed_message(&headers(&body, NOW), &body, NOW).unwrap();
    let wrong = SigningKey::from_bytes(&[8; 32]);
    let rotated = json!({"keys":[{"x":"bad"},{"kty":"RSA","x":"bad"},
        {"x":URL_SAFE_NO_PAD.encode(wrong.verifying_key().as_bytes())}, jwks()["keys"][0]]});
    assert!(verify(&decode_keys(rotated).unwrap(), &message, &sig).is_ok());
    assert!(verify(&[wrong.verifying_key()], &message, &sig).is_err());
    assert!(
        verify(
            &decode_keys(jwks()).unwrap(),
            &message,
            &Signature::from_bytes(&[0; 64])
        )
        .is_err()
    );
}

/// Missing or unusable keys are invalid provider responses, not authentication successes.
#[test]
fn malformed_jwks_rejected() {
    for raw in [
        json!({}),
        json!({"keys":[]}),
        json!({"keys":[{"x":"!"}]}),
        json!({"keys":[{"crv":"X25519","x":jwks()["keys"][0]["x"]}]}),
    ] {
        assert_eq!(
            decode_keys(raw).unwrap_err().kind(),
            ErrorKind::InvalidResponse
        );
    }
}

/// Incomplete success, unexpected statuses and malformed authenticated JSON never fabricate failure.
#[tokio::test]
async fn incomplete_authenticated_results_are_errors() {
    let bodies = [
        br#"{"request_id":"job","status":"OK","payload":null,"payload_error":"missing"}"#
            .as_slice(),
        br#"{"request_id":"job","status":"OK","payload":{}}"#,
        br#"{"request_id":"job","status":"ALIEN"}"#,
        br#"{"request_id":"job","status":"ERROR"}"#,
        br#"{"request_id":"","status":"OK"}"#,
        br#"{"request_id":"job","status":"OK","payload":{"error":"failure"}}"#,
        br#"{"request_id":"job","status":"OK","error":"failure","payload":{}}"#,
        b"not json PRIVATE FAKE-SECRET",
    ];
    for body in bodies {
        let (base, calls) = serve(vec![reply(jwks())]);
        let err = parse_at(&client(), &headers(body, NOW), body, &base, NOW)
            .await
            .unwrap_err();
        assert!(matches!(
            err.kind(),
            ErrorKind::InvalidResponse | ErrorKind::Deserialize
        ));
        assert!(!format!("{err} {err:?}").contains("PRIVATE"));
        assert!(
            !String::from_utf8_lossy(err.response_body().unwrap().bytes()).contains("FAKE-SECRET")
        );
        assert_eq!(calls.try_iter().count(), 1);
    }
}

/// JWKS failures retain HTTP/transport categories rather than becoming signature failures.
#[tokio::test]
async fn key_fetch_failures_preserve_categories() {
    let body = success_body();
    let (base, _calls) = serve(vec![(
        503,
        "application/json",
        r#"{"error":"keys unavailable"}"#.into(),
    )]);
    let err = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Http);
    assert_eq!(err.http_status(), Some(503));
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    drop(socket);
    let err = parse_at(
        &client(),
        &headers(&body, NOW),
        &body,
        &format!("http://{address}"),
        NOW,
    )
    .await
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Transport);
}

/// Public parsing rejects an invalid notification before reaching the fixed JWKS service.
#[tokio::test]
async fn public_parser_preflight_never_fetches_untrusted_keys() {
    let err = parse(&client(), &HeaderMap::new(), b"PRIVATE")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.response_body().is_none());
}

/// Key-service redirects are rejected without following even a valid alternative JWKS response.
#[tokio::test]
async fn key_service_redirects_are_not_followed() {
    let body = success_body();
    let (sink, sink_calls) = serve(vec![reply(jwks())]);
    let mime = format!("text/plain\r\nLocation: {sink}");
    let (base, calls) = serve(vec![(302, &mime, "redirect".into())]);
    let error = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(302));
    assert_eq!(calls.try_iter().count(), 1);
    assert_eq!(sink_calls.try_iter().count(), 0);
}

/// Gateway retry IDs do not change stable identity or the output decoded by polling.
#[tokio::test]
async fn gateway_retries_preserve_identity_and_polling_output() {
    let mut raw: Value = serde_json::from_slice(&success_body()).unwrap();
    let VideoJobStatus::Succeeded { response: polled } =
        video::completed_video(raw["payload"].clone(), &client()).unwrap()
    else {
        panic!("expected polling success")
    };
    let (base, calls) = serve(vec![reply(jwks()), reply(jwks())]);
    for gateway in ["attempt-2", "attempt-3"] {
        raw["gateway_request_id"] = json!(gateway);
        let body = serde_json::to_vec(&raw).unwrap();
        let event = parse_at(&client(), &headers(&body, NOW), &body, &base, NOW)
            .await
            .unwrap();
        assert_eq!(event.job_id, "stable-job");
        let VideoJobStatus::Succeeded { response } = event.status else {
            panic!("expected webhook success")
        };
        assert_eq!(
            serde_json::to_value(response.videos).unwrap(),
            serde_json::to_value(&polled.videos).unwrap()
        );
        assert_eq!(
            response.raw_metadata.unwrap()["gateway_request_id"],
            gateway
        );
    }
    assert_eq!(calls.try_iter().count(), 2);
}
