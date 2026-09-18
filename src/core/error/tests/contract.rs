use super::super::*;
use serde_json::json;

/// Wrapping preserves each cause separately and formats identical messages only once.
#[test]
fn cause_chain_is_inspectable() {
    let error = RathError::new(ErrorKind::Provider, "dispatch failed")
        .with_context(Provider::Fal, "audio submit")
        .with_source(
            RathError::new(ErrorKind::Transport, "connection refused")
                .with_source(RathError::new(ErrorKind::Other, "connection refused")),
        );
    assert_eq!(error.source().unwrap().message(), "connection refused");
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(error.to_string().matches("connection refused").count(), 1);
    assert!(error.source().unwrap().source().is_some());
}

/// Retained bodies preserve completeness and binary bytes but remain absent from formatting.
#[test]
fn binary_body_is_explicit() {
    let bytes = vec![0xff, 0, 128, b'x'];
    for body in [
        ErrorBody::Complete(bytes.clone()),
        ErrorBody::Incomplete(bytes.clone()),
    ] {
        let complete = body.is_complete();
        let error = RathError::new(ErrorKind::Deserialize, "bad response").with_body(body);
        assert_eq!(error.response_body().unwrap().bytes(), bytes);
        assert_eq!(error.response_body().unwrap().is_complete(), complete);
        assert!(!format!("{error:?}").contains("128"));
    }
}

/// Known credentials and encoded forms disappear from explicit bodies as well as all cause levels.
#[test]
fn sanitization_covers_all_retained_fields() {
    let key = "SECRET/a+b";
    let body = json!({"error":{"message":"invalid voice", "code":"bad_voice"},
        "api_key":"unknown-secret", "detail":"SECRET/a+b SECRET%2Fa%2Bb", "payload":"PRIVATE-PROSE"});
    let error = RathError::new(ErrorKind::Provider, key)
        .with_source(RathError::new(
            ErrorKind::Other,
            "https://u:p@host.test/path?token=secret#fragment",
        ))
        .with_response(&body)
        .sanitized(&[key]);
    let rendered = format!("{error} {error:?} {:?}", error.source());
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    for value in [
        key,
        "SECRET%2Fa%2Bb",
        "unknown-secret",
        "u:p",
        "token=secret",
        "#fragment",
    ] {
        assert!(!rendered.contains(value), "{rendered}");
        assert!(!body.contains(value), "{body}");
    }
    assert!(body.contains("PRIVATE-PROSE"));
    assert!(!rendered.contains("PRIVATE-PROSE"));
}

/// Structured provider messages remain visible while the entire response is explicitly accessible.
#[test]
fn provider_failure_extracts_useful_fields() {
    let error = provider_failure(
        Provider::Fal,
        "audio execution",
        &json!({
            "error":{"message":"voice unavailable", "code":"invalid_voice"},
            "request_id":"req-123", "private":"PRIVATE-PROSE"
        }),
        &[],
    );
    assert_eq!(error.provider_code(), Some("invalid_voice"));
    assert_eq!(error.request_id(), Some("req-123"));
    assert!(error.to_string().contains("voice unavailable"));
    assert!(!format!("{error:?}").contains("PRIVATE-PROSE"));
}

/// Redaction handles lowercase percent escapes, JSON quoting, form escaping and nested secret values.
#[test]
fn escaped_and_structured_secrets_are_removed() {
    let key = "UPPER/a+b c\"";
    let raw = r#"{"details":"UPPER%2fa%2bb+c%22 UPPER/a+b c\"","cookie":{"a":"COOKIE","b":["SECOND"]},"safe":{"count":2}}"#;
    let error = RathError::new(ErrorKind::Http, "unavailable")
        .with_body(ErrorBody::Complete(raw.as_bytes().to_vec()))
        .sanitized(&[key]);
    let body = std::str::from_utf8(error.response_body().unwrap().bytes()).unwrap();
    for marker in ["UPPER", "COOKIE", "SECOND"] {
        assert!(!body.contains(marker), "{body}");
    }
    let decoded: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(decoded["safe"]["count"], 2);
    assert_eq!(decoded["cookie"], "[REDACTED]");
}

/// Native errors that embed the source text do not duplicate it in Rath's normalized chain.
#[test]
fn native_context_is_not_flattened() {
    #[derive(Debug, thiserror::Error)]
    #[error("dispatch failed: {0}")]
    struct Native(#[source] std::io::Error);
    let native = Native(std::io::Error::other("connection refused"));
    let error = RathError::from_error(ErrorKind::Transport, &native);
    assert_eq!(error.message(), "dispatch failed");
    assert_eq!(error.source().unwrap().message(), "connection refused");
    assert_eq!(error.to_string(), "dispatch failed: connection refused");
}

/// Escaped JSON field names and escaped-slash URLs cannot bypass credential-location redaction.
#[test]
fn escaped_json_locations_are_redacted() {
    let raw = r#"{"api\u005fkey":"HIDDENKEY","url":"https:\/\/user:PASS@host.test\/path?signature=VALUE#FRAGMENT","other":3}"#;
    let error = RathError::new(ErrorKind::Http, "rejected")
        .with_body(ErrorBody::Complete(raw.as_bytes().to_vec()));
    let body = std::str::from_utf8(error.response_body().unwrap().bytes()).unwrap();
    for marker in ["HIDDENKEY", "PASS", "VALUE", "FRAGMENT"] {
        assert!(!body.contains(marker), "{body}");
    }
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(value["other"], 3);
    assert_eq!(
        value["url"],
        "https://host.test/path?signature=[REDACTED]#[REDACTED]"
    );
}

/// Invalid UTF-8 inside a credential field cannot expose the remainder of that field.
#[test]
fn binary_secret_field_is_redacted_as_a_whole() {
    let raw = b"{\"password\":\"HIDDEN-PREFIX \xff HIDDEN-TAIL\",\"other\":\"\xfe KEEP\"}";
    let error =
        RathError::new(ErrorKind::Http, "rejected").with_body(ErrorBody::Complete(raw.to_vec()));
    let body = error.response_body().unwrap().bytes();
    assert_eq!(
        body,
        b"{\"password\":\"[REDACTED]\",\"other\":\"\xfe KEEP\"}"
    );
}

/// Fal-style validation envelopes expose their useful messages without implicitly printing input data.
#[test]
fn structured_validation_messages_are_visible() {
    let error = sdk_http(
        422,
        Some(
            r#"{"detail":[{"loc":["body","voice"],"msg":"voice is unsupported","input":"PRIVATE-VOICE"}]}"#,
        ),
        &[],
    );
    assert!(error.to_string().contains("voice is unsupported"));
    assert!(!format!("{error:?}").contains("PRIVATE-VOICE"));
    assert!(
        String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("PRIVATE-VOICE")
    );
}
