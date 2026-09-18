use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use super::super::*;
use serde_json::json;

/// Runs one bounded local response with caller-selected headers and exact bytes.
fn serve(status: u16, headers: &str, body: Vec<u8>, declared: Option<usize>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let headers = headers.to_string();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        let mut request = [0; 4096];
        let _ = stream.read(&mut request);
        write!(
            stream,
            "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",
            declared.unwrap_or(body.len())
        )
        .unwrap();
        stream.write_all(&body).unwrap();
    });
    address
}

/// HTTP failures retain useful metadata and provider messages, while bodies remain explicit.
#[tokio::test]
async fn http_details_and_redaction() {
    let body = json!({"error":{"code":"quota", "message":"quota exceeded KEY/123"},
        "unknown":"PRIVATE-PROSE", "password":"do-not-retain"})
    .to_string()
    .into_bytes();
    let url = serve(
        429,
        "X-Request-Id: request-42\r\nRetry-After: 15\r\nSet-Cookie: secret-cookie\r\n",
        body,
        None,
    );
    let error = http::mapped(
        reqwest::Client::new().get(url),
        Provider::OpenAi,
        "generation",
        &["KEY/123"],
        Ok,
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(429));
    assert_eq!(error.request_id(), Some("request-42"));
    assert_eq!(error.retry_after(), Some("15"));
    assert_eq!(error.provider_code(), Some("quota"));
    assert!(error.to_string().contains("quota exceeded [REDACTED]"));
    let formatted = format!("{error} {error:?}");
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    for secret in ["KEY/123", "do-not-retain", "secret-cookie"] {
        assert!(!formatted.contains(secret));
        assert!(!body.contains(secret));
    }
    assert!(!formatted.contains("PRIVATE-PROSE"));
    assert!(body.contains("PRIVATE-PROSE"));
}

/// Interrupted transfers keep already-received bytes, status, and the transport cause chain.
#[tokio::test]
async fn interrupted_body_is_retained() {
    let url = serve(
        500,
        "X-Request-Id: interrupted\r\n",
        b"partial-response".to_vec(),
        Some(999),
    );
    let error = http::mapped(
        reqwest::Client::new().get(url),
        Provider::Fal,
        "result",
        &[],
        Ok,
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(error.http_status(), Some(500));
    assert_eq!(error.request_id(), Some("interrupted"));
    assert_eq!(error.response_body().unwrap().bytes(), b"partial-response");
    assert!(!error.response_body().unwrap().is_complete());
    assert!(error.source().is_some());
}

/// Malformed successful responses retain status, exact non-UTF-8 bytes, and decoding causes.
#[tokio::test]
async fn malformed_json_keeps_exact_bytes() {
    let bytes = vec![0xff, 0x80, b'{', b'x'];
    let url = serve(200, "", bytes.clone(), None);
    let error = http::mapped(
        reqwest::Client::new().get(url),
        Provider::Anthropic,
        "generation",
        &[],
        Ok,
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Deserialize);
    assert_eq!(error.http_status(), Some(200));
    assert_eq!(error.response_body().unwrap().bytes(), bytes);
    assert!(error.response_body().unwrap().is_complete());
    assert!(error.source().is_some());
}

/// Semantic failures retain the successful HTTP response metadata and original wire body.
#[tokio::test]
async fn missing_field_keeps_response_metadata() {
    let bytes = br#"{ "extra": "keep this" }"#.to_vec();
    let url = serve(200, "Request-Id: r-1\r\n", bytes.clone(), None);
    let error = http::mapped::<()>(
        reqwest::Client::new().get(url),
        Provider::Ollama,
        "embeddings",
        &[],
        |value| Err(RathError::invalid("missing embeddings", &value)),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert_eq!(error.request_id(), Some("r-1"));
    assert_eq!(error.response_body().unwrap().bytes(), bytes);
}

/// Non-ASCII request identifiers survive as reversible byte escapes instead of disappearing.
#[tokio::test]
async fn non_text_metadata_is_preserved() {
    let url = serve(400, "X-Request-Id: req-ÿ\r\n", b"{}".to_vec(), None);
    let error = http::mapped(
        reqwest::Client::new().get(url),
        Provider::Fal,
        "result",
        &[],
        Ok,
    )
    .await
    .unwrap_err();
    assert_eq!(error.request_id(), Some(r"req-\xc3\xbf"));
}
