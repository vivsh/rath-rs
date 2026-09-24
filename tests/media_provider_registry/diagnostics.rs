use rath::{ErrorBody, ErrorKind, Provider, RathError};

/// Public error construction preserves normalized source order, binary bodies and metadata.
#[test]
fn public_diagnostic_snapshots_are_complete() {
    #[derive(Debug, thiserror::Error)]
    #[error("outer: {0}")]
    struct Native(#[source] std::io::Error);
    let native = Native(std::io::Error::other("inner SECRET/a+b"));
    let error = RathError::from_error(ErrorKind::Transport, &native, &["SECRET/a+b"])
        .with_context(Provider::External("acme".into()), "download")
        .with_http_status(503)
        .with_provider_code("unavailable", &[])
        .with_request_id("request-1", &[])
        .with_retry_after("20", &[])
        .with_body(ErrorBody::Incomplete(vec![0xff, 0, b'A']), &[]);
    assert_eq!(error.message(), "outer");
    assert_eq!(error.source().unwrap().message(), "inner [REDACTED]");
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "inner [REDACTED]"
    );
    assert_eq!(error.http_status(), Some(503));
    assert_eq!(error.provider_code(), Some("unavailable"));
    assert_eq!(error.request_id(), Some("request-1"));
    assert_eq!(error.retry_after(), Some("20"));
    assert_eq!(error.response_body().unwrap().bytes(), [0xff, 0, b'A']);
    assert!(!error.response_body().unwrap().is_complete());
}

/// Known credentials are absent from all public diagnostic fields, including external identity.
#[test]
fn public_diagnostics_redact_secrets_without_hiding_private_body_access() {
    let secret = "SECRET/a+b";
    let error = RathError::new(ErrorKind::Http, format!("bad credential {secret}"))
        .with_context(Provider::External(format!("acme-{secret}")), "synthesis")
        .with_source(RathError::new(ErrorKind::Other, secret)
            .with_context(Provider::External(secret.into()), "source"))
        .with_provider_code(secret, &[secret]).with_request_id(secret, &[secret])
        .with_retry_after(secret, &[secret])
        .with_body(ErrorBody::Complete(b"{\"detail\":\"SECRET%2Fa%2Bb\",\"cookie\":\"hidden\",\"private\":\"PRIVATE-MEDIA\"}".to_vec()), &[secret]);
    let rendered = format!("{error} {error:?} {:?}", error.source());
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    for marker in [secret, "SECRET%2Fa%2Bb", "hidden"] {
        assert!(!rendered.contains(marker));
        assert!(!body.contains(marker));
    }
    assert!(!rendered.contains("PRIVATE-MEDIA"));
    assert!(body.contains("PRIVATE-MEDIA"));
    assert_eq!(error.provider().unwrap().as_str(), "acme-[REDACTED]");
    assert_eq!(error.provider_code(), Some("[REDACTED]"));
    assert_eq!(error.request_id(), Some("[REDACTED]"));
    assert_eq!(error.retry_after(), Some("[REDACTED]"));
    assert!(error.response_body().unwrap().is_complete());
}

/// Ordinary tracing never prints retained media or credentials in external provider identities.
#[test]
fn external_error_logging_is_redacted() {
    let path = std::env::temp_dir().join(format!("rath-external-log-{}.txt", uuid::Uuid::now_v7()));
    let file = std::sync::Arc::new(std::fs::File::create(&path).unwrap());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(file)
        .finish();
    let error = RathError::new(ErrorKind::Http, "voice unavailable SECRET-CREDENTIAL")
        .with_context(
            Provider::External("acme-SECRET-CREDENTIAL".into()),
            "synthesis",
        )
        .with_body(
            ErrorBody::Complete(b"PRIVATE-MEDIA SECRET-CREDENTIAL".to_vec()),
            &["SECRET-CREDENTIAL"],
        );
    tracing::subscriber::with_default(subscriber, || {
        tracing::error!(error = ?error, display = %error, "synthesis failed");
    });
    let log = std::fs::read_to_string(&path).unwrap();
    assert!(log.contains("voice unavailable"));
    assert!(!log.contains("PRIVATE-MEDIA"));
    assert!(!log.contains("SECRET-CREDENTIAL"));
    std::fs::remove_file(path).unwrap();
}
