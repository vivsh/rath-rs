use crate::core::{Provider, RathError};

/// Normal diagnostic formatting and error chaining never disclose partial output.
#[test]
fn partial_response_is_redacted_but_accessible() {
    let error = RathError::OutputLimitReached {
        provider: Provider::OpenAi,
        response: serde_json::json!({"output": "PRIVATE-CONTENT-729"}),
    };
    assert!(!format!("{error} {error:?} {error:#?}").contains("PRIVATE-CONTENT-729"));
    assert!(std::error::Error::source(&error).is_none());
    match error {
        RathError::OutputLimitReached { response, .. } => {
            assert_eq!(response["output"], "PRIVATE-CONTENT-729")
        }
        _ => panic!("expected partial response"),
    }
}

/// A normal tracing event formats the error without logging raw partial content.
#[test]
fn tracing_redacts_partial_response() {
    let path = std::env::temp_dir().join(format!("rath-log-{}.txt", uuid::Uuid::now_v7()));
    let file = std::sync::Arc::new(std::fs::File::create(&path).unwrap());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(file)
        .finish();
    let error = RathError::OutputLimitReached {
        provider: Provider::Gemini,
        response: serde_json::json!({"text":"PRIVATE-CONTENT-729"}),
    };
    tracing::subscriber::with_default(subscriber, || {
        tracing::error!(error = ?error, display = %error, "generation failed");
    });
    let log = std::fs::read_to_string(&path).unwrap();
    assert!(log.contains("generation failed"));
    assert!(!log.contains("PRIVATE-CONTENT-729"));
    std::fs::remove_file(path).unwrap();
}
