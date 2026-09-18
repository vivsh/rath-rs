use super::super::errors::normalize;
use crate::core::ErrorKind;
use gemini_rust::client::Error;

/// SDK HTTP envelopes retain available status, codes and bytes without exposing body contents.
#[test]
fn sdk_http_error_preserves_available_evidence() {
    let native = Error::BadResponse {
        code: 403,
        description: Some(
            r#"{"error":{"code":403,"message":"key SECRET rejected"},"private":"PRIVATE-PROSE"}"#
                .into(),
        ),
    };
    let error = normalize(&native, "generation", Some("SECRET"));
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(403));
    assert_eq!(error.provider_code(), Some("403"));
    assert!(error.to_string().contains("key [REDACTED] rejected"));
    assert!(!format!("{error:?}").contains("PRIVATE-PROSE"));
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    assert!(body.contains("PRIVATE-PROSE"));
    assert!(!body.contains("SECRET"));
}

/// SDK parser errors become a stable Rath chain without pretending discarded bodies are available.
#[test]
fn sdk_parser_cause_is_preserved() {
    let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let message = source.to_string();
    let error = normalize(&Error::Deserialize { source }, "generation", None);
    assert_eq!(error.kind(), ErrorKind::Deserialize);
    assert_eq!(error.source().unwrap().message(), message);
    assert!(error.response_body().is_none());
}

/// Native operation codes remain separately inspectable rather than embedded only in a message.
#[test]
fn sdk_job_failure_keeps_native_fields() {
    let error = normalize(
        &Error::OperationFailed {
            name: "jobs/123".into(),
            code: 9,
            message: "resource exhausted".into(),
        },
        "generation",
        None,
    );
    assert_eq!(error.kind(), ErrorKind::Provider);
    assert_eq!(error.provider_code(), Some("9"));
    assert_eq!(error.message(), "resource exhausted");
    assert!(String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("jobs/123"));
}

/// Generation and embeddings normalize HTTP rejections from the actual official SDK dispatch path.
#[tokio::test]
async fn sdk_dispatch_keeps_failure_without_extra_requests() {
    use super::super::*;
    for embedding in [false, true] {
        let (base, requests) = crate::providers::tests::http::serve(
            422,
            serde_json::json!({
                "error":{"message":"unsupported model KNOWN-KEY", "code":"invalid_model"},
                "private":"PRIVATE-PROSE"
            }),
        );
        let mut url = ModelUrl::parse("gemini:///selected-model").unwrap();
        url.api_key = Some("KNOWN-KEY".into());
        let client = GeminiClient {
            client: gemini_rust::client::GeminiBuilder::new("KNOWN-KEY")
                .with_model(GeminiModel::Custom("models/selected-model".into()))
                .with_base_url(base.parse().unwrap())
                .build()
                .unwrap(),
            options: LlmOptions::default(),
            url,
            exit_tool_name: None,
        };
        let error = if embedding {
            client
                .embed(&EmbedRequest {
                    input: "hello".into(),
                    ..Default::default()
                })
                .await
                .unwrap_err()
        } else {
            client.execute(&[Message::user("hello")]).await.unwrap_err()
        };
        assert_eq!(error.kind(), ErrorKind::Http);
        assert_eq!(error.http_status(), Some(422));
        assert_eq!(error.provider_code(), Some("invalid_model"));
        assert!(error.to_string().contains("unsupported model [REDACTED]"));
        assert!(!format!("{error:?}").contains("PRIVATE-PROSE"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}
