use super::http::serve;
use crate::core::{ErrorKind, ModelUrl, Provider, RathError};
use crate::llm::{LlmOptions, Message};
use serde_json::json;

/// Supplies a local endpoint and explicit secret without process-wide environment changes.
fn model(provider: &str, base: String) -> ModelUrl {
    let mut url = ModelUrl::parse(&format!("{provider}:///test-model")).unwrap();
    url.base_url = Some(base);
    url.api_key = Some("KNOWN-CREDENTIAL".into());
    url
}

/// Verifies provider diagnostics remain useful while raw response details require explicit access.
fn assert_failure(error: &RathError) {
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(422));
    assert_eq!(error.provider_code(), Some("bad_model"));
    assert_eq!(error.request_id(), Some("req-42"));
    let display = format!("{error} {error:?}");
    assert!(display.contains("choose another model"));
    assert!(!display.contains("KNOWN-CREDENTIAL"));
    assert!(!display.contains("PRIVATE-PROSE"));
    let body = String::from_utf8_lossy(error.response_body().unwrap().bytes());
    assert!(body.contains("PRIVATE-PROSE"));
    assert!(!body.contains("KNOWN-CREDENTIAL"));
}

/// All direct-HTTP LLM adapters share the same rejection contract and dispatch only once.
#[tokio::test]
async fn generation_errors_match_across_adapters() {
    for provider in ["openai", "openrouter", "anthropic", "ollama"] {
        let (base, requests) = serve(
            422,
            json!({"error":{"message":"choose another model KNOWN-CREDENTIAL", "code":"bad_model"},"request_id":"req-42","private":"PRIVATE-PROSE"}),
        );
        let url = model(provider, base);
        let client = crate::providers::create_llm_client(&url, LlmOptions::default()).unwrap();
        let error = client.execute(&[Message::user("hello")]).await.unwrap_err();
        assert_failure(&error);
        assert_eq!(error.provider(), Some(&url.provider));
        assert_eq!(error.operation(), Some("generation"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Embedding failures retain the same status, cause/body contract and do not retry.
#[tokio::test]
async fn embedding_errors_match_across_adapters() {
    for provider in ["openai", "ollama"] {
        let (base, requests) = serve(
            422,
            json!({"error":{"message":"choose another model KNOWN-CREDENTIAL", "code":"bad_model"},"request_id":"req-42","private":"PRIVATE-PROSE"}),
        );
        let url = model(provider, base);
        let client = crate::providers::create_embedding_client(&url, Default::default()).unwrap();
        let error = client
            .embed(&crate::embeddings::EmbedRequest {
                input: "hello".into(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_failure(&error);
        assert_eq!(error.operation(), Some("embeddings"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// A missing tool identifier cannot turn a partly malformed Anthropic response into successful output.
#[tokio::test]
async fn malformed_anthropic_tool_retains_response() {
    let (base, _requests) = serve(
        200,
        json!({"content":[{"type":"text","text":"PRIVATE-PROSE"},{"type":"tool_use","name":"search","input":{}}]}),
    );
    let client =
        crate::providers::create_llm_client(&model("anthropic", base), LlmOptions::default())
            .unwrap();
    let error = client.execute(&[Message::user("hello")]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert_eq!(error.provider(), Some(&Provider::Anthropic));
    assert!(error.message().contains("missing id"));
    assert!(!error.to_string().contains("PRIVATE-PROSE"));
    assert!(
        String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("PRIVATE-PROSE")
    );
}
