//! Built-in factory injection through the public registry API, without provider calls.
use rath::llm::{
    LlmBackend, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ThinkingLevel,
};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ErrorKind, ModelUrl, Provider, RathError};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Factory;
struct Backend {
    url: ModelUrl,
    options: LlmOptions,
}
impl ProviderFactory for Factory {
    async fn llm(&self, url: &ModelUrl, options: LlmOptions) -> Result<LlmClient, RathError> {
        Ok(LlmClient::from_backend(Backend {
            url: url.clone(),
            options,
        }))
    }
}
impl LlmBackend for Backend {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }
    fn options(&self) -> &LlmOptions {
        &self.options
    }
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        Ok(LlmResponse::new(
            self.url.provider().clone(),
            LlmOutput::Output(serde_json::json!({"messages": messages.len()})),
        ))
    }
}

struct Reject(Arc<AtomicUsize>);
impl ProviderFactory for Reject {
    async fn llm(&self, url: &ModelUrl, _: LlmOptions) -> Result<LlmClient, RathError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(
            RathError::new(ErrorKind::UnsupportedCapability, "custom factory refusal")
                .with_http_status(418)
                .with_provider_code("custom-code", &[])
                .with_source(RathError::new(
                    ErrorKind::Other,
                    format!("cause {}", url.api_key().unwrap_or("no credential")),
                )),
        )
    }
}

/// Built-in aliases retain identity, URL precedence and an executable client after registry disposal.
#[tokio::test]
async fn builtin_interception_precedence_and_lifetime() {
    let registry = ProviderRegistry::with_builtin_factory(Factory);
    let client = registry
        .llm(
            "claude:///nested/model?thinking=off&temperature=0.2&base_url=http://127.0.0.1:1",
            LlmOptions::default()
                .with_thinking(Some(ThinkingLevel::High))
                .with_temperature(0.9),
        )
        .await
        .unwrap();
    assert_eq!(client.model_url().provider(), &Provider::Anthropic);
    assert_eq!(client.model_url().model(), "nested/model");
    assert_eq!(client.model_url().base_url(), Some("http://127.0.0.1:1"));
    assert_eq!(client.options().temperature, Some(0.2));
    assert_eq!(client.options().thinking, Some(ThinkingLevel::Off));
    drop(registry);
    let messages = [Message::user("hello")];
    let response = client.execute(&messages).await.unwrap();
    assert!(matches!(response.output, LlmOutput::Output(value) if value["messages"] == 1));
}

/// External routes never reach the injected built-in factory, and reserved names stay reserved.
#[tokio::test]
async fn external_registrations_remain_independent() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = ProviderRegistry::with_builtin_factory(Reject(calls.clone()))
        .register("acme", Factory)
        .unwrap();
    let client = registry
        .llm("acme:///model", LlmOptions::default())
        .await
        .unwrap();
    assert_eq!(
        client.model_url().provider(),
        &Provider::External("acme".into())
    );
    let error = registry
        .llm("unknown:///model", LlmOptions::default())
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        registry.register("ollama", Factory).err().unwrap().kind(),
        ErrorKind::Validation
    );
}

/// A custom refusal is returned once with metadata, sanitized causes and no default fallback.
#[tokio::test]
async fn custom_error_is_preserved_without_fallback() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = ProviderRegistry::with_builtin_factory(Reject(calls.clone()));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let error = registry
        .llm("ollama:///model?api_key_env=PATH", LlmOptions::default())
        .await
        .err()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(error.provider(), Some(&Provider::Ollama));
    assert_eq!(error.http_status(), Some(418));
    assert_eq!(error.provider_code(), Some("custom-code"));
    assert_eq!(error.source().unwrap().message(), "cause [REDACTED]");
    assert!(!format!("{error} {error:?}").contains(&std::env::var("PATH").unwrap()));
    let error = registry
        .embeddings("ollama:///model", Default::default())
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Invalid locators and conflicting native settings fail before custom factory dispatch.
#[tokio::test]
async fn validation_precedes_custom_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = ProviderRegistry::with_builtin_factory(Reject(calls.clone()));
    let error = registry
        .llm(
            "ollama:///model?base_url=ftp://example.com",
            Default::default(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidUrl);
    let error = registry
        .llm(
            "ollama:///model",
            LlmOptions::default().with_provider_config(serde_json::json!({"model":"other"})),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
