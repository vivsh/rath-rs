//! Public stabilization and configuration contracts, without provider calls.
use rath::embeddings::{EmbeddingOptions, FastEmbedOptions};
use rath::llm::{
    LlmBackend, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ResponseFormat,
    ThinkingLevel,
};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ErrorKind, ModelUrl, RathError};
use serde_json::json;

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
            LlmOutput::Output(json!(messages.len())),
        ))
    }
}

/// External LLM factories receive resolved URL precedence and return independent, borrowing clients.
#[tokio::test]
async fn external_llm_precedence_and_lifetimes() {
    let registry = ProviderRegistry::with_builtins()
        .register("acme", Factory)
        .unwrap();
    let client = registry
        .llm(
            "acme:///nested/model?thinking=off&temperature=0.2",
            LlmOptions::default()
                .with_thinking(Some(ThinkingLevel::High))
                .with_temperature(0.9),
        )
        .await
        .unwrap();
    assert_eq!(client.options().thinking, Some(ThinkingLevel::Off));
    assert_eq!(client.options().temperature, Some(0.2));
    assert_eq!(client.model_url().model(), "nested/model");
    drop(registry);
    let messages = vec![Message::user("hi")];
    fn send<T: Send>(_: &T) {}
    let future = client.execute(&messages);
    send(&future);
    assert!(matches!(future.await.unwrap().output, LlmOutput::Output(_)));
}

/// Structured format setters replace the entire choice; typed native aliases cannot compete.
#[tokio::test]
async fn format_replacement_and_config_conflicts() {
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_response_format(ResponseFormat::Text);
    assert_eq!(options.response_format, ResponseFormat::Text);
    for config in [
        json!({"model":"other"}),
        json!({"stream":true}),
        json!({"api_key":"secret"}),
        json!({"options":{"num_predict":1}}),
        json!({"generationConfig":{"maxOutputTokens":1}}),
        json!([]),
    ] {
        let error = LlmOptions::default()
            .with_provider_config(config)
            .create("ollama:///m")
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}

/// Endpoint validation and credential-aware debug apply to direct builtin parsing as well.
#[test]
fn locator_is_private_and_redacted() {
    assert!(ModelUrl::parse("openai:///m?base_url=ftp://example.com").is_err());
    let url = ModelUrl::parse(
        "openai:///m?api_key_env=PATH&base_url=https://example.com/api?secret=PRIVATE",
    )
    .unwrap();
    let debug = format!("{url:?}");
    assert!(!debug.contains(&std::env::var("PATH").unwrap()));
    assert!(!debug.contains("PRIVATE"));
    assert!(url.api_key().is_some());
}

/// Every non-LLM capability rejects language-model URL controls before factory dispatch.
#[tokio::test]
async fn media_rejects_llm_url_controls() {
    let error = EmbeddingOptions::default()
        .create("ollama:///m?thinking=off")
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidUrl);
    let registry = ProviderRegistry::new().register("acme", Factory).unwrap();
    assert_eq!(
        registry
            .tts(
                "acme:///m?temperature=0.1",
                rath::audio::tts::TtsOptions::default()
            )
            .await
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidUrl
    );
}

/// The local scheme is always reserved and setup controls have no competing JSON path.
#[tokio::test]
async fn fastembed_setup_is_not_native_configuration() {
    assert_eq!(
        ProviderRegistry::new()
            .register("fastembed", Factory)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Validation
    );
    for key in [
        "cache_dir",
        "download",
        "revision",
        "max_length",
        "intra_threads",
        "batch_size",
    ] {
        let error = EmbeddingOptions::default()
            .with_provider_config(json!({key:1}))
            .create("openai:///model")
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    let error = EmbeddingOptions::default()
        .with_fastembed(FastEmbedOptions::new(""))
        .create("fastembed:///BGESmallENV15")
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

/// Disabled optional inference yields an informative capability error without model or runtime loading.
#[cfg(not(feature = "fastembed"))]
#[tokio::test]
async fn feature_disabled_is_explicit() {
    let error = EmbeddingOptions::default()
        .with_fastembed(FastEmbedOptions::new("models"))
        .create("fastembed:///BGESmallENV15")
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert!(error.to_string().contains("Cargo feature"));
}

/// Native prompt effects not represented by a counting projection fail explicitly.
#[tokio::test]
async fn native_prompts_cannot_disappear_from_estimates() {
    let client = LlmOptions::default()
        .with_provider_config(json!({"deployment_prompt":"hidden instructions"}))
        .create("ollama:///model")
        .await
        .unwrap();
    assert_eq!(
        client
            .estimate_tokens(&[Message::user("hi")])
            .unwrap_err()
            .kind(),
        ErrorKind::UnsupportedCapability
    );
}
