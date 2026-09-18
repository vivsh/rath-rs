use serde_json::Value;

use super::super::*;
use async_trait::async_trait;

struct DummyFactory;

struct DummyClient {
    url: ModelUrl,
}

impl DummyClient {
    fn new() -> Self {
        Self {
            url: ModelUrl::parse("openai:///test-model").expect("valid test URL"),
        }
    }
}

struct MarkerLayer;

struct MarkedFactory<F> {
    inner: F,
    marked: bool,
}

#[async_trait]
impl LlmClient for DummyClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &LlmOptions {
        static OPTS: std::sync::OnceLock<LlmOptions> = std::sync::OnceLock::new();
        OPTS.get_or_init(LlmOptions::default)
    }

    async fn execute(&self, _messages: &[Message]) -> Result<LlmResponse, RathError> {
        Ok(LlmResponse::new(
            Provider::OpenAi,
            LlmOutput::Output(serde_json::json!({ "ok": true })),
        ))
    }
}

impl LlmClientFactory for DummyFactory {
    fn create(
        &self,
        _model_url: &str,
        _options: LlmOptions,
    ) -> Result<Box<dyn LlmClient>, RathError> {
        Ok(Box::new(DummyClient::new()))
    }
}

impl<F: LlmClientFactory> LlmClientFactory for MarkedFactory<F> {
    fn create(
        &self,
        model_url: &str,
        options: LlmOptions,
    ) -> Result<Box<dyn LlmClient>, RathError> {
        self.inner.create(model_url, options)
    }
}

impl<F: LlmClientFactory> LlmClientFactoryLayer<F> for MarkerLayer {
    type Factory = MarkedFactory<F>;

    fn layer(self, inner: F) -> Self::Factory {
        MarkedFactory {
            inner,
            marked: true,
        }
    }
}

/// `ModelUrl::parse` correctly parses a gemini URL without an API key.
#[test]
fn parse_gemini_url_no_key() {
    let url = ModelUrl::parse("gemini:///gemini-2.5-flash-lite").unwrap();
    assert_eq!(url.provider, Provider::Gemini);
    assert_eq!(url.model, "gemini-2.5-flash-lite");
    assert!(url.api_key.is_none());
    assert!(url.base_url.is_none());
}

/// `ModelUrl::parse` extracts model and custom base_url from an ollama locator.
#[test]
fn parse_ollama_url() {
    let url = ModelUrl::parse("ollama:///qwen3:8b?base_url=http://localhost:11434").unwrap();
    assert_eq!(url.provider, Provider::Ollama);
    assert_eq!(url.model, "qwen3:8b");
    assert_eq!(url.base_url.as_deref(), Some("http://localhost:11434"));
    assert!(url.api_key.is_none());
}

/// `ModelUrl::parse` resolves `api_key_env` query params before the client is built.
#[test]
fn parse_query_api_key_env() {
    let expected = std::env::var("PATH").expect("PATH should be set during tests");
    let url = ModelUrl::parse("anthropic:///claude-haiku-4-5?api_key_env=PATH").unwrap();
    assert_eq!(url.provider, Provider::Anthropic);
    assert_eq!(url.api_key.as_deref(), Some(expected.as_str()));
}

/// `anthropic://` and `claude://` both select the Anthropic provider.
#[test]
fn parse_anthropic_aliases() {
    let anthropic = ModelUrl::parse("anthropic:///claude-sonnet-4-5").unwrap();
    let claude = ModelUrl::parse("claude:///claude-sonnet-4-5").unwrap();
    assert_eq!(anthropic.provider, Provider::Anthropic);
    assert_eq!(claude.provider, Provider::Anthropic);
}

/// Tool schemas must be JSON objects and duplicate names are rejected before provider calls.
#[test]
fn validate_tools_rejects_bad_definitions() {
    let non_object = vec![ToolDefinition {
        name: "bad".into(),
        description: "bad".into(),
        parameters: serde_json::json!(true),
    }];
    assert!(matches!(
        validate_tools(Provider::OpenAi, &non_object),
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));

    let duplicate = vec![
        ToolDefinition {
            name: "dup".into(),
            description: "one".into(),
            parameters: serde_json::json!({ "type": "object" }),
        },
        ToolDefinition {
            name: "dup".into(),
            description: "two".into(),
            parameters: serde_json::json!({ "type": "object" }),
        },
    ];
    assert!(matches!(
        validate_tools(Provider::OpenAi, &duplicate),
        Err(error) if error.kind() == crate::core::ErrorKind::Validation
    ));
}

/// `ModelUrl::parse` returns an error for an unknown provider scheme.
#[test]
fn parse_unknown_scheme_errors() {
    assert!(matches!(
        ModelUrl::parse("unknown:///model"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Missing `api_key_env` variables fail early with a clear URL-configuration error.
#[test]
fn parse_missing_api_key_env_errors() {
    assert!(matches!(
        ModelUrl::parse("openai:///gpt-4o?api_key_env=__PRAVAH_MISSING_ENV__"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// `ModelUrl::parse` returns an error when no scheme separator is present.
#[test]
fn parse_missing_scheme_errors() {
    assert!(matches!(
        ModelUrl::parse("gemini-2.5-flash-lite"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Verifies effective preamble appends input schema hint.
#[test]
fn effective_preamble_appends_input_schema_hint() {
    let options = LlmOptions::default()
        .with_preamble("You are helpful.")
        .with_input_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "kind": { "type": "string" }
            },
            "required": ["kind"]
        }));

    let preamble = options
        .effective_preamble()
        .expect("effective preamble should be present");
    assert!(preamble.contains("You are helpful."));
    assert!(preamble.contains("The user message is JSON."));
    assert!(preamble.contains("\"required\":[\"kind\"]"));
}

/// Explicit response mode enables JSON decoding.
#[test]
fn wants_json_output_uses_explicit_response_format() {
    assert!(!LlmOptions::default().wants_json_output());
    assert!(
        LlmOptions::default()
            .with_response_format(ResponseFormat::Json)
            .wants_json_output()
    );
}

/// Input schema alone does not force JSON response mode.
#[test]
fn input_schema_does_not_force_json_output() {
    assert!(
        !LlmOptions::default()
            .with_input_schema(serde_json::json!({ "type": "object" }))
            .wants_json_output()
    );
}

/// Output schema opts the client into JSON response mode.
#[test]
fn output_schema_enables_json_output() {
    assert!(
        LlmOptions::default()
            .with_output_schema(serde_json::json!({ "type": "object" }))
            .wants_json_output()
    );
}

/// Verifies provider config builder stores config.
#[test]
fn provider_config_builder_stores_config() {
    let config = serde_json::json!({
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            }
        ]
    });
    let options = LlmOptions::default().with_provider_config(config.clone());

    assert_eq!(options.provider_config, Some(config));
}

/// Verifies decode output text returns plain text when json mode disabled.
#[test]
fn decode_output_text_returns_plain_text_when_json_mode_disabled() {
    assert_eq!(
        decode_output_text("hello", false).unwrap(),
        Value::String("hello".into())
    );
}

/// Verifies decode output text parses json when json mode enabled.
#[test]
fn decode_output_text_parses_json_when_json_mode_enabled() {
    assert_eq!(
        decode_output_text(r#"{"ok":true}"#, true).unwrap(),
        serde_json::json!({ "ok": true })
    );
}

/// `LlmClientFactory::layer` wraps a concrete factory with the supplied decorator.
#[tokio::test]
async fn layer_wraps_factory() {
    let factory = DummyFactory.layer(MarkerLayer);
    assert!(factory.marked);

    let client = factory
        .create("openai:///test-model", LlmOptions::default())
        .expect("layered factory should create a client");
    let response = client
        .execute(&[Message::user("hi")])
        .await
        .expect("layered client should execute");
    assert!(matches!(response.output, LlmOutput::Output(_)));
}

/// Existing custom implementations retain unsupported defaults for every counting method.
#[tokio::test]
async fn custom_client_defaults_are_unsupported() {
    let c = DummyClient::new();
    assert!(matches!(
        c.estimate_tokens(&[Message::user("x")]),
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
    assert!(matches!(
        c.estimate_content_tokens("x"),
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
    assert!(matches!(
        c.count_tokens(&[Message::user("x")]).await,
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
    assert!(matches!(
        c.count_content_tokens("x").await,
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
}
