use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::core::{CacheControl, ModelUrl, Provider, RathError, ThinkingLevel, TokenUsage};

mod tool;

/// A binary or URL attachment that can accompany a message.
/// Attachments are carried through the history layer and translated into
/// provider-specific wire formats by each LLM adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Attachment {
    /// Inline binary data (e.g. a screenshot).
    /// `data` must be base64-encoded.
    Inline { mime_type: String, data: String },
    /// File path that should be materialized by the caller before dispatch.
    File { mime_type: String, path: String },
    /// Reference to a publicly accessible URL.
    Url { mime_type: String, url: String },
}

pub mod schema;

pub use tool::ToolDefinition;

#[deprecated(note = "use rath::core::ModelUrl")]
pub type LlmUrl = ModelUrl;

/// Role of a message in provider-facing history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Role {
    /// System-level instructions prepended before user history.
    System,
    /// Human turn.
    User,
    /// Model turn.
    Assistant,
    /// Assistant turn that carries tool calls.
    /// The enclosing [`Message`] keeps any accompanying text in `content`.
    AssistantToolCalls { calls: Vec<ToolCall> },
    /// Tool result fed back to the model.
    /// `call_id` must match the originating [`ToolCall`].
    Tool { call_id: String },
}

/// One history message prepared for provider dispatch.
/// File attachments should be converted to inline or URL attachments before
/// calling a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role that produced this message.
    pub role: Role,
    /// Text body of the message.
    pub content: String,
    /// Attachments (images, files) to send alongside the message content.
    /// Serialization is skipped when empty so existing stored history is unaffected.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
    /// Provider-reported token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl Message {
    /// Creates a user-role message with the given text.
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Creates an assistant-role message with the given text.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Creates a tool-result message. `call_id` must match the originating [`ToolCall::id`].
    pub fn tool_output(call_id: String, content: impl Into<String>) -> Self {
        Message {
            role: Role::Tool { call_id },
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Builds a message by JSON-encoding `value`.
    pub fn from_json(role: Role, value: &impl serde::Serialize) -> Result<Self, serde_json::Error> {
        Ok(Message {
            role,
            content: serde_json::to_string(value)?,
            attachments: Vec::new(),
            usage: None,
        })
    }

    /// Attaches token usage reported by the provider.
    pub fn with_usage(self, usage: TokenUsage) -> Self {
        Message {
            usage: Some(usage),
            ..self
        }
    }

    /// Appends a pre-built attachment.
    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Appends an inline binary attachment; `bytes` are base64-encoded internally.
    pub fn with_inline(mut self, mime_type: impl Into<String>, bytes: impl AsRef<[u8]>) -> Self {
        self.attachments.push(Attachment::Inline {
            mime_type: mime_type.into(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        self
    }

    /// Appends a file attachment. Callers should materialize it before dispatch.
    pub fn with_file(mut self, mime_type: impl Into<String>, path: impl Into<String>) -> Self {
        self.attachments.push(Attachment::File {
            mime_type: mime_type.into(),
            path: path.into(),
        });
        self
    }

    /// Appends a URL attachment.
    pub fn with_url(mut self, mime_type: impl Into<String>, url: impl Into<String>) -> Self {
        self.attachments.push(Attachment::Url {
            mime_type: mime_type.into(),
            url: url.into(),
        });
        self
    }
}

/// Tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id that must be echoed back in [`Role::Tool`].
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// JSON arguments for the tool call.
    pub args: Value,
    /// Provider-specific continuation data from Gemini thinking models.
    /// Echo this back unchanged on the next turn when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signatures: Option<Vec<String>>,
}

/// Output from a single model call.
#[derive(Debug)]
pub enum LlmOutput {
    /// Structured output payload, or plain text wrapped as `Value::String`.
    Output(Value),
    /// Tool calls requested by the model.
    ToolCalls {
        /// Accompanying text emitted with the tool calls.
        thought: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Provider-normalized result from one model call.
#[derive(Debug)]
pub struct LlmResponse {
    /// Parsed model output.
    pub output: LlmOutput,
    /// Token counts for this call, if reported.
    pub usage: Option<TokenUsage>,
    /// Provider that produced this response.
    pub provider: Provider,
    /// Model identifier echoed by the provider, if available.
    pub provider_model: Option<String>,
    /// Raw provider-specific metadata (e.g. finish reason, safety ratings).
    pub raw_metadata: Option<Value>,
}

impl LlmResponse {
    /// Constructs a minimal response with no usage or metadata.
    pub fn new(provider: Provider, output: LlmOutput) -> Self {
        Self {
            output,
            usage: None,
            provider,
            provider_model: None,
            raw_metadata: None,
        }
    }

    /// Attaches token usage to the response.
    pub fn with_usage(mut self, usage: Option<TokenUsage>) -> Self {
        self.usage = usage;
        self
    }

    /// Sets the provider-echoed model identifier.
    pub fn with_provider_model(mut self, provider_model: Option<String>) -> Self {
        self.provider_model = provider_model;
        self
    }

    /// Attaches raw provider metadata.
    pub fn with_raw_metadata(mut self, raw_metadata: Option<Value>) -> Self {
        self.raw_metadata = raw_metadata;
        self
    }
}

pub(crate) fn required_api_key(url: &ModelUrl, default_env: &str) -> Result<String, RathError> {
    url.api_key
        .clone()
        .or_else(|| std::env::var(default_env).ok())
        .ok_or_else(|| RathError::Provider(format!("{default_env} is not set")))
}

pub(crate) fn optional_api_key(url: &ModelUrl, default_env: &str) -> Option<String> {
    url.api_key
        .clone()
        .or_else(|| std::env::var(default_env).ok())
}

pub(crate) fn configured_base_url(url: &ModelUrl, default_base_url: &str) -> String {
    url.base_url
        .clone()
        .unwrap_or_else(|| default_base_url.to_string())
}

/// Controls whether the model may call tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// Let the provider decide.
    #[default]
    Auto,
    /// Require at least one tool call.
    Required,
    /// Disable tool calls.
    Disabled,
}

/// Controls how providers should format assistant output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Return plain text.
    #[default]
    Text,
    /// Return JSON content.
    Json,
}

/// Per-call client settings.
#[derive(Debug, Clone, Default)]
pub struct LlmOptions {
    /// Optional label for tracing.
    pub name: Option<String>,
    /// Preamble sent before history.
    pub preamble: Option<String>,
    /// Tools available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Reasoning depth. `None` means no thinking mode.
    pub thinking: Option<ThinkingLevel>,
    /// Tool-call policy.
    pub tool_choice: ToolChoice,
    /// JSON Schema for the user payload.
    pub input_schema: Option<Value>,
    /// JSON Schema for structured output.
    pub output_schema: Option<Value>,
    /// Preferred assistant output format.
    pub response_format: ResponseFormat,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Provider-specific request configuration for knobs Rath does not model.
    pub provider_config: Option<Value>,
    /// Prompt caching policy. `None` means no explicit cache control.
    /// Currently only used by Anthropic; other providers cache automatically.
    pub cache: Option<CacheControl>,
    /// Maximum LLM dispatch turns. When `Some(n)`, a last-turn reminder is
    /// injected on the final turn. A factory layer may override the value set
    /// by `AgentConfig` by writing to this field before returning the client.
    pub turn_budget: Option<u32>,
    /// Overrides the default last-turn reminder injected when `turn_budget` is
    /// reached. `None` uses the provider-appropriate default.
    pub turn_budget_message: Option<String>,
    /// Name of the output type expected from this agent run.
    /// Used by clients that need to inject an exit tool.
    pub output_type_name: String,
}

impl LlmOptions {
    /// Sets the system preamble sent before user history.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Registers the tools available to the model for this call.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Enables extended thinking. `None` disables it.
    pub fn with_thinking(mut self, thinking: Option<ThinkingLevel>) -> Self {
        self.thinking = thinking;
        self
    }

    /// Sets the tool-call policy for this request.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Sets a label used in tracing spans.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the input schema.
    pub fn with_input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    pub(crate) fn effective_preamble(&self) -> Option<String> {
        match (&self.preamble, &self.input_schema) {
            (None, None) => None,
            (Some(preamble), None) => Some(preamble.clone()),
            (None, Some(schema)) => Some(Self::input_schema_hint(schema)),
            (Some(preamble), Some(schema)) => {
                Some(format!("{preamble}\n\n{}", Self::input_schema_hint(schema)))
            }
        }
    }

    fn input_schema_hint(schema: &Value) -> String {
        format!("The user message is JSON. Interpret it using this JSON Schema: {schema}")
    }

    pub(crate) fn wants_json_output(&self) -> bool {
        self.response_format == ResponseFormat::Json
    }

    /// Sets the structured-output schema and enables JSON output mode.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self.response_format = ResponseFormat::Json;
        self
    }

    /// Sets the preferred assistant output format.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the sampling temperature from an `Option`.
    pub fn with_temperature_opt(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets provider-specific request configuration.
    pub fn with_provider_config(mut self, provider_config: Value) -> Self {
        self.provider_config = Some(provider_config);
        self
    }

    /// Sets provider-specific request configuration from an `Option`.
    pub fn with_provider_config_opt(mut self, provider_config: Option<Value>) -> Self {
        self.provider_config = provider_config;
        self
    }

    /// Builds a provider client for the given model URL.
    pub fn create(mut self, llm_url: &str) -> Result<Box<dyn LlmClient>, RathError> {
        let url = ModelUrl::parse(llm_url)?;
        if url.temperature.is_some() {
            self.temperature = url.temperature;
        }
        if url.thinking.is_some() {
            self.thinking = url.thinking.clone();
        }
        if url.cache.is_some() {
            self.cache = url.cache.clone();
        }
        crate::providers::create_llm_client(&url, self)
    }
}

#[allow(dead_code)]
pub(crate) fn validate_tools(
    provider: Provider,
    tools: &[ToolDefinition],
) -> Result<(), RathError> {
    let mut seen = std::collections::HashSet::new();
    for tool in tools {
        if tool.name.trim().is_empty() {
            return Err(RathError::Validation("tool name must not be empty".into()));
        }
        if !seen.insert(tool.name.as_str()) {
            return Err(RathError::Validation(format!(
                "duplicate tool name '{}'",
                tool.name
            )));
        }
        if !tool.parameters.is_object() {
            return Err(RathError::UnsupportedCapability {
                provider,
                capability: format!("tool '{}' has a non-object JSON schema", tool.name),
            });
        }
    }
    Ok(())
}

/// Injects a synthetic exit-tool into `options`, converting structured-output
/// delivery into a required tool call.
///
/// Moves `output_schema` into a [`ToolDefinition`], sets `tool_choice` to
/// [`ToolChoice::Required`], clears `output_schema`, and resets
/// `response_format` to [`ResponseFormat::Text`].
pub(crate) fn inject_exit_tool(options: &mut LlmOptions) {
    if options.output_type_name.is_empty() {
        return;
    }
    let name = options.output_type_name.clone();
    let parameters = options
        .output_schema
        .take()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));
    options.tools.push(ToolDefinition {
        name,
        description: "Submit your final answer.".to_string(),
        parameters,
    });
    options.tool_choice = ToolChoice::Required;
    options.response_format = ResponseFormat::Text;
}

/// Searches `calls` for a tool call whose name matches `name`.
///
/// Returns the call's argument payload when found.
pub(crate) fn extract_exit_tool_call(calls: &[ToolCall], name: &str) -> Option<Value> {
    calls
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.args.clone())
}

#[allow(dead_code)]
pub(crate) fn parse_json_output(text: &str) -> Result<Value, RathError> {
    serde_json::from_str(text).map_err(|e| {
        tracing::error!(model_output = %text, parse_error = %e, "LLM output deserialization failed");
        RathError::Deserialize {
            source: e,
            raw: text.to_string(),
        }
    })
}

pub(crate) fn decode_output_text(text: &str, wants_json_output: bool) -> Result<Value, RathError> {
    if wants_json_output {
        parse_json_output(text)
    } else {
        Ok(Value::String(text.to_owned()))
    }
}

// ── LlmClient trait ──────────────────────────────────────────────────────────────

/// Provider-agnostic stateless LLM client.
///
/// Options are fixed at construction time and owned by the implementation.
/// Callers push input messages to history before calling `execute`, and
/// push tool-result messages after dispatch.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// The parsed model URL used to construct this client.
    fn model_url(&self) -> &ModelUrl;

    /// The options this client was constructed with.
    ///
    /// Factory layers may override fields such as `turn_budget` or
    /// `turn_budget_message` on the `LlmOptions` passed to `create`, and
    /// the runtime reads them back through this accessor.
    fn options(&self) -> &LlmOptions;

    /// The provider backing this client instance.
    fn provider(&self) -> Provider {
        self.model_url().provider.clone()
    }

    /// Returns `true` when this client uses an exit-tool strategy to collect
    /// structured output (Ollama always; Gemini before version 3.1).
    fn uses_exit_tool(&self) -> bool {
        self.model_url().needs_exit_tool()
    }

    /// Wraps a reminder message in a provider-appropriate envelope.
    ///
    /// Anthropic and Gemini use an XML `<system-reminder>` wrapper; all other
    /// providers return the text unchanged.
    fn wrap_system_reminder(&self, text: &str) -> String {
        match self.provider() {
            Provider::Anthropic | Provider::Gemini => {
                format!("<system-reminder><critical>{text}</critical></system-reminder>")
            }
            _ => text.to_string(),
        }
    }

    /// Returns a default turn-budget reminder message for this provider.
    ///
    /// When `exit_tool_name` is `Some`, the message names the exit tool to call.
    fn default_turn_budget_message(&self, exit_tool_name: Option<&str>) -> String {
        if let Some(name) = exit_tool_name {
            let msg = format!(
                "This is your final response turn. \
                 Call the `{name}` tool with your final answer now."
            );
            return self.wrap_system_reminder(&msg);
        }
        match self.provider() {
            Provider::Anthropic | Provider::Gemini => {
                "<system-reminder>\
                 <critical>TURN LIMIT REACHED</critical>\
                 <constraint>This is your final response turn. \
                 Do not call any more tools. \
                 Provide your best answer now, following the output format already specified.</constraint>\
                 </system-reminder>"
                    .to_string()
            }
            _ => {
                "FINAL TURN: do not call any more tools. \
                 Provide your best answer now, following the output format already specified."
                    .to_string()
            }
        }
    }

    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError>;
}

/// Creates a [`LlmClient`] from a model URL and call-time options.
///
/// Implement this trait to inject alternative backends (e.g. mocks) into a
/// [`crate::flows`] pipeline. The default implementation delegates to
/// [`LlmOptions::create`].
pub trait LlmClientFactory: Send + Sync + 'static {
    fn create(&self, model_url: &str, options: LlmOptions)
    -> Result<Box<dyn LlmClient>, RathError>;

    /// Wraps this factory with `layer`.
    /// The most recently added layer becomes the outermost wrapper.
    fn layer<L>(self, layer: L) -> L::Factory
    where
        Self: Sized,
        L: LlmClientFactoryLayer<Self>,
    {
        layer.layer(self)
    }
}

/// Decorates one [`LlmClientFactory`] with another.
pub trait LlmClientFactoryLayer<F> {
    type Factory: LlmClientFactory;

    fn layer(self, inner: F) -> Self::Factory;
}

/// Default factory — creates real provider clients via [`LlmOptions::create`].
pub struct DefaultLlmClientFactory;

impl LlmClientFactory for DefaultLlmClientFactory {
    fn create(
        &self,
        model_url: &str,
        options: LlmOptions,
    ) -> Result<Box<dyn LlmClient>, RathError> {
        options.create(model_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Err(RathError::UnsupportedCapability { .. })
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
            Err(RathError::Validation(_))
        ));
    }

    /// `ModelUrl::parse` returns an error for an unknown provider scheme.
    #[test]
    fn parse_unknown_scheme_errors() {
        assert!(matches!(
            ModelUrl::parse("unknown:///model"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Missing `api_key_env` variables fail early with a clear URL-configuration error.
    #[test]
    fn parse_missing_api_key_env_errors() {
        assert!(matches!(
            ModelUrl::parse("openai:///gpt-4o?api_key_env=__PRAVAH_MISSING_ENV__"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// `ModelUrl::parse` returns an error when no scheme separator is present.
    #[test]
    fn parse_missing_scheme_errors() {
        assert!(matches!(
            ModelUrl::parse("gemini-2.5-flash-lite"),
            Err(RathError::InvalidUrl(_))
        ));
    }

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

    #[test]
    fn decode_output_text_returns_plain_text_when_json_mode_disabled() {
        assert_eq!(
            decode_output_text("hello", false).unwrap(),
            Value::String("hello".into())
        );
    }

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
}
