use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::core::{
    CacheControl, ErrorBody, ErrorKind, ModelUrl, Provider, RathError, ThinkingLevel, TokenUsage,
};

pub use super::tool::ToolDefinition;

/// A binary or URL attachment that can accompany a message.
/// Attachments are carried through the history layer and translated into
/// provider-specific wire formats by each LLM adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Attachment {
    /// Inline binary data (e.g. a screenshot).
    /// `data` must be base64-encoded.
    Inline {
        /// Media MIME type.
        mime_type: String,
        /// Standard base64-encoded bytes.
        data: String,
    },
    /// File path that should be materialized by the caller before dispatch.
    File {
        /// Media MIME type.
        mime_type: String,
        /// Caller-owned path, requiring materialization before provider dispatch.
        path: String,
    },
    /// Reference to a publicly accessible URL.
    Url {
        /// Media MIME type.
        mime_type: String,
        /// Provider-accessible URL.
        url: String,
    },
}

/// Legacy alias for the shared model locator.
#[deprecated(note = "use rath::core::ModelUrl")]
pub type LlmUrl = ModelUrl;

/// Role of a message in provider-facing history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    /// System-level instructions prepended before user history.
    System,
    /// Human turn.
    User,
    /// Model turn.
    Assistant,
    /// Assistant turn that carries tool calls.
    /// The enclosing [`Message`] keeps any accompanying text in `content`.
    AssistantToolCalls {
        /// Calls emitted in this assistant turn.
        calls: Vec<ToolCall>,
    },
    /// Tool result fed back to the model.
    /// `call_id` must match the originating [`ToolCall`].
    Tool {
        /// Identifier of the originating tool call.
        call_id: String,
    },
}

/// One history message prepared for provider dispatch.
/// File attachments should be converted to inline or URL attachments before
/// calling a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    /// Opaque application identity, assigned and interpreted by the caller.
    /// Preserved in storage serialization; excluded from provider requests and token counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
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
    /// Sets or replaces the application key without changing provider-visible content.
    /// Rath does not generate keys, validate uniqueness, or deduplicate messages.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Creates a user-role message with the given text.
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            key: None,
            role: Role::User,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Creates an assistant-role message with the given text.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            key: None,
            role: Role::Assistant,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Creates a tool-result message. `call_id` must match the originating [`ToolCall::id`].
    pub fn tool_output(call_id: String, content: impl Into<String>) -> Self {
        Message {
            key: None,
            role: Role::Tool { call_id },
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Builds a message by JSON-encoding `value`.
    pub fn from_json(role: Role, value: &impl serde::Serialize) -> Result<Self, RathError> {
        Ok(Message {
            key: None,
            role,
            content: serde_json::to_string(value)
                .map_err(|e| RathError::from_native_error(ErrorKind::Serialize, &e))?,
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
#[non_exhaustive]
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
        text: Option<String>,
        /// Calls to dispatch, subject to the application tool policy.
        calls: Vec<ToolCall>,
    },
}

/// Provider-normalized result from one model call.
#[derive(Debug)]
#[non_exhaustive]
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
    if let Some(key) = &url.api_key {
        return Ok(key.clone());
    }
    std::env::var(default_env).map_err(|error| {
        RathError::new(
            ErrorKind::Validation,
            format!("cannot read credential environment variable {default_env}"),
        )
        .with_source(crate::core::error::credential_cause(&error))
    })
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Return plain text.
    #[default]
    Text,
    /// Return JSON content.
    Json,
    /// Requests JSON conforming to this schema; parsing is not full schema validation.
    JsonSchema {
        /// Schema used by the adapter to request structured output.
        schema: Value,
    },
}

/// Per-call client settings.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LlmOptions {
    /// Provider generation-token cap, including reasoning where applicable.
    /// Zero, unrepresentable values, and provider_config cap aliases are rejected.
    pub max_output_tokens: Option<u32>,
    /// Optional label for tracing.
    pub name: Option<String>,
    /// Preamble sent before history.
    pub preamble: Option<String>,
    /// Tools available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Reasoning depth. `None` leaves the adapter/provider default unspecified.
    pub thinking: Option<ThinkingLevel>,
    /// Tool-call policy.
    pub tool_choice: ToolChoice,
    /// Prompt hint describing user JSON; Rath does not validate input against this schema.
    pub input_schema: Option<Value>,
    /// Preferred assistant output format.
    pub response_format: ResponseFormat,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Provider-specific request configuration for knobs Rath does not model.
    pub provider_config: Option<Value>,
    /// Prompt caching policy. `None` means no explicit cache control.
    /// Currently only used by Anthropic; other providers cache automatically.
    pub cache: Option<CacheControl>,
}

impl LlmOptions {
    /// Sets the generation budget; validated at construction and request preparation.
    pub fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

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

    /// Requests a reasoning level; `None` leaves the adapter/provider default unspecified.
    /// Use `Some(ThinkingLevel::Off)` to explicitly request disabling reasoning.
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
        !matches!(self.response_format, ResponseFormat::Text)
    }

    /// Sets the structured-output schema and enables JSON output mode.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.response_format = ResponseFormat::JsonSchema { schema };
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

    /// Builds a provider client; explicit URL temperature, thinking and cache override these options.
    pub async fn create(self, model_url: &str) -> Result<LlmClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .llm(model_url, self)
            .await
    }
}

/// Validates tool names and parameter objects before provider dispatch.
#[allow(dead_code)]
pub(crate) fn validate_tools(
    provider: Provider,
    tools: &[ToolDefinition],
) -> Result<(), RathError> {
    let mut seen = std::collections::HashSet::new();
    for tool in tools {
        if tool.name.trim().is_empty() {
            return Err(RathError::new(
                crate::core::ErrorKind::Validation,
                "tool name must not be empty",
            ));
        }
        if !seen.insert(tool.name.as_str()) {
            return Err(RathError::new(
                crate::core::ErrorKind::Validation,
                format!("duplicate tool name '{}'", tool.name),
            ));
        }
        if !tool.parameters.is_object() {
            return Err(RathError::unsupported(
                provider,
                format!("tool '{}' has a non-object JSON schema", tool.name),
            ));
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn parse_json_output(text: &str) -> Result<Value, RathError> {
    serde_json::from_str(text).map_err(|e| RathError::deserialize(&e, text))
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
use super::client::LlmClient;

impl LlmOptions {
    /// Returns the schema from the sole authoritative response format.
    pub(crate) fn output_schema(&self) -> Option<&Value> {
        match &self.response_format {
            ResponseFormat::JsonSchema { schema } => Some(schema),
            _ => None,
        }
    }
}

impl Message {
    /// Constructs required values; optional controls start unset.
    pub fn new(role: Role, content: String) -> Self {
        Self {
            key: None,
            role,
            content,
            attachments: Vec::new(),
            usage: None,
        }
    }
}

impl ToolCall {
    /// Constructs required values; optional controls start unset.
    pub fn new(id: String, name: String, args: Value) -> Self {
        Self {
            id,
            name,
            args,
            thought_signatures: None,
        }
    }
    /// Sets `thought_signatures` for this value.
    pub fn with_thought_signatures(mut self, value: Vec<String>) -> Self {
        self.thought_signatures = Some(value);
        self
    }
}

impl LlmOptions {
    /// Sets `cache` for this value.
    pub fn with_cache(mut self, value: CacheControl) -> Self {
        self.cache = Some(value);
        self
    }
}

/// Merges native defaults underneath explicitly constructed wire fields; callers validate first.
pub(crate) fn native_defaults(payload: Value, native: Option<&Value>) -> Value {
    let mut merged = native
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Value::Object(fields) = payload {
        merged.extend(fields);
    }
    Value::Object(merged)
}
