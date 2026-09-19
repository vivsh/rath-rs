use crate::core::error::http;
mod measurement;
mod tool_calls;

use tool_calls::{collect_tool_calls, validate_call_policy, validate_tool_request};

use std::borrow::Cow;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::Serialize;
use serde_json::{Value, json};

use crate::embeddings::{EmbedRequest, EmbedResponse, EmbeddingClient, EmbeddingOptions};

use crate::llm::{
    Attachment, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ModelUrl, Provider,
    RathError, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice, ToolDefinition,
    configured_base_url, decode_output_text, extract_exit_tool_call, inject_exit_tool,
    optional_api_key, parse_json_output, validate_tools,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

struct OllamaClient {
    http: HttpClient,
    api_key: Option<String>,
    base_url: String,
    model: String,
    options: LlmOptions,
    url: ModelUrl,
    exit_tool_name: Option<String>,
}

impl OllamaClient {
    /// Sends JSON to the configured endpoint with optional bearer authentication.
    async fn post_json<T: Serialize + ?Sized, R>(
        &self,
        endpoint: &str,
        payload: &T,
        operation: &'static str,
        decode: impl FnOnce(Value) -> Result<R, RathError>,
    ) -> Result<R, RathError> {
        http::mapped(
            with_bearer_auth(self.http.post(endpoint), self.api_key.as_deref()).json(payload),
            Provider::Ollama,
            operation,
            &self.api_key.as_deref().into_iter().collect::<Vec<_>>(),
            decode,
        )
        .await
    }
}

/// Creates an LLM client after validating output caps and resolving credentials.
pub fn new_client(
    url: &ModelUrl,
    mut options: LlmOptions,
) -> Result<Box<dyn LlmClient>, RathError> {
    crate::llm::counting::validate_options(Provider::Ollama, &options)?;
    let exit_tool_name = if url.needs_exit_tool()
        && !options.output_type_name.is_empty()
        && !options.tools.is_empty()
    {
        let name = options.output_type_name.clone();
        inject_exit_tool(&mut options);
        Some(name)
    } else {
        None
    };
    validate_tool_request(&options)?;
    Ok(Box::new(OllamaClient {
        http: HttpClient::new(),
        api_key: optional_api_key(url, "OLLAMA_API_KEY"),
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
        url: url.clone(),
        exit_tool_name,
    }))
}

/// Creates the provider embedding client or returns a configuration error.
pub fn new_embedding_client(
    url: &ModelUrl,
    _options: EmbeddingOptions,
) -> Result<Box<dyn EmbeddingClient>, RathError> {
    Ok(Box::new(OllamaClient {
        http: HttpClient::new(),
        api_key: optional_api_key(url, "OLLAMA_API_KEY"),
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options: LlmOptions::default(),
        url: url.clone(),
        exit_tool_name: None,
    }))
}

#[async_trait]
impl LlmClient for OllamaClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &crate::llm::LlmOptions {
        &self.options
    }

    fn estimate_tokens(&self, messages: &[Message]) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&self.options, messages)
            .map_err(|error| {
                error
                    .with_context(Provider::Ollama, "token estimation")
                    .sanitized(&self.api_key.as_deref().into_iter().collect::<Vec<_>>())
            })
    }

    fn estimate_content_tokens(&self, content: &str) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&LlmOptions::default(), &[Message::user(content)])
            .map_err(|error| {
                error
                    .with_context(Provider::Ollama, "token estimation")
                    .sanitized(&self.api_key.as_deref().into_iter().collect::<Vec<_>>())
            })
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        crate::llm::counting::validate_options(Provider::Ollama, &self.options).map_err(
            |error| {
                error
                    .with_context(Provider::Ollama, "generation")
                    .sanitized(&self.api_key.as_deref().into_iter().collect::<Vec<_>>())
            },
        )?;
        validate_history(messages).map_err(|error| {
            error
                .with_context(Provider::Ollama, "generation")
                .sanitized(&self.api_key.as_deref().into_iter().collect::<Vec<_>>())
        })?;
        validate_tool_request(&self.options).map_err(|error| {
            error
                .with_context(Provider::Ollama, "generation")
                .sanitized(&self.api_key.as_deref().into_iter().collect::<Vec<_>>())
        })?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let endpoint = chat_completions_endpoint(&self.base_url);

        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let result = self
            .post_json(&endpoint, &payload, "generation", |response| {
                map_response(response, &self.options)
            })
            .await?;

        if let Some(ref name) = self.exit_tool_name
            && let LlmOutput::ToolCalls { calls, .. } = &result.output
            && let Some(args) = extract_exit_tool_call(calls, name)
        {
            return Ok(LlmResponse::new(Provider::Ollama, LlmOutput::Output(args))
                .with_usage(result.usage)
                .with_provider_model(result.provider_model)
                .with_raw_metadata(result.raw_metadata));
        }

        Ok(result)
    }
}

#[async_trait]
impl EmbeddingClient for OllamaClient {
    /// Sends the embedding request to the configured provider and normalizes its result.
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        let endpoint = embed_endpoint(&self.base_url);
        let payload = json!({ "model": self.model, "input": request.input });
        self.post_json(&endpoint, &payload, "embeddings", |response| {
            let values: Vec<f32> = response["embeddings"][0]
                .as_array()
                .ok_or_else(|| RathError::invalid("missing embeddings[0]", &response))?
                .iter()
                .map(|v| {
                    v.as_f64().map(|v| v as f32).ok_or_else(|| {
                        RathError::invalid("embedding contains a non-number", &response)
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(EmbedResponse { values })
        })
        .await
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    format!("{}/v1/chat/completions", base_url.trim_end_matches('/'))
}

fn embed_endpoint(base_url: &str) -> String {
    format!("{}/api/embed", base_url.trim_end_matches('/'))
}

fn with_bearer_auth(
    request: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    match api_key {
        Some(api_key) => request.bearer_auth(api_key),
        None => request,
    }
}

/// Rejects empty history and a final assistant tool call without subsequent results.
fn validate_history(messages: &[Message]) -> Result<(), RathError> {
    if messages.is_empty() {
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            "messages must not be empty",
        ));
    }
    if matches!(
        messages.last().map(|m| &m.role),
        Some(Role::AssistantToolCalls { .. })
    ) {
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            "history ends with assistant tool calls without tool results",
        ));
    }
    Ok(())
}

/// Constructs provider wire input, including enabled tools, schemas and generation settings.
fn build_payload(
    model: &str,
    options: &LlmOptions,
    messages: &[Message],
    tools_enabled: bool,
) -> Value {
    let thinking_requested = options
        .thinking
        .as_ref()
        .is_some_and(|t| *t != ThinkingLevel::Off);
    let schema_hint = if options.wants_json_output() {
        options.output_schema.as_ref()
    } else {
        None
    };

    let mut payload = json!({
        "model": model,
        "messages": build_messages(
            messages,
            options.effective_preamble().as_deref(),
            schema_hint,
        ),
        "stream": false,
    });

    if let Some(level) = &options.thinking {
        payload["reasoning_effort"] = json!(reasoning_effort(level));
    }
    if let Some(cap) = options.max_output_tokens {
        payload["max_tokens"] = json!(cap);
    }
    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        if options.tool_choice == ToolChoice::Required {
            payload["tool_choice"] = Value::String("required".into());
        }
    }
    // Preserve JSON-mode selection without treating an unspecified setting as Off.
    if options.wants_json_output() && !thinking_requested {
        payload["response_format"] = json!({ "type": "json_object" });
    }

    payload
}

/// Maps every typed level to Ollama's compatibility API; deployment rejection stays an error.
fn reasoning_effort(level: &ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "none",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "max",
    }
}

/// Serializes retained history and adapter instructions into provider messages.
fn build_messages(
    history: &[Message],
    preamble: Option<&str>,
    schema_hint: Option<&Value>,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(
        history.len()
            + if preamble.is_some() || schema_hint.is_some() {
                1
            } else {
                0
            },
    );
    if let Some(system) = combined_system_message(preamble, schema_hint) {
        out.push(json!({ "role": "system", "content": system }));
    }

    for msg in history {
        match &msg.role {
            Role::System => out.push(json!({ "role": "system", "content": msg.content })),
            Role::User => out.push(build_user_message(msg)),
            Role::Assistant => out.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                let tool_calls = build_call_history(calls);
                out.push(json!({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }));
            }
            Role::Tool { call_id } => {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": msg.content,
                }));
                push_tool_attachment_messages(&mut out, &msg.attachments);
            }
        }
    }
    out
}

/// Serializes retained assistant tool calls into the compatible wire format.
fn build_call_history(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.args.to_string(),
                }
            })
        })
        .collect()
}

/// Combines persona and schema instructions without inventing absent content.
fn combined_system_message(preamble: Option<&str>, schema_hint: Option<&Value>) -> Option<String> {
    match (preamble, schema_hint) {
        (Some(preamble), Some(schema)) => Some(format!(
            "{preamble}\n\nRespond with JSON that matches the following schema:\n{schema}"
        )),
        (Some(preamble), None) => Some(preamble.to_owned()),
        (None, Some(schema)) => Some(format!(
            "Respond with JSON that matches the following schema:\n{schema}"
        )),
        (None, None) => None,
    }
}

/// Serializes user text and supported attachments in provider content format.
fn build_user_message(message: &Message) -> Value {
    let content = &message.content;
    if message.attachments.is_empty() {
        return json!({ "role": "user", "content": content });
    }

    let mut parts =
        Vec::with_capacity(message.attachments.len() + if content.is_empty() { 0 } else { 1 });
    parts.extend(message.attachments.iter().filter_map(ollama_image_part));
    if !content.is_empty() {
        parts.push(json!({ "type": "text", "text": content }));
    }
    json!({ "role": "user", "content": parts })
}

fn push_tool_attachment_messages(out: &mut Vec<Value>, attachments: &[Attachment]) {
    for part in attachments.iter().filter_map(ollama_image_part) {
        out.push(json!({
            "role": "user",
            "content": [part]
        }));
    }
}

/// Serializes supported images and omits unsupported attachment forms.
fn ollama_image_part(att: &Attachment) -> Option<Value> {
    let url = match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => {
            Some(format!("data:{mime_type};base64,{data}"))
        }
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(url.clone()),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "Ollama attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Ollama serialization; dropping");
            None
        }
    }?;
    Some(json!({
        "type": "image_url",
        "image_url": {
            "url": url,
        }
    }))
}

/// Serializes enabled tool definitions and their parameter schemas for this provider.
fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

/// Rejects token-limited output before normalizing content, calls and usage.
fn map_response(response: Value, options: &LlmOptions) -> Result<LlmResponse, RathError> {
    crate::llm::counting::check_output_limit(Provider::Ollama, &response)?;
    let message = response
        .pointer("/choices/0/message")
        .ok_or_else(|| RathError::invalid("missing choices[0].message", &response))?;
    let enabled = !options.tools.is_empty() && options.tool_choice != ToolChoice::Disabled;
    let calls = collect_tool_calls(message, enabled)?;
    validate_call_policy(
        &calls,
        options,
        response.pointer("/choices/0/finish_reason"),
    )?;
    let output = if calls.is_empty() {
        let text = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RathError::invalid(
                    "missing choices[0].message.content or tool_calls",
                    &response,
                )
            })?;
        LlmOutput::Output(decode_ollama_output(
            strip_thinking(text),
            options.wants_json_output(),
        )?)
    } else {
        LlmOutput::ToolCalls {
            thought: message
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_owned),
            calls,
        }
    };
    Ok(LlmResponse::new(Provider::Ollama, output)
        .with_usage(response.get("usage").map(usage_from_value))
        .with_provider_model(
            response
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
        )
        .with_raw_metadata(Some(response_metadata(&response))))
}

/// Retains explicit provider completion evidence without inventing missing usage or reasoning.
fn response_metadata(response: &Value) -> Value {
    json!({
        "id": response.get("id"),
        "finish_reason": response.pointer("/choices/0/finish_reason"),
        "reasoning": response.pointer("/choices/0/message/reasoning"),
        "usage": response.get("usage"),
    })
}

/// Preserves valid JSON verbatim in value form and explains failures after legacy markdown repair.
fn decode_ollama_output(text: &str, wants_json_output: bool) -> Result<Value, RathError> {
    if !wants_json_output {
        return decode_output_text(text, false);
    }
    if let Ok(value) = serde_json::from_str(strip_json_code_fence(text).unwrap_or(text)) {
        return Ok(value);
    }

    let sanitized = sanitize_json_markdown(text);
    parse_json_output(sanitized.as_ref())
        .map(strip_markdown_json_keys)
        .map_err(|cause| {
            RathError::new(
                crate::core::ErrorKind::Deserialize,
                "Ollama returned invalid JSON output",
            )
            .with_source(cause)
        })
}

/// Removes only a leading reasoning block, preserving delimiters inside ordinary text or JSON.
fn strip_thinking(text: &str) -> &str {
    if text.trim_start().starts_with("<think>")
        && let Some((_, answer)) = text.split_once("</think>")
    {
        answer.trim_start()
    } else {
        text
    }
}

/// Removes model-produced JSON markdown wrappers before decoding.
fn sanitize_json_markdown(text: &str) -> Cow<'_, str> {
    let unfenced = strip_json_code_fence(text).unwrap_or(text);
    let repaired = repair_markdown_wrapped_keys(unfenced);

    if unfenced == text {
        repaired
    } else {
        match repaired {
            Cow::Borrowed(_) => Cow::Owned(unfenced.to_owned()),
            Cow::Owned(value) => Cow::Owned(value),
        }
    }
}

/// Borrows the content inside a recognized JSON code fence when present.
fn strip_json_code_fence(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return None;
    }

    let body_start = trimmed.find('\n')? + 1;
    let body = &trimmed[body_start..];
    let body_end = body.rfind("\n```")?;
    Some(body[..body_end].trim())
}

/// Repairs markdown-wrapped JSON keys while preserving quoted string contents.
fn repair_markdown_wrapped_keys(text: &str) -> Cow<'_, str> {
    let mut repaired = String::with_capacity(text.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;
    while index < text.len() {
        let rest = &text[index..];
        let Some(ch) = rest.chars().next() else {
            break;
        };

        if in_string {
            repaired.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += ch.len_utf8();
            continue;
        }

        if ch == '"' {
            in_string = true;
            repaired.push(ch);
            index += ch.len_utf8();
            continue;
        }

        if let Some((close_end, colon_index, key)) = markdown_wrapped_key(rest) {
            push_json_key(&mut repaired, key);
            repaired.push_str(&rest[close_end..colon_index]);
            index += colon_index;
            changed = true;
            continue;
        }

        repaired.push(ch);
        index += ch.len_utf8();
    }

    if changed {
        Cow::Owned(repaired)
    } else {
        Cow::Borrowed(text)
    }
}

/// Finds a markdown-wrapped object key followed by a JSON colon.
fn markdown_wrapped_key(text: &str) -> Option<(usize, usize, &str)> {
    let marker = if text.starts_with("**") {
        "**"
    } else if text.starts_with("__") {
        "__"
    } else {
        return None;
    };

    let inner = &text[marker.len()..];
    let close_start = marker.len() + inner.find(marker)?;
    let key = &text[marker.len()..close_start];
    if key.is_empty() || key.contains('\n') {
        return None;
    }

    let close_end = close_start + marker.len();
    for (offset, ch) in text[close_end..].char_indices() {
        if ch.is_whitespace() {
            continue;
        }
        return (ch == ':').then_some((close_end, close_end + offset, key));
    }
    None
}

/// Writes an escaped JSON key into the local repair buffer.
fn push_json_key(output: &mut String, key: &str) {
    output.push('"');
    for ch in key.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            _ => output.push(ch),
        }
    }
    output.push('"');
}

/// Removes recognized markdown key markers before parsing model JSON.
fn strip_markdown_json_keys(value: Value) -> Value {
    match value {
        Value::Array(items) => {
            Value::Array(items.into_iter().map(strip_markdown_json_keys).collect())
        }
        Value::Object(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(key, value)| {
                    (
                        strip_markdown_key(&key).to_owned(),
                        strip_markdown_json_keys(value),
                    )
                })
                .collect(),
        ),
        other => other,
    }
}

fn strip_markdown_key(key: &str) -> &str {
    if key.len() > 4 {
        for marker in ["**", "__"] {
            if key.starts_with(marker) && key.ends_with(marker) {
                return &key[marker.len()..key.len() - marker.len()];
            }
        }
    }
    key
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: token_count(value, "prompt_tokens"),
        output: token_count(value, "completion_tokens"),
    }
}

fn token_count(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
}

#[cfg(test)]
#[path = "tests/ollama.rs"]
mod tests;
