mod measurement;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use crate::llm::{
    Attachment, CacheControl, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ModelUrl,
    Provider, RathError, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice, ToolDefinition,
    configured_base_url, decode_output_text, required_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

struct AnthropicClient {
    http: HttpClient,
    api_key: String,
    base_url: String,
    model: String,
    options: LlmOptions,
    url: ModelUrl,
}

/// Creates an LLM client after validating output caps and resolving credentials.
pub fn new_client(url: &ModelUrl, options: LlmOptions) -> Result<Box<dyn LlmClient>, RathError> {
    crate::llm::counting::validate_options(Provider::Anthropic, &options)?;
    let api_key = required_api_key(url, "ANTHROPIC_API_KEY")?;
    Ok(Box::new(AnthropicClient {
        http: HttpClient::new(),
        api_key,
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
        url: url.clone(),
    }))
}

#[async_trait]
impl LlmClient for AnthropicClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &crate::llm::LlmOptions {
        &self.options
    }

    fn estimate_tokens(&self, messages: &[Message]) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&self.options, messages)
    }

    fn estimate_content_tokens(&self, content: &str) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&LlmOptions::default(), &[Message::user(content)])
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&self.options, messages).await
    }

    async fn count_content_tokens(
        &self,
        content: &str,
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&LlmOptions::default(), &[Message::user(content)])
            .await
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        crate::llm::counting::validate_options(Provider::Anthropic, &self.options)?;
        validate_history(messages)?;
        validate_tools(Provider::Anthropic, &self.options.tools)?;

        if matches!(&self.options.thinking, Some(t) if *t != ThinkingLevel::Off) {
            return Err(RathError::UnsupportedCapability {
                provider: Provider::Anthropic,
                capability: "thinking is not exposed by the Anthropic adapter yet".into(),
            });
        }

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let response = send_messages_request(
            &self.http,
            messages_endpoint(&self.base_url),
            &self.api_key,
            &payload,
        )
        .await?;

        map_response(response, wants_json_output)
    }
}

/// Sends a single Anthropic request without retrying or relaxing its output cap.
async fn send_messages_request(
    http: &HttpClient,
    endpoint: String,
    api_key: &str,
    payload: &Value,
) -> Result<Value, RathError> {
    let response = http
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(payload)
        .send()
        .await
        .map_err(|e| RathError::Provider(e.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| RathError::Provider(e.to_string()))?;
    if !status.is_success() {
        return Err(RathError::Provider(format_anthropic_http_error(
            status.as_u16(),
            &body,
        )));
    }

    serde_json::from_str(&body).map_err(|e| RathError::Deserialize {
        source: e,
        raw: body,
    })
}

fn messages_endpoint(base_url: &str) -> String {
    format!("{}/messages", base_url.trim_end_matches('/'))
}

/// Formats the structured Anthropic error when present, otherwise retains the error body.
fn format_anthropic_http_error(status: u16, body: &str) -> String {
    let response: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                return format!("Anthropic API request failed with HTTP {status}");
            }
            return format!("Anthropic API request failed with HTTP {status}: {trimmed}");
        }
    };

    let error = response.get("error").unwrap_or(&response);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown Anthropic error");
    let error_type = error.get("type").and_then(Value::as_str);
    match error_type {
        Some(error_type) => {
            format!("Anthropic API request failed with HTTP {status} ({error_type}): {message}")
        }
        None => format!("Anthropic API request failed with HTTP {status}: {message}"),
    }
}

/// Rejects empty history and a final assistant tool call without subsequent results.
fn validate_history(messages: &[Message]) -> Result<(), RathError> {
    if messages.is_empty() {
        return Err(RathError::Validation("messages must not be empty".into()));
    }
    if matches!(
        messages.last().map(|m| &m.role),
        Some(Role::AssistantToolCalls { .. })
    ) {
        return Err(RathError::Validation(
            "history ends with assistant tool calls without tool results".into(),
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
    let mut payload = json!({
        "model": model,
        "max_tokens": options.max_output_tokens.unwrap_or(4096),
        "messages": build_messages(messages),
    });

    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if let Some(system) = build_system(options, messages) {
        payload["system"] = system;
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        if options.tool_choice == ToolChoice::Required {
            payload["tool_choice"] = json!({ "type": "any" });
        }
    }

    payload
}

/// Renders system instructions, schema guidance, and optional cache control once.
fn build_system(options: &LlmOptions, messages: &[Message]) -> Option<Value> {
    let mut system = Vec::new();
    if let Some(preamble) = options.effective_preamble() {
        system.push(preamble);
    }
    for msg in messages {
        if matches!(msg.role, Role::System) {
            system.push(msg.content.clone());
        }
    }
    if options.wants_json_output() {
        let schema_hint = options
            .output_schema
            .as_ref()
            .map(|schema| format!("\n\nReturn only valid JSON matching this JSON Schema: {schema}"))
            .unwrap_or_else(|| "\n\nReturn only valid JSON.".to_string());
        // Append schema hint to last system block (or create one).
        if let Some(last) = system.last_mut() {
            last.push_str(&schema_hint);
        } else {
            system.push(schema_hint.trim_start_matches('\n').to_string());
        }
    }
    if !system.is_empty() {
        if let Some(cache) = &options.cache {
            let cache_control = match cache {
                CacheControl::Ephemeral5m => json!({"type": "ephemeral"}),
                CacheControl::Ephemeral1h => json!({"type": "ephemeral", "ttl": "1h"}),
            };
            return Some(json!([{
                "type": "text",
                "text": system.join("\n\n"),
                "cache_control": cache_control,
            }]));
        } else {
            return Some(Value::String(system.join("\n\n")));
        }
    }

    None
}

/// Serializes retained history and adapter instructions into provider messages.
fn build_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match &msg.role {
            Role::System => {}
            Role::User => out.push(build_user_message(msg)),
            Role::Assistant => out.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                let mut content = Vec::new();
                if !msg.content.is_empty() {
                    content.push(json!({ "type": "text", "text": msg.content }));
                }
                for call in calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.args,
                    }));
                }
                out.push(json!({ "role": "assistant", "content": content }));
            }
            Role::Tool { call_id } => {
                let mut content = anthropic_image_blocks(&msg.attachments);
                content.push(json!({ "type": "text", "text": msg.content }));
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    }]
                }));
            }
        }
    }
    out
}

/// Serializes user text and supported attachments in provider content format.
fn build_user_message(msg: &Message) -> Value {
    if msg.attachments.is_empty() {
        return json!({ "role": "user", "content": msg.content });
    }

    let mut content = anthropic_image_blocks(&msg.attachments);
    if !msg.content.is_empty() {
        content.push(json!({ "type": "text", "text": msg.content }));
    }
    json!({ "role": "user", "content": content })
}

fn anthropic_image_blocks(attachments: &[Attachment]) -> Vec<Value> {
    attachments
        .iter()
        .filter_map(anthropic_image_block)
        .collect()
}

/// Serializes supported image sources and omits unsupported attachment forms.
fn anthropic_image_block(att: &Attachment) -> Option<Value> {
    match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => Some(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            }
        })),
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(json!({
            "type": "image",
            "source": {
                "type": "url",
                "media_type": mime_type,
                "url": url,
            }
        })),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "Anthropic attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Anthropic serialization; dropping");
            None
        }
    }
}

/// Serializes enabled tool definitions and their parameter schemas for this provider.
fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

/// Rejects token-limited output before normalizing content, calls and usage.
fn map_response(response: Value, wants_json_output: bool) -> Result<LlmResponse, RathError> {
    crate::llm::counting::check_output_limit(Provider::Anthropic, &response)?;
    let usage = response.get("usage").map(usage_from_value);
    let provider_model = response
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let metadata = Some(json!({
        "id": response.get("id").cloned().unwrap_or(Value::Null),
        "stop_reason": response.get("stop_reason").cloned().unwrap_or(Value::Null),
    }));

    let (text, calls) = collect_content(&response);
    if !calls.is_empty() {
        return Ok(LlmResponse::new(
            Provider::Anthropic,
            LlmOutput::ToolCalls {
                thought: text,
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }
    let text = text.ok_or(RathError::EmptyResponse)?;
    Ok(LlmResponse::new(
        Provider::Anthropic,
        LlmOutput::Output(decode_output_text(&text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

/// Collects provider text and function calls from returned content blocks.
fn collect_content(response: &Value) -> (Option<String>, Vec<ToolCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    for part in response
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                if let (Some(id), Some(name)) = (
                    part.get("id").and_then(Value::as_str),
                    part.get("name").and_then(Value::as_str),
                ) {
                    calls.push(ToolCall {
                        id: id.to_string(),
                        name: name.to_string(),
                        args: part.get("input").cloned().unwrap_or_else(|| json!({})),
                        thought_signatures: None,
                    });
                }
            }
            _ => {}
        }
    }
    ((!text.is_empty()).then_some(text), calls)
}

/// Extracts provider-reported input/output usage when present.
fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
#[path = "anthropic/tests/mod.rs"]
mod tests;
