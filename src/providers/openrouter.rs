use crate::core::error::http;
mod measurement;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use crate::llm::{
    Attachment, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ModelUrl, Provider,
    RathError, Role, TokenUsage, ToolCall, ToolChoice, ToolDefinition, configured_base_url,
    decode_output_text, required_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

struct OpenRouterClient {
    http: HttpClient,
    api_key: String,
    base_url: String,
    model: String,
    options: LlmOptions,
    url: ModelUrl,
}

/// Creates an LLM client after validating output caps and resolving credentials.
pub fn new_client(url: &ModelUrl, options: LlmOptions) -> Result<Box<dyn LlmClient>, RathError> {
    crate::llm::counting::validate_options(Provider::OpenRouter, &options)?;
    let api_key = required_api_key(url, "OPENROUTER_API_KEY")?;
    Ok(Box::new(OpenRouterClient {
        http: HttpClient::new(),
        api_key,
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
        url: url.clone(),
    }))
}

#[async_trait]
impl LlmClient for OpenRouterClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &LlmOptions {
        &self.options
    }

    fn estimate_tokens(&self, messages: &[Message]) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&self.options, messages)
            .map_err(|error| {
                error
                    .with_context(Provider::OpenRouter, "token estimation")
                    .sanitized(&[&self.api_key])
            })
    }

    fn estimate_content_tokens(&self, content: &str) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&LlmOptions::default(), &[Message::user(content)])
            .map_err(|error| {
                error
                    .with_context(Provider::OpenRouter, "token estimation")
                    .sanitized(&[&self.api_key])
            })
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        crate::llm::counting::validate_options(Provider::OpenRouter, &self.options).map_err(
            |error| {
                error
                    .with_context(Provider::OpenRouter, "generation")
                    .sanitized(&[&self.api_key])
            },
        )?;
        validate_history(messages).map_err(|error| {
            error
                .with_context(Provider::OpenRouter, "generation")
                .sanitized(&[&self.api_key])
        })?;
        validate_tools(Provider::OpenRouter, &self.options.tools).map_err(|error| {
            error
                .with_context(Provider::OpenRouter, "generation")
                .sanitized(&[&self.api_key])
        })?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        http::mapped(
            self.http
                .post(chat_completions_endpoint(&self.base_url))
                .bearer_auth(&self.api_key)
                .json(&payload),
            Provider::OpenRouter,
            "generation",
            &[&self.api_key],
            |response| map_response(response, wants_json_output),
        )
        .await
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
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
    let mut payload = json!({
        "model": model,
        "messages": build_messages(messages, options.effective_preamble().as_deref()),
        "stream": false,
    });

    if let Some(cap) = options.max_output_tokens {
        payload["max_tokens"] = json!(cap);
    }
    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        payload["tool_choice"] = match options.tool_choice {
            ToolChoice::Required => Value::String("required".into()),
            ToolChoice::Auto => Value::String("auto".into()),
            ToolChoice::Disabled => Value::String("none".into()),
        };
    }

    if options.wants_json_output() {
        payload["response_format"] = match &options.output_schema {
            Some(schema) => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "strict": true,
                    "schema": schema,
                }
            }),
            None => json!({ "type": "json_object" }),
        };
    }

    payload
}

/// Serializes retained history and adapter instructions into provider messages.
fn build_messages(history: &[Message], preamble: Option<&str>) -> Vec<Value> {
    let mut out = Vec::with_capacity(history.len() + usize::from(preamble.is_some()));
    if let Some(system) = preamble {
        out.push(json!({ "role": "system", "content": system }));
    }

    for msg in history {
        match &msg.role {
            Role::System => out.push(json!({ "role": "system", "content": msg.content })),
            Role::User => out.push(build_user_message(msg)),
            Role::Assistant => out.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                let tool_calls: Vec<Value> = calls
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
                    .collect();
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

/// Serializes user text and supported attachments in provider content format.
fn build_user_message(message: &Message) -> Value {
    if message.attachments.is_empty() {
        return json!({ "role": "user", "content": message.content });
    }

    let mut parts = Vec::with_capacity(
        message.attachments.len() + if message.content.is_empty() { 0 } else { 1 },
    );
    parts.extend(message.attachments.iter().filter_map(openrouter_image_part));
    if !message.content.is_empty() {
        parts.push(json!({ "type": "text", "text": message.content }));
    }
    json!({ "role": "user", "content": parts })
}

fn push_tool_attachment_messages(out: &mut Vec<Value>, attachments: &[Attachment]) {
    for part in attachments.iter().filter_map(openrouter_image_part) {
        out.push(json!({
            "role": "user",
            "content": [part]
        }));
    }
}

/// Serializes supported image attachments and omits unsupported attachment forms.
fn openrouter_image_part(att: &Attachment) -> Option<Value> {
    let url = match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => {
            Some(format!("data:{mime_type};base64,{data}"))
        }
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(url.clone()),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "OpenRouter attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before OpenRouter serialization; dropping");
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
fn map_response(response: Value, wants_json_output: bool) -> Result<LlmResponse, RathError> {
    crate::llm::counting::check_output_limit(Provider::OpenRouter, &response)?;
    let usage = response.get("usage").map(usage_from_value);
    let provider_model = response
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let metadata = Some(json!({
        "id": response.get("id").cloned().unwrap_or(Value::Null),
        "object": response.get("object").cloned().unwrap_or(Value::Null),
    }));
    let message = response
        .pointer("/choices/0/message")
        .ok_or_else(|| RathError::invalid("missing choices[0].message", &response))?;

    let calls = collect_tool_calls(message)?;
    if !calls.is_empty() {
        return Ok(LlmResponse::new(
            Provider::OpenRouter,
            LlmOutput::ToolCalls {
                thought: message
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }

    let text = message
        .get("content")
        .and_then(Value::as_str)
        .ok_or(RathError::new(
            crate::core::ErrorKind::InvalidResponse,
            "missing choices[0].message.content or tool_calls",
        ))?;
    Ok(LlmResponse::new(
        Provider::OpenRouter,
        LlmOutput::Output(decode_output_text(text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

/// Parses provider function calls and rejects malformed arguments or missing identifiers.
fn collect_tool_calls(message: &Value) -> Result<Vec<ToolCall>, RathError> {
    let mut calls = Vec::new();
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = call.get("id").and_then(Value::as_str).ok_or_else(|| {
            RathError::new(
                crate::core::ErrorKind::InvalidResponse,
                "OpenRouter tool call missing id",
            )
        })?;
        let function = call.get("function").ok_or_else(|| {
            RathError::new(
                crate::core::ErrorKind::InvalidResponse,
                "OpenRouter tool call missing function",
            )
        })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RathError::new(
                    crate::core::ErrorKind::InvalidResponse,
                    "OpenRouter tool call missing function name",
                )
            })?;
        let raw_args = function
            .get("arguments")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RathError::new(
                    crate::core::ErrorKind::InvalidResponse,
                    "function call missing string arguments",
                )
            })?;
        let args =
            serde_json::from_str(raw_args).map_err(|e| RathError::deserialize(&e, raw_args))?;
        calls.push(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            args,
            thought_signatures: None,
        });
    }
    Ok(calls)
}

/// Extracts provider-reported input/output usage when present.
fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
#[path = "openrouter/tests/mod.rs"]
mod tests;
