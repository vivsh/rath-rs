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

pub fn new_client(url: &ModelUrl, options: LlmOptions) -> Result<Box<dyn LlmClient>, RathError> {
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

    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        validate_history(messages)?;
        validate_tools(Provider::OpenRouter, &self.options.tools)?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        let response: Value = self
            .http
            .post(chat_completions_endpoint(&self.base_url))
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?
            .error_for_status()
            .map_err(|e| RathError::Provider(e.to_string()))?
            .json()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;

        map_response(response, wants_json_output)
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

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

fn map_response(response: Value, wants_json_output: bool) -> Result<LlmResponse, RathError> {
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
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or(RathError::EmptyResponse)?;

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
        .ok_or(RathError::EmptyResponse)?;
    Ok(LlmResponse::new(
        Provider::OpenRouter,
        LlmOutput::Output(decode_output_text(text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn collect_tool_calls(message: &Value) -> Result<Vec<ToolCall>, RathError> {
    let mut calls = Vec::new();
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| RathError::Validation("OpenRouter tool call missing id".into()))?;
        let function = call
            .get("function")
            .ok_or_else(|| RathError::Validation("OpenRouter tool call missing function".into()))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RathError::Validation("OpenRouter tool call missing function name".into())
            })?;
        let raw_args = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let args = serde_json::from_str(raw_args).map_err(|e| RathError::Deserialize {
            source: e,
            raw: raw_args.to_string(),
        })?;
        calls.push(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            args,
            thought_signatures: None,
        });
    }
    Ok(calls)
}

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
mod tests {
    use super::*;

    #[test]
    fn endpoint_uses_chat_completions() {
        assert_eq!(
            chat_completions_endpoint("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn payload_uses_openrouter_model_slug() {
        let payload = build_payload(
            "openai/gpt-5.2",
            &LlmOptions::default(),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["model"], "openai/gpt-5.2");
        assert_eq!(payload["messages"][0]["content"], "hi");
    }

    #[test]
    fn payload_with_schema_uses_openrouter_json_schema_shape() {
        let payload = build_payload(
            "openai/gpt-5.2",
            &LlmOptions::default().with_output_schema(json!({
                "type": "object",
                "properties": {
                    "ok": { "type": "boolean" }
                }
            })),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["response_format"]["type"], "json_schema");
        assert_eq!(
            payload["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn maps_tool_call_response() {
        let response = json!({
            "id": "gen-1",
            "object": "chat.completion",
            "model": "openai/gpt-5.2",
            "usage": { "prompt_tokens": 10, "completion_tokens": 4 },
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"x\"}"
                        }
                    }]
                }
            }]
        });
        let mapped = map_response(response, false).unwrap();
        assert_eq!(mapped.provider, Provider::OpenRouter);
        assert_eq!(mapped.usage.unwrap().total(), Some(14));
        match mapped.output {
            LlmOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].name, "lookup"),
            _ => panic!("expected tool calls"),
        }
    }
}
