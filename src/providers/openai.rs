use async_trait::async_trait;
use reqwest::Client as HttpClient;
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};

use crate::audio::stt::{SttClient, SttOptions, SttRequest, SttResponse};
use crate::audio::tts::{TtsClient, TtsOptions, TtsRequest, TtsResponse};
use crate::embeddings::{EmbedRequest, EmbedResponse, EmbeddingClient, EmbeddingOptions};

use crate::llm::{
    Attachment, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ModelUrl, Provider,
    RathError, Role, TokenUsage, ToolCall, ToolChoice, ToolDefinition, configured_base_url,
    decode_output_text, required_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

struct OpenAiClient {
    http: HttpClient,
    api_key: String,
    base_url: String,
    model: String,
    options: LlmOptions,
    url: ModelUrl,
    provider_config: Option<Value>,
}

pub fn new_client(url: &ModelUrl, options: LlmOptions) -> Result<Box<dyn LlmClient>, RathError> {
    let provider_config = options.provider_config.clone();
    Ok(Box::new(build_client(url, options, provider_config)?))
}

pub fn new_embedding_client(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<Box<dyn EmbeddingClient>, RathError> {
    Ok(Box::new(build_client(
        url,
        LlmOptions::default(),
        options.provider_config,
    )?))
}

pub fn new_tts_client(
    url: &ModelUrl,
    options: TtsOptions,
) -> Result<Box<dyn TtsClient>, RathError> {
    Ok(Box::new(build_client(
        url,
        LlmOptions::default(),
        options.provider_config,
    )?))
}

pub fn new_stt_client(
    url: &ModelUrl,
    options: SttOptions,
) -> Result<Box<dyn SttClient>, RathError> {
    Ok(Box::new(build_client(
        url,
        LlmOptions::default(),
        options.provider_config,
    )?))
}

fn build_client(
    url: &ModelUrl,
    options: LlmOptions,
    provider_config: Option<Value>,
) -> Result<OpenAiClient, RathError> {
    let api_key = required_api_key(url, "OPENAI_API_KEY")?;
    Ok(OpenAiClient {
        http: HttpClient::new(),
        api_key,
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
        url: url.clone(),
        provider_config,
    })
}

#[async_trait]
impl LlmClient for OpenAiClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &LlmOptions {
        &self.options
    }

    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        validate_history(messages)?;
        validate_tools(Provider::OpenAi, &self.options.tools)?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        let response: Value = self
            .http
            .post(responses_endpoint(&self.base_url))
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

#[async_trait]
impl EmbeddingClient for OpenAiClient {
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        let payload = json!({
            "model": self.model,
            "input": request.input,
            "encoding_format": "float",
        });
        let response: Value = self
            .http
            .post(embeddings_endpoint(&self.base_url))
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
        let values: Vec<f32> = response["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| RathError::Provider("embedding missing in response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        Ok(EmbedResponse { values })
    }
}

#[async_trait]
impl TtsClient for OpenAiClient {
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let mut payload = json_object_from(&self.provider_config);
        merge_json_object(&mut payload, &request.provider_config);
        payload.insert(
            "model".to_string(),
            Value::String(request.model.clone().unwrap_or_else(|| self.model.clone())),
        );
        payload.insert("input".to_string(), Value::String(request.input.clone()));
        if let Some(voice) = &request.voice {
            payload.insert("voice".to_string(), Value::String(voice.clone()));
        }
        if let Some(format) = &request.format {
            payload.insert("response_format".to_string(), Value::String(format.clone()));
        }

        let response = self
            .http
            .post(speech_endpoint(&self.base_url))
            .bearer_auth(&self.api_key)
            .json(&Value::Object(payload))
            .send()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?
            .error_for_status()
            .map_err(|e| RathError::Provider(e.to_string()))?;
        let mime_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_string();
        let data = response
            .bytes()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?
            .to_vec();
        Ok(TtsResponse {
            mime_type,
            data,
            raw_metadata: None,
        })
    }
}

#[async_trait]
impl SttClient for OpenAiClient {
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let file = Part::bytes(request.data.clone())
            .file_name("audio")
            .mime_str(&request.mime_type)
            .map_err(|e| RathError::Validation(e.to_string()))?;
        let form = Form::new().text("model", model).part("file", file);
        let form = add_form_fields(form, &self.provider_config);
        let form = add_form_fields(form, &request.provider_config);

        let response: Value = self
            .http
            .post(transcriptions_endpoint(&self.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?
            .error_for_status()
            .map_err(|e| RathError::Provider(e.to_string()))?
            .json()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;
        let text = response
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| RathError::Provider("transcription text missing in response".into()))?
            .to_string();
        Ok(SttResponse {
            text,
            raw_metadata: Some(response),
        })
    }
}

fn responses_endpoint(base_url: &str) -> String {
    format!("{}/responses", base_url.trim_end_matches('/'))
}

fn embeddings_endpoint(base_url: &str) -> String {
    format!("{}/embeddings", base_url.trim_end_matches('/'))
}

fn speech_endpoint(base_url: &str) -> String {
    format!("{}/audio/speech", base_url.trim_end_matches('/'))
}

fn transcriptions_endpoint(base_url: &str) -> String {
    format!("{}/audio/transcriptions", base_url.trim_end_matches('/'))
}

fn json_object_from(value: &Option<Value>) -> serde_json::Map<String, Value> {
    match value {
        Some(Value::Object(map)) => map.clone(),
        _ => serde_json::Map::new(),
    }
}

fn merge_json_object(payload: &mut serde_json::Map<String, Value>, value: &Option<Value>) {
    if let Some(Value::Object(map)) = value {
        for (key, value) in map {
            payload.insert(key.clone(), value.clone());
        }
    }
}

fn add_form_fields(mut form: Form, value: &Option<Value>) -> Form {
    if let Some(Value::Object(map)) = value {
        for (key, value) in map {
            let field_value = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            form = form.text(key.clone(), field_value);
        }
    }
    form
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
        "input": build_input(messages),
    });

    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if let Some(preamble) = options.effective_preamble() {
        payload["instructions"] = Value::String(preamble);
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        payload["tool_choice"] = match options.tool_choice {
            ToolChoice::Required => Value::String("required".to_string()),
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::Disabled => Value::String("none".to_string()),
        };
    }
    if options.wants_json_output() {
        payload["text"] = json!({
            "format": match &options.output_schema {
                Some(schema) => json!({
                    "type": "json_schema",
                    "name": "agent_output",
                    "schema": schema,
                    "strict": true
                }),
                None => json!({ "type": "json_object" }),
            }
        });
    }

    payload
}

fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for msg in messages {
        match &msg.role {
            Role::System => input.push(json!({ "role": "system", "content": msg.content })),
            Role::User => input.push(build_user_input(msg)),
            Role::Assistant => input.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                for call in calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.args.to_string(),
                    }));
                }
            }
            Role::Tool { call_id } => {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": msg.content,
                }));
                push_tool_attachment_inputs(&mut input, &msg.attachments);
            }
        }
    }
    input
}

fn build_user_input(msg: &Message) -> Value {
    if msg.attachments.is_empty() {
        return json!({ "role": "user", "content": msg.content });
    }

    let mut content = msg
        .attachments
        .iter()
        .filter_map(openai_image_content)
        .collect::<Vec<_>>();
    if !msg.content.is_empty() {
        content.push(json!({ "type": "input_text", "text": msg.content }));
    }
    json!({ "role": "user", "content": content })
}

fn push_tool_attachment_inputs(input: &mut Vec<Value>, attachments: &[Attachment]) {
    for image_url in attachments.iter().filter_map(openai_image_url) {
        input.push(json!({
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": image_url,
            }]
        }));
    }
}

fn openai_image_content(att: &Attachment) -> Option<Value> {
    openai_image_url(att).map(|image_url| {
        json!({
            "type": "input_image",
            "image_url": image_url,
        })
    })
}

fn openai_image_url(att: &Attachment) -> Option<String> {
    match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => {
            Some(format!("data:{mime_type};base64,{data}"))
        }
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(url.clone()),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "OpenAI attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before OpenAI serialization; dropping");
            None
        }
    }
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": true,
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
        "status": response.get("status").cloned().unwrap_or(Value::Null),
    }));

    let calls = collect_tool_calls(&response)?;
    if !calls.is_empty() {
        return Ok(LlmResponse::new(
            Provider::OpenAi,
            LlmOutput::ToolCalls {
                thought: collect_text(&response),
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }

    let text = collect_text(&response).ok_or(RathError::EmptyResponse)?;
    Ok(LlmResponse::new(
        Provider::OpenAi,
        LlmOutput::Output(decode_output_text(&text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn collect_tool_calls(response: &Value) -> Result<Vec<ToolCall>, RathError> {
    let mut calls = Vec::new();
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        for item in output {
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                continue;
            }
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    RathError::Validation("OpenAI function call missing call_id".into())
                })?;
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| RathError::Validation("OpenAI function call missing name".into()))?;
            let raw_args = item
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
    }
    Ok(calls)
}

fn collect_text(response: &Value) -> Option<String> {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    let mut out = String::new();
    for item in response.get("output").and_then(Value::as_array)? {
        for content in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if matches!(
                content.get("type").and_then(Value::as_str),
                Some("output_text" | "text")
            ) && let Some(text) = content.get("text").and_then(Value::as_str)
            {
                out.push_str(text);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("input_tokens")
            .or_else(|| value.get("prompt_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("output_tokens")
            .or_else(|| value.get("completion_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn custom_base_url_builds_openai_responses_endpoint() {
        assert_eq!(
            responses_endpoint("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/responses"
        );
        assert_eq!(
            embeddings_endpoint("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/embeddings"
        );
    }

    #[test]
    fn responses_payload_uses_schema_and_required_tools() {
        let options = LlmOptions::default()
            .with_tool_choice(ToolChoice::Required)
            .with_tools(vec![ToolDefinition {
                name: "lookup".into(),
                description: "Lookup a thing.".into(),
                parameters: json!({"type":"object","properties":{}}),
            }]);
        let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);
        assert_eq!(payload["model"], "custom-model");
        assert_eq!(payload["tool_choice"], "required");
        assert_eq!(payload["tools"][0]["name"], "lookup");
    }

    #[test]
    fn responses_payload_appends_input_schema_to_instructions() {
        let options = LlmOptions::default()
            .with_preamble("You are helpful.")
            .with_input_schema(json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" }
                },
                "required": ["kind"]
            }));

        let payload = build_payload("custom-model", &options, &[Message::user("hi")], false);

        let instructions = payload["instructions"]
            .as_str()
            .expect("instructions should be a string");
        assert!(instructions.contains("You are helpful."));
        assert!(instructions.contains("The user message is JSON."));
        assert!(instructions.contains("\"required\":[\"kind\"]"));
    }

    #[test]
    fn payload_without_input_schema_uses_text_mode() {
        let payload = build_payload(
            "custom-model",
            &LlmOptions::default(),
            &[Message::user("hi")],
            false,
        );
        assert!(payload.get("text").is_none());
    }

    /// Explicit JSON response mode uses the OpenAI json_object format when no schema is supplied.
    #[test]
    fn payload_with_json_response_and_no_output_schema_uses_json_object_mode() {
        let payload = build_payload(
            "custom-model",
            &LlmOptions::default().with_response_format(crate::llm::ResponseFormat::Json),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["text"]["format"]["type"], "json_object");
    }

    /// Non-image attachments are ignored on the OpenAI image-only wire path.
    #[test]
    fn non_image_user_attachments_are_dropped() {
        let payload = build_payload(
            "custom-model",
            &LlmOptions::default(),
            &[Message {
                role: Role::User,
                content: "describe this".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "application/pdf".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            false,
        );

        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"].as_array().unwrap().len(), 1);
        assert_eq!(payload["input"][0]["content"][0]["type"], "input_text");
    }

    /// User attachments are encoded as OpenAI `input_image` content items.
    #[test]
    fn user_attachments_use_content_array() {
        let payload = build_payload(
            "custom-model",
            &LlmOptions::default(),
            &[Message {
                role: Role::User,
                content: "describe this".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            false,
        );

        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"][0]["type"], "input_image");
        assert_eq!(payload["input"][0]["content"][1]["type"], "input_text");
    }

    /// Tool outputs remain unchanged when a reminder is appended as a later user turn.
    #[test]
    fn build_input_keeps_tool_output_and_reminder_separate() {
        let input = build_input(&[
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        args: json!({"q":"x"}),
                        thought_signatures: None,
                    }],
                },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
            Message::tool_output("call_1".into(), r#"{"ok":true}"#),
            Message::user("FINAL TURN: call final_answer"),
        ]);

        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["output"], r#"{"ok":true}"#);
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"], "FINAL TURN: call final_answer");
    }

    #[test]
    fn schema_and_tools_openai_prefers_tools_over_structured_output() {
        let options = LlmOptions::default()
            .with_tool_choice(ToolChoice::Required)
            .with_output_schema(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }))
            .with_tools(vec![ToolDefinition {
                name: "submit".into(),
                description: "Submit the final answer.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"]
                }),
            }]);

        let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);

        assert!(
            payload.get("text").is_some(),
            "text format should be set alongside tools"
        );
        assert_eq!(payload["tools"][0]["name"], "submit");
        assert_eq!(payload["tool_choice"], "required");
    }

    #[test]
    fn maps_response_usage_and_tool_call() {
        let response = json!({
            "id": "resp_1",
            "model": "gpt-x",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "output": [{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}]
        });
        let mapped = map_response(response, false).unwrap();
        assert_eq!(mapped.usage.unwrap().total(), Some(15));
        assert_eq!(mapped.provider_model.as_deref(), Some("gpt-x"));
        match mapped.output {
            LlmOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "call_1"),
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn map_response_without_json_mode_returns_string() {
        let response = json!({
            "id": "resp_1",
            "model": "gpt-x",
            "output_text": "plain text"
        });
        let mapped = map_response(response, false).unwrap();
        match mapped.output {
            LlmOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
            _ => panic!("expected string output"),
        }
    }
}
