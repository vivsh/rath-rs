use crate::core::error::http;
mod measurement;

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
    crate::llm::counting::validate_options(Provider::OpenAi, &options)?;
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

/// Constructs the provider client with resolved credentials and endpoint settings.
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

    fn estimate_tokens(&self, messages: &[Message]) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&self.options, messages)
            .map_err(|error| {
                error
                    .with_context(Provider::OpenAi, "token estimation")
                    .sanitized(&[&self.api_key])
            })
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&self.options, messages)
            .await
            .map_err(|error| {
                error
                    .with_context(Provider::OpenAi, "token counting")
                    .sanitized(&[&self.api_key])
            })
    }

    fn estimate_content_tokens(&self, content: &str) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&LlmOptions::default(), &[Message::user(content)])
            .map_err(|error| {
                error
                    .with_context(Provider::OpenAi, "token estimation")
                    .sanitized(&[&self.api_key])
            })
    }

    async fn count_content_tokens(
        &self,
        content: &str,
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&LlmOptions::default(), &[Message::user(content)])
            .await
            .map_err(|error| {
                error
                    .with_context(Provider::OpenAi, "token counting")
                    .sanitized(&[&self.api_key])
            })
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        crate::llm::counting::validate_options(Provider::OpenAi, &self.options).map_err(
            |error| {
                error
                    .with_context(Provider::OpenAi, "generation")
                    .sanitized(&[&self.api_key])
            },
        )?;
        validate_history(messages).map_err(|error| {
            error
                .with_context(Provider::OpenAi, "generation")
                .sanitized(&[&self.api_key])
        })?;
        validate_tools(Provider::OpenAi, &self.options.tools).map_err(|error| {
            error
                .with_context(Provider::OpenAi, "generation")
                .sanitized(&[&self.api_key])
        })?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        http::mapped(
            self.http
                .post(responses_endpoint(&self.base_url))
                .bearer_auth(&self.api_key)
                .json(&payload),
            Provider::OpenAi,
            "generation",
            &[&self.api_key],
            |response| map_response(response, wants_json_output),
        )
        .await
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiClient {
    /// Sends the embedding request to the configured provider and normalizes its result.
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        let payload = json!({
            "model": self.model,
            "input": request.input,
            "encoding_format": "float",
        });
        http::mapped(
            self.http
                .post(embeddings_endpoint(&self.base_url))
                .bearer_auth(&self.api_key)
                .json(&payload),
            Provider::OpenAi,
            "embeddings",
            &[&self.api_key],
            |response| {
                let values: Vec<f32> = response["data"][0]["embedding"]
                    .as_array()
                    .ok_or_else(|| RathError::invalid("missing data[0].embedding", &response))?
                    .iter()
                    .map(|v| {
                        v.as_f64().map(|v| v as f32).ok_or_else(|| {
                            RathError::invalid("embedding contains a non-number", &response)
                        })
                    })
                    .collect::<Result<_, _>>()?;
                Ok(EmbedResponse { values })
            },
        )
        .await
    }
}

#[async_trait]
impl TtsClient for OpenAiClient {
    /// Sends speech synthesis input and returns audio bytes or a provider error.
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let payload = speech_payload(self, request)?;
        let response = http::send(
            self.http
                .post(speech_endpoint(&self.base_url))
                .bearer_auth(&self.api_key)
                .json(&Value::Object(payload)),
            Provider::OpenAi,
            "speech synthesis",
            &[&self.api_key],
        )
        .await?;
        let mime_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_string();
        let data = http::read(
            response,
            Provider::OpenAi,
            "speech synthesis",
            &[&self.api_key],
        )
        .await?;
        Ok(TtsResponse {
            mime_type,
            data,
            raw_metadata: None,
        })
    }
}

/// Preserves preset synthesis while rejecting incompatible custom voices and unsupported controls.
fn speech_payload(
    client: &OpenAiClient,
    request: &TtsRequest,
) -> Result<serde_json::Map<String, Value>, RathError> {
    crate::audio::voice::validate_config(client.provider_config.as_ref())?;
    crate::audio::voice::validate_config(request.provider_config.as_ref())?;
    if request.input.trim().is_empty() {
        return Err(crate::audio::voice::invalid(
            "speech input must not be blank",
        ));
    }
    if request.language.is_some() || request.instructions.is_some() {
        return Err(RathError::unsupported(
            Provider::OpenAi,
            "typed TTS language/instructions",
        ));
    }
    let mut payload = json_object_from(&client.provider_config);
    merge_json_object(&mut payload, &request.provider_config);
    payload.insert(
        "model".into(),
        json!(request.model.as_deref().unwrap_or(&client.model)),
    );
    payload.insert("input".into(), json!(request.input));
    if let Some(voice) = &request.voice {
        voice.validate("openai")?;
        let crate::audio::voice::VoiceData::Id(id) = &voice.data else {
            return Err(RathError::unsupported(
                Provider::OpenAi,
                "selected voice representation",
            ));
        };
        payload.insert("voice".into(), json!(id));
    }
    if let Some(format) = &request.format {
        payload.insert("response_format".into(), json!(format));
    }
    Ok(payload)
}

#[async_trait]
impl SttClient for OpenAiClient {
    /// Uploads audio for transcription and normalizes the returned text and metadata.
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let file = Part::bytes(request.data.clone())
            .file_name("audio")
            .mime_str(&request.mime_type)
            .map_err(|e| {
                RathError::from_error(crate::core::ErrorKind::Validation, &e)
                    .with_context(Provider::OpenAi, "transcription input")
                    .sanitized(&[&self.api_key])
            })?;
        let form = Form::new().text("model", model).part("file", file);
        let form = add_form_fields(form, &self.provider_config);
        let form = add_form_fields(form, &request.provider_config);

        http::mapped(
            self.http
                .post(transcriptions_endpoint(&self.base_url))
                .bearer_auth(&self.api_key)
                .multipart(form),
            Provider::OpenAi,
            "transcription",
            &[&self.api_key],
            |response| {
                let text = response
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| RathError::invalid("missing text", &response))?
                    .to_string();
                Ok(SttResponse {
                    text,
                    raw_metadata: Some(response),
                })
            },
        )
        .await
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

/// Adds supported scalar provider fields to the multipart request.
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
        "input": build_input(messages),
    });

    if let Some(cap) = options.max_output_tokens {
        payload["max_output_tokens"] = json!(cap);
    }
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

/// Preserves history text, tool exchanges and supported images in Responses API input.
fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for msg in messages {
        match &msg.role {
            Role::System => input.push(json!({ "role": "system", "content": msg.content })),
            Role::User => input.push(build_user_input(msg)),
            Role::Assistant => input.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                if !msg.content.is_empty() {
                    input.push(json!({ "role": "assistant", "content": msg.content }));
                }
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

/// Serializes a user message with supported images and text.
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

/// Adds supported tool-produced images as user content after the tool result.
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

/// Resolves supported image data or URLs; unsupported attachment forms are omitted.
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

/// Serializes enabled tool definitions and their parameter schemas for this provider.
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

/// Rejects token-limited output before normalizing content, calls and usage.
fn map_response(response: Value, wants_json_output: bool) -> Result<LlmResponse, RathError> {
    crate::llm::counting::check_output_limit(Provider::OpenAi, &response)?;
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

    let text = collect_text(&response).ok_or(RathError::new(
        crate::core::ErrorKind::InvalidResponse,
        "missing output_text or output[].content[].text",
    ))?;
    Ok(LlmResponse::new(
        Provider::OpenAi,
        LlmOutput::Output(decode_output_text(&text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

/// Parses provider function calls and rejects malformed arguments or missing identifiers.
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
                    RathError::new(
                        crate::core::ErrorKind::InvalidResponse,
                        "OpenAI function call missing call_id",
                    )
                })?;
            let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
                RathError::new(
                    crate::core::ErrorKind::InvalidResponse,
                    "OpenAI function call missing name",
                )
            })?;
            let raw_args = item
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
    }
    Ok(calls)
}

/// Collects text from the Responses API output blocks when present.
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

/// Extracts provider-reported input/output usage when present.
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
#[path = "openai/tests/mod.rs"]
mod tests;
