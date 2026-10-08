#![allow(deprecated)]
use crate::llm::LlmBackend;
// Keep the existing generateContent wire contract while using the official SDK.
mod embedding;
mod image;
mod media;
pub(crate) use image::new_image_client;
mod stt;
mod tts;
pub(crate) use stt::new_stt_client;
pub(crate) use tts::new_tts_client;
mod errors;
mod measurement;
pub(crate) use embedding::new_embedding_client;
mod request;

use gemini_rust::{
    Blob, Content, FileData as GeminiFileData, FunctionCall as GeminiFunctionCall,
    FunctionCallingMode, FunctionDeclaration, FunctionResponse as GeminiFunctionResponse, Gemini,
    GenerationResponse, Message as GeminiMessage, Part, Role as GeminiRole, SafetySetting,
    Tool as GeminiTool, client::Model as GeminiModel,
};
use serde_json::Value;

use crate::llm::schema;
use crate::llm::{
    Attachment, LlmClient, LlmOptions, LlmOutput, LlmResponse, Message, ModelUrl, Provider,
    RathError, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice, ToolDefinition,
    decode_output_text, validate_tools,
};

/// Constructs the provider client with resolved credentials and endpoint settings.
fn build_client(url: &ModelUrl) -> Result<Gemini, RathError> {
    if url.base_url.is_some() {
        return Err(RathError::unsupported(Provider::Gemini, "custom endpoint"));
    }
    let api_key = if let Some(key) = &url.api_key {
        key.clone()
    } else {
        std::env::var("GEMINI_API_KEY").map_err(|error| {
            RathError::new(
                crate::core::ErrorKind::Validation,
                "GEMINI_API_KEY is not set",
            )
            .with_source(crate::core::error::credential_cause(&error))
        })?
    };
    let model_id = if url.model.starts_with("models/") {
        url.model.clone()
    } else {
        format!("models/{}", url.model)
    };
    let model = GeminiModel::Custom(model_id);
    Gemini::with_model(&api_key, model)
        .map_err(|e| errors::normalize(&e, "client construction", Some(&api_key)))
}

struct GeminiClient {
    client: Gemini,
    options: LlmOptions,
    url: ModelUrl,
}

/// Builds Gemini messages from history.
fn build_gemini_messages(history: &[Message]) -> Vec<GeminiMessage> {
    let mut msgs = Vec::new();
    let mut i = 0;
    while i < history.len() {
        match &history[i].role {
            Role::System => {
                i += 1;
            }
            Role::User => {
                msgs.push(user_to_message(&history[i]));
                i += 1;
            }
            Role::Assistant => {
                msgs.push(GeminiMessage::model(&history[i].content));
                i += 1;
            }
            Role::AssistantToolCalls { calls } => {
                let mut message = tool_calls_to_message(calls);
                if !history[i].content.is_empty() {
                    message.content.parts.get_or_insert_default().insert(
                        0,
                        Part::Text {
                            text: history[i].content.clone(),
                            thought: None,
                            thought_signature: None,
                        },
                    );
                }
                msgs.push(message);
                i += 1;
            }
            Role::Tool { .. } => {
                let (msg, consumed) = tool_responses_to_message(history, i);
                msgs.push(msg);
                i += consumed;
            }
        }
    }
    msgs
}

/// Translates supported attachments into Gemini parts; unmaterialized files are omitted.
fn gemini_part_from_attachment(att: &Attachment) -> Option<Part> {
    match att {
        Attachment::Inline { mime_type, data } => Some(Part::InlineData {
            inline_data: Blob::new(mime_type, data),
            media_resolution: None,
        }),
        Attachment::Url { mime_type, url } => Some(Part::FileData {
            file_data: GeminiFileData {
                mime_type: mime_type.clone(),
                file_uri: url.clone(),
            },
        }),
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Gemini serialization; dropping");
            None
        }
    }
}

/// Preserves user text and supported attachments in a Gemini content message.
fn user_to_message(message: &Message) -> GeminiMessage {
    if message.attachments.is_empty() {
        return GeminiMessage::user(message.content.clone());
    }

    let mut parts = message
        .attachments
        .iter()
        .filter_map(gemini_part_from_attachment)
        .collect::<Vec<_>>();
    if !message.content.is_empty() {
        parts.push(Part::Text {
            text: message.content.clone(),
            thought: None,
            thought_signature: None,
        });
    }
    GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::User),
        },
        role: GeminiRole::User,
    }
}

/// Converts enabled function definitions to the Gemini tool schema or returns an error.
fn build_tools_spec(tools: &[ToolDefinition]) -> Result<Option<GeminiTool>, RathError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let fns: Vec<FunctionDeclaration> = tools
        .iter()
        .map(build_fn_decl)
        .collect::<Result<Vec<_>, _>>()?;
    if fns.is_empty() {
        Ok(None)
    } else {
        Ok(Some(GeminiTool::with_functions(fns)))
    }
}

/// Converts `AssistantToolCalls` history into a model-role message.
fn tool_calls_to_message(calls: &[ToolCall]) -> GeminiMessage {
    let parts: Vec<Part> = calls
        .iter()
        .map(|c| {
            let thought_sig = c
                .thought_signatures
                .as_ref()
                .and_then(|v| v.first())
                .cloned();
            Part::FunctionCall {
                function_call: GeminiFunctionCall::new(&c.name, c.args.clone()),
                thought_signature: thought_sig,
            }
        })
        .collect();
    GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::Model),
        },
        role: GeminiRole::Model,
    }
}

/// Groups consecutive `Tool` history entries into one user-role message.
fn tool_responses_to_message(history: &[Message], start: usize) -> (GeminiMessage, usize) {
    let mut parts = Vec::new();
    let mut i = start;
    while i < history.len() {
        let Role::Tool { call_id } = &history[i].role else {
            break;
        };
        let name = resolve_call_name(history, call_id);
        let val: Value = serde_json::from_str(&history[i].content)
            .unwrap_or_else(|_| Value::String(history[i].content.clone()));
        parts.push(Part::FunctionResponse {
            function_response: GeminiFunctionResponse::new(name, val),
        });
        // Append any attachments produced by this tool call as additional parts.
        for part in history[i]
            .attachments
            .iter()
            .filter_map(gemini_part_from_attachment)
        {
            parts.push(part);
        }
        i += 1;
    }
    let msg = GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::User),
        },
        role: GeminiRole::User,
    };
    (msg, i - start)
}

/// Resolves a tool call name by walking history backwards.
fn resolve_call_name<'a>(history: &'a [Message], call_id: &'a str) -> &'a str {
    for msg in history.iter().rev() {
        if let Role::AssistantToolCalls { calls } = &msg.role {
            for c in calls {
                if c.id == call_id {
                    return &c.name;
                }
            }
        }
    }
    tracing::error!(
        call_id,
        "could not resolve tool call name from history; using call_id as fallback"
    );
    call_id
}

/// Converts a `ToolDefinition` into a Gemini function declaration.
fn build_fn_decl(tool: &ToolDefinition) -> Result<FunctionDeclaration, RathError> {
    let sanitized = schema::sanitize_strict(tool.parameters.clone())?;
    let json = serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": sanitized,
    });
    serde_json::from_value(json)
        .map_err(|error| RathError::from_native_error(crate::core::ErrorKind::Serialize, &error))
}

/// Maps the raw Gemini response into a [`LlmOutput`].
fn map_response(
    response: GenerationResponse,
    wants_json_output: bool,
) -> Result<LlmResponse, RathError> {
    check_generation_limit(&response)?;
    let usage = response.usage_metadata.as_ref().map(|usage| TokenUsage {
        input: usage.prompt_token_count.map(|v| v as u32),
        output: usage.candidates_token_count.map(|v| v as u32),
    });
    let provider_model = response.model_version.clone();
    let raw_metadata = Some(serde_json::json!({
        "response_id": response.response_id.clone(),
    }));
    let fcs = response.function_calls_with_thoughts();
    if !fcs.is_empty() {
        let thought_text = response.text();
        let thought = (!thought_text.is_empty()).then_some(thought_text);
        let calls: Vec<ToolCall> = fcs
            .iter()
            .enumerate()
            .map(|(idx, (fc, sig))| ToolCall {
                id: format!("{}_{}", fc.name, idx),
                name: fc.name.clone(),
                args: fc.args.clone(),
                thought_signatures: sig.map(|s| vec![s.to_string()]),
            })
            .collect();
        return Ok(LlmResponse::new(
            Provider::Gemini,
            LlmOutput::ToolCalls {
                text: thought,
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(raw_metadata));
    }
    let text = response.text();
    Ok(LlmResponse::new(
        Provider::Gemini,
        LlmOutput::Output(decode_nonempty(&text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(raw_metadata))
}

fn wants_json_output(options: &LlmOptions) -> bool {
    options.wants_json_output()
}

fn response_schema(options: &LlmOptions) -> Result<Option<Value>, RathError> {
    if !wants_json_output(options) {
        return Ok(None);
    }
    options
        .output_schema()
        .map(|value| schema::sanitize_strict(value.clone()))
        .transpose()
}

/// Validates and decodes the Gemini safety settings supported by this adapter.
fn gemini_safety_settings_from_provider_config(
    provider_config: &Option<Value>,
) -> Result<Option<Vec<SafetySetting>>, RathError> {
    let Some(config) = provider_config else {
        return Ok(None);
    };
    let Value::Object(map) = config else {
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            "Gemini provider_config must be a JSON object",
        ));
    };

    if let Some(key) = map.keys().find(|key| key.as_str() != "safetySettings") {
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            format!("unsupported Gemini provider_config key '{key}'"),
        ));
    }

    map.get("safetySettings")
        .map(|value| {
            serde_json::from_value::<Vec<SafetySetting>>(value.clone()).map_err(|e| {
                RathError::new(
                    crate::core::ErrorKind::Validation,
                    "invalid Gemini provider_config.safetySettings",
                )
                .with_source(RathError::from_native_error(
                    crate::core::ErrorKind::Deserialize,
                    &e,
                ))
            })
        })
        .transpose()
}

impl LlmBackend for GeminiClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &crate::llm::LlmOptions {
        &self.options
    }

    fn estimate_tokens(&self, messages: &[Message]) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&self.options, messages, false)
            .map_err(|error| {
                errors::context(error, "token estimation", self.url.api_key.as_deref())
            })
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&self.options, messages, false)
            .await
            .map_err(|error| errors::context(error, "token counting", self.url.api_key.as_deref()))
    }

    fn estimate_content_tokens(&self, content: &str) -> Result<crate::llm::TokenCount, RathError> {
        self.estimate_request(&LlmOptions::default(), &[Message::user(content)], true)
            .map_err(|error| {
                errors::context(error, "token estimation", self.url.api_key.as_deref())
            })
    }

    async fn count_content_tokens(
        &self,
        content: &str,
    ) -> Result<crate::llm::TokenCount, RathError> {
        self.count_request(&LlmOptions::default(), &[Message::user(content)], true)
            .await
            .map_err(|error| errors::context(error, "token counting", self.url.api_key.as_deref()))
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        let wants_json_output = wants_json_output(&self.options);
        let response = request::build_request(
            &self.client,
            &self.url.model,
            &self.options,
            messages,
            false,
        )
        .map_err(|error| errors::context(error, "generation", self.url.api_key.as_deref()))?
        .execute()
        .await
        .map_err(|e| errors::normalize(&e, "generation", self.url.api_key.as_deref()))?;
        let raw = serde_json::to_value(&response)
            .map_err(|e| RathError::from_native_error(crate::core::ErrorKind::Serialize, &e))?;
        let result = map_response(response, wants_json_output).map_err(|error| {
            errors::context(
                error.with_response(&raw),
                "generation",
                self.url.api_key.as_deref(),
            )
        })?;

        crate::llm::output::finish(&self.options, result)
            .map_err(|error| errors::context(error, "generation", self.url.api_key.as_deref()))
    }
}

/// Creates a Gemini client.
/// Fails when the API key cannot be resolved.
pub fn new_client(url: &ModelUrl, options: LlmOptions) -> Result<LlmClient, RathError> {
    crate::llm::counting::validate_options(Provider::Gemini, &options)?;
    let mut url = url.clone();
    url.api_key = Some(crate::llm::required_api_key(&url, "GEMINI_API_KEY")?);
    let client = build_client(&url)?;
    crate::llm::output::prepare(url.provider.clone(), &url.model, &options)?;
    Ok(LlmClient::from_backend(GeminiClient {
        client,
        options,
        url: url.clone(),
    }))
}

#[cfg(test)]
#[path = "gemini/tests/mod.rs"]
mod tests;

fn check_generation_limit(response: &GenerationResponse) -> Result<(), RathError> {
    let raw = serde_json::to_value(response)
        .map_err(|error| RathError::from_native_error(crate::core::ErrorKind::Serialize, &error))?;
    crate::llm::counting::check_output_limit(Provider::Gemini, &raw)
}

/// Missing provider text is invalid before structured-output decoding.
fn decode_nonempty(text: &str, wants_json_output: bool) -> Result<Value, RathError> {
    if text.is_empty() {
        return Err(RathError::new(
            crate::core::ErrorKind::InvalidResponse,
            "missing nonempty candidates[].content.parts[].text or functionCall",
        ));
    }
    decode_output_text(text, wants_json_output)
}
