use super::{TokenCount, TokenCountSource, validate_options};
use crate::llm::{Attachment, LlmOptions, Message, Provider, RathError, Role, validate_tools};
use serde_json::{Map, Value};

pub(crate) fn unsupported(provider: Provider, capability: &str) -> RathError {
    RathError::UnsupportedCapability {
        provider,
        capability: capability.into(),
    }
}

/// Ensures measurement never silently drops unsupported input components.
pub(crate) fn validate_measurement(
    provider: Provider,
    options: &LlmOptions,
    messages: &[Message],
    remote: bool,
) -> Result<(), RathError> {
    validate_options(provider.clone(), options)?;
    validate_tools(provider.clone(), &options.tools)?;
    if messages.is_empty()
        || matches!(
            messages.last().map(|m| &m.role),
            Some(Role::AssistantToolCalls { .. })
        )
    {
        return Err(RathError::Validation(
            "measurement requires nonempty history with resolved final tool calls".into(),
        ));
    }
    for message in messages {
        for attachment in &message.attachments {
            let supported = remote
                && match (&provider, &message.role, attachment) {
                    (
                        Provider::Gemini,
                        Role::User | Role::Tool { .. },
                        Attachment::Inline { .. } | Attachment::Url { .. },
                    ) => true,
                    (
                        Provider::OpenAi,
                        Role::User | Role::Tool { .. },
                        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. },
                    ) => mime_type.starts_with("image/"),
                    (
                        Provider::Anthropic,
                        Role::User | Role::Tool { .. },
                        Attachment::Inline { mime_type, .. },
                    ) => mime_type.starts_with("image/"),
                    _ => false,
                };
            if !supported {
                return Err(unsupported(
                    provider,
                    "counting this attachment without omitting content",
                ));
            }
        }
    }
    Ok(())
}

/// Selects the already-built request fields accepted by a prompt/counting projection.
pub(crate) fn project(payload: &Value, fields: &[&str]) -> Value {
    let mut result = Map::new();
    for field in fields {
        if let Some(value) = payload.get(*field) {
            result.insert((*field).into(), value.clone());
        }
    }
    Value::Object(result)
}

/// Decodes a count without exposing provider bodies in ordinary errors.
pub(crate) async fn count_response(
    provider: Provider,
    request: reqwest::RequestBuilder,
) -> Result<TokenCount, RathError> {
    let response = request
        .send()
        .await
        .map_err(|e| RathError::Provider(e.to_string()))?;
    if matches!(response.status().as_u16(), 404 | 405 | 501) {
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;
        if status == 404 && missing_model(&body) {
            return Err(RathError::Provider(
                "token counting returned HTTP 404 for the selected model".into(),
            ));
        }
        return Err(unsupported(provider, "provider token counting endpoint"));
    }
    let response = response
        .error_for_status()
        .map_err(|e| RathError::Provider(e.to_string()))?;
    let value: Value = response
        .json()
        .await
        .map_err(|e| RathError::TokenCounting {
            message: e.to_string(),
        })?;
    let input_tokens = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| RathError::TokenCounting {
            message: "provider omitted a valid input_tokens count".into(),
        })?;
    Ok(TokenCount {
        input_tokens,
        source: TokenCountSource::ProviderReported,
    })
}

/// Distinguishes known model-not-found responses from an unavailable counting route.
pub(crate) fn missing_model(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    body.contains("model")
        && [
            "not found",
            "not_found",
            "does not exist",
            "not supported",
            "not available",
            "model:",
        ]
        .iter()
        .any(|marker| body.contains(marker))
}
