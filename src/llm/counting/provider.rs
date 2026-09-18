use super::{TokenCount, TokenCountSource, validate_options};
use crate::llm::{Attachment, LlmOptions, Message, Provider, RathError, Role, validate_tools};
use serde_json::{Map, Value};

pub(crate) fn unsupported(provider: Provider, capability: &str) -> RathError {
    RathError::unsupported(provider, capability)
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
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            "measurement requires nonempty history with resolved final tool calls",
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
    secrets: &[&str],
) -> Result<TokenCount, RathError> {
    let result =
        crate::core::error::http::mapped(request, provider, "token counting", secrets, |value| {
            let input_tokens = value
                .get("input_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| RathError::invalid("missing valid input_tokens", &value))?;
            Ok(TokenCount {
                input_tokens,
                source: TokenCountSource::ProviderReported,
            })
        })
        .await;
    result.map_err(|error| {
        let missing_model = error.http_status() == Some(404)
            && error
                .response_body()
                .is_some_and(|body| missing_model(&String::from_utf8_lossy(body.bytes())));
        if matches!(error.http_status(), Some(404 | 405 | 501)) && !missing_model {
            error.classified(crate::core::ErrorKind::UnsupportedCapability)
        } else {
            error
        }
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
