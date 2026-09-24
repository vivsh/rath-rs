use crate::llm::LlmOptions;
use crate::{ErrorKind, ModelUrl, RathError};
use serde_json::Value;

/// Resolves URL precedence before adapter or external factory dispatch.
pub(crate) fn llm(url: &ModelUrl, mut options: LlmOptions) -> Result<LlmOptions, RathError> {
    crate::core::url::validate_endpoint(url)?;
    native(options.provider_config.as_ref())?;
    if let Some(value) = url.temperature {
        options.temperature = Some(value);
    }
    if let Some(value) = &url.thinking {
        options.thinking = Some(value.clone());
    }
    if let Some(value) = &url.cache {
        options.cache = Some(value.clone());
    }
    crate::llm::counting::validate_options(url.provider.clone(), &options)?;
    Ok(options)
}

/// Rejects controls that would compete with Rath's model binding or transport.
pub(crate) fn native(config: Option<&Value>) -> Result<(), RathError> {
    let Some(config) = config else {
        return Ok(());
    };
    let object = config.as_object().ok_or_else(|| {
        RathError::new(ErrorKind::Validation, "provider_config must be an object")
    })?;
    for key in object.keys() {
        if reserved(key) {
            return Err(RathError::new(
                ErrorKind::Validation,
                format!("provider_config.{key} is reserved; use the model URL or typed setting"),
            ));
        }
    }
    for (parent, keys) in [
        ("generationConfig", &["maxOutputTokens"][..]),
        ("options", &["num_predict"][..]),
    ] {
        if let Some(value) = object.get(parent) {
            for key in keys {
                if value.get(key).is_some() {
                    return Err(RathError::new(
                        ErrorKind::Validation,
                        format!(
                            "provider_config.{parent}.{key} is reserved; use max_output_tokens"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Identifies native fields that compete with model binding, transport or typed controls.
fn reserved(key: &str) -> bool {
    matches!(
        key,
        "thinking"
            | "think"
            | "reasoning_effort"
            | "input"
            | "messages"
            | "contents"
            | "tools"
            | "tool_choice"
            | "file"
            | "model"
            | "base_url"
            | "endpoint"
            | "url"
            | "api_key"
            | "api_key_env"
            | "authorization"
            | "headers"
            | "stream"
            | "streaming"
            | "max_tokens"
            | "max_completion_tokens"
            | "max_output_tokens"
    )
}
