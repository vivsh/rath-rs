use crate::llm::{LlmOptions, Provider, RathError};
use serde_json::Value;

/// Validates the typed cap and rejects competing provider configuration locations.
pub(crate) fn validate_options(provider: Provider, options: &LlmOptions) -> Result<(), RathError> {
    if let Some(cap) = options.max_output_tokens
        && (cap == 0 || (provider == Provider::Gemini && cap > i32::MAX as u32))
    {
        return Err(RathError::Validation(
            "max_output_tokens is zero or outside the provider's representable range".into(),
        ));
    }
    let Some(config) = &options.provider_config else {
        return Ok(());
    };
    for path in [
        "/max_tokens",
        "/max_completion_tokens",
        "/max_output_tokens",
        "/maxOutputTokens",
        "/num_predict",
        "/generationConfig/maxOutputTokens",
        "/generation_config/max_output_tokens",
        "/options/num_predict",
    ] {
        if config.pointer(path).is_some() {
            return Err(RathError::Validation(format!(
                "provider_config{} is reserved; use typed max_output_tokens",
                path.replace('/', ".")
            )));
        }
    }
    Ok(())
}

/// Stops partial text/JSON/tool responses before any output interpretation or logging.
pub(crate) fn check_output_limit(provider: Provider, response: &Value) -> Result<(), RathError> {
    let limited = match provider {
        Provider::OpenAi => {
            response.get("status").and_then(Value::as_str) == Some("incomplete")
                && response
                    .pointer("/incomplete_details/reason")
                    .and_then(Value::as_str)
                    == Some("max_output_tokens")
        }
        Provider::Anthropic => {
            response.get("stop_reason").and_then(Value::as_str) == Some("max_tokens")
        }
        Provider::OpenRouter | Provider::Ollama => {
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                == Some("length")
        }
        Provider::Gemini => response
            .get("candidates")
            .and_then(Value::as_array)
            .is_some_and(|cs| {
                cs.iter()
                    .any(|c| c.get("finishReason").and_then(Value::as_str) == Some("MAX_TOKENS"))
            }),
        _ => false,
    };
    if limited {
        return Err(RathError::OutputLimitReached {
            provider,
            response: response.clone(),
        });
    }
    Ok(())
}
