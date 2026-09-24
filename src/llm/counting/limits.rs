use crate::llm::{LlmOptions, Provider, RathError};
use serde_json::Value;

/// Validates the typed cap and rejects competing provider configuration locations.
pub(crate) fn validate_options(provider: Provider, options: &LlmOptions) -> Result<(), RathError> {
    crate::registry::prepare::native(options.provider_config.as_ref())?;
    if options
        .tools
        .iter()
        .any(|t| t.name == "__rath_final_output")
        && options.wants_json_output()
    {
        return Err(RathError::new(
            crate::ErrorKind::Validation,
            "tool name __rath_final_output is reserved",
        ));
    }
    if options.cache.is_some()
        && matches!(
            provider,
            Provider::OpenAi | Provider::OpenRouter | Provider::Ollama | Provider::Gemini
        )
    {
        return Err(RathError::unsupported(provider, "explicit cache policy"));
    }
    if options.thinking.is_some() && provider == Provider::OpenAi {
        return Err(RathError::unsupported(
            provider,
            "typed thinking control on this adapter",
        ));
    }
    if let Some(cap) = options.max_output_tokens
        && (cap == 0 || (provider == Provider::Gemini && cap > i32::MAX as u32))
    {
        return Err(RathError::new(
            crate::core::ErrorKind::Validation,
            "max_output_tokens is zero or outside the provider's representable range",
        ));
    }
    validate_cap_aliases(options.provider_config.as_ref())
}

/// Rejects competing output budgets without scanning arbitrary nested schema property names.
fn validate_cap_aliases(config: Option<&Value>) -> Result<(), RathError> {
    let Some(config) = config else {
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
            return Err(RathError::new(
                crate::core::ErrorKind::Validation,
                format!(
                    "provider_config{} is reserved; use typed max_output_tokens",
                    path.replace('/', ".")
                ),
            ));
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
        return Err(RathError::new(
            crate::core::ErrorKind::OutputLimitReached,
            "generation reached the output token limit",
        )
        .with_context(provider, "generation")
        .with_response(response));
    }
    Ok(())
}
