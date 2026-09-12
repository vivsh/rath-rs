use super::*;
use crate::llm::{TokenCount, counting};

const PROMPT_FIELDS: &[&str] = &["messages", "tools", "tool_choice", "response_format"];

impl OllamaClient {
    /// Estimates the prompt projection of the same request builder used for generation.
    pub(super) fn estimate_request(
        &self,
        options: &LlmOptions,
        messages: &[Message],
    ) -> Result<TokenCount, RathError> {
        let payload = measurement_payload(&self.model, options, messages, false)?;
        counting::estimate(
            &counting::project(&payload, PROMPT_FIELDS),
            payload["messages"].as_array().map_or(0, Vec::len),
            payload["tools"].as_array().map_or(0, Vec::len),
        )
    }
}

/// Validates measurement support and builds the same payload used for generation.
fn measurement_payload(
    model: &str,
    options: &LlmOptions,
    messages: &[Message],
    remote: bool,
) -> Result<Value, RathError> {
    counting::validate_measurement(Provider::Ollama, options, messages, remote)?;
    validate_history(messages)?;
    let enabled = !options.tools.is_empty() && options.tool_choice != ToolChoice::Disabled;
    Ok(build_payload(model, options, messages, enabled))
}
