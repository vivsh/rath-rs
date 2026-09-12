use super::*;
use crate::llm::{TokenCount, counting};

const PROMPT_FIELDS: &[&str] = &["messages", "system", "tools", "tool_choice"];

impl AnthropicClient {
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

    /// Counts the complete supported request at the configured endpoint without fallback.
    pub(super) async fn count_request(
        &self,
        options: &LlmOptions,
        messages: &[Message],
    ) -> Result<TokenCount, RathError> {
        let payload = measurement_payload(&self.model, options, messages, true)?;
        let body = counting::project(
            &payload,
            &["model", "messages", "system", "tools", "tool_choice"],
        );
        counting::count_response(
            Provider::Anthropic,
            self.http
                .post(format!(
                    "{}/messages/count_tokens",
                    self.base_url.trim_end_matches('/')
                ))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&body),
        )
        .await
    }
}

/// Validates measurement support and builds the same payload used for generation.
fn measurement_payload(
    model: &str,
    options: &LlmOptions,
    messages: &[Message],
    remote: bool,
) -> Result<Value, RathError> {
    counting::validate_measurement(Provider::Anthropic, options, messages, remote)?;
    validate_history(messages)?;
    if matches!(&options.thinking, Some(t) if *t != ThinkingLevel::Off) {
        return Err(counting::unsupported(
            Provider::Anthropic,
            "thinking is not exposed by the Anthropic adapter yet",
        ));
    }
    let enabled = !options.tools.is_empty() && options.tool_choice != ToolChoice::Disabled;
    Ok(build_payload(model, options, messages, enabled))
}
