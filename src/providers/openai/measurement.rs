use super::*;
use crate::llm::{TokenCount, counting};

const PROMPT_FIELDS: &[&str] = &["input", "instructions", "tools", "tool_choice", "text"];

impl OpenAiClient {
    /// Estimates the prompt projection of the same request builder used for generation.
    pub(super) fn estimate_request(
        &self,
        options: &LlmOptions,
        messages: &[Message],
    ) -> Result<TokenCount, RathError> {
        let payload = measurement_payload(&self.model, options, messages, false)?;
        counting::estimate(
            &counting::project(&payload, PROMPT_FIELDS),
            payload["input"].as_array().map_or(0, Vec::len),
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
            &[
                "model",
                "input",
                "instructions",
                "tools",
                "tool_choice",
                "text",
            ],
        );
        counting::count_response(
            Provider::OpenAi,
            self.http
                .post(format!(
                    "{}/responses/input_tokens",
                    self.base_url.trim_end_matches('/')
                ))
                .bearer_auth(&self.api_key)
                .json(&body),
            &[&self.api_key],
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
    counting::validate_measurement(Provider::OpenAi, options, messages, remote)?;
    validate_history(messages)?;
    let enabled = !options.tools.is_empty() && options.tool_choice != ToolChoice::Disabled;
    Ok(build_payload(model, options, messages, enabled))
}
