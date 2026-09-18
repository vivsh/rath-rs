use super::*;
use crate::llm::{TokenCount, TokenCountSource, counting};

impl GeminiClient {
    /// Estimates the prompt projection of the same request builder used for generation.
    pub(super) fn estimate_request(
        &self,
        options: &LlmOptions,
        messages: &[Message],
        minimal: bool,
    ) -> Result<TokenCount, RathError> {
        counting::validate_measurement(Provider::Gemini, options, messages, false)?;
        let request =
            request::build_request(&self.client, &self.url.model, options, messages, minimal)?
                .build();
        let payload = serde_json::to_value(request)
            .map_err(|error| RathError::from_error(crate::core::ErrorKind::Serialize, &error))?;
        let mut prompt = counting::project(
            &payload,
            &["contents", "systemInstruction", "tools", "toolConfig"],
        );
        if let Some(config) = payload.get("generationConfig") {
            prompt["generationConfig"] = counting::project(
                config,
                &["responseMimeType", "responseSchema", "responseJsonSchema"],
            );
        }
        // An absent generation configuration and an empty projection have identical prompt meaning.
        if prompt
            .get("generationConfig")
            .is_some_and(|v| v.as_object().is_some_and(|m| m.is_empty()))
        {
            prompt.as_object_mut().map(|m| m.remove("generationConfig"));
        }
        let tools = payload["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| t["functionDeclarations"].as_array().map_or(0, Vec::len))
            .sum();
        counting::estimate(
            &prompt,
            payload["contents"].as_array().map_or(0, Vec::len),
            tools,
        )
    }

    /// Counts the complete supported request at the configured endpoint without fallback.
    pub(super) async fn count_request(
        &self,
        options: &LlmOptions,
        messages: &[Message],
        minimal: bool,
    ) -> Result<TokenCount, RathError> {
        counting::validate_measurement(Provider::Gemini, options, messages, true)?;
        let result =
            request::build_request(&self.client, &self.url.model, options, messages, minimal)?
                .count_tokens()
                .await
                .map_err(|error| count_error(error, self.url.api_key.as_deref()))?;
        Ok(TokenCount {
            input_tokens: u64::from(result.total_tokens),
            source: TokenCountSource::ProviderReported,
        })
    }
}

/// Distinguishes unavailable counting endpoints from other Gemini provider errors.
fn count_error(error: gemini_rust::client::Error, credential: Option<&str>) -> RathError {
    if let gemini_rust::client::Error::BadResponse { code, description } = &error {
        let model_error =
            *code == 404 && description.as_deref().is_some_and(counting::missing_model);
        if matches!(code, 404 | 405 | 501) && !model_error {
            return errors::normalize(&error, "token counting", credential)
                .classified(crate::core::ErrorKind::UnsupportedCapability);
        }
    }
    errors::normalize(&error, "token counting", credential)
}
