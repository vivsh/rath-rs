use super::super::*;
use crate::llm::{LlmOptions, Provider};
use serde_json::json;

/// Typed fields and all reserved configuration locations are validated directly.
#[test]
fn rejects_conflicts_and_invalid_values() {
    for config in [
        json!({"max_tokens": 1}),
        json!({"max_output_tokens": null}),
        json!({"generationConfig": {"maxOutputTokens": 2}}),
        json!({"options": {"num_predict": -1}}),
    ] {
        assert!(
            validate_options(
                Provider::OpenAi,
                &LlmOptions::default().with_provider_config(config)
            )
            .is_err()
        );
    }
    let options = LlmOptions {
        max_output_tokens: Some(0),
        ..Default::default()
    };
    assert!(validate_options(Provider::OpenAi, &options).is_err());
    assert!(
        validate_options(
            Provider::Gemini,
            &LlmOptions::default().with_max_output_tokens(u32::MAX)
        )
        .is_err()
    );
    assert!(
        validate_options(
            Provider::OpenAi,
            &LlmOptions::default()
                .with_provider_config(json!({"schema": {"properties": {"max_tokens": {}}}}))
        )
        .is_ok()
    );
}

/// Equal usage, unrelated incomplete reasons and absent finish metadata are not token-limit evidence.
#[test]
fn usage_alone_never_implies_truncation() {
    for provider in [
        Provider::Gemini,
        Provider::OpenAi,
        Provider::Anthropic,
        Provider::OpenRouter,
        Provider::Ollama,
    ] {
        assert!(
            check_output_limit(
                provider,
                &json!({"usage":{"output_tokens":10}, "max_output_tokens":10})
            )
            .is_ok()
        );
    }
    assert!(
        check_output_limit(
            Provider::OpenAi,
            &json!({"status":"incomplete", "incomplete_details":{"reason":"content_filter"}})
        )
        .is_ok()
    );
}
