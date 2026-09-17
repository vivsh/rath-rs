use super::super::*;
use serde_json::json;

/// Build the shared generation/counting payload without making provider calls.
fn payload(model: &str, thinking: Option<ThinkingLevel>, minimal: bool) -> Value {
    let client = Gemini::with_model("test", GeminiModel::Custom(model.into())).unwrap();
    let options = LlmOptions::default().with_thinking(thinking);
    serde_json::to_value(
        request::build_request(&client, model, &options, &[Message::user("hello")], minimal)
            .unwrap()
            .build(),
    )
    .unwrap()
}

/// Gemma 4 disables reasoning with a level, never the unsupported numeric budget.
#[test]
fn gemma_off_and_default_use_minimal_level() {
    for model in ["gemma-4-26b-a4b-it", "models/gemma-4-31b-it"] {
        for thinking in [None, Some(ThinkingLevel::Off)] {
            let body = payload(model, thinking, false);
            assert_eq!(
                body["generationConfig"]["thinkingConfig"],
                json!({"thinkingLevel":"MINIMAL"})
            );
        }
    }
}

/// Gemma's binary reasoning switch maps all enabled depths to its supported high level.
#[test]
fn gemma_enabled_thinking_uses_high_level() {
    for thinking in [
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::XHigh,
    ] {
        let body = payload("gemma-4-26b-a4b-it", Some(thinking), false);
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel":"HIGH"})
        );
    }
}

/// Existing Gemini budget mappings are unchanged by Gemma-specific handling.
#[test]
fn other_models_keep_existing_budgets() {
    for model in ["models/gemini-3-flash-preview", "gemini-2.5-flash"] {
        for (thinking, budget) in [
            (None, 0),
            (Some(ThinkingLevel::Off), 0),
            (Some(ThinkingLevel::Low), 512),
            (Some(ThinkingLevel::Medium), 4096),
            (Some(ThinkingLevel::High), 16384),
            (Some(ThinkingLevel::XHigh), i32::MAX),
        ] {
            let body = payload(model, thinking, false);
            assert_eq!(
                body["generationConfig"]["thinkingConfig"],
                json!({"thinkingBudget":budget})
            );
        }
    }
}

/// Standalone content counting still omits thinking regardless of model or configured depth.
#[test]
fn minimal_counting_omits_thinking() {
    for model in [
        "gemma-4-26b-a4b-it",
        "models/gemma-4-31b-it",
        "gemini-3-flash-preview",
    ] {
        let body = payload(model, Some(ThinkingLevel::High), true);
        assert!(body["generationConfig"]["thinkingConfig"].is_null());
    }
}
