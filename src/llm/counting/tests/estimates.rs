use super::super::*;
use crate::llm::{LlmOptions, Message, Provider};
use serde_json::json;

/// Padding uses ceiling division and fails safely at integer overflow.
#[test]
fn padding_is_checked() {
    assert_eq!(super::super::estimate::padded(1, 1).unwrap(), 342);
    assert!(super::super::estimate::padded(u64::MAX, 1).is_err());
    assert!(super::super::estimate::padded(1, u64::MAX).is_err());
}

/// Unicode and special-token spellings are ordinary text and yield stable estimates.
#[test]
fn ordinary_unicode_is_deterministic() {
    let prompt = json!({"messages": [{"content": "नमस्ते 世界 <|endoftext|>"}]});
    let a = estimate(&prompt, 1, 0).unwrap();
    assert_eq!(a, estimate(&prompt, 1, 0).unwrap());
    assert_eq!(a.source, TokenCountSource::Estimated);
    assert!(a.input_tokens > 340);
}

/// Local estimates never mistake a media URL or base64 blob for full media usage.
#[test]
fn media_cannot_be_partially_estimated() {
    let messages = [Message::user("image").with_url("image/png", "https://example.com/i.png")];
    assert!(
        validate_measurement(Provider::OpenAi, &LlmOptions::default(), &messages, false).is_err()
    );
    assert!(
        validate_measurement(Provider::OpenAi, &LlmOptions::default(), &messages, true).is_ok()
    );
    assert!(
        validate_measurement(Provider::Anthropic, &LlmOptions::default(), &messages, true).is_err()
    );
}
