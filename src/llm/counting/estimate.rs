use super::{TokenCount, TokenCountSource};
use crate::llm::RathError;
use serde_json::Value;
use std::sync::OnceLock;
use tiktoken_rs::{CoreBPE, cl100k_base};

static TOKENIZER: OnceLock<Result<CoreBPE, RathError>> = OnceLock::new();

/// Measures only the adapter's prompt projection with fixed reference-tokenizer padding.
pub(crate) fn estimate(
    prompt: &Value,
    messages: usize,
    tools: usize,
) -> Result<TokenCount, RathError> {
    let tokenizer = TOKENIZER.get_or_init(|| {
        cl100k_base().map_err(|error| {
            RathError::from_error(crate::core::ErrorKind::TokenCounting, error.as_ref())
        })
    });
    let tokenizer = tokenizer.as_ref().map_err(Clone::clone)?;
    let text = serde_json::to_string(prompt)
        .map_err(|error| RathError::from_error(crate::core::ErrorKind::Serialize, &error))?;
    let tokens = u64::try_from(tokenizer.encode_ordinary(&text).len()).map_err(|error| {
        overflow().with_source(RathError::from_error(crate::core::ErrorKind::Other, &error))
    })?;
    let units = messages.checked_add(tools).ok_or_else(overflow)?;
    let units = u64::try_from(units).map_err(|error| {
        overflow().with_source(RathError::from_error(crate::core::ErrorKind::Other, &error))
    })?;
    Ok(TokenCount {
        input_tokens: padded(tokens, units)?,
        source: TokenCountSource::Estimated,
    })
}

/// Applies framing allowance and a 25-percent margin without wrapping arithmetic.
pub(super) fn padded(tokens: u64, units: u64) -> Result<u64, RathError> {
    units
        .checked_mul(16)
        .and_then(|n| n.checked_add(256))
        .and_then(|n| n.checked_add(tokens))
        .and_then(|n| n.checked_mul(5))
        .and_then(|n| n.checked_add(3))
        .map(|n| n / 4)
        .ok_or_else(overflow)
}

fn overflow() -> RathError {
    RathError::new(
        crate::core::ErrorKind::TokenCounting,
        "token estimate overflow",
    )
}
