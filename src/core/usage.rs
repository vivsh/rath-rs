use serde::{Deserialize, Serialize};

/// Token counts reported for a single model call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input-side tokens.
    pub input: Option<u32>,
    /// Output-side tokens.
    pub output: Option<u32>,
}

impl TokenUsage {
    /// Returns `input + output` if both are present, `None` otherwise.
    pub fn total(&self) -> Option<u32> {
        match (self.input, self.output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }
}
