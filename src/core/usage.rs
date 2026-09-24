use serde::{Deserialize, Serialize};

/// Token counts reported for a single model call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
            (Some(i), Some(o)) => i.checked_add(o),
            _ => None,
        }
    }
}

impl TokenUsage {
    /// Constructs required values; optional controls start unset.
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
        }
    }
    /// Sets `input` for this value.
    pub fn with_input(mut self, value: u32) -> Self {
        self.input = Some(value);
        self
    }
    /// Sets `output` for this value.
    pub fn with_output(mut self, value: u32) -> Self {
        self.output = Some(value);
        self
    }
}
