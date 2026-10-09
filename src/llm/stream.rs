use std::pin::Pin;

use futures_core::Stream;

use super::{LlmResponse, RathError};

/// Pull-driven events from one generation, borrowing the immutable backend.
///
/// A successful built-in stream emits one `Completed`, then permanent EOF.
/// A failed stream emits one error, then EOF. Dropping it releases local reads,
/// without promising remote cancellation or stopped billing.
pub type LlmStream<'a> = Pin<Box<dyn Stream<Item = Result<LlmEvent, RathError>> + Send + 'a>>;

/// Provisional generation progress and the authoritative completed response.
///
/// Fragments may be followed by failure. Execute tools and persist complete
/// structured output only from `Completed`, after terminal validation.
#[non_exhaustive]
pub enum LlmEvent {
    /// Nonempty visible answer text, in provider order; not necessarily a token.
    TextDelta {
        /// Append this fragment to the displayed answer.
        text: String,
    },
    /// Append-only tool progress; argument fragments need not be valid JSON.
    /// Provider/SDK granularity varies; private final-output calls are hidden.
    ToolCallDelta {
        /// Stable stream-local index, possibly sparse; not a final vector offset.
        index: usize,
        /// Available complete call ID, which may repeat unchanged.
        id: Option<String>,
        /// Append this fragment to the call name when present.
        name_delta: Option<String>,
        /// Append this fragment to the argument JSON string.
        arguments_delta: String,
    },
    /// Terminal success, including completed calls, output and reported usage.
    Completed {
        /// Existing normalized response; authoritative over provisional progress.
        response: LlmResponse,
    },
}

#[cfg(test)]
#[path = "tests/stream.rs"]
mod tests;
