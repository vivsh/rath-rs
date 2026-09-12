use thiserror::Error;

use super::Provider;

/// Errors returned by Rath provider calls.
#[derive(Error)]
pub enum RathError {
    /// Local tokenizer initialization, arithmetic, or count decoding failed.
    #[error("token counting failed: {message}")]
    TokenCounting { message: String },
    /// Generation stopped at the provider token limit. Partial content is not success.
    /// The response is available explicitly, but omitted from Display and Debug.
    #[error("provider '{provider:?}' reached the output token limit")]
    OutputLimitReached {
        provider: Provider,
        response: serde_json::Value,
    },
    /// Request payload could not be serialized.
    #[error("failed to serialize input: {0}")]
    Serialize(#[source] serde_json::Error),
    /// Provider response could not be deserialized; `raw` contains the original text.
    #[error("failed to deserialize output: {source}\nraw response: {raw}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    /// The provider returned an HTTP or API error.
    #[error("provider call failed: {0}")]
    Provider(String),
    /// The provider returned an empty response body.
    #[error("provider returned an empty response")]
    EmptyResponse,
    /// Input failed pre-dispatch validation.
    #[error("validation failed: {0}")]
    Validation(String),
    /// A tool-call response was expected but the model returned none.
    #[error("No tool calls found: {0:?}")]
    MissingToolCalls(Option<String>),
    /// The requested capability is not available for this provider.
    #[error("provider '{provider:?}' does not support capability '{capability}'")]
    UnsupportedCapability {
        provider: Provider,
        capability: String,
    },
    /// The model URL could not be parsed into a known provider + model.
    #[error("invalid model URL: {0}")]
    InvalidUrl(String),
    /// Catch-all for errors from third-party provider libraries.
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

// Keep partial provider output out of ordinary diagnostic formatting.
impl std::fmt::Debug for RathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputLimitReached { provider, .. } => f
                .debug_struct("OutputLimitReached")
                .field("provider", provider)
                .finish_non_exhaustive(),
            other => std::fmt::Display::fmt(other, f),
        }
    }
}

#[cfg(test)]
#[path = "error/tests/mod.rs"]
mod tests;
