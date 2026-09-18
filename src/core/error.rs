//! Structured diagnostics owned by the returned error, never by a client.

mod body;
mod format;
pub(crate) mod http;
mod normalize;
pub(crate) use normalize::{
    credential_cause, provider_failure, provider_failure_message, sdk_http,
};
mod redact;

use super::Provider;
pub use body::ErrorBody;

/// Stable, provider-independent classification of a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Caller input or configuration is invalid.
    Validation,
    /// A model locator cannot be parsed.
    InvalidUrl,
    /// A provider or deployment does not support the requested operation.
    UnsupportedCapability,
    /// A request or response transfer failed.
    Transport,
    /// An operation exceeded its deadline.
    Timeout,
    /// A provider returned a failing HTTP status.
    Http,
    /// A provider reported a failed operation inside a successful response.
    Provider,
    /// Input could not be encoded.
    Serialize,
    /// Output could not be decoded.
    Deserialize,
    /// Output is missing required fields or has an invalid shape.
    InvalidResponse,
    /// Generation stopped at the configured output budget.
    OutputLimitReached,
    /// Local or remote token measurement failed.
    TokenCounting,
    /// An underlying cause has no more specific Rath classification.
    Other,
}

/// An immutable diagnostic snapshot. Response bodies require explicit access.
/// Known credentials must be supplied to adapter normalization before return.
#[derive(Clone, thiserror::Error)]
pub struct RathError {
    kind: ErrorKind,
    provider: Option<Provider>,
    operation: Option<&'static str>,
    message: String,
    http_status: Option<u16>,
    provider_code: Option<String>,
    request_id: Option<String>,
    retry_after: Option<String>,
    response_body: Option<ErrorBody>,
    #[source]
    source: Option<Box<RathError>>,
}

impl RathError {
    /// Creates a local failure, sanitizing credential fields and URL components.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            provider: None,
            operation: None,
            message: redact::text(&message.into(), &[]),
            http_status: None,
            provider_code: None,
            request_id: None,
            retry_after: None,
            response_body: None,
            source: None,
        }
    }

    /// Adds operation context. Existing distinct context remains in the cause chain.
    pub fn with_context(mut self, provider: Provider, operation: &'static str) -> Self {
        if self.operation.is_some()
            && (self.operation != Some(operation) || self.provider.as_ref() != Some(&provider))
        {
            return Self::new(self.kind, "operation failed")
                .with_context(provider, operation)
                .with_source(self);
        }
        self.provider = Some(provider);
        self.operation = Some(operation);
        self
    }

    /// Appends a cause without discarding any existing cause chain.
    pub fn with_source(mut self, source: RathError) -> Self {
        let mut tail = &mut self.source;
        while let Some(node) = tail {
            tail = &mut node.source;
        }
        *tail = Some(Box::new(source));
        self
    }

    /// Returns the stable failure classification.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    /// Returns the provider when the failure occurred at a provider boundary.
    pub fn provider(&self) -> Option<&Provider> {
        self.provider.as_ref()
    }
    /// Returns the static operation name, excluding caller data.
    pub fn operation(&self) -> Option<&'static str> {
        self.operation
    }
    /// Returns this level's useful message, without flattening its causes.
    pub fn message(&self) -> &str {
        &self.message
    }
    /// Returns the response status when available.
    pub fn http_status(&self) -> Option<u16> {
        self.http_status
    }
    /// Returns the native provider error code when available.
    pub fn provider_code(&self) -> Option<&str> {
        self.provider_code.as_deref()
    }
    /// Returns the request identifier when available.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }
    /// Returns the native Retry-After header without interpreting its units.
    pub fn retry_after(&self) -> Option<&str> {
        self.retry_after.as_deref()
    }
    /// Explicitly accesses response or partial-output bytes, which may contain private content.
    pub fn response_body(&self) -> Option<&ErrorBody> {
        self.response_body.as_ref()
    }
    /// Returns the next normalized cause; also exposed through std::error::Error::source.
    pub fn source(&self) -> Option<&RathError> {
        self.source.as_deref()
    }
}

#[cfg(test)]
mod tests;
