//! Normalization at the official SDK boundary, without exposing SDK types to callers.
use crate::core::{ErrorKind, Provider, RathError};
use gemini_rust::client::Error;

/// Preserves everything the SDK exposes; it may already have discarded response-read errors.
pub(super) fn normalize(
    error: &Error,
    operation: &'static str,
    credential: Option<&str>,
) -> RathError {
    let result = match error {
        Error::BadResponse { code, description } => {
            crate::core::error::sdk_http(*code, description.as_deref(), &[])
        }
        Error::OperationTimeout { .. } => RathError::from_error(ErrorKind::Timeout, error),
        Error::PerformRequest { source, .. } | Error::PerformRequestNew { source } => {
            let kind = if source.is_timeout() {
                ErrorKind::Timeout
            } else {
                ErrorKind::Transport
            };
            RathError::from_error(kind, error)
        }
        Error::Deserialize { .. } | Error::DecodeResponse { .. } => {
            RathError::from_error(ErrorKind::Deserialize, error)
        }
        Error::InvalidApiKey { .. }
        | Error::ConstructUrl { .. }
        | Error::UrlParse { .. }
        | Error::InvalidResourceName { .. } => RathError::from_error(ErrorKind::Validation, error),
        Error::MissingResponseHeader { header } => RathError::new(
            ErrorKind::InvalidResponse,
            format!("missing response header {header}"),
        ),
        Error::OperationFailed {
            name,
            code,
            message,
        } => crate::core::error::provider_failure(
            Provider::Gemini,
            operation,
            &serde_json::json!({"name":name, "error":{"code":code, "message":message}}),
            &[],
        ),
        _ => RathError::from_error(ErrorKind::Provider, error),
    };
    context(result, operation, credential)
}

/// Sanitizes SDK and local Gemini diagnostics without storing another copy of credentials.
pub(super) fn context(
    error: RathError,
    operation: &'static str,
    credential: Option<&str>,
) -> RathError {
    let fallback = if credential.is_none() {
        std::env::var("GEMINI_API_KEY").ok()
    } else {
        None
    };
    let secrets: Vec<_> = credential.or(fallback.as_deref()).into_iter().collect();
    error
        .with_context(Provider::Gemini, operation)
        .sanitized(&secrets)
}
