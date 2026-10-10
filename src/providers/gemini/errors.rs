//! Normalization at the official SDK boundary, without exposing SDK types to callers.
use crate::core::{ErrorKind, Provider, RathError};
use gemini_rust::client::Error;

/// Replaces reduced mapping metadata with the available full decoded generation response.
/// The caller must apply operation-wide credential sanitization before returning this error.
pub(super) fn with_response(error: RathError, response: &serde_json::Value) -> RathError {
    match serde_json::to_vec(response) {
        Ok(bytes) => error.with_raw_body(crate::ErrorBody::Complete(bytes)),
        Err(source) => {
            error.with_source(RathError::from_native_error(ErrorKind::Serialize, &source))
        }
    }
}

/// Preserves everything the SDK exposes; it may already have discarded response-read errors.
pub(super) fn normalize(
    error: &Error,
    operation: &'static str,
    credential: Option<&str>,
) -> RathError {
    context(classify(error, operation), operation, credential)
}

/// Maps exposed SDK variants to stable Rath error categories and native causes.
fn classify(error: &Error, operation: &'static str) -> RathError {
    match error {
        Error::BadResponse { code, description } => {
            crate::core::error::sdk_http(*code, description.as_deref(), &[])
        }
        Error::OperationTimeout { .. } => RathError::from_native_error(ErrorKind::Timeout, error),
        Error::PerformRequest { source, .. } | Error::PerformRequestNew { source } => {
            let kind = if source.is_timeout() {
                ErrorKind::Timeout
            } else {
                ErrorKind::Transport
            };
            RathError::from_native_error(kind, error)
        }
        Error::Deserialize { .. } | Error::DecodeResponse { .. } => {
            RathError::from_native_error(ErrorKind::Deserialize, error)
        }
        Error::BadPart { source } => match source {
            eventsource_stream::EventStreamError::Transport(cause) => {
                let kind = if cause.is_timeout() {
                    ErrorKind::Timeout
                } else {
                    ErrorKind::Transport
                };
                RathError::from_native_error(kind, error)
                    .with_source(RathError::from_native_error(kind, cause))
            }
            _ => RathError::from_native_error(ErrorKind::Deserialize, error),
        },
        _ => local_error(error, operation),
    }
}

/// Classifies local validation, protocol and failed-operation SDK diagnostics.
fn local_error(error: &Error, operation: &'static str) -> RathError {
    match error {
        Error::InvalidApiKey { .. }
        | Error::ConstructUrl { .. }
        | Error::UrlParse { .. }
        | Error::InvalidResourceName { .. } => {
            RathError::from_native_error(ErrorKind::Validation, error)
        }
        Error::MissingResponseHeader { header } => RathError::new(
            ErrorKind::InvalidResponse,
            format!("missing response header {header}"),
        ),
        Error::OperationFailed {
            name,
            code,
            message,
        } => operation_failed(operation, name, *code, message),
        _ => RathError::from_native_error(ErrorKind::Provider, error),
    }
}

/// Preserves the provider's structured failed-operation diagnostic.
fn operation_failed(operation: &'static str, name: &str, code: i32, message: &str) -> RathError {
    crate::core::error::provider_failure(
        Provider::Gemini,
        operation,
        &serde_json::json!({"name":name, "error":{"code":code, "message":message}}),
        &[],
    )
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
