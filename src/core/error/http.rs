//! Shared stateless request/response diagnostics using the adapter's existing HTTP client.
use super::{ErrorBody, ErrorKind, RathError, normalize::details};
use crate::core::Provider;
use reqwest::{RequestBuilder, Response};
use serde_json::Value;

/// Normalizes transport errors without flattening their source chain.
pub(crate) fn transport(
    provider: Provider,
    operation: &'static str,
    error: reqwest::Error,
    secrets: &[&str],
) -> RathError {
    let kind = if error.is_timeout() {
        ErrorKind::Timeout
    } else {
        ErrorKind::Transport
    };
    RathError::from_error(kind, &error)
        .with_context(provider, operation)
        .sanitized(secrets)
}

/// Sends once. Existing adapter authentication, endpoint selection and retry policy are preserved.
pub(crate) async fn send(
    request: RequestBuilder,
    provider: Provider,
    operation: &'static str,
    secrets: &[&str],
) -> Result<Response, RathError> {
    request
        .send()
        .await
        .map_err(|error| transport(provider, operation, error, secrets))
}

/// Reads bytes incrementally, retaining status, identifiers and partial bytes when transfer fails.
pub(crate) async fn read(
    mut response: Response,
    provider: Provider,
    operation: &'static str,
    secrets: &[&str],
) -> Result<Vec<u8>, RathError> {
    let mut error = metadata(&response, provider, operation);
    let mut bytes = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => bytes.extend_from_slice(&chunk),
            Ok(None) => break,
            Err(cause) => {
                error.kind = if cause.is_timeout() {
                    ErrorKind::Timeout
                } else {
                    ErrorKind::Transport
                };
                error.message =
                    "response transfer failed; incomplete response details available".into();
                return Err(error
                    .with_source(RathError::from_error(error_kind(&cause), &cause))
                    .with_body(ErrorBody::Incomplete(bytes))
                    .sanitized(secrets));
            }
        }
    }
    if !response.status().is_success() {
        attach_details(&mut error, &bytes);
        return Err(error
            .with_body(ErrorBody::Complete(bytes))
            .sanitized(secrets));
    }
    Ok(bytes)
}

/// Validates binary response content before returning bytes, preserving the response on failure.
pub(crate) async fn read_checked(
    response: Response,
    provider: Provider,
    operation: &'static str,
    secrets: &[&str],
    validate: impl FnOnce(&[u8]) -> Result<(), RathError>,
) -> Result<Vec<u8>, RathError> {
    let meta = metadata(&response, provider.clone(), operation);
    let bytes = read(response, provider, operation, secrets).await?;
    if let Err(error) = validate(&bytes) {
        return Err(enrich(error, meta, bytes, secrets));
    }
    Ok(bytes)
}

/// Preserves documented request and retry identifiers, excluding unrelated headers.
fn metadata(response: &Response, provider: Provider, operation: &'static str) -> RathError {
    let mut error = RathError::new(
        ErrorKind::Http,
        "request failed; response details available",
    )
    .with_context(provider, operation);
    error.http_status = Some(response.status().as_u16());
    for name in [
        "request-id",
        "x-request-id",
        "x-goog-request-id",
        "x-fal-request-id",
    ] {
        if let Some(value) = response.headers().get(name) {
            error.request_id = Some(header_text(value));
            break;
        }
    }
    error.retry_after = response.headers().get("retry-after").map(header_text);
    error
}

/// Keeps non-text metadata losslessly inspectable using byte escapes rather than dropping it.
fn header_text(value: &reqwest::header::HeaderValue) -> String {
    value
        .to_str()
        .map(str::to_owned)
        .unwrap_or_else(|_| value.as_bytes().escape_ascii().to_string())
}

/// Extracts useful provider messages; malformed error bodies retain the decoding cause.
fn attach_details(error: &mut RathError, bytes: &[u8]) {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) => details(error, &value),
        Err(cause) => {
            error.source = Some(Box::new(RathError::from_error(
                ErrorKind::Deserialize,
                &cause,
            )))
        }
    }
}

fn error_kind(error: &reqwest::Error) -> ErrorKind {
    if error.is_timeout() {
        ErrorKind::Timeout
    } else {
        ErrorKind::Transport
    }
}

/// Decodes and maps a response while preserving original bytes and headers on semantic failure.
pub(crate) async fn mapped<T>(
    request: RequestBuilder,
    provider: Provider,
    operation: &'static str,
    secrets: &[&str],
    decode: impl FnOnce(Value) -> Result<T, RathError>,
) -> Result<T, RathError> {
    let response = send(request, provider.clone(), operation, secrets).await?;
    mapped_response(response, provider.clone(), operation, secrets, |value| {
        if value.get("error").is_some_and(|e| !e.is_null()) {
            Err(super::provider_failure(
                provider, operation, &value, secrets,
            ))
        } else {
            decode(value)
        }
    })
    .await
}

/// Shares error enrichment for JSON decoding and provider-specific response interpretation.
pub(crate) async fn mapped_response<T>(
    response: Response,
    provider: Provider,
    operation: &'static str,
    secrets: &[&str],
    decode: impl FnOnce(Value) -> Result<T, RathError>,
) -> Result<T, RathError> {
    let meta = metadata(&response, provider.clone(), operation);
    let bytes = read(response, provider.clone(), operation, secrets).await?;
    let result = serde_json::from_slice::<Value>(&bytes)
        .map_err(|cause| {
            RathError::new(
                ErrorKind::Deserialize,
                "invalid JSON response; response details available",
            )
            .with_source(RathError::from_error(ErrorKind::Deserialize, &cause))
        })
        .and_then(decode);
    result.map_err(|error| {
        enrich(
            error.with_context(provider, operation),
            meta,
            bytes,
            secrets,
        )
    })
}

/// Attaches original wire bytes and selected response headers to a decoding/validation error.
fn enrich(mut error: RathError, meta: RathError, bytes: Vec<u8>, secrets: &[&str]) -> RathError {
    error.http_status = meta.http_status;
    if error.request_id.is_none() {
        error.request_id = meta.request_id;
    }
    error.retry_after = meta.retry_after;
    error
        .with_body(ErrorBody::Complete(bytes))
        .sanitized(secrets)
}
