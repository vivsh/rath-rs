//! Shared stateless request/response diagnostics using the adapter's existing HTTP client.
use super::{ErrorBody, ErrorKind, RathError, normalize::details};
use crate::core::Provider;
use reqwest::{RequestBuilder, Response, header::HeaderValue};
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
    let mut bytes = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => bytes.extend_from_slice(&chunk),
            Ok(None) => break,
            Err(cause) => {
                let error = RathError::new(
                    error_kind(&cause),
                    "response transfer failed; incomplete response details available",
                )
                .with_context(provider, operation)
                .with_source(RathError::from_error(error_kind(&cause), &cause));
                return Err(enrich(
                    error,
                    metadata(&response),
                    ErrorBody::Incomplete(bytes),
                    secrets,
                ));
            }
        }
    }
    if !response.status().is_success() {
        let mut error = RathError::new(
            ErrorKind::Http,
            "request failed; response details available",
        )
        .with_context(provider, operation);
        let meta = metadata(&response);
        error.inner.request_id = meta.1.as_ref().map(header_text);
        attach_details(&mut error, &bytes);
        return Err(enrich(error, meta, ErrorBody::Complete(bytes), secrets));
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
    let meta = metadata(&response);
    let bytes = read(response, provider, operation, secrets).await?;
    if let Err(error) = validate(&bytes) {
        return Err(enrich(error, meta, ErrorBody::Complete(bytes), secrets));
    }
    Ok(bytes)
}

/// Borrows/clones only selected headers; no error snapshot is allocated on successful responses.
fn metadata(response: &Response) -> (u16, Option<HeaderValue>, Option<HeaderValue>) {
    let request_id = [
        "request-id",
        "x-request-id",
        "x-goog-request-id",
        "x-fal-request-id",
    ]
    .into_iter()
    .find_map(|name| response.headers().get(name))
    .cloned();
    (
        response.status().as_u16(),
        request_id,
        response.headers().get("retry-after").cloned(),
    )
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
            error.inner.source = Some(RathError::from_error(ErrorKind::Deserialize, &cause))
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
    let meta = metadata(&response);
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
            ErrorBody::Complete(bytes),
            secrets,
        )
    })
}

/// Attaches original wire bytes and selected response headers to a decoding/validation error.
fn enrich(
    mut error: RathError,
    (status, request_id, retry_after): (u16, Option<HeaderValue>, Option<HeaderValue>),
    body: ErrorBody,
    secrets: &[&str],
) -> RathError {
    error.inner.http_status = Some(status);
    if error.inner.request_id.is_none() {
        error.inner.request_id = request_id.as_ref().map(header_text);
    }
    error.inner.retry_after = retry_after.as_ref().map(header_text);
    error.with_body(body).sanitized(secrets)
}
