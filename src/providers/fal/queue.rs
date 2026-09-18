//! Operation-local queue execution for Fal audio requests.

use std::time::Duration;

use reqwest::Url;
use serde_json::Value;

use super::{FalClient, build_endpoint};
use crate::core::error::http;
use crate::core::{ErrorKind, Provider, RathError};

pub(super) const AUDIO_TIMEOUT: Duration = Duration::from_secs(300);

/// Submits once and polls the returned URLs without retaining job state in the client.
pub(super) async fn run<T>(
    client: &FalClient,
    model: &str,
    payload: Value,
    decode: impl FnOnce(Value) -> Result<T, RathError>,
) -> Result<T, RathError> {
    let endpoint = build_endpoint(&client.queue_base_url, model);
    let (status_url, response_url) = json(
        client.http.post(endpoint).json(&payload),
        client,
        "submit",
        |raw| {
            Ok((
                queue_url(client, &raw, "status_url")?,
                queue_url(client, &raw, "response_url")?,
            ))
        },
    )
    .await?;
    loop {
        let complete = json(
            client.http.get(status_url.clone()),
            client,
            "status",
            |raw| match raw.get("status").and_then(Value::as_str) {
                Some("IN_QUEUE" | "IN_PROGRESS") => Ok(false),
                Some("COMPLETED") => Ok(true),
                _ => Err(failed("status decoding").with_response(&raw)),
            },
        )
        .await?;
        if complete {
            return json(client.http.get(response_url), client, "result", decode).await;
        }
        tokio::time::sleep(client.poll_interval).await;
    }
}

/// Rejects queue URLs outside the configured service before credentials are attached.
fn queue_url(client: &FalClient, raw: &Value, field: &str) -> Result<Url, RathError> {
    let value = raw
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| RathError::invalid(format!("missing {field} (queue URLs)"), raw))?;
    let url = parse_url(value, "queue URLs")?;
    let base = parse_url(&client.queue_base_url, "queue configuration")?;
    if url.origin() != base.origin() {
        return Err(failed("queue URL origin"));
    }
    Ok(url)
}

/// Allows HTTP(S) only, excluding embedded credentials and URL fragments.
pub(super) fn parse_url(value: &str, stage: &'static str) -> Result<Url, RathError> {
    let url = Url::parse(value).map_err(|error| {
        RathError::from_error(ErrorKind::InvalidResponse, &error).with_context(Provider::Fal, stage)
    })?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(failed(stage));
    }
    Ok(url)
}

/// Preserve Fal HTTP/decoding failures for callers; response details may contain private data.
async fn json<T>(
    request: reqwest::RequestBuilder,
    client: &FalClient,
    stage: &'static str,
    decode: impl FnOnce(Value) -> Result<T, RathError>,
) -> Result<T, RathError> {
    http::mapped(
        request.header("Authorization", format!("Key {}", client.api_key)),
        Provider::Fal,
        stage,
        &[&client.api_key],
        decode,
    )
    .await
}

/// Produces a diagnostic containing only a static operation stage.
pub(super) fn failed(stage: &'static str) -> RathError {
    RathError::new(
        ErrorKind::InvalidResponse,
        format!("invalid or missing {stage}"),
    )
    .with_context(Provider::Fal, stage)
}
