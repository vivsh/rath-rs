//! Operation-local queue execution for Fal audio requests.

use std::time::Duration;

use reqwest::Url;
use serde_json::Value;

use super::{FalClient, build_endpoint};
use crate::core::RathError;

pub(super) const AUDIO_TIMEOUT: Duration = Duration::from_secs(300);

/// Submits once and polls the returned URLs without retaining job state in the client.
pub(super) async fn run(
    client: &FalClient,
    model: &str,
    payload: Value,
) -> Result<Value, RathError> {
    let endpoint = build_endpoint(&client.queue_base_url, model);
    let submitted = json(client.http.post(endpoint).json(&payload), client, "submit").await?;
    let status_url = queue_url(client, &submitted, "status_url")?;
    let response_url = queue_url(client, &submitted, "response_url")?;
    loop {
        let status = json(client.http.get(status_url.clone()), client, "status").await?;
        match status.get("status").and_then(Value::as_str) {
            Some("IN_QUEUE" | "IN_PROGRESS") => tokio::time::sleep(client.poll_interval).await,
            Some("COMPLETED") => {
                if status.get("error").is_some_and(|error| !error.is_null()) {
                    return Err(failed("execution"));
                }
                let result = json(client.http.get(response_url), client, "result").await?;
                if result.get("error").is_some_and(|error| !error.is_null()) {
                    return Err(failed("execution"));
                }
                return Ok(result);
            }
            _ => return Err(failed("status decoding")),
        }
    }
}

/// Rejects queue URLs outside the configured service before credentials are attached.
fn queue_url(client: &FalClient, raw: &Value, field: &str) -> Result<Url, RathError> {
    let value = raw
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| failed("queue URLs"))?;
    let url = parse_url(value, "queue URLs")?;
    let base = parse_url(&client.queue_base_url, "queue configuration")?;
    if url.origin() != base.origin() {
        return Err(failed("queue URL origin"));
    }
    Ok(url)
}

/// Allows HTTP(S) only, excluding embedded credentials and URL fragments.
pub(super) fn parse_url(value: &str, stage: &str) -> Result<Url, RathError> {
    let url = Url::parse(value).map_err(|_| failed(stage))?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(failed(stage));
    }
    Ok(url)
}

/// Reads queue JSON without exposing provider bodies, credentials or signed URLs in errors.
async fn json(
    request: reqwest::RequestBuilder,
    client: &FalClient,
    stage: &str,
) -> Result<Value, RathError> {
    let response = request
        .header("Authorization", format!("Key {}", client.api_key))
        .send()
        .await
        .map_err(|_| failed(stage))?;
    if !response.status().is_success() {
        return Err(RathError::Provider(format!(
            "Fal audio {stage} failed (HTTP {})",
            response.status().as_u16()
        )));
    }
    response.json().await.map_err(|_| failed(stage))
}

/// Produces a diagnostic containing only a static operation stage.
pub(super) fn failed(stage: &str) -> RathError {
    RathError::Provider(format!("Fal audio {stage} failed"))
}
