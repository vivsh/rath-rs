//! Stateless Gemini media transport, configuration and completed-candidate validation.
use crate::{ErrorKind, ModelUrl, Provider, RathError};
use serde_json::{Map, Value};
use std::time::Duration;

/// Resolves the existing credential once into the client-owned locator, without another key field.
pub(super) fn resolved_url(url: &ModelUrl) -> Result<ModelUrl, RathError> {
    let mut url = url.clone();
    url.api_key = Some(crate::llm::required_api_key(&url, "GEMINI_API_KEY")?);
    Ok(url)
}

/// Builds a bounded HTTP client that never follows credential-bearing redirects or retries.
pub(super) fn transport() -> Result<reqwest::Client, RathError> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| RathError::from_error(ErrorKind::Transport, &e, &[]))
}

/// Uses the configured API root and bound native model, never a request-level routing override.
pub(super) fn endpoint(url: &ModelUrl) -> String {
    format!(
        "{}/models/{}:generateContent",
        url.base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
            .trim_end_matches('/'),
        url.model.trim_start_matches("models/")
    )
}

pub(super) fn invalid(message: impl Into<String>) -> RathError {
    RathError::new(ErrorKind::Validation, message)
}

/// Merges native top-level defaults then request settings, rejecting every unhandled control.
pub(super) fn config(
    defaults: Option<&Value>,
    request: Option<&Value>,
    allowed: &[&str],
) -> Result<Map<String, Value>, RathError> {
    let mut merged = Map::new();
    for value in [defaults, request].into_iter().flatten() {
        crate::audio::voice::validate_config(Some(value))?;
        for (key, value) in value.as_object().into_iter().flatten() {
            if !allowed.contains(&key.as_str()) {
                return Err(invalid(format!(
                    "unsupported Gemini media provider_config.{key}"
                )));
            }
            merged.insert(key.clone(), value.clone());
        }
    }
    Ok(merged)
}

/// Only one complete candidate can become a capability output; partial/blocked output fails.
pub(super) fn parts(raw: &Value) -> Result<&[Value], RathError> {
    crate::llm::counting::check_output_limit(Provider::Gemini, raw)?;
    if raw.pointer("/promptFeedback/blockReason").is_some() {
        return Err(RathError::new(
            ErrorKind::Provider,
            "Gemini blocked the media request",
        ));
    }
    let candidates = raw["candidates"]
        .as_array()
        .filter(|v| v.len() == 1)
        .ok_or_else(|| RathError::invalid("expected one Gemini media candidate", raw))?;
    let candidate = &candidates[0];
    match candidate["finishReason"].as_str() {
        Some("STOP") => {}
        Some(_) => {
            return Err(RathError::new(
                ErrorKind::Provider,
                "Gemini media generation did not complete",
            ));
        }
        None => {
            return Err(RathError::invalid(
                "missing Gemini candidate finishReason",
                raw,
            ));
        }
    }
    candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .map(Vec::as_slice)
        .ok_or_else(|| RathError::invalid("missing Gemini candidate content.parts", raw))
}

/// Validates thought metadata before excluding reasoning from user-visible output.
pub(super) fn thought(part: &Value, raw: &Value) -> Result<bool, RathError> {
    match part.get("thought") {
        None | Some(Value::Bool(false)) => Ok(false),
        Some(Value::Bool(true)) => Ok(true),
        _ => Err(RathError::invalid("invalid Gemini part.thought", raw)),
    }
}
