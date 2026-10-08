//! Optional Standard Webhooks verification for project-registered Gemini video callbacks.
use super::{super::media, response};
use crate::video::{VideoData, VideoEvent, VideoJobStatus, VideoOptions, VideoResponse};
use crate::{ErrorBody, ErrorKind, ModelUrl, Provider, RathError};
use base64::{Engine, engine::general_purpose::STANDARD};
use hmac::{Hmac, Mac};
use http::HeaderMap;
use serde_json::{Value, json};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

const ENV_SETTING: &str = "webhook_signing_secret_env";
const OPERATION: &str = "video webhook";

/// Reads only the configured variable name; generation never transports this local setting.
pub(super) fn configured_env(options: &VideoOptions) -> Result<Option<&str>, RathError> {
    let Some(value) = options
        .provider_config
        .as_ref()
        .and_then(|value| value.get(ENV_SETTING))
    else {
        return Ok(None);
    };
    let name = value
        .as_str()
        .filter(|name| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .ok_or_else(|| {
            media::invalid("webhook_signing_secret_env must name an environment variable")
        })?;
    Ok(Some(name))
}

/// Resolves the caller-owned secret only for this parse; no callback/key state is retained.
pub(super) fn parse(
    url: &ModelUrl,
    options: &VideoOptions,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<VideoEvent, RathError> {
    let name = configured_env(options)?.ok_or_else(|| {
        RathError::unsupported(
            Provider::Gemini,
            "video webhook verification is not configured",
        )
    })?;
    let secret = std::env::var(name).map_err(|error| {
        media::invalid("Gemini webhook signing-secret environment variable is unavailable")
            .with_source(crate::core::error::credential_cause(&error))
    })?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| media::invalid("system clock is before the Unix epoch"))?
        .as_secs();
    parse_at(url, headers, body, &secret, now)
}

/// Provides deterministic verification without exposing caller-chosen clocks as public API.
fn parse_at(
    url: &ModelUrl,
    headers: &HeaderMap,
    body: &[u8],
    secret: &str,
    now: u64,
) -> Result<VideoEvent, RathError> {
    let credentials = &[
        url.api_key.as_deref().unwrap_or_default(),
        secret,
        secret.strip_prefix("whsec_").unwrap_or(secret),
    ];
    verify(headers, body, secret, now).map_err(|error| {
        error
            .with_context(Provider::Gemini, OPERATION)
            .sanitized(credentials)
    })?;
    decode(url, body, header(headers, "webhook-id")?).map_err(|error| {
        error
            .with_context(Provider::Gemini, OPERATION)
            .with_raw_body(ErrorBody::Complete(body.to_vec()))
            .sanitized(credentials)
    })
}

/// Authenticates the exact bytes with constant-time MAC comparison before interpreting JSON.
fn verify(headers: &HeaderMap, body: &[u8], secret: &str, now: u64) -> Result<(), RathError> {
    let id = header(headers, "webhook-id")?;
    let timestamp = header(headers, "webhook-timestamp")?;
    let signatures = header(headers, "webhook-signature")?;
    if !timestamp.bytes().all(|b| b.is_ascii_digit()) {
        return Err(media::invalid("invalid webhook timestamp"));
    }
    let seconds = timestamp
        .parse::<u64>()
        .map_err(|_| media::invalid("invalid webhook timestamp"))?;
    if now.abs_diff(seconds) > 300 {
        return Err(media::invalid(
            "webhook timestamp is outside the accepted window",
        ));
    }
    let key = STANDARD
        .decode(secret.strip_prefix("whsec_").unwrap_or(secret))
        .map_err(|_| media::invalid("invalid webhook signing-secret encoding"))?;
    if key.is_empty() {
        return Err(media::invalid("empty webhook signing secret"));
    }
    let mut mac = Hmac::<Sha256>::new_from_slice(&key)
        .map_err(|_| media::invalid("invalid webhook signing secret"))?;
    mac.update(id.as_bytes());
    mac.update(b".");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    for candidate in signatures.split_ascii_whitespace() {
        if let Some(encoded) = candidate.strip_prefix("v1,")
            && let Ok(signature) = STANDARD.decode(encoded)
            && mac.clone().verify_slice(&signature).is_ok()
        {
            return Ok(());
        }
    }
    Err(media::invalid("webhook signature verification failed"))
}

/// Rejects ambiguous duplicate headers while permitting signature lists in one signed header.
fn header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str, RathError> {
    let mut values = headers.get_all(name).iter();
    let value = values
        .next()
        .ok_or_else(|| media::invalid("missing signed webhook header"))?;
    if values.next().is_some() {
        return Err(media::invalid("duplicate signed webhook header"));
    }
    let value = value
        .to_str()
        .map_err(|_| media::invalid("invalid signed webhook header"))?;
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_graphic() || (name == "webhook-signature" && b == b' '))
    {
        return Err(media::invalid("invalid signed webhook header"));
    }
    Ok(value)
}

/// Converts only an authenticated video completion into the existing event/status contract.
fn decode(url: &ModelUrl, body: &[u8], delivery_id: &str) -> Result<VideoEvent, RathError> {
    let raw: Value = serde_json::from_slice(body).map_err(|_| {
        RathError::new(ErrorKind::Deserialize, "invalid authenticated webhook JSON")
    })?;
    if raw["type"].as_str() != Some("video.generated") {
        return Err(RathError::invalid(
            "unsupported Gemini webhook event type",
            &raw,
        ));
    }
    if raw["version"].as_str() != Some("v1") {
        return Err(RathError::invalid(
            "unsupported Gemini webhook envelope version",
            &raw,
        ));
    }
    let id = raw
        .pointer("/data/id")
        .and_then(Value::as_str)
        .ok_or_else(|| RathError::invalid("missing video webhook operation ID", &raw))?;
    response::operation_url(url, id)?;
    let uri = raw
        .pointer("/data/output_file_uri")
        .and_then(Value::as_str)
        .ok_or_else(|| RathError::invalid("missing video webhook output URI", &raw))?;
    response::video_uri(uri)?;
    Ok(VideoEvent::new(
        id.into(),
        VideoJobStatus::Succeeded {
            response: VideoResponse::new(vec![VideoData::Url { url: uri.into() }])
                .with_raw_metadata(json!({"event":raw, "webhook_id":delivery_id})),
        },
    ))
}

#[cfg(test)]
#[path = "tests/webhook.rs"]
mod tests;
