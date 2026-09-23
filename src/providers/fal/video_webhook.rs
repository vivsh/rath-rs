//! Verifies Fal completion notifications without retaining keys or job state.
use super::{FalClient, video};
use crate::core::error::http as transport;
use crate::core::{ErrorBody, ErrorKind, Provider, RathError};
use crate::video::{VideoEvent, VideoJobStatus};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signature, VerifyingKey};
use http::HeaderMap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const JWKS_URL: &str = "https://rest.fal.ai/.well-known/jwks.json";
const OPERATION: &str = "video webhook";

/// Authenticates with Fal's fixed key service and the current clock, without retaining state.
pub(super) async fn parse(
    client: &FalClient,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<VideoEvent, RathError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock is before the Unix epoch"))?
        .as_secs();
    parse_at(client, headers, body, JWKS_URL, now).await
}

/// A private injection seam keeps tests offline without exposing caller-chosen verification keys.
async fn parse_at(
    client: &FalClient,
    headers: &HeaderMap,
    body: &[u8],
    jwks_url: &str,
    now: u64,
) -> Result<VideoEvent, RathError> {
    let (message, signature) = signed_message(headers, body, now)?;
    let keys = fetch_keys(jwks_url).await?;
    verify(&keys, &message, &signature)?;
    decode(client, body).map_err(|error| {
        error
            .with_context(Provider::Fal, OPERATION)
            .with_body(ErrorBody::Complete(body.to_vec()))
            .sanitized(&[&client.api_key])
    })
}

/// Parses each signed header once and constructs the exact provider-defined signed message.
fn signed_message(
    headers: &HeaderMap,
    body: &[u8],
    now: u64,
) -> Result<(Vec<u8>, Signature), RathError> {
    let request_id = header(headers, "x-fal-webhook-request-id")?;
    let user_id = header(headers, "x-fal-webhook-user-id")?;
    let timestamp = header(headers, "x-fal-webhook-timestamp")?;
    if !timestamp.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("invalid webhook timestamp"));
    }
    let seconds: u64 = timestamp
        .parse()
        .map_err(|_| invalid("invalid webhook timestamp"))?;
    if now.abs_diff(seconds) > 300 {
        return Err(invalid("webhook timestamp is outside the accepted window"));
    }
    let encoded = header(headers, "x-fal-webhook-signature")?;
    let signature = signature(encoded)?;
    let hash = Sha256::digest(body);
    Ok((
        format!("{request_id}\n{user_id}\n{timestamp}\n{hash:x}").into_bytes(),
        signature,
    ))
}

/// HeaderMap preserves duplicates, which are rejected instead of selecting an ambiguous value.
fn header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str, RathError> {
    let mut values = headers.get_all(name).iter();
    let value = values
        .next()
        .ok_or_else(|| invalid("missing signed webhook header"))?;
    if values.next().is_some() {
        return Err(invalid("duplicate signed webhook header"));
    }
    let value = value
        .to_str()
        .map_err(|_| invalid("invalid signed webhook header"))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(invalid("invalid signed webhook header"));
    }
    Ok(value)
}

/// Decodes exactly 64 signature bytes without adding a separate hex dependency.
fn signature(encoded: &str) -> Result<Signature, RathError> {
    if encoded.len() != 128 || !encoded.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("invalid webhook signature encoding"));
    }
    let mut bytes = [0u8; 64];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
            .map_err(|_| invalid("invalid webhook signature encoding"))?;
    }
    Ok(Signature::from_bytes(&bytes))
}

/// Fetches only trusted public keys; no Fal API key, redirects, retries, or cache.
async fn fetch_keys(url: &str) -> Result<Vec<VerifyingKey>, RathError> {
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|error| transport::transport(Provider::Fal, OPERATION, error, &[]))?;
    transport::mapped(
        http.get(url),
        Provider::Fal,
        "video webhook keys",
        &[],
        decode_keys,
    )
    .await
}

/// Accepts valid Ed25519 signing keys, ignoring unrelated or malformed entries during rotation.
fn decode_keys(raw: Value) -> Result<Vec<VerifyingKey>, RathError> {
    let entries = raw
        .get("keys")
        .and_then(Value::as_array)
        .ok_or_else(|| RathError::invalid("JWKS is missing keys", &raw))?;
    let keys: Vec<_> = entries.iter().filter_map(decode_key).collect();
    if keys.is_empty() {
        return Err(RathError::invalid(
            "JWKS contains no usable Ed25519 signing keys",
            &raw,
        ));
    }
    Ok(keys)
}

/// Rejects incompatible key metadata and decodes the Ed25519 public key.
fn decode_key(raw: &Value) -> Option<VerifyingKey> {
    for (field, expected) in [("kty", "OKP"), ("crv", "Ed25519"), ("use", "sig")] {
        if raw
            .get(field)
            .is_some_and(|value| value.as_str() != Some(expected))
        {
            return None;
        }
    }
    let data = URL_SAFE_NO_PAD.decode(raw.get("x")?.as_str()?).ok()?;
    let bytes: [u8; 32] = data.try_into().ok()?;
    VerifyingKey::from_bytes(&bytes).ok()
}

/// Accepts a strict signature from any current signing key, including during rotation.
fn verify(keys: &[VerifyingKey], message: &[u8], signature: &Signature) -> Result<(), RathError> {
    if keys
        .iter()
        .any(|key| key.verify_strict(message, signature).is_ok())
    {
        Ok(())
    } else {
        Err(invalid("webhook signature verification failed"))
    }
}

/// Decodes only authenticated bytes; incomplete success is not a remote generation failure.
fn decode(client: &FalClient, body: &[u8]) -> Result<VideoEvent, RathError> {
    let raw: Value = serde_json::from_slice(body).map_err(|cause| {
        RathError::new(ErrorKind::Deserialize, "invalid authenticated webhook JSON")
            .with_source(RathError::from_error(ErrorKind::Deserialize, &cause))
    })?;
    let job_id = raw
        .get("request_id")
        .and_then(Value::as_str)
        .filter(|id| {
            !id.is_empty()
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
        })
        .ok_or_else(|| RathError::invalid("missing or invalid webhook request_id", &raw))?
        .to_owned();
    if raw
        .get("payload_error")
        .is_some_and(|value| !value.is_null())
    {
        return Err(RathError::invalid(
            "webhook payload unavailable; reconcile the submitted job",
            &raw,
        ));
    }
    let status = match raw.get("status").and_then(Value::as_str) {
        Some("OK") => success(client, raw)?,
        Some("ERROR") => video::video_failure(&raw, client)
            .ok_or_else(|| RathError::invalid("webhook ERROR is missing error details", &raw))?,
        _ => return Err(RathError::invalid("unknown webhook status", &raw)),
    };
    Ok(VideoEvent { job_id, status })
}

/// Shares polling's media decoder but retains the complete authenticated notification as metadata.
fn success(client: &FalClient, raw: Value) -> Result<VideoJobStatus, RathError> {
    if raw.get("error").is_some_and(|value| !value.is_null()) {
        return Err(RathError::invalid(
            "webhook OK contains contradictory error details",
            &raw,
        ));
    }
    let payload = raw
        .get("payload")
        .filter(|value| value.is_object())
        .ok_or_else(|| RathError::invalid("webhook OK is missing its output payload", &raw))?;
    if payload.get("error").is_some_and(|value| !value.is_null()) {
        return Err(RathError::invalid(
            "webhook OK contains a failed output payload",
            &raw,
        ));
    }
    let mut status = video::completed_video(payload.clone(), client)?;
    if let VideoJobStatus::Succeeded { response } = &mut status {
        response.raw_metadata = Some(raw);
    }
    Ok(status)
}

fn invalid(message: &str) -> RathError {
    RathError::new(ErrorKind::Validation, message).with_context(Provider::Fal, OPERATION)
}

#[cfg(test)]
#[path = "tests/video_webhook.rs"]
mod tests;
