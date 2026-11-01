//! Stateless queue operations for sound effects and music.
use super::{FalClient, audio, build_endpoint, merged_config, queue};
use crate::audio::generation::{AudioData, AudioJob, AudioJobStatus, AudioResponse};
use crate::audio::music::{MusicBackend, MusicClient, MusicOptions, MusicRequest};
use crate::audio::sound::{SoundBackend, SoundClient, SoundOptions, SoundRequest};
use crate::core::error::{http, provider_failure};
use crate::{ErrorKind, ModelUrl, Provider, RathError};
use serde_json::{Map, Value};
use std::time::Duration;

const MUSIC: &str = "fal-ai/elevenlabs/music";
const SOUND: &str = "fal-ai/elevenlabs/sound-effects/v2";

/// Constructs the supported model with redirects disabled and bounded requests.
pub(crate) fn new_sound(url: &ModelUrl, options: SoundOptions) -> Result<SoundClient, RathError> {
    if url.model != SOUND {
        return Err(RathError::unsupported(Provider::Fal, "sound model"));
    }
    Ok(SoundClient::from_backend(client(
        url,
        options.provider_config,
    )?))
}

/// Constructs the supported music model without submitting work.
pub(crate) fn new_music(url: &ModelUrl, options: MusicOptions) -> Result<MusicClient, RathError> {
    if url.model != MUSIC {
        return Err(RathError::unsupported(Provider::Fal, "music model"));
    }
    Ok(MusicClient::from_backend(client(
        url,
        options.provider_config,
    )?))
}

/// Builds an immutable audio transport with safe redirects and a bounded request timeout.
fn client(url: &ModelUrl, config: Option<Value>) -> Result<FalClient, RathError> {
    let mut client = audio::audio_client(url, config)?;
    if queue::parse_url(&client.queue_base_url, "queue configuration")?
        .query()
        .is_some()
    {
        return Err(invalid("queue base URL cannot contain a query"));
    }
    client.http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| http::transport(Provider::Fal, "audio construction", e, &[&client.api_key]))?;
    Ok(client)
}

impl SoundBackend for FalClient {
    /// Validates and submits one sound-effects request.
    async fn submit_sound(&self, request: &SoundRequest) -> Result<AudioJob, RathError> {
        let mut payload = config(self, &request.provider_config, &request.prompt)?;
        payload.insert("text".into(), Value::String(request.prompt.clone()));
        if let Some(value) = request.duration_seconds {
            payload.insert("duration_seconds".into(), finite(value)?);
        }
        if let Some(value) = request.looped {
            payload.insert("loop".into(), Value::Bool(value));
        }
        duration(&payload, "duration_seconds", 0.5, 22.0)?;
        boolean(&payload, "loop")?;
        submit(self, payload).await
    }
    async fn get_sound(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        get(self, job).await
    }
}

impl MusicBackend for FalClient {
    /// Validates and submits one music request.
    async fn submit_music(&self, request: &MusicRequest) -> Result<AudioJob, RathError> {
        let mut payload = config(self, &request.provider_config, &request.prompt)?;
        if payload.contains_key("composition_plan") {
            return Err(invalid(
                "composition_plan is incompatible with prompt-based music",
            ));
        }
        payload.insert("prompt".into(), Value::String(request.prompt.clone()));
        if let Some(seconds) = request.duration_seconds {
            if !seconds.is_finite() || !(3.0..=600.0).contains(&seconds) {
                return Err(invalid("music duration is out of range"));
            }
            let millis = (seconds * 1000.0).round();
            if !(3000.0..=600000.0).contains(&millis) {
                return Err(invalid("music duration is out of range"));
            }
            payload.insert("music_length_ms".into(), Value::from(millis as u64));
        }
        if let Some(value) = request.instrumental {
            payload.insert("force_instrumental".into(), Value::Bool(value));
        }
        if let Some(value) = payload.get("music_length_ms") {
            let ms = value
                .as_u64()
                .ok_or_else(|| invalid("music_length_ms must be an integer"))?;
            if !(3000..=600000).contains(&ms) {
                return Err(invalid("music duration is out of range"));
            }
        }
        boolean(&payload, "force_instrumental")?;
        submit(self, payload).await
    }
    async fn get_music(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        get(self, job).await
    }
}

fn invalid(message: &str) -> RathError {
    RathError::new(ErrorKind::Validation, message)
}
fn finite(value: f64) -> Result<Value, RathError> {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| invalid("duration must be finite"))
}
fn duration(map: &Map<String, Value>, key: &str, min: f64, max: f64) -> Result<(), RathError> {
    if let Some(v) = map.get(key) {
        let n = v.as_f64().ok_or_else(|| invalid("invalid duration"))?;
        if !n.is_finite() || !(min..=max).contains(&n) {
            return Err(invalid("duration is out of range"));
        }
    }
    Ok(())
}
fn boolean(map: &Map<String, Value>, key: &str) -> Result<(), RathError> {
    if map.get(key).is_some_and(|v| !v.is_boolean()) {
        return Err(invalid("invalid boolean control"));
    }
    Ok(())
}
/// Merges native defaults and enforces a nonblank prompt and fixed output format.
fn config(
    client: &FalClient,
    native: &Option<Value>,
    prompt: &str,
) -> Result<Map<String, Value>, RathError> {
    if prompt.trim().is_empty() {
        return Err(invalid("prompt must not be blank"));
    }
    crate::registry::prepare::native(client.provider_config.as_ref())?;
    crate::registry::prepare::native(native.as_ref())?;
    let mut payload = merged_config(&client.provider_config, native);
    payload.insert(
        "output_format".into(),
        Value::String("mp3_44100_128".into()),
    );
    Ok(payload)
}

/// Submits once and returns only validated, caller-owned correlation data.
async fn submit(client: &FalClient, payload: Map<String, Value>) -> Result<AudioJob, RathError> {
    let endpoint = build_endpoint(&client.queue_base_url, &client.model);
    let request = client
        .http
        .post(endpoint)
        .json(&payload)
        .header("Authorization", format!("Key {}", client.api_key));
    let response = http::send(request, Provider::Fal, "audio submit", &[&client.api_key]).await?;
    http::mapped_response(
        response,
        Provider::Fal,
        "audio submit",
        &[&client.api_key],
        |raw| {
            if raw.get("error").is_some_and(|e| !e.is_null()) {
                return Err(provider_failure(
                    Provider::Fal,
                    "audio submit",
                    &raw,
                    &[&client.api_key],
                ));
            }
            receipt(client, &raw).map_err(|error| error.classified(ErrorKind::InvalidResponse))
        },
    )
    .await
}

/// Validates the acknowledged receipt before returning its owned correlation data.
fn receipt(client: &FalClient, raw: &Value) -> Result<AudioJob, RathError> {
    let id = field(raw, "request_id")?;
    validate_id(id)?;
    if !client.api_key.is_empty() && id.contains(&client.api_key) {
        return Err(invalid("job ID contains a credential"));
    }
    let status = field(raw, "status_url")?;
    let response = field(raw, "response_url")?;
    validate_url(client, status, id, true)?;
    validate_url(client, response, id, false)?;
    Ok(AudioJob {
        id: id.into(),
        provider: Provider::Fal,
        provider_model: Some(client.model.clone()),
        status_url: Some(status.into()),
        response_url: Some(response.into()),
        raw_metadata: None,
    })
}
fn field<'a>(raw: &'a Value, key: &str) -> Result<&'a str, RathError> {
    raw.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| RathError::invalid(format!("missing {key}"), raw))
}
fn validate_id(id: &str) -> Result<(), RathError> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        return Err(invalid("invalid job ID"));
    }
    Ok(())
}
/// Binds an authenticated queue URL to its configured origin, app, and job ID.
fn validate_url(client: &FalClient, value: &str, id: &str, status: bool) -> Result<(), RathError> {
    if !client.api_key.is_empty() && value.contains(&client.api_key) {
        return Err(invalid("queue URL contains a credential"));
    }
    let url = queue::parse_url(value, "queue URL")?;
    if url.query().is_some() {
        return Err(invalid("queue URL cannot contain a query"));
    }
    let base = queue::parse_url(&client.queue_base_url, "queue configuration")?;
    if url.origin() != base.origin() {
        return Err(invalid("queue URL origin mismatch"));
    }
    let app = super::video::queue_model(&client.model);
    let suffix = if status { "/status" } else { "" };
    let expected = format!(
        "{}/{}/requests/{id}{suffix}",
        client.queue_base_url.trim_end_matches('/'),
        app
    );
    let expected = queue::parse_url(&expected, "queue configuration")?;
    if url.path() != expected.path() {
        return Err(invalid("queue URL does not match job ID and model"));
    }
    Ok(())
}
/// Performs one authenticated JSON request with credential-aware diagnostics.
async fn request(
    client: &FalClient,
    req: reqwest::RequestBuilder,
    stage: &'static str,
) -> Result<Value, RathError> {
    let response = http::send(
        req.header("Authorization", format!("Key {}", client.api_key)),
        Provider::Fal,
        stage,
        &[&client.api_key],
    )
    .await?;
    http::mapped_response(response, Provider::Fal, stage, &[&client.api_key], Ok).await
}

/// Reads a job snapshot, never submitting work or retaining lifecycle state.
async fn get(client: &FalClient, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
    if job.provider != Provider::Fal || job.provider_model.as_deref() != Some(&client.model) {
        return Err(invalid("job provider or model mismatch"));
    }
    validate_id(&job.id)?;
    let status = job
        .status_url
        .as_deref()
        .ok_or_else(|| invalid("missing status URL"))?;
    let response = job
        .response_url
        .as_deref()
        .ok_or_else(|| invalid("missing response URL"))?;
    validate_url(client, status, &job.id, true)?;
    validate_url(client, response, &job.id, false)?;
    let raw = request(client, client.http.get(status), "audio status").await?;
    if raw
        .get("request_id")
        .is_some_and(|id| id.as_str() != Some(&job.id))
    {
        return Err(RathError::invalid("audio status request ID mismatch", &raw)
            .sanitized(&[&client.api_key]));
    }
    if raw.get("error").is_some_and(|e| !e.is_null()) {
        return Ok(failure(client, &raw));
    }
    match raw.get("status").and_then(Value::as_str) {
        Some("IN_QUEUE") => Ok(AudioJobStatus::Queued {
            queue_position: raw.get("queue_position").and_then(Value::as_u64),
            raw_metadata: None,
        }),
        Some("IN_PROGRESS") => Ok(AudioJobStatus::Running { raw_metadata: None }),
        Some("COMPLETED") => result(client, response, &job.id).await,
        _ => {
            Err(RathError::invalid("invalid audio job status", &raw).sanitized(&[&client.api_key]))
        }
    }
}
/// Retrieves a terminal result without downloading or retaining the output media.
async fn result(
    client: &FalClient,
    endpoint: &str,
    job_id: &str,
) -> Result<AudioJobStatus, RathError> {
    let raw = match request(client, client.http.get(endpoint), "audio result").await {
        Ok(raw) => raw,
        Err(error) => return classify_result_error(client, error, job_id),
    };
    if raw.get("error").is_some_and(|e| !e.is_null()) {
        return Ok(failure(client, &raw));
    }
    let url = raw
        .get("audio")
        .and_then(|v| v.get("url"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            RathError::invalid("missing audio.url", &raw).sanitized(&[&client.api_key])
        })?;
    queue::parse_url(url, "audio URL")?;
    if !client.api_key.is_empty() && url.contains(&client.api_key) {
        return Err(invalid("audio URL contains a credential"));
    }
    if let Some(mime) = raw.get("audio").and_then(|v| v.get("content_type")) {
        let valid = mime
            .as_str()
            .is_some_and(crate::audio::voice::valid_audio_mime);
        if !valid {
            return Err(RathError::invalid("invalid audio content_type", &raw)
                .sanitized(&[&client.api_key]));
        }
    }
    Ok(AudioJobStatus::Succeeded {
        response: AudioResponse {
            audio: AudioData::Url { url: url.into() },
            raw_metadata: None,
        },
    })
}
/// Converts only explicitly identified completed-job failures.
fn classify_result_error(
    client: &FalClient,
    error: RathError,
    job_id: &str,
) -> Result<AudioJobStatus, RathError> {
    let Some(status) = error.http_status() else {
        return Err(error);
    };
    if matches!(status, 401 | 403 | 404 | 429) {
        return Err(error);
    }
    let Some(raw) = error
        .response_body()
        .filter(|body| body.is_complete())
        .and_then(|body| serde_json::from_slice::<Value>(body.bytes()).ok())
    else {
        return Err(error);
    };
    if raw
        .get("request_id")
        .is_some_and(|id| id.as_str() != Some(job_id))
    {
        return Err(error);
    }
    if explicit_failure(&raw, job_id) {
        Ok(failure(client, &raw))
    } else {
        Err(error)
    }
}

/// Requires a matching request ID and an explicit completed-job failure envelope.
fn explicit_failure(raw: &Value, job_id: &str) -> bool {
    raw.get("request_id").and_then(Value::as_str) == Some(job_id)
        && raw.get("status").and_then(Value::as_str) == Some("COMPLETED")
        && raw.get("error").is_some_and(|error| !error.is_null())
}

fn failure(client: &FalClient, raw: &Value) -> AudioJobStatus {
    let err = provider_failure(Provider::Fal, "audio job", raw, &[&client.api_key]);
    AudioJobStatus::Failed {
        message: err.message().into(),
        raw_metadata: err
            .response_body()
            .and_then(|body| serde_json::from_slice(body.bytes()).ok()),
    }
}

#[cfg(test)]
#[path = "tests/generation.rs"]
mod tests;
