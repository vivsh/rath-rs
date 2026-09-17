//! Model-specific Fal speech synthesis and transcription mappings.

use async_trait::async_trait;
use base64::Engine;
use serde_json::{Map, Value};

use super::{FalClient, merged_config, queue};
use crate::audio::stt::{SttClient, SttOptions, SttRequest, SttResponse};
use crate::audio::tts::{TtsClient, TtsOptions, TtsRequest, TtsResponse};
use crate::core::{ModelUrl, Provider, RathError};

const KOKORO: &str = "fal-ai/kokoro/american-english";
const ELEVENLABS: &str = "fal-ai/elevenlabs/tts/turbo-v2.5";
const WIZPER: &str = "fal-ai/wizper";
const SCRIBE: &str = "fal-ai/elevenlabs/speech-to-text/scribe-v2";

/// Constructs a speech client with immutable options and credential-safe redirect behavior.
pub(crate) fn new_tts_client(
    url: &ModelUrl,
    options: TtsOptions,
) -> Result<Box<dyn TtsClient>, RathError> {
    tts_input_field(&url.model)?;
    Ok(Box::new(audio_client(url, options.provider_config)?))
}

/// Constructs a transcription client for an explicitly supported Fal endpoint.
pub(crate) fn new_stt_client(
    url: &ModelUrl,
    options: SttOptions,
) -> Result<Box<dyn SttClient>, RathError> {
    validate_stt_model(&url.model)?;
    Ok(Box::new(audio_client(url, options.provider_config)?))
}

/// Reuses the Fal client without changing redirect behavior for existing image/video clients.
fn audio_client(url: &ModelUrl, config: Option<Value>) -> Result<FalClient, RathError> {
    validate_config(&config)?;
    let mut client = FalClient::new(url, config, "audio")?;
    queue::parse_url(&client.queue_base_url, "queue configuration")?;
    client.http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| queue::failed("HTTP client construction"))?;
    Ok(client)
}

#[async_trait]
impl TtsClient for FalClient {
    /// Synthesizes and downloads audio within a five-minute overall deadline.
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let model = request.model.as_deref().unwrap_or(&self.model);
        let payload = tts_payload(model, &self.provider_config, request)?;
        tokio::time::timeout(queue::AUDIO_TIMEOUT, async {
            let raw = queue::run(self, model, Value::Object(payload)).await?;
            let (mime_type, data) = download(self, &raw).await?;
            Ok(TtsResponse {
                mime_type,
                data,
                raw_metadata: Some(raw),
            })
        })
        .await
        .map_err(|_| queue::failed("timeout (remote job may still run)"))?
    }
}

#[async_trait]
impl SttClient for FalClient {
    /// Sends audio as a data URI and waits up to five minutes for transcription.
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        let model = request.model.as_deref().unwrap_or(&self.model);
        let payload = stt_payload(model, &self.provider_config, request)?;
        tokio::time::timeout(queue::AUDIO_TIMEOUT, async {
            let raw = queue::run(self, model, Value::Object(payload)).await?;
            let text = raw
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| queue::failed("transcript decoding"))?
                .to_string();
            Ok(SttResponse {
                text,
                raw_metadata: Some(raw),
            })
        })
        .await
        .map_err(|_| queue::failed("timeout (remote job may still run)"))?
    }
}

/// Builds a verified endpoint payload; typed fields override provider configuration.
fn tts_payload(
    model: &str,
    config: &Option<Value>,
    request: &TtsRequest,
) -> Result<Map<String, Value>, RathError> {
    let input_field = tts_input_field(model)?;
    validate_config(config)?;
    validate_config(&request.provider_config)?;
    if request.input.trim().is_empty() {
        return Err(RathError::Validation(
            "Fal speech input must not be empty".into(),
        ));
    }
    if request.format.is_some() {
        return Err(RathError::Validation(
            "this Fal endpoint does not support selecting an audio format".into(),
        ));
    }
    let mut payload = merged_config(config, &request.provider_config);
    payload.insert(input_field.into(), Value::String(request.input.clone()));
    if let Some(voice) = &request.voice {
        payload.insert("voice".into(), Value::String(voice.clone()));
    }
    Ok(payload)
}

/// Maps text only for endpoints with verified synthesis input/output contracts.
fn tts_input_field(model: &str) -> Result<&'static str, RathError> {
    match model {
        KOKORO => Ok("prompt"),
        ELEVENLABS => Ok("text"),
        _ => Err(unsupported("text-to-speech")),
    }
}

/// Accepts only endpoints with verified transcription input/output contracts.
fn validate_stt_model(model: &str) -> Result<(), RathError> {
    match model {
        WIZPER | SCRIBE => Ok(()),
        _ => Err(unsupported("speech-to-text")),
    }
}

/// Reports unsupported endpoint selection without echoing caller-controlled text.
fn unsupported(capability: &str) -> RathError {
    RathError::UnsupportedCapability {
        provider: Provider::Fal,
        capability: format!("{capability} for selected endpoint"),
    }
}

/// Encodes bytes in operation-local storage and retains native model configuration.
fn stt_payload(
    model: &str,
    config: &Option<Value>,
    request: &SttRequest,
) -> Result<Map<String, Value>, RathError> {
    validate_stt_model(model)?;
    validate_config(config)?;
    validate_config(&request.provider_config)?;
    if request.data.is_empty() {
        return Err(RathError::Validation(
            "Fal transcription audio must not be empty".into(),
        ));
    }
    let mime = audio_mime(&request.mime_type)
        .filter(|_| !request.mime_type.contains(';'))
        .ok_or_else(|| {
            RathError::Validation(
                "Fal transcription requires an audio MIME type without parameters".into(),
            )
        })?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&request.data);
    let mut payload = merged_config(config, &request.provider_config);
    payload.insert(
        "audio_url".into(),
        Value::String(format!("data:{mime};base64,{encoded}")),
    );
    Ok(payload)
}

/// Rejects non-object configuration rather than silently discarding it.
fn validate_config(config: &Option<Value>) -> Result<(), RathError> {
    if config.as_ref().is_some_and(|value| !value.is_object()) {
        return Err(RathError::Validation(
            "Fal audio provider_config must be an object".into(),
        ));
    }
    Ok(())
}

/// Downloads output without authorization headers and requires an identifiable audio MIME type.
async fn download(client: &FalClient, raw: &Value) -> Result<(String, Vec<u8>), RathError> {
    let audio = raw
        .get("audio")
        .ok_or_else(|| queue::failed("audio decoding"))?;
    let url = audio
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| queue::failed("audio URL"))?;
    let response = client
        .http
        .get(queue::parse_url(url, "audio URL")?)
        .send()
        .await
        .map_err(|_| queue::failed("download"))?;
    if !response.status().is_success() {
        return Err(queue::failed("download HTTP status"));
    }
    let header = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    let metadata = audio.get("content_type").and_then(Value::as_str);
    let mime_type = download_mime(header, metadata)?;
    let data = response
        .bytes()
        .await
        .map_err(|_| queue::failed("download body"))?
        .to_vec();
    if data.is_empty() {
        return Err(queue::failed("empty audio"));
    }
    Ok((mime_type, data))
}

/// Uses metadata only for absent or generic download types, never to disguise non-audio content.
fn download_mime(header: Option<&str>, metadata: Option<&str>) -> Result<String, RathError> {
    if let Some(header) = header {
        if let Some(mime) = audio_mime(header) {
            return Ok(mime);
        }
        if !header
            .split(';')
            .next()
            .is_some_and(|h| h.trim().eq_ignore_ascii_case("application/octet-stream"))
        {
            return Err(queue::failed("audio MIME type"));
        }
    }
    metadata
        .and_then(audio_mime)
        .ok_or_else(|| queue::failed("audio MIME type"))
}

/// Normalizes a syntactically valid audio MIME type without parameters.
fn audio_mime(value: &str) -> Option<String> {
    let mime = value.split(';').next()?.trim().to_ascii_lowercase();
    let subtype = mime.strip_prefix("audio/")?;
    if subtype.is_empty()
        || !subtype
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$&^_.+-".contains(&b))
    {
        return None;
    }
    Some(mime)
}

#[cfg(test)]
#[path = "audio/tests/mod.rs"]
mod tests;
