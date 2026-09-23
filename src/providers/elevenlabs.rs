//! Native ElevenLabs voice operations; no local persistence or remote-resource cleanup.
use crate::audio::{
    tts::{TtsClient, TtsOptions, TtsRequest, TtsResponse},
    voice::{Voice, VoiceData, invalid, validate_config, validate_samples},
    voice_clone::{VoiceCloneClient, VoiceCloneOptions, VoiceCloneRequest},
    voice_design::{
        VoiceDesignClient, VoiceDesignOptions, VoiceDesignRequest, VoiceDesignResponse,
        VoicePreview,
    },
};
use crate::core::{ErrorKind, ModelUrl, Provider, RathError, error::http};
use crate::llm::{configured_base_url, required_api_key};
use async_trait::async_trait;
use reqwest::{
    Client,
    multipart::{Form, Part},
};
use serde_json::{Map, Value, json};
mod design;
#[cfg(test)]
#[path = "tests/elevenlabs.rs"]
mod tests;

const SCOPE: &str = "elevenlabs";
const PREVIEW_SCOPE: &str = "elevenlabs/voice-design";

/// Immutable provider connection and model selection; operation state stays local.
struct ElevenLabsClient {
    http: Client,
    api_key: String,
    base_url: String,
    model: String,
    provider_config: Option<Value>,
}

/// Constructs a verified native voice-design model adapter.
pub(crate) fn new_design(
    url: &ModelUrl,
    options: VoiceDesignOptions,
) -> Result<Box<dyn VoiceDesignClient>, RathError> {
    if !matches!(
        url.model.as_str(),
        "eleven_multilingual_ttv_v2" | "eleven_ttv_v3"
    ) {
        return Err(unsupported("voice design model"));
    }
    Ok(Box::new(build(url, options.provider_config)?))
}

/// Constructs instant voice cloning; training and verification workflows are not implicit.
pub(crate) fn new_clone(
    url: &ModelUrl,
    options: VoiceCloneOptions,
) -> Result<Box<dyn VoiceCloneClient>, RathError> {
    if url.model != "ivc" {
        return Err(unsupported("voice cloning model"));
    }
    Ok(Box::new(build(url, options.provider_config)?))
}

/// Constructs a verified native speech model adapter.
pub(crate) fn new_tts(
    url: &ModelUrl,
    options: TtsOptions,
) -> Result<Box<dyn TtsClient>, RathError> {
    validate_tts_model(&url.model)?;
    Ok(Box::new(build(url, options.provider_config)?))
}

/// Builds a redirect-denying bounded client, without performing any I/O or registration.
fn build(url: &ModelUrl, config: Option<Value>) -> Result<ElevenLabsClient, RathError> {
    validate_config(config.as_ref())?;
    let base_url = configured_base_url(url, "https://api.elevenlabs.io/v1");
    let base = reqwest::Url::parse(&base_url)
        .map_err(|error| RathError::from_error(ErrorKind::InvalidUrl, &error))?;
    if !matches!(base.scheme(), "https" | "http")
        || base.host_str().is_none()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
    {
        return Err(invalid("invalid ElevenLabs API base URL"));
    }
    Ok(ElevenLabsClient {
        http: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|error| {
                http::transport(Provider::ElevenLabs, "client construction", error, &[])
            })?,
        api_key: required_api_key(url, "ELEVENLABS_API_KEY")?,
        base_url,
        model: url.model.clone(),
        provider_config: config,
    })
}

impl ElevenLabsClient {
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http
            .post(format!("{}/{path}", self.base_url.trim_end_matches('/')))
            .header("xi-api-key", &self.api_key)
    }

    /// Merges request defaults without retaining a second mutable configuration.
    fn config(&self, config: Option<&Value>) -> Result<Map<String, Value>, RathError> {
        validate_config(config)?;
        let mut merged = self
            .provider_config
            .as_ref()
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(object) = config.and_then(Value::as_object) {
            merged.extend(object.clone());
        }
        Ok(merged)
    }
}

#[async_trait]
impl VoiceCloneClient for ElevenLabsClient {
    async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError> {
        let form = clone_form(self, request)?;
        http::mapped(self.post("voices/add").multipart(form), Provider::ElevenLabs,
            "voice cloning; remote resource may be created", &[&self.api_key], |raw| {
                match raw["requires_verification"].as_bool() {
                    Some(false) => decode_voice(&raw),
                    Some(true) => Err(RathError::new(ErrorKind::Provider,
                        "created voice requires verification; remote resource retained, do not blindly retry").with_response(&raw)),
                    None => Err(RathError::invalid("missing requires_verification", &raw)),
                }
            }).await
    }
}

/// Encodes reference samples in memory, with explicit names overriding native defaults.
fn clone_form(client: &ElevenLabsClient, request: &VoiceCloneRequest) -> Result<Form, RathError> {
    validate_samples(&request.samples)?;
    let name = request
        .name
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("ElevenLabs cloning requires a name"))?;
    let mut config = client.config(request.provider_config.as_ref())?;
    config.remove("name");
    if config.contains_key("files") || config.contains_key("files[]") {
        return Err(invalid("reference files must be supplied through samples"));
    }
    let mut form = Form::new().text("name", name.to_owned());
    for (key, value) in config {
        if !value.is_null() {
            let text = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            form = form.text(key, text);
        }
    }
    for (index, sample) in request.samples.iter().enumerate() {
        let file = Part::bytes(sample.data.clone())
            .file_name(format!("sample-{index}"))
            .mime_str(&sample.mime_type)
            .map_err(|error| {
                http::transport(
                    Provider::ElevenLabs,
                    "clone audio",
                    error,
                    &[&client.api_key],
                )
            })?;
        form = form.part("files", file);
    }
    Ok(form)
}

#[async_trait]
impl VoiceDesignClient for ElevenLabsClient {
    async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError> {
        design::generate(self, request).await
    }

    async fn register_voice(
        &self,
        preview: &VoicePreview,
        name: &str,
        config: Option<&Value>,
    ) -> Result<Voice, RathError> {
        design::register(self, preview, name, config).await
    }
}

#[async_trait]
impl TtsClient for ElevenLabsClient {
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let voice = request
            .voice
            .as_ref()
            .ok_or_else(|| invalid("ElevenLabs requires a voice"))?;
        voice.validate(SCOPE)?;
        let VoiceData::Id(id) = &voice.data else {
            return Err(unsupported("selected voice representation"));
        };
        validate_id(id)?;
        let payload = speech_payload(self, request)?;
        let format = request.format.as_deref().unwrap_or("mp3_44100_128");
        if !matches!(format, "mp3_44100_128" | "mp3_22050_32") {
            return Err(unsupported("selected output format"));
        }
        let response = http::send(
            self.post(&format!("text-to-speech/{id}?output_format={format}"))
                .json(&payload),
            Provider::ElevenLabs,
            "speech synthesis",
            &[&self.api_key],
        )
        .await?;
        read_audio(self, response).await
    }
}

/// Validates audio before dropping response metadata and returns only caller-owned bytes.
async fn read_audio(
    client: &ElevenLabsClient,
    response: reqwest::Response,
) -> Result<TtsResponse, RathError> {
    let mime_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("audio/mpeg")
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let data = http::read_checked(
        response,
        Provider::ElevenLabs,
        "speech synthesis",
        &[&client.api_key],
        |bytes| {
            if bytes.is_empty() || !crate::audio::voice::valid_audio_mime(&mime_type) {
                return Err(RathError::new(
                    ErrorKind::InvalidResponse,
                    "invalid speech audio response",
                ));
            }
            Ok(())
        },
    )
    .await?;
    Ok(TtsResponse {
        mime_type,
        data,
        raw_metadata: None,
    })
}

/// Applies supported typed synthesis controls; unsupported controls never disappear silently.
fn speech_payload(
    client: &ElevenLabsClient,
    request: &TtsRequest,
) -> Result<Map<String, Value>, RathError> {
    let model = request.model.as_deref().unwrap_or(&client.model);
    validate_tts_model(model)?;
    if request.input.trim().is_empty() {
        return Err(invalid("speech input must not be blank"));
    }
    if request.instructions.is_some() || request.language.is_some() {
        return Err(unsupported(
            "explicit language or instructions; use native text/audio tags where supported",
        ));
    }
    let mut payload = client.config(request.provider_config.as_ref())?;
    payload.insert("text".into(), json!(request.input));
    payload.insert("model_id".into(), json!(model));
    Ok(payload)
}

fn validate_tts_model(model: &str) -> Result<(), RathError> {
    match model {
        "eleven_multilingual_v2" | "eleven_v3" | "eleven_turbo_v2_5" => Ok(()),
        _ => Err(unsupported("speech model")),
    }
}

/// Restricts path-bound IDs so they cannot alter the authenticated request destination.
fn validate_id(id: &str) -> Result<(), RathError> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        return Err(invalid("invalid ElevenLabs voice identifier"));
    }
    Ok(())
}

/// Returns an identity only after validating its wire shape; preserves evidence on failure.
fn decode_voice(raw: &Value) -> Result<Voice, RathError> {
    let id = raw["voice_id"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| RathError::invalid("missing voice_id", raw))?;
    validate_id(id).map_err(|_| RathError::invalid("invalid voice_id", raw))?;
    Ok(Voice::id(SCOPE, id))
}

fn unsupported(capability: &str) -> RathError {
    RathError::unsupported(Provider::ElevenLabs, capability)
}
