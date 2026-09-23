//! Qwen speaker-embedding extraction using the existing Fal queue and credential boundaries.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;

use super::{FalClient, merged_config, queue};
use crate::audio::{
    tts::TtsRequest,
    voice::{Voice, VoiceData, VoiceSample, invalid, validate_config, validate_samples},
    voice_clone::{VoiceCloneClient, VoiceCloneOptions, VoiceCloneRequest},
    voice_design::{
        VoiceDesignClient, VoiceDesignOptions, VoiceDesignRequest, VoiceDesignResponse,
        VoicePreview,
    },
};
use crate::core::{ErrorKind, Provider, RathError, error::http};
use async_trait::async_trait;
use serde_json::{Map, Value};

pub(super) const DESIGN: &str = "fal-ai/qwen-3-tts/voice-design/1.7b";
pub(super) const CLONE: &str = "fal-ai/qwen-3-tts/clone-voice/1.7b";
pub(super) const SPEECH: &str = "fal-ai/qwen-3-tts/text-to-speech/1.7b";
const SCOPE: &str = "fal/qwen3-tts-1.7b";
const FORMAT: &str = "qwen3-tts-1.7b/safetensors";

/// Constructs only the verified Qwen design endpoint.
pub(crate) fn new_design(
    url: &crate::core::ModelUrl,
    options: VoiceDesignOptions,
) -> Result<Box<dyn VoiceDesignClient>, RathError> {
    if url.model != DESIGN {
        return Err(RathError::unsupported(
            Provider::Fal,
            "voice design endpoint",
        ));
    }
    Ok(Box::new(super::audio::audio_client(
        url,
        options.provider_config,
    )?))
}

/// Constructs only the verified Qwen embedding-extraction endpoint.
pub(crate) fn new_clone(
    url: &crate::core::ModelUrl,
    options: VoiceCloneOptions,
) -> Result<Box<dyn VoiceCloneClient>, RathError> {
    if url.model != CLONE {
        return Err(RathError::unsupported(
            Provider::Fal,
            "voice clone endpoint",
        ));
    }
    Ok(Box::new(super::audio::audio_client(
        url,
        options.provider_config,
    )?))
}

#[async_trait]
impl VoiceDesignClient for FalClient {
    async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError> {
        validate_config(request.provider_config.as_ref())?;
        if request.description.trim().is_empty() || request.text.trim().is_empty() {
            return Err(invalid(
                "voice description and sample text must not be blank",
            ));
        }
        let mut payload = merged_config(&self.provider_config, &request.provider_config);
        payload.insert("prompt".into(), json!(request.description));
        payload.insert("text".into(), json!(request.text));
        payload.entry("max_new_tokens").or_insert(json!(2048));
        if let Some(language) = &request.language {
            payload.insert("language".into(), json!(qwen_language(language)?));
        }
        tokio::time::timeout(queue::AUDIO_TIMEOUT, async {
            let raw = queue::run(self, DESIGN, payload.into(), |raw| {
                if !raw["audio"]["url"].as_str().is_some_and(|s| !s.is_empty()) {
                    return Err(RathError::invalid("missing audio.url", &raw));
                }
                Ok(raw)
            })
            .await?;
            let (mime_type, data) = super::audio::download(self, &raw).await?;
            Ok(VoiceDesignResponse {
                previews: vec![VoicePreview {
                    sample: VoiceSample {
                        mime_type,
                        data,
                        transcript: Some(request.text.clone()),
                    },
                    scope: SCOPE.into(),
                    registration_token: None,
                }],
                raw_metadata: Some(raw),
            })
        })
        .await
        .map_err(|error| {
            RathError::from_error(ErrorKind::Timeout, &error).with_context(
                Provider::Fal,
                "voice design deadline; remote job may still run",
            )
        })?
    }
}

#[async_trait]
impl VoiceCloneClient for FalClient {
    async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError> {
        validate_samples(&request.samples)?;
        validate_config(request.provider_config.as_ref())?;
        if request.samples.len() != 1 || request.name.is_some() {
            return Err(invalid(
                "Qwen cloning requires one sample and does not support a name",
            ));
        }
        let sample = &request.samples[0];
        let transcript = sample
            .transcript
            .as_deref()
            .ok_or_else(|| invalid("Qwen cloning requires a transcript"))?;
        let data = clone(
            self,
            &sample.data,
            &sample.mime_type,
            transcript,
            &request.provider_config,
        )
        .await?;
        Ok(Voice {
            scope: SCOPE.into(),
            data: VoiceData::Embedding {
                format: FORMAT.into(),
                data,
                reference_text: Some(transcript.into()),
            },
        })
    }
}

/// Applies typed controls after native settings, preventing native voice settings from shadowing them.
pub(super) fn condition(
    model: &str,
    request: &TtsRequest,
    payload: &mut Map<String, Value>,
) -> Result<(), RathError> {
    if model == SPEECH {
        payload.entry("max_new_tokens").or_insert(json!(4096));
    }
    if let Some(voice) = &request.voice {
        apply_voice(model, voice, payload)?;
    }
    if let Some(language) = &request.language {
        if model != SPEECH {
            return Err(RathError::unsupported(
                Provider::Fal,
                "typed language for selected model",
            ));
        }
        payload.insert("language".into(), json!(qwen_language(language)?));
    }
    if let Some(instructions) = &request.instructions {
        if model != SPEECH || payload.contains_key("speaker_voice_embedding_file_url") {
            return Err(RathError::unsupported(
                Provider::Fal,
                "delivery instructions for selected voice",
            ));
        }
        if instructions.trim().is_empty() {
            return Err(invalid("instructions must not be blank"));
        }
        payload.insert("prompt".into(), json!(instructions));
    }
    Ok(())
}

/// Replaces all competing native identity fields with the explicit typed voice.
fn apply_voice(
    model: &str,
    voice: &Voice,
    payload: &mut Map<String, Value>,
) -> Result<(), RathError> {
    let scope = match model {
        SPEECH => SCOPE,
        "fal-ai/kokoro/american-english" => "fal/kokoro/american-english",
        _ => "fal/elevenlabs",
    };
    voice.validate(scope)?;
    for key in [
        "voice",
        "speaker_voice_embedding_file_url",
        "reference_text",
    ] {
        payload.remove(key);
    }
    match &voice.data {
        VoiceData::Id(id) => {
            payload.insert("voice".into(), json!(id));
        }
        VoiceData::Embedding {
            format,
            data,
            reference_text,
        } if model == SPEECH && format == FORMAT => {
            payload.insert(
                "speaker_voice_embedding_file_url".into(),
                json!(format!(
                    "data:application/octet-stream;base64,{}",
                    STANDARD.encode(data)
                )),
            );
            if let Some(text) = reference_text {
                payload.insert("reference_text".into(), json!(text));
            }
        }
        _ => {
            return Err(RathError::unsupported(
                Provider::Fal,
                "selected voice representation",
            ));
        }
    }
    Ok(())
}

/// Maps only supported base language tags; regional accent is not inferred from a language.
fn qwen_language(tag: &str) -> Result<&'static str, RathError> {
    match tag {
        "en" => Ok("English"),
        "zh" => Ok("Chinese"),
        "es" => Ok("Spanish"),
        "fr" => Ok("French"),
        "de" => Ok("German"),
        "it" => Ok("Italian"),
        "ja" => Ok("Japanese"),
        "ko" => Ok("Korean"),
        "pt" => Ok("Portuguese"),
        "ru" => Ok("Russian"),
        _ => Err(invalid("unsupported Qwen language tag")),
    }
}

/// Extracts and downloads one opaque embedding without retaining remote job state.
pub(super) async fn clone(
    client: &FalClient,
    data: &[u8],
    mime_type: &str,
    reference_text: &str,
    config: &Option<Value>,
) -> Result<Vec<u8>, RathError> {
    validate(client, data, mime_type, reference_text)?;
    tokio::time::timeout(queue::AUDIO_TIMEOUT, async {
        let mut payload = merged_config(&client.provider_config, config);
        payload.insert(
            "audio_url".into(),
            json!(format!("data:{mime_type};base64,{}", STANDARD.encode(data))),
        );
        payload.insert("reference_text".into(), json!(reference_text));
        let url = queue::run(client, CLONE, payload.into(), |raw| {
            let url = raw["speaker_embedding"]["url"]
                .as_str()
                .ok_or_else(|| RathError::invalid("missing speaker_embedding.url", &raw))?;
            queue::parse_url(url, "speaker embedding URL")
        })
        .await?;
        download(client, url).await
    })
    .await
    .map_err(|error| {
        RathError::from_error(ErrorKind::Timeout, &error).with_context(
            Provider::Fal,
            "voice cloning deadline; remote job may still run",
        )
    })?
}

/// Rejects unsupported endpoints and malformed inputs before submitting a paid job.
fn validate(
    client: &FalClient,
    data: &[u8],
    mime_type: &str,
    reference_text: &str,
) -> Result<(), RathError> {
    if client.model != CLONE {
        return Err(RathError::unsupported(
            Provider::Fal,
            "voice cloning for selected endpoint",
        ));
    }
    if data.is_empty()
        || reference_text.trim().is_empty()
        || !mime_type.strip_prefix("audio/").is_some_and(|subtype| {
            !subtype.is_empty()
                && subtype
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"!#$&^_.+-".contains(&b))
        })
    {
        return Err(RathError::new(
            ErrorKind::Validation,
            "voice cloning requires audio and a transcript",
        ));
    }
    Ok(())
}

/// Downloads opaque model data without credentials, rejecting empty or non-binary responses.
async fn download(client: &FalClient, url: reqwest::Url) -> Result<Vec<u8>, RathError> {
    let response = http::send(
        client.http.get(url),
        Provider::Fal,
        "embedding download",
        &[&client.api_key],
    )
    .await?;
    let mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .split(';')
        .next()
        .unwrap_or_default()
        .to_owned();
    http::read_checked(
        response,
        Provider::Fal,
        "embedding download",
        &[&client.api_key],
        |bytes| {
            if bytes.is_empty()
                || !matches!(
                    mime.as_str(),
                    "application/octet-stream" | "application/x-safetensors"
                )
            {
                return Err(queue::failed("speaker embedding bytes"));
            }
            Ok(())
        },
    )
    .await
}
