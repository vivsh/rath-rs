//! Explicit preview generation and registration; previews are never auto-registered.
use super::*;
use crate::audio::voice::VoiceSample;
use base64::{Engine as _, engine::general_purpose::STANDARD};

/// Generates all native previews and preserves the provider's exact transcript.
pub(super) async fn generate(
    client: &ElevenLabsClient,
    request: &VoiceDesignRequest,
) -> Result<VoiceDesignResponse, RathError> {
    if !(20..=1000).contains(&request.description.chars().count())
        || !(100..=1000).contains(&request.text.chars().count())
        || request.description.trim().is_empty()
        || request.text.trim().is_empty()
    {
        return Err(invalid(
            "ElevenLabs requires a 20-1000 character description and 100-1000 character sample text",
        ));
    }
    if request.language.is_some() {
        return Err(unsupported("explicit design language"));
    }
    let mut payload = client.config(request.provider_config.as_ref())?;
    payload.insert("voice_description".into(), json!(request.description));
    payload.insert("text".into(), json!(request.text));
    payload.insert("model_id".into(), json!(client.model));
    payload.insert("auto_generate_text".into(), json!(false));
    payload.insert("stream_previews".into(), json!(false));
    http::mapped(
        client.post("text-to-voice/design").json(&payload),
        Provider::ElevenLabs,
        "voice design",
        &[&client.api_key],
        decode_previews,
    )
    .await
}

/// Registers only on an explicit call, requiring the provider's description through native config.
pub(super) async fn register(
    client: &ElevenLabsClient,
    preview: &VoicePreview,
    name: &str,
    config: Option<&Value>,
) -> Result<Voice, RathError> {
    if preview.scope != PREVIEW_SCOPE || name.trim().is_empty() {
        return Err(invalid("incompatible preview or empty registration name"));
    }
    let token = preview
        .registration_token
        .as_deref()
        .ok_or_else(|| invalid("preview has no native registration token"))?;
    validate_id(token)?;
    let mut payload = client.config(config)?;
    let description = payload
        .get("voice_description")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("registration requires provider_config.voice_description"))?;
    if !(20..=1000).contains(&description.chars().count()) || description.trim().is_empty() {
        return Err(invalid(
            "registration description must contain 20-1000 characters",
        ));
    }
    payload.insert("voice_name".into(), json!(name));
    payload.insert("generated_voice_id".into(), json!(token));
    http::mapped(
        client.post("text-to-voice").json(&payload),
        Provider::ElevenLabs,
        "voice registration; remote resource may be created",
        &[&client.api_key],
        decode_registered,
    )
    .await
}

/// Does not expose verification-gated remote resources as ready voices.
fn decode_registered(raw: Value) -> Result<Voice, RathError> {
    if raw["voice_verification"]["requires_verification"].as_bool() == Some(true)
        && raw["voice_verification"]["is_verified"].as_bool() != Some(true)
    {
        return Err(RathError::new(
            ErrorKind::Provider,
            "registered voice requires verification; remote resource retained",
        )
        .with_response(&raw));
    }
    decode_voice(&raw)
}

/// Decodes all candidates before publishing any, retaining response evidence for malformed audio.
fn decode_previews(raw: Value) -> Result<VoiceDesignResponse, RathError> {
    let transcript = raw["text"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| RathError::invalid("missing design transcript", &raw))?;
    let candidates = raw["previews"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| RathError::invalid("missing voice previews", &raw))?;
    let previews = candidates
        .iter()
        .map(|candidate| decode_preview(candidate, transcript, &raw))
        .collect::<Result<_, _>>()?;
    Ok(VoiceDesignResponse {
        previews,
        raw_metadata: Some(raw),
    })
}

/// Validates a preview token and encoded audio without confusing it with a registered voice.
fn decode_preview(
    candidate: &Value,
    transcript: &str,
    raw: &Value,
) -> Result<VoicePreview, RathError> {
    let token = candidate["generated_voice_id"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing generated_voice_id", raw))?;
    validate_id(token).map_err(|_| RathError::invalid("invalid generated_voice_id", raw))?;
    let mime = candidate["media_type"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing preview media_type", raw))?;
    let encoded = candidate["audio_base_64"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing preview audio_base_64", raw))?;
    let data = STANDARD.decode(encoded).map_err(|error| {
        RathError::invalid("invalid preview base64", raw)
            .with_source(RathError::from_error(ErrorKind::Deserialize, &error))
    })?;
    let sample = VoiceSample {
        mime_type: mime.into(),
        data,
        transcript: Some(transcript.into()),
    };
    validate_samples(std::slice::from_ref(&sample))
        .map_err(|_| RathError::invalid("invalid preview audio", raw))?;
    Ok(VoicePreview {
        sample,
        scope: PREVIEW_SCOPE.into(),
        registration_token: Some(token.into()),
    })
}
