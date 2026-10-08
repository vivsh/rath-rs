//! Gemini-specific single-speaker configuration and generation payloads.
use super::media;
use crate::audio::{
    tts::{TtsOptions, TtsRequest},
    voice::VoiceData,
};
use crate::{ModelUrl, Provider, RathError};
use serde_json::{Map, Value, json};

const VOICES: &[&str] = &[
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
];
const LANGUAGES: &[&str] = &[
    "de-DE", "en-AU", "en-GB", "en-IN", "en-US", "es-US", "fr-FR", "hi-IN", "pt-BR", "ar-XA",
    "es-ES", "fr-CA", "id-ID", "it-IT", "ja-JP", "tr-TR", "vi-VN", "bn-IN", "gu-IN", "kn-IN",
    "ml-IN", "mr-IN", "ta-IN", "te-IN", "nl-NL", "ko-KR", "cmn-CN", "pl-PL", "ru-RU", "th-TH",
];

/// Binds only documented TTS models; the boolean selects Gemini 3.8's structured style contract.
pub(super) fn model(model: &str) -> Result<bool, RathError> {
    match model.trim_start_matches("models/") {
        "gemini-3.8-flash-tts" | "gemini-3.8-flash-lite-tts" => Ok(true),
        "gemini-2.5-flash-preview-tts"
        | "gemini-2.5-pro-preview-tts"
        | "gemini-3.1-flash-tts-preview" => Ok(false),
        _ => Err(RathError::unsupported(
            Provider::Gemini,
            "speech generation for this model",
        )),
    }
}

/// Validates supported native controls separately before the established shallow precedence merge.
pub(super) fn config(
    defaults: Option<&Value>,
    request: Option<&Value>,
) -> Result<Map<String, Value>, RathError> {
    for value in [defaults, request].into_iter().flatten() {
        let config = media::config(Some(value), None, &["generationConfig", "safetySettings"])?;
        if let Some(value) = config.get("generationConfig") {
            generation(value)?;
        }
        if let Some(value) = config.get("safetySettings") {
            serde_json::from_value::<Vec<gemini_rust::SafetySetting>>(value.clone())
                .map_err(|_| media::invalid("invalid Gemini audio safetySettings"))?;
        }
    }
    media::config(defaults, request, &["generationConfig", "safetySettings"])
}

/// Restricts native generation fields to supported sampling and single-speaker controls.
fn generation(value: &Value) -> Result<(), RathError> {
    let fields = media::config(
        Some(value),
        None,
        &["temperature", "topP", "topK", "seed", "speechConfig"],
    )?;
    for (key, value) in &fields {
        let valid = match key.as_str() {
            "temperature" => value.as_f64().is_some_and(|v| (0.0..=2.0).contains(&v)),
            "topP" => value.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v)),
            "topK" => value
                .as_u64()
                .is_some_and(|v| v > 0 && v <= i32::MAX as u64),
            "seed" => value.as_i64().is_some_and(|v| i32::try_from(v).is_ok()),
            "speechConfig" => {
                speech(value)?;
                true
            }
            _ => false,
        };
        if !valid {
            return Err(media::invalid(format!(
                "invalid Gemini generationConfig.{key}"
            )));
        }
    }
    Ok(())
}

/// Supports one native preset and language; cloning, multi-speaker and arbitrary IDs remain explicit failures.
fn speech(value: &Value) -> Result<(), RathError> {
    let fields = media::config(Some(value), None, &["voiceConfig", "languageCode"])?;
    if let Some(voice) = fields.get("voiceConfig") {
        let fields = media::config(Some(voice), None, &["prebuiltVoiceConfig"])?;
        let preset = fields
            .get("prebuiltVoiceConfig")
            .ok_or_else(|| media::invalid("Gemini voiceConfig requires prebuiltVoiceConfig"))?;
        let preset = media::config(Some(preset), None, &["voiceName"])?;
        validate_voice(preset.get("voiceName").and_then(Value::as_str))?;
    }
    if let Some(language) = fields.get("languageCode") {
        validate_language(language.as_str())?;
    }
    Ok(())
}

fn validate_voice(voice: Option<&str>) -> Result<(), RathError> {
    if voice.is_some_and(|voice| VOICES.contains(&voice)) {
        Ok(())
    } else {
        Err(media::invalid("unsupported Gemini preset voice"))
    }
}

fn validate_language(language: Option<&str>) -> Result<(), RathError> {
    if language.is_some_and(|language| LANGUAGES.contains(&language)) {
        Ok(())
    } else {
        Err(media::invalid(
            "unsupported Gemini speech languageCode; use a supported BCP-47 tag",
        ))
    }
}

/// Overlays typed controls without modifying native defaults or the caller's request.
fn effective_speech(
    generation: &mut Value,
    request: &TtsRequest,
    current: bool,
) -> Result<(), RathError> {
    let mut speech = generation
        .get("speechConfig")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let preset = match &request.voice {
        Some(voice) => {
            voice.validate("gemini")?;
            let VoiceData::Id(id) = &voice.data else {
                return Err(media::invalid("Gemini speech requires a preset voice ID"));
            };
            id.as_str()
        }
        None => speech
            .pointer("/voiceConfig/prebuiltVoiceConfig/voiceName")
            .and_then(Value::as_str)
            .unwrap_or("Kore"),
    };
    validate_voice(Some(preset))?;
    let voice = if current {
        json!({"voice":preset})
    } else {
        json!({"prebuiltVoiceConfig":{"voiceName":preset}})
    };
    speech["voiceConfig"] = voice;
    if let Some(language) = &request.language {
        validate_language(Some(language))?;
        speech["languageCode"] = json!(language);
    }
    generation["speechConfig"] = speech;
    Ok(())
}

/// Distinguishes structured Gemini 3.8 style metadata from legacy prompt-based directions.
fn text(request: &TtsRequest, current: bool) -> Result<Value, RathError> {
    if request.input.trim().is_empty()
        || request
            .instructions
            .as_ref()
            .is_some_and(|s| s.trim().is_empty())
    {
        return Err(media::invalid(
            "Gemini speech input and supplied instructions must not be blank",
        ));
    }
    let mut part = json!({"text":request.input});
    if let Some(style) = &request.instructions {
        if current {
            part["speechMetadata"] = json!({"style":style});
        } else {
            part["text"] = json!(format!(
                "{style}\n\nRead aloud the following text verbatim:\n{}",
                request.input
            ));
        }
    }
    Ok(part)
}

/// Produces the documented generateContent request, requesting raw PCM on current models.
pub(super) fn payload(
    url: &ModelUrl,
    options: &TtsOptions,
    request: &TtsRequest,
) -> Result<Value, RathError> {
    if !matches!(request.format.as_deref(), None | Some("wav" | "pcm")) {
        return Err(media::invalid(
            "Gemini speech supports only wav or pcm output",
        ));
    }
    let current = model(&url.model)?;
    let mut payload = Value::Object(config(
        options.provider_config.as_ref(),
        request.provider_config.as_ref(),
    )?);
    let mut generation = payload
        .get("generationConfig")
        .cloned()
        .unwrap_or_else(|| json!({}));
    effective_speech(&mut generation, request, current)?;
    generation["responseModalities"] = json!(["AUDIO"]);
    if current {
        generation["responseFormat"] =
            json!({"audio":{"mimeType":"AUDIO_L16", "delivery":"INLINE", "sampleRate":24000}});
    }
    payload["generationConfig"] = generation;
    payload["contents"] = json!([{"role":"user", "parts":[text(request, current)?]}]);
    Ok(payload)
}
