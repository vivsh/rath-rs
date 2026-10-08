//! Inline, text-only transcription using Gemini's generateContent audio input contract.
use super::media;
use crate::audio::stt::{SttBackend, SttClient, SttOptions, SttRequest, SttResponse};
use crate::core::error::http;
use crate::{ModelUrl, Provider, RathError};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

struct GeminiSttBackend {
    http: reqwest::Client,
    url: ModelUrl,
    options: SttOptions,
}

/// Prepares an independent transcription client; construction performs no provider I/O.
pub(crate) fn new_stt_client(url: &ModelUrl, options: SttOptions) -> Result<SttClient, RathError> {
    media::config(options.provider_config.as_ref(), None, &[])?;
    Ok(SttClient::from_backend(GeminiSttBackend {
        http: media::transport()?,
        url: media::resolved_url(url)?,
        options,
    }))
}

impl SttBackend for GeminiSttBackend {
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        let key = self.url.api_key.as_deref().unwrap_or_default();
        let payload = payload(&self.options, request).map_err(|e| {
            e.with_context(Provider::Gemini, "speech-to-text")
                .sanitized(&[key])
        })?;
        http::mapped(
            self.http
                .post(media::endpoint(&self.url))
                .header("x-goog-api-key", key)
                .json(&payload),
            Provider::Gemini,
            "speech-to-text",
            &[key],
            decode,
        )
        .await
    }
}

/// Uses inline audio only and caps the encoded request below the documented 20 MB boundary.
fn payload(options: &SttOptions, request: &SttRequest) -> Result<Value, RathError> {
    media::config(
        options.provider_config.as_ref(),
        request.provider_config.as_ref(),
        &[],
    )?;
    if request.data.is_empty() {
        return Err(media::invalid(
            "Gemini transcription requires nonempty audio",
        ));
    }
    let mime = input_mime(&request.mime_type)?;
    if request.data.len() > 14_000_000 {
        return Err(media::invalid(
            "Gemini inline transcription supports at most 14,000,000 audio bytes; Files API uploads are not supported",
        ));
    }
    Ok(json!({
        "contents":[{"role":"user", "parts":[
            {"text":"Transcribe the speech in this audio verbatim in its original language. Return only the transcript, without commentary, translation, timestamps, speaker labels or Markdown."},
            {"inlineData":{"mimeType":mime, "data":STANDARD.encode(&request.data)}}
        ]}],
        "generationConfig":{"responseModalities":["TEXT"]}
    }))
}

/// Canonicalizes encoded-file MIME aliases and browser codec parameters without guessing bytes.
fn input_mime(mime: &str) -> Result<String, RathError> {
    if mime.bytes().any(|b| b.is_ascii_control()) {
        return Err(media::invalid(
            "invalid Gemini audio MIME control character",
        ));
    }
    let mut tokens = mime.split(';');
    let base = tokens
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let mut codecs = false;
    for parameter in tokens {
        let (name, value) = parameter
            .trim()
            .split_once('=')
            .ok_or_else(|| media::invalid("invalid Gemini audio MIME parameter"))?;
        if codecs || !name.eq_ignore_ascii_case("codecs") || !valid_codecs(value) {
            return Err(media::invalid(
                "Gemini encoded audio only supports a codecs MIME parameter",
            ));
        }
        codecs = true;
    }
    let canonical = match base.as_str() {
        "audio/wav" | "audio/x-wav" => "audio/wav",
        "audio/mp3" | "audio/mpeg" => "audio/mpeg",
        "audio/aiff" | "audio/x-aiff" => "audio/aiff",
        "audio/m4a" | "audio/x-m4a" | "audio/mp4" => "audio/m4a",
        "audio/aac" => "audio/aac",
        "audio/ogg" => "audio/ogg",
        "audio/flac" => "audio/flac",
        "audio/webm" => "audio/webm",
        "audio/opus" => "audio/opus",
        _ => return Err(media::invalid("unsupported Gemini encoded audio MIME type")),
    };
    Ok(canonical.into())
}

/// Allows one well-formed codec list, including the quoted lists emitted by browser recorders.
fn valid_codecs(value: &str) -> bool {
    let value = value.trim();
    let value = if let Some(quoted) = value.strip_prefix('"') {
        quoted.strip_suffix('"').unwrap_or_default()
    } else {
        value
    };
    value.split(',').all(|codec| {
        let codec = codec.trim();
        !codec.is_empty()
            && codec
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    })
}

/// Concatenates only completed non-thought text parts; empty or mixed outputs are failures.
fn decode(raw: Value) -> Result<SttResponse, RathError> {
    let mut text = String::new();
    for part in media::parts(&raw)? {
        if media::thought(part, &raw)? {
            continue;
        }
        let value = part["text"]
            .as_str()
            .ok_or_else(|| RathError::invalid("missing Gemini transcription part.text", &raw))?;
        if part.get("inlineData").is_some() || part.get("functionCall").is_some() {
            return Err(RathError::invalid(
                "unexpected non-text Gemini transcription output",
                &raw,
            ));
        }
        text.push_str(value);
    }
    if text.trim().is_empty() {
        return Err(RathError::invalid(
            "empty Gemini transcription output",
            &raw,
        ));
    }
    Ok(SttResponse::new(text).with_raw_metadata(raw))
}

#[cfg(test)]
#[path = "tests/stt.rs"]
mod tests;
