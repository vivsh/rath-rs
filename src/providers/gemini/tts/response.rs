//! Decodes Gemini PCM or complete WAV without mislabelling headerless audio.
use super::media;
use crate::RathError;
use crate::audio::tts::TtsResponse;
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::Value;

const PCM_MIME: &str = "audio/L16;codec=pcm;rate=24000";

/// Decodes all audio parts from one completed candidate, checking MIME agreement and PCM alignment.
pub(super) fn decode(raw: Value, pcm: bool) -> Result<TtsResponse, RathError> {
    let mut data = Vec::new();
    let mut format = None;
    let mut count = 0;
    for part in media::parts(&raw)? {
        if media::thought(part, &raw)? {
            continue;
        }
        if part.get("text").is_some() || part.get("functionCall").is_some() {
            return Err(RathError::invalid(
                "unexpected non-audio Gemini speech output",
                &raw,
            ));
        }
        let blob = &part["inlineData"];
        let wav = mime(blob["mimeType"].as_str(), &raw)?;
        if format.is_some_and(|f| f != wav) || (wav && count != 0) {
            return Err(RathError::invalid(
                "inconsistent or multiple WAV Gemini speech parts",
                &raw,
            ));
        }
        let encoded = blob["data"]
            .as_str()
            .ok_or_else(|| RathError::invalid("missing Gemini speech inlineData.data", &raw))?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| RathError::invalid("invalid Gemini speech base64 data", &raw))?;
        if bytes.is_empty() || (!wav && bytes.len() % 2 != 0) {
            return Err(RathError::invalid(
                "empty or unaligned Gemini speech audio",
                &raw,
            ));
        }
        data.extend_from_slice(&bytes);
        format = Some(wav);
        count += 1;
    }
    let (mime_type, data) = output(data, format, pcm, &raw)?;
    Ok(TtsResponse::new(mime_type.into(), data).with_raw_metadata(raw))
}

/// Accepts only the documented 24 kHz mono signed little-endian PCM or complete WAV envelope.
fn mime(value: Option<&str>, raw: &Value) -> Result<bool, RathError> {
    let value = value
        .ok_or_else(|| RathError::invalid("missing Gemini speech inlineData.mimeType", raw))?;
    let mut tokens = value.split(';');
    let base = tokens.next().unwrap_or_default().trim();
    if base.eq_ignore_ascii_case("audio/wav") && tokens.next().is_none() {
        return Ok(true);
    }
    if !base.eq_ignore_ascii_case("audio/l16") {
        return Err(RathError::invalid(
            "unsupported Gemini speech audio MIME type",
            raw,
        ));
    }
    let mut seen = Vec::new();
    for parameter in tokens {
        let (key, value) = parameter
            .trim()
            .split_once('=')
            .ok_or_else(|| RathError::invalid("invalid Gemini PCM MIME parameter", raw))?;
        let key = key.to_ascii_lowercase();
        let valid = match key.as_str() {
            "codec" => value.eq_ignore_ascii_case("pcm"),
            "rate" => value == "24000",
            "channels" => value == "1",
            _ => false,
        };
        if !valid || seen.contains(&key) {
            return Err(RathError::invalid(
                "unsupported Gemini PCM encoding, rate or channels",
                raw,
            ));
        }
        seen.push(key);
    }
    Ok(false)
}

/// Returns raw PCM only on explicit request; container bytes already returned by Gemini remain intact.
fn output(
    data: Vec<u8>,
    wav: Option<bool>,
    pcm: bool,
    raw: &Value,
) -> Result<(&'static str, Vec<u8>), RathError> {
    match wav {
        Some(true) if !pcm => {
            validate_wav(&data, raw)?;
            Ok(("audio/wav", data))
        }
        Some(false) if pcm => Ok((PCM_MIME, data)),
        Some(false) => Ok(("audio/wav", wrap_wav(data, raw)?)),
        Some(true) => Err(RathError::invalid(
            "Gemini returned WAV when raw PCM was requested",
            raw,
        )),
        None => Err(RathError::invalid("missing Gemini speech audio", raw)),
    }
}

/// Prepends one valid RIFF/WAVE PCM header; sample bytes and little-endian order are unchanged.
fn wrap_wav(data: Vec<u8>, raw: &Value) -> Result<Vec<u8>, RathError> {
    let len = u32::try_from(data.len())
        .ok()
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| RathError::invalid("Gemini PCM is too large for RIFF/WAVE", raw))?;
    let mut wav = Vec::with_capacity(44 + data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(len + 36).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&24000u32.to_le_bytes());
    wav.extend_from_slice(&48000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&len.to_le_bytes());
    wav.extend_from_slice(&data);
    Ok(wav)
}

/// Checks complete RIFF chunk bounds, nonempty aligned data and the documented PCM format.
fn validate_wav(data: &[u8], raw: &Value) -> Result<(), RathError> {
    let invalid = || RathError::invalid("invalid Gemini WAV container or PCM format", raw);
    if data.len() < 44
        || data.get(..4) != Some(b"RIFF")
        || data.get(8..12) != Some(b"WAVE")
        || u32_at(data, 4).map(|n| n as u64 + 8) != Some(data.len() as u64)
    {
        return Err(invalid());
    }
    let mut offset = 12usize;
    let mut format = false;
    let mut samples = false;
    while offset < data.len() {
        let size = u32_at(data, offset + 4).ok_or_else(invalid)? as usize;
        let start = offset.checked_add(8).ok_or_else(invalid)?;
        let end = start
            .checked_add(size)
            .filter(|n| *n <= data.len())
            .ok_or_else(invalid)?;
        match &data[offset..offset + 4] {
            b"fmt " if !format => {
                format = valid_format(&data[start..end]);
                if !format {
                    return Err(invalid());
                }
            }
            b"data" if !samples => {
                samples = size > 0 && size.is_multiple_of(2);
                if !samples {
                    return Err(invalid());
                }
            }
            b"fmt " | b"data" => return Err(invalid()),
            _ => {}
        }
        offset = end
            .checked_add(size % 2)
            .filter(|n| *n <= data.len())
            .ok_or_else(invalid)?;
    }
    if format && samples {
        Ok(())
    } else {
        Err(invalid())
    }
}

fn u32_at(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn valid_format(data: &[u8]) -> bool {
    data.len() >= 16
        && data.get(..4) == Some(&[1, 0, 1, 0])
        && u32_at(data, 4) == Some(24000)
        && u32_at(data, 8) == Some(48000)
        && data.get(12..16) == Some(&[2, 0, 16, 0])
}
