//! Text-to-image Gemini generation through the existing model-bound image contract.
use super::media;
use crate::core::error::http;
use crate::images::{
    ImageBackend, ImageClient, ImageData, ImageOptions, ImageRequest, ImageResponse,
};
use crate::{ModelUrl, Provider, RathError};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::Value;
mod request;

struct GeminiImageBackend {
    http: reqwest::Client,
    url: ModelUrl,
    options: ImageOptions,
}

/// Prepares native image generation with the same resolved Gemini credential as chat and audio.
pub(crate) fn new_image_client(
    url: &ModelUrl,
    options: ImageOptions,
) -> Result<ImageClient, RathError> {
    request::model(&url.model)?;
    request::config(&url.model, options.provider_config.as_ref(), None)?;
    Ok(ImageClient::from_backend(GeminiImageBackend {
        http: media::transport()?,
        url: media::resolved_url(url)?,
        options,
    }))
}

impl ImageBackend for GeminiImageBackend {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        let key = self.url.api_key.as_deref().unwrap_or_default();
        let payload = request::payload(&self.url, &self.options, request).map_err(|e| {
            e.with_context(Provider::Gemini, "image generation")
                .sanitized(&[key])
        })?;
        http::mapped(
            self.http
                .post(media::endpoint(&self.url))
                .header("x-goog-api-key", key)
                .json(&payload),
            Provider::Gemini,
            "image generation",
            &[key],
            decode,
        )
        .await
    }
}

/// Preserves generated image order and native text metadata, excluding interim thought images.
fn decode(raw: Value) -> Result<ImageResponse, RathError> {
    let mut images = Vec::new();
    for part in media::parts(&raw)? {
        if media::thought(part, &raw)? {
            continue;
        }
        if let Some(blob) = part.get("inlineData") {
            if part.get("text").is_some() {
                return Err(RathError::invalid("mixed Gemini image part", &raw));
            }
            images.push(image(blob, &raw)?);
        } else if part["text"].as_str().is_none() {
            return Err(RathError::invalid(
                "unexpected Gemini image output part",
                &raw,
            ));
        }
    }
    if images.is_empty() {
        return Err(RathError::invalid("missing Gemini generated image", &raw));
    }
    Ok(ImageResponse::new(images).with_raw_metadata(raw))
}

/// Checks base64, MIME and encoded-file signatures without adding a full image decoder dependency.
fn image(blob: &Value, raw: &Value) -> Result<ImageData, RathError> {
    let mime = blob["mimeType"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing Gemini image inlineData.mimeType", raw))?;
    let data = blob["data"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing Gemini image inlineData.data", raw))?;
    let bytes = STANDARD
        .decode(data)
        .map_err(|_| RathError::invalid("invalid Gemini image base64 data", raw))?;
    let valid = match mime {
        "image/png" => {
            bytes.len() >= 33
                && bytes.starts_with(b"\x89PNG\r\n\x1a\n")
                && bytes.get(12..16) == Some(b"IHDR")
        }
        "image/jpeg" => {
            bytes.len() > 4 && bytes.starts_with(&[255, 216, 255]) && bytes.ends_with(&[255, 217])
        }
        "image/webp" => {
            bytes.len() > 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP")
        }
        _ => false,
    };
    if !valid {
        return Err(RathError::invalid(
            "empty, unsupported or mismatched Gemini image bytes/MIME",
            raw,
        ));
    }
    Ok(ImageData::Base64 {
        mime_type: mime.into(),
        data: data.into(),
    })
}

#[cfg(test)]
#[path = "tests/image.rs"]
mod tests;
