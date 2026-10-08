//! Validated Veo inputs and native parameter controls; no input URLs are fetched.
use super::super::media;
use crate::video::{VideoInput, VideoOptions, VideoRequest};
use crate::{ModelUrl, Provider, RathError, images::ImageData};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};
mod controls;

/// Restricts typed generation to the documented Veo 3.1 variants.
pub(super) fn model(value: &str) -> Result<&str, RathError> {
    let value = value.strip_prefix("models/").unwrap_or(value);
    match value {
        "veo-3.1-generate-preview"
        | "veo-3.1-fast-generate-preview"
        | "veo-3.1-lite-generate-preview" => Ok(value),
        _ => Err(RathError::unsupported(Provider::Gemini, "video model")),
    }
}

/// Uses the bound API root and model; requests cannot override routing.
pub(super) fn submit_url(url: &ModelUrl) -> String {
    format!(
        "{}/models/{}:predictLongRunning",
        api_root(url),
        url.model.trim_start_matches("models/")
    )
}

pub(super) fn api_root(url: &ModelUrl) -> &str {
    url.base_url
        .as_deref()
        .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
        .trim_end_matches('/')
}

/// Merges immutable defaults with generation controls while excluding verification configuration.
pub(super) fn config(
    defaults: Option<&Value>,
    request: Option<&Value>,
) -> Result<Map<String, Value>, RathError> {
    let mut config = media::config(
        defaults,
        None,
        &["parameters", "webhook_signing_secret_env"],
    )?;
    config.remove("webhook_signing_secret_env");
    config.extend(media::config(None, request, &["parameters"])?);
    if config
        .get("parameters")
        .is_some_and(|value| !value.is_object())
    {
        return Err(media::invalid("Veo parameters must be an object"));
    }
    Ok(config)
}

/// Builds exactly one typed instance, rejecting every input/control absent from this adapter.
pub(super) fn payload(
    url: &ModelUrl,
    options: &VideoOptions,
    request: &VideoRequest,
) -> Result<Value, RathError> {
    if request.webhook_url.is_some() {
        return Err(RathError::unsupported(
            Provider::Gemini,
            "dynamic video webhooks; register a project webhook",
        ));
    }
    if request.prompt.trim().is_empty() {
        return Err(media::invalid("Veo requires a nonblank prompt"));
    }
    let mut instance = Map::from_iter([("prompt".into(), json!(request.prompt))]);
    match &request.input {
        VideoInput::TextToVideo => {}
        VideoInput::ImageToVideo {
            start_image,
            end_image,
        } => {
            instance.insert("image".into(), image(start_image)?);
            if let Some(image_data) = end_image {
                instance.insert("lastFrame".into(), image(image_data)?);
            }
        }
        _ => return Err(RathError::unsupported(Provider::Gemini, "video input")),
    }
    let mut config = config(
        options.provider_config.as_ref(),
        request.provider_config.as_ref(),
    )?;
    let parameters = config.entry("parameters").or_insert_with(|| json!({}));
    controls::validate(model(&url.model)?, parameters, &request.input)?;
    config.insert("instances".into(), json!([instance]));
    Ok(Value::Object(config))
}

/// Encodes validated inline image bytes in the official SDK's Veo REST shape.
fn image(value: &ImageData) -> Result<Value, RathError> {
    let ImageData::Base64 { mime_type, data } = value else {
        return Err(RathError::unsupported(
            Provider::Gemini,
            "Veo requires inline base64 images",
        ));
    };
    if !matches!(mime_type.as_str(), "image/png" | "image/jpeg") {
        return Err(media::invalid(
            "Veo input images must be image/png or image/jpeg",
        ));
    }
    if data.is_empty() || data.len() > 28_000_000 {
        return Err(media::invalid(
            "Veo input images must be nonempty and at most 28 MB encoded",
        ));
    }
    STANDARD
        .decode(data)
        .map_err(|_| media::invalid("invalid Veo image base64"))?;
    Ok(json!({"bytesBase64Encoded":data, "mimeType":mime_type}))
}
