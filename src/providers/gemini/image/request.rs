//! Model-specific Gemini text-to-image size and aspect ratio validation.
use super::media;
use crate::images::{ImageOptions, ImageRequest};
use crate::{ModelUrl, Provider, RathError};
use serde_json::{Map, Value, json};

/// Restricts construction to documented native image models, without runtime model discovery.
pub(super) fn model(model: &str) -> Result<(), RathError> {
    match model.trim_start_matches("models/") {
        "gemini-2.5-flash-image"
        | "gemini-3.1-flash-image"
        | "gemini-3.1-flash-lite-image"
        | "gemini-3-pro-image"
        | "gemini-nano-banana-2.1" => Ok(()),
        _ => Err(RathError::unsupported(
            Provider::Gemini,
            "image generation for this model",
        )),
    }
}

/// Rejects native schema extensions that cannot be represented faithfully by this adapter.
pub(super) fn config(
    model: &str,
    defaults: Option<&Value>,
    request: Option<&Value>,
) -> Result<Map<String, Value>, RathError> {
    for value in [defaults, request].into_iter().flatten() {
        let config = media::config(Some(value), None, &["generationConfig", "safetySettings"])?;
        if let Some(generation) = config.get("generationConfig") {
            let fields = media::config(Some(generation), None, &["imageConfig"])?;
            if let Some(image) = fields.get("imageConfig") {
                image_config(model, image)?;
            }
        }
        if let Some(value) = config.get("safetySettings") {
            serde_json::from_value::<Vec<gemini_rust::SafetySetting>>(value.clone())
                .map_err(|_| media::invalid("invalid Gemini image safetySettings"))?;
        }
    }
    media::config(defaults, request, &["generationConfig", "safetySettings"])
}

/// Validates recognized native dimensions, including model-specific aspect ratio restrictions.
fn image_config(model: &str, image: &Value) -> Result<(), RathError> {
    let fields = media::config(Some(image), None, &["aspectRatio", "imageSize"])?;
    if let Some(size) = fields.get("imageSize") {
        validate_size(model, size.as_str())?;
    }
    if let Some(ratio) = fields.get("aspectRatio") {
        let ratio = ratio
            .as_str()
            .ok_or_else(|| media::invalid("Gemini aspectRatio must be a string"))?;
        let ordinary = [
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
        ];
        let extended = ["1:4", "4:1", "1:8", "8:1"];
        let model = model.trim_start_matches("models/");
        if !ordinary.contains(&ratio)
            && !(matches!(model, "gemini-3.1-flash-image" | "gemini-nano-banana-2.1")
                && extended.contains(&ratio))
        {
            return Err(media::invalid(
                "unsupported Gemini image aspectRatio for this model",
            ));
        }
    }
    Ok(())
}

/// Size is a Gemini resolution label, never guessed from arbitrary pixel dimensions.
fn validate_size(model: &str, size: Option<&str>) -> Result<(), RathError> {
    let valid = match model.trim_start_matches("models/") {
        "gemini-2.5-flash-image" => false,
        "gemini-3.1-flash-lite-image" => size == Some("1K"),
        "gemini-3.1-flash-image" => matches!(size, Some("512" | "1K" | "2K" | "4K")),
        "gemini-3-pro-image" | "gemini-nano-banana-2.1" => matches!(size, Some("1K" | "2K" | "4K")),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(media::invalid(
            "unsupported Gemini imageSize for this model",
        ))
    }
}

/// Supplies text-only input and overlays typed size on native request/default image configuration.
pub(super) fn payload(
    url: &ModelUrl,
    options: &ImageOptions,
    request: &ImageRequest,
) -> Result<Value, RathError> {
    if request.prompt.trim().is_empty() {
        return Err(media::invalid("Gemini image prompt must not be blank"));
    }
    let mut payload = Value::Object(config(
        &url.model,
        options.provider_config.as_ref(),
        request.provider_config.as_ref(),
    )?);
    let mut generation = payload
        .get("generationConfig")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(size) = &request.size {
        validate_size(&url.model, Some(size))?;
        let mut image = generation
            .get("imageConfig")
            .cloned()
            .unwrap_or_else(|| json!({}));
        image["imageSize"] = json!(size);
        generation["imageConfig"] = image;
    }
    generation["responseModalities"] = json!(["TEXT", "IMAGE"]);
    payload["generationConfig"] = generation;
    payload["contents"] = json!([{"role":"user", "parts":[{"text":request.prompt}]}]);
    Ok(payload)
}
