//! Explicit endpoint mappings; native mode deliberately bypasses model-input translation.
use super::{merged_config, video_media};
use crate::core::{ErrorKind, Provider, RathError};
use crate::video::{VideoInput, VideoRequest};
use serde_json::{Map, Value};

pub(super) const KLING_T2V: &str = "fal-ai/kling-video/v3/pro/text-to-video";
pub(super) const KLING_I2V: &str = "fal-ai/kling-video/v3/pro/image-to-video";
pub(super) const KLING_MOTION: &str = "fal-ai/kling-video/v3/pro/motion-control";
pub(super) const WAN_T2V: &str = "fal-ai/wan/v2.2-a14b/text-to-video";

pub(super) fn invalid(message: &str) -> RathError {
    RathError::new(ErrorKind::Validation, message).with_context(Provider::Fal, "video input")
}

pub(super) fn validate_config(config: &Option<Value>) -> Result<(), RathError> {
    if config.as_ref().is_some_and(|value| !value.is_object()) {
        return Err(invalid("provider_config must be an object"));
    }
    Ok(())
}

/// Merges operation-local settings, with explicit native or typed inputs taking precedence.
pub(super) fn payload(
    model: &str,
    defaults: &Option<Value>,
    request: &VideoRequest,
) -> Result<Map<String, Value>, RathError> {
    validate_config(defaults)?;
    validate_config(&request.provider_config)?;
    let mut payload = merged_config(defaults, &request.provider_config);
    if let VideoInput::Native { payload: native } = &request.input {
        if !request.prompt.is_empty() {
            return Err(invalid("Native input requires an empty top-level prompt"));
        }
        let native = native
            .as_object()
            .ok_or_else(|| invalid("Native payload must be an object"))?;
        payload.extend(native.clone());
        return Ok(payload);
    }
    validate_operation(model, request, &payload)?;
    for key in [
        "prompt",
        "image_url",
        "start_image_url",
        "end_image_url",
        "video_url",
    ] {
        payload.remove(key);
    }
    if !request.prompt.trim().is_empty() {
        payload.insert("prompt".into(), Value::String(request.prompt.clone()));
    }
    apply_media(&mut payload, &request.input)?;
    Ok(payload)
}

/// Rejects unsupported operations and semantic conflicts before any network request.
fn validate_operation(
    model: &str,
    request: &VideoRequest,
    payload: &Map<String, Value>,
) -> Result<(), RathError> {
    let compatible = match request.input {
        VideoInput::TextToVideo => matches!(model, KLING_T2V | WAN_T2V),
        VideoInput::ImageToVideo { .. } => model == KLING_I2V,
        VideoInput::MotionTransfer { .. } => model == KLING_MOTION,
        VideoInput::Native { .. } => true,
    };
    if !compatible {
        return Err(RathError::new(ErrorKind::UnsupportedCapability,
            "typed video operation is unsupported for this endpoint; use Native for explicit native input")
            .with_context(Provider::Fal, "video input"));
    }
    if matches!(
        request.input,
        VideoInput::TextToVideo | VideoInput::ImageToVideo { .. }
    ) {
        if request.prompt.trim().is_empty() {
            return Err(invalid("typed T2V/I2V requires a nonblank prompt"));
        }
        if payload.contains_key("multi_prompt") {
            return Err(invalid("multi_prompt requires Native input"));
        }
    }
    if matches!(request.input, VideoInput::MotionTransfer { .. })
        && !matches!(
            payload.get("character_orientation").and_then(Value::as_str),
            Some("image" | "video")
        )
    {
        return Err(invalid(
            "motion transfer requires character_orientation: image or video",
        ));
    }
    Ok(())
}

/// Translates input roles into endpoint-specific field names without retaining media.
fn apply_media(payload: &mut Map<String, Value>, input: &VideoInput) -> Result<(), RathError> {
    match input {
        VideoInput::ImageToVideo {
            start_image,
            end_image,
        } => {
            payload.insert(
                "start_image_url".into(),
                Value::String(video_media::image(start_image)?),
            );
            if let Some(image) = end_image {
                payload.insert(
                    "end_image_url".into(),
                    Value::String(video_media::image(image)?),
                );
            }
        }
        VideoInput::MotionTransfer {
            character_image,
            driving_video,
        } => {
            payload.insert(
                "image_url".into(),
                Value::String(video_media::image(character_image)?),
            );
            payload.insert(
                "video_url".into(),
                Value::String(video_media::video(driving_video)?),
            );
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/video_input.rs"]
mod tests;
