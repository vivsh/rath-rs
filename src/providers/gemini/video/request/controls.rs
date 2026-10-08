//! Documented Veo generation controls and model/input constraints.
use super::{VideoInput, media};
use crate::RathError;
use serde_json::Value;

/// Rejects unknown native fields instead of silently ignoring routing or generation controls.
pub(super) fn validate(model: &str, value: &Value, input: &VideoInput) -> Result<(), RathError> {
    let fields = value
        .as_object()
        .ok_or_else(|| media::invalid("Veo parameters must be an object"))?;
    for (key, value) in fields {
        let valid = match key.as_str() {
            "aspectRatio" => matches!(value.as_str(), Some("16:9" | "9:16")),
            "durationSeconds" => matches!(value.as_u64(), Some(4 | 6 | 8)),
            "sampleCount" => value.as_u64() == Some(1),
            "resolution" => matches!(value.as_str(), Some("720p" | "1080p" | "4k")),
            "personGeneration" => matches!(value.as_str(), Some("allow_all" | "allow_adult")),
            "negativePrompt" => value.is_string(),
            "enhancePrompt" => value.is_boolean(),
            _ => return Err(media::invalid(format!("unsupported Veo parameter {key}"))),
        };
        if !valid {
            return Err(media::invalid(format!("invalid Veo parameter {key}")));
        }
    }
    constraints(model, value, input)
}

/// Applies documented resolution/duration and image/person constraints without inferring region.
fn constraints(model: &str, parameters: &Value, input: &VideoInput) -> Result<(), RathError> {
    let resolution = parameters["resolution"].as_str().unwrap_or("720p");
    if resolution == "4k" && model == "veo-3.1-lite-generate-preview" {
        return Err(media::invalid("Veo 3.1 Lite does not support 4k"));
    }
    let duration = parameters["durationSeconds"].as_u64().unwrap_or(8);
    if resolution != "720p" && duration != 8 {
        return Err(media::invalid(
            "high-resolution Veo generation requires 8 seconds",
        ));
    }
    if matches!(input, VideoInput::ImageToVideo { .. })
        && parameters["personGeneration"].as_str() == Some("allow_all")
    {
        return Err(media::invalid(
            "Veo image-to-video requires personGeneration allow_adult when specified",
        ));
    }
    Ok(())
}
