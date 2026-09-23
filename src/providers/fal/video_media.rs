//! Operation-local media validation and data-URI encoding.
use super::video_input::invalid;
use crate::core::RathError;
use crate::images::ImageData;
use crate::video::VideoData;
use base64::{Engine, engine::general_purpose::STANDARD};

/// Validates a remote input without downloading it or revealing signed query values.
pub(super) fn url(value: &str) -> Result<reqwest::Url, RathError> {
    let url = reqwest::Url::parse(value).map_err(|_| invalid("invalid media/transport URL"))?;
    if value.trim() != value
        || !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "URL must be HTTP(S), without credentials or fragments",
        ));
    }
    Ok(url)
}

pub(super) fn image(value: &ImageData) -> Result<String, RathError> {
    match value {
        ImageData::Url { url: value } => Ok(url(value)?.to_string()),
        ImageData::Base64 { mime_type, data } => data_uri(mime_type, data, "image/"),
    }
}

pub(super) fn video(value: &VideoData) -> Result<String, RathError> {
    match value {
        VideoData::Url { url: value } => Ok(url(value)?.to_string()),
        VideoData::Base64 { mime_type, data } => data_uri(mime_type, data, "video/"),
    }
}

/// Ensures the complete base64 and MIME category are valid before any paid submission.
fn data_uri(mime: &str, data: &str, prefix: &str) -> Result<String, RathError> {
    let subtype = mime.strip_prefix(prefix).unwrap_or("");
    if subtype.is_empty()
        || !subtype
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$&^_.+-".contains(&b))
    {
        return Err(invalid("incompatible or malformed media MIME type"));
    }
    let bytes = STANDARD
        .decode(data)
        .map_err(|_| invalid("invalid media base64"))?;
    if bytes.is_empty() {
        return Err(invalid("media must not be empty"));
    }
    Ok(format!("data:{mime};base64,{data}"))
}
