//! Stateless Veo receipt/status decoding with operation routing confined to the bound model.
use super::{super::media, request};
use crate::video::{VideoData, VideoJob, VideoJobStatus, VideoResponse};
use crate::{ModelUrl, Provider, RathError};
use serde_json::Value;

/// Validates a provider operation name before constructing an authenticated polling URL.
pub(super) fn operation_url(url: &ModelUrl, name: &str) -> Result<String, RathError> {
    let prefix = format!("models/{}/operations/", request::model(&url.model)?);
    let id = name
        .strip_prefix(&prefix)
        .ok_or_else(|| media::invalid("Veo operation does not match the bound model"))?;
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        return Err(media::invalid("invalid Veo operation name"));
    }
    Ok(format!("{}/{name}", request::api_root(url)))
}

/// Preserves the operation receipt for caller-owned tracking without retaining it in the backend.
pub(super) fn job(url: &ModelUrl, raw: Value) -> Result<VideoJob, RathError> {
    let id = raw["name"]
        .as_str()
        .ok_or_else(|| RathError::invalid("missing Veo operation name", &raw))?;
    let location = operation_url(url, id)?;
    Ok(VideoJob::new(id.into(), Provider::Gemini)
        .with_provider_model(request::model(&url.model)?.into())
        .with_status_url(location)
        .with_raw_metadata(raw))
}

/// Distinguishes remote execution failures from malformed or incomplete operation responses.
pub(super) fn status(
    url: &ModelUrl,
    expected: &str,
    raw: Value,
) -> Result<VideoJobStatus, RathError> {
    if raw["name"].as_str() != Some(expected) {
        return Err(RathError::invalid(
            "Veo returned a different operation name",
            &raw,
        ));
    }
    let done = match raw.get("done") {
        None | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        _ => return Err(RathError::invalid("invalid Veo operation done flag", &raw)),
    };
    let error = raw.get("error").filter(|value| !value.is_null());
    let response = raw.get("response").filter(|value| !value.is_null());
    if !done {
        if error.is_some() || response.is_some() {
            return Err(RathError::invalid(
                "incomplete Veo operation contains a terminal result",
                &raw,
            ));
        }
        return Ok(VideoJobStatus::Running {
            raw_metadata: Some(raw),
        });
    }
    if error.is_some() && response.is_some() {
        return Err(RathError::invalid(
            "Veo operation contains both result and error",
            &raw,
        ));
    }
    if let Some(error) = error {
        if !error.is_object()
            || error["message"]
                .as_str()
                .is_none_or(|s| s.trim().is_empty())
        {
            return Err(RathError::invalid("invalid Veo operation error", &raw));
        }
        return failed(url, &raw, error["message"].as_str().unwrap_or_default());
    }
    completed(url, raw)
}

/// Maps explicit safety filtering to a remote failure and requires nonempty completed videos.
fn completed(url: &ModelUrl, raw: Value) -> Result<VideoJobStatus, RathError> {
    let result = raw
        .pointer("/response/generateVideoResponse")
        .ok_or_else(|| RathError::invalid("missing Veo generation result", &raw))?;
    if result["raiMediaFilteredCount"].as_u64().unwrap_or(0) > 0 {
        return failed(
            url,
            &raw,
            "Veo filtered the generated video; response details available",
        );
    }
    let samples = result["generatedSamples"]
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| RathError::invalid("missing generated Veo videos", &raw))?;
    let videos = samples
        .iter()
        .map(|sample| {
            let uri = sample
                .pointer("/video/uri")
                .and_then(Value::as_str)
                .ok_or_else(|| RathError::invalid("missing generated Veo video URI", &raw))?;
            video_uri(uri)?;
            Ok(VideoData::Url { url: uri.into() })
        })
        .collect::<Result<Vec<_>, RathError>>()?;
    Ok(VideoJobStatus::Succeeded {
        response: VideoResponse::new(videos).with_raw_metadata(raw),
    })
}

/// Keeps provider failure evidence through the existing redaction boundary.
fn failed(url: &ModelUrl, raw: &Value, message: &str) -> Result<VideoJobStatus, RathError> {
    let error = crate::core::error::provider_failure_message(
        Provider::Gemini,
        "video status",
        raw,
        &[url.api_key.as_deref().unwrap_or_default()],
        message,
    );
    let metadata = error
        .response_body()
        .and_then(|body| serde_json::from_slice(body.bytes()).ok());
    Ok(VideoJobStatus::Failed {
        message: error.message().into(),
        raw_metadata: metadata,
    })
}

/// Validates locations without downloading them or treating them as authenticated API endpoints.
pub(super) fn video_uri(value: &str) -> Result<(), RathError> {
    let url = reqwest::Url::parse(value).map_err(|_| media::invalid("invalid Veo video URI"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(media::invalid(
            "Veo video URI must use HTTPS without userinfo or a fragment",
        ));
    }
    Ok(())
}
