//! Fal video jobs; transport configuration is immutable and lifecycle state is caller-owned.
use super::{FalClient, build_endpoint, video_input, video_media, video_webhook};
use crate::core::error::{http, provider_failure};
use crate::core::{ErrorKind, ModelUrl, Provider, RathError};
use crate::video::{
    VideoClient, VideoData, VideoEvent, VideoJob, VideoJobStatus, VideoOptions, VideoRequest,
    VideoResponse,
};
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

/// Constructs a video-only transport without changing other Fal clients.
pub(crate) fn new_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<Box<dyn VideoClient>, RathError> {
    video_input::validate_config(&options.provider_config)?;
    let mut client = FalClient::new(url, options.provider_config, "video")?;
    if video_media::url(&client.queue_base_url)?.query().is_some() {
        return Err(video_input::invalid(
            "video base_url must not contain a query",
        ));
    }
    client.http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| {
            http::transport(
                Provider::Fal,
                "video configuration",
                error,
                &[&client.api_key],
            )
        })?;
    client.poll_interval = options.poll_interval;
    Ok(Box::new(client))
}

impl FalClient {
    /// Validates GET responses before their original bytes and headers leave the HTTP boundary.
    async fn get<T>(
        &self,
        endpoint: &str,
        operation: &'static str,
        decode: impl FnOnce(Value) -> Result<T, RathError>,
    ) -> Result<T, RathError> {
        validate_queue_url(self, endpoint)?;
        let response = http::send(
            self.http
                .get(endpoint)
                .header("Authorization", format!("Key {}", self.api_key)),
            Provider::Fal,
            operation,
            &[&self.api_key],
        )
        .await?;
        http::mapped_response(response, Provider::Fal, operation, &[&self.api_key], decode).await
    }

    /// Preserves get_video's Failed status, but enriches wait_video failures before dropping headers.
    async fn fetch_video(
        &self,
        job_id: &str,
        fail_on_job_error: bool,
    ) -> Result<VideoJobStatus, RathError> {
        validate_job_id(job_id)?;
        let endpoint = build_status_endpoint(&self.queue_base_url, &self.model, job_id);
        let raw = self
            .get(&endpoint, "video status", |raw| {
                if fail_on_job_error && let Some(message) = failed_status_message(&raw) {
                    return Err(crate::core::error::provider_failure_message(
                        Provider::Fal,
                        "video status",
                        &raw,
                        &[&self.api_key],
                        &message,
                    ));
                }
                validate_queue_urls(self, &raw)?;
                Ok(raw)
            })
            .await?;
        video_status_from_status(job_id, raw, self, fail_on_job_error).await
    }
}

#[async_trait]
impl VideoClient for FalClient {
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        let payload = video_input::payload(&self.model, &self.provider_config, request)?;
        let endpoint = submission_url(self, request)?;
        self.post(
            endpoint.as_str(),
            Value::Object(payload),
            "video submit",
            |raw| {
                validate_queue_urls(self, &raw)?;
                video_job_from_submit(&self.model, raw)
            },
        )
        .await
    }

    async fn parse_webhook(
        &self,
        headers: &::http::HeaderMap,
        body: &[u8],
    ) -> Result<VideoEvent, RathError> {
        video_webhook::parse(self, headers, body).await
    }

    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        self.fetch_video(job_id, false).await
    }

    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        loop {
            match self.fetch_video(job_id, true).await? {
                VideoJobStatus::Queued { .. } | VideoJobStatus::Running { .. } => {
                    tokio::time::sleep(self.poll_interval).await;
                }
                VideoJobStatus::Succeeded { response } => return Ok(response),
                VideoJobStatus::Failed {
                    message,
                    raw_metadata,
                } => {
                    let response =
                        raw_metadata.unwrap_or_else(|| serde_json::json!({"error": &message}));
                    return Err(crate::core::error::provider_failure_message(
                        Provider::Fal,
                        "video wait",
                        &response,
                        &[&self.api_key],
                        &message,
                    ));
                }
            }
        }
    }
}

pub(super) fn build_status_endpoint(queue_base_url: &str, model: &str, request_id: &str) -> String {
    format!(
        "{}/{}/requests/{}/status",
        queue_base_url.trim_end_matches('/'),
        model.trim_start_matches('/'),
        request_id
    )
}

fn build_response_endpoint(queue_base_url: &str, model: &str, request_id: &str) -> String {
    format!(
        "{}/{}/requests/{}/response",
        queue_base_url.trim_end_matches('/'),
        model.trim_start_matches('/'),
        request_id
    )
}

/// Extracts existing supported result shapes without following media URLs.
pub(super) fn extract_videos(raw: &Value) -> Vec<VideoData> {
    let mut videos = Vec::new();
    if let Some(items) = raw.get("videos").and_then(Value::as_array) {
        videos.extend(items.iter().filter_map(extract_video));
    }
    if let Some(video) = raw.get("video").and_then(extract_video) {
        videos.push(video);
    }
    if let Some(url) = raw.get("video_url").and_then(Value::as_str) {
        videos.push(VideoData::Url {
            url: url.to_string(),
        });
    }
    videos
}

/// Returns an owned receipt after checking the provider's stable request identifier.
pub(super) fn video_job_from_submit(model: &str, raw: Value) -> Result<VideoJob, RathError> {
    let id = raw
        .get("request_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RathError::invalid("Fal queue submit response missing request_id", &raw))?
        .to_string();
    validate_job_id(&id).map_err(|_| RathError::invalid("invalid request_id", &raw))?;
    Ok(VideoJob {
        id,
        provider: Provider::Fal,
        provider_model: Some(model.to_string()),
        status_url: raw
            .get("status_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        response_url: raw
            .get("response_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        cancel_url: raw
            .get("cancel_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        raw_metadata: Some(raw),
    })
}

/// Maps queue lifecycle status while preserving provider failures and required result fields.
async fn video_status_from_status(
    job_id: &str,
    raw: Value,
    client: &FalClient,
    fail_on_job_error: bool,
) -> Result<VideoJobStatus, RathError> {
    let status = raw["status"].as_str().unwrap_or("UNKNOWN");
    match status {
        "IN_QUEUE" => Ok(VideoJobStatus::Queued {
            queue_position: raw.get("queue_position").and_then(Value::as_u64),
            raw_metadata: Some(raw),
        }),
        "IN_PROGRESS" => Ok(VideoJobStatus::Running {
            raw_metadata: Some(raw),
        }),
        "COMPLETED" => {
            if let Some(failure) = video_failure(&raw, client) {
                return Ok(failure);
            }
            fetch_completed(client, job_id, raw, fail_on_job_error).await
        }
        other => Ok(VideoJobStatus::Failed {
            message: format!("Fal video job returned unknown status '{other}'"),
            raw_metadata: Some(raw),
        }),
    }
}

/// Retrieves the terminal result once, keeping provider errors distinct from invalid output.
async fn fetch_completed(
    client: &FalClient,
    job_id: &str,
    raw: Value,
    fail_on_job_error: bool,
) -> Result<VideoJobStatus, RathError> {
    let endpoint = raw
        .get("response_url")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| build_response_endpoint(&client.queue_base_url, &client.model, job_id));
    client
        .get(&endpoint, "video result", |response_raw| {
            if let Some(failure) = video_failure(&response_raw, client) {
                if fail_on_job_error {
                    return Err(provider_failure(
                        Provider::Fal,
                        "video result",
                        &response_raw,
                        &[&client.api_key],
                    ));
                }
                return Ok(failure);
            }
            completed_video(response_raw, client)
        })
        .await
}

/// Identifies the existing Failed states without changing queue polling or retry decisions.
fn failed_status_message(raw: &Value) -> Option<String> {
    match raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN")
    {
        "IN_QUEUE" | "IN_PROGRESS" => None,
        "COMPLETED" => raw
            .get("error")
            .filter(|value| !value.is_null())
            .map(|_| "video job failed; response details available".into()),
        other => Some(format!("Fal video job returned unknown status '{other}'")),
    }
}

/// Validates required video output fields and preserves the full result on failure.
pub(super) fn completed_video(
    response_raw: Value,
    client: &FalClient,
) -> Result<VideoJobStatus, RathError> {
    let videos = extract_videos(&response_raw);
    if videos.is_empty() {
        return Err(
            RathError::invalid("missing video(s), video_url or video data", &response_raw)
                .with_context(Provider::Fal, "video result")
                .sanitized(&[&client.api_key]),
        );
    }
    for video in &videos {
        video_media::video(video).map_err(|_| {
            RathError::invalid("invalid video media", &response_raw)
                .with_context(Provider::Fal, "video result")
                .sanitized(&[&client.api_key])
        })?;
    }
    Ok(VideoJobStatus::Succeeded {
        response: VideoResponse {
            videos,
            raw_metadata: Some(response_raw),
        },
    })
}

/// Recognizes failed-job envelopes without treating partial job output as a successful video.
pub(super) fn video_failure(raw: &Value, client: &FalClient) -> Option<VideoJobStatus> {
    raw.get("error").filter(|error| !error.is_null())?;
    let error = provider_failure(Provider::Fal, "video result", raw, &[&client.api_key]);
    Some(VideoJobStatus::Failed {
        message: error.message().to_owned(),
        raw_metadata: error
            .response_body()
            .and_then(|body| serde_json::from_slice(body.bytes()).ok()),
    })
}

#[cfg(test)]
/// Maps pending fixtures using the existing public state representation.
pub(super) fn video_status_from_pending(raw: Value) -> VideoJobStatus {
    let status = raw["status"].as_str().unwrap_or("UNKNOWN");
    match status {
        "IN_QUEUE" => VideoJobStatus::Queued {
            queue_position: raw.get("queue_position").and_then(Value::as_u64),
            raw_metadata: Some(raw),
        },
        "IN_PROGRESS" => VideoJobStatus::Running {
            raw_metadata: Some(raw),
        },
        other => VideoJobStatus::Failed {
            message: format!("Fal video job returned unknown status '{other}'"),
            raw_metadata: Some(raw),
        },
    }
}

/// Recognizes a remote video or encoded media in native Fal result objects.
fn extract_video(value: &Value) -> Option<VideoData> {
    let url = value.get("url").and_then(Value::as_str);
    if let Some(url) = url {
        return Some(VideoData::Url {
            url: url.to_string(),
        });
    }
    let data = value
        .get("base64")
        .or_else(|| value.get("data"))
        .and_then(Value::as_str)?;
    let mime_type = value
        .get("content_type")
        .or_else(|| value.get("mime_type"))
        .and_then(Value::as_str)
        .unwrap_or("video/mp4");
    Some(VideoData::Base64 {
        mime_type: mime_type.to_string(),
        data: data.to_string(),
    })
}

/// Adds callbacks as transport query parameters, never model input.
fn submission_url(client: &FalClient, request: &VideoRequest) -> Result<reqwest::Url, RathError> {
    let mut url = video_media::url(&build_endpoint(&client.queue_base_url, &client.model))?;
    if let Some(callback) = &request.webhook_url {
        let callback = video_media::url(callback)?;
        if callback.scheme() != "https" {
            return Err(video_input::invalid("webhook_url must use HTTPS"));
        }
        url.query_pairs_mut()
            .append_pair("fal_webhook", callback.as_str());
    }
    Ok(url)
}

/// Rejects path/query injection in IDs before constructing authenticated URLs.
fn validate_job_id(id: &str) -> Result<(), RathError> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        return Err(video_input::invalid("invalid video job ID"));
    }
    Ok(())
}

/// Checks every supplied queue URL before returning a receipt or polling status.
fn validate_queue_urls(client: &FalClient, raw: &Value) -> Result<(), RathError> {
    for field in ["status_url", "response_url", "cancel_url"] {
        if let Some(value) = raw.get(field) {
            let endpoint = value
                .as_str()
                .ok_or_else(|| RathError::invalid(format!("invalid {field}"), raw))?;
            validate_queue_url(client, endpoint).map_err(|error| error.with_response(raw))?;
        }
    }
    Ok(())
}

/// Never attaches credentials to a provider-returned URL outside the queue origin.
fn validate_queue_url(client: &FalClient, endpoint: &str) -> Result<(), RathError> {
    let url = video_media::url(endpoint)?;
    let base = video_media::url(&client.queue_base_url)?;
    if url.origin() != base.origin() {
        return Err(RathError::new(
            ErrorKind::InvalidResponse,
            "video queue URL origin mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/video.rs"]
mod tests;
