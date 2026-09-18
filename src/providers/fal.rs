use crate::core::error::{http, provider_failure};
mod audio;
mod queue;

pub(crate) use audio::{new_stt_client, new_tts_client};

use async_trait::async_trait;
use serde_json::{Map, Value};
use std::time::Duration;

use crate::core::RathError;
use crate::core::{ModelUrl, Provider};
use crate::images::{ImageClient, ImageData, ImageOptions, ImageRequest, ImageResponse};
use crate::video::{
    VideoClient, VideoData, VideoJob, VideoJobStatus, VideoOptions, VideoRequest, VideoResponse,
};

const DEFAULT_BASE_URL: &str = "https://fal.run";
const DEFAULT_QUEUE_BASE_URL: &str = "https://queue.fal.run";
const DEFAULT_API_KEY_ENV: &str = "FAL_KEY";

pub(crate) fn new_image_client(
    url: &ModelUrl,
    options: ImageOptions,
) -> Result<Box<dyn ImageClient>, RathError> {
    Ok(Box::new(FalClient::new(
        url,
        options.provider_config,
        "image",
    )?))
}

pub(crate) fn new_video_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<Box<dyn VideoClient>, RathError> {
    let mut client = FalClient::new(url, options.provider_config, "video")?;
    client.poll_interval = options.poll_interval;
    Ok(Box::new(client))
}

struct FalClient {
    http: reqwest::Client,
    api_key: String,
    endpoint: String,
    queue_base_url: String,
    model: String,
    provider_config: Option<Value>,
    poll_interval: Duration,
}

impl FalClient {
    fn new(
        url: &ModelUrl,
        provider_config: Option<Value>,
        capability: &str,
    ) -> Result<Self, RathError> {
        let api_key =
            match &url.api_key {
                Some(key) => key.clone(),
                None => std::env::var(DEFAULT_API_KEY_ENV).map_err(|error| {
                    RathError::new(crate::core::ErrorKind::Validation, format!(
                    "set {DEFAULT_API_KEY_ENV} or pass api_key_env for Fal {capability} calls"
                )).with_source(crate::core::error::credential_cause(&error))
                })?,
            };
        let base_url = url
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let endpoint = build_endpoint(&base_url, &url.model);
        let queue_base_url = queue_base_url(&base_url);
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            endpoint,
            queue_base_url,
            model: url.model.clone(),
            provider_config,
            poll_interval: Duration::from_secs(5),
        })
    }

    async fn post<T>(
        &self,
        endpoint: &str,
        payload: Value,
        operation: &'static str,
        decode: impl FnOnce(Value) -> Result<T, RathError>,
    ) -> Result<T, RathError> {
        http::mapped(
            self.http
                .post(endpoint)
                .header("Authorization", format!("Key {}", self.api_key))
                .json(&payload),
            Provider::Fal,
            operation,
            &[&self.api_key],
            decode,
        )
        .await
    }

    /// Validates GET responses before their original bytes and headers leave the HTTP boundary.
    async fn get<T>(
        &self,
        endpoint: &str,
        operation: &'static str,
        decode: impl FnOnce(Value) -> Result<T, RathError>,
    ) -> Result<T, RathError> {
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
                Ok(raw)
            })
            .await?;
        video_status_from_status(
            &self.queue_base_url,
            &self.model,
            job_id,
            raw,
            self,
            fail_on_job_error,
        )
        .await
    }
}

#[async_trait]
impl ImageClient for FalClient {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        let payload = image_payload(&self.model, &self.provider_config, request);
        self.post(
            &self.endpoint,
            Value::Object(payload),
            "image generation",
            |raw| {
                let images = extract_images(&raw);
                if images.is_empty() {
                    return Err(RathError::invalid(
                        "missing image(s), image_url or image data",
                        &raw,
                    ));
                }
                Ok(ImageResponse {
                    images,
                    raw_metadata: Some(raw),
                })
            },
        )
        .await
    }
}

#[async_trait]
impl VideoClient for FalClient {
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        let payload = video_payload(&self.provider_config, request);
        let endpoint = build_endpoint(&self.queue_base_url, &self.model);
        self.post(&endpoint, Value::Object(payload), "video submit", |raw| {
            video_job_from_submit(&self.model, raw)
        })
        .await
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

fn build_endpoint(base_url: &str, model: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        model.trim_start_matches('/')
    )
}

fn queue_base_url(base_url: &str) -> String {
    match base_url.trim_end_matches('/') {
        DEFAULT_BASE_URL => DEFAULT_QUEUE_BASE_URL.to_string(),
        DEFAULT_QUEUE_BASE_URL => DEFAULT_QUEUE_BASE_URL.to_string(),
        other => other.to_string(),
    }
}

fn build_status_endpoint(queue_base_url: &str, model: &str, request_id: &str) -> String {
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

fn image_payload(
    model: &str,
    options_config: &Option<Value>,
    request: &ImageRequest,
) -> Map<String, Value> {
    let mut payload = merged_config(options_config, &request.provider_config);
    if !request.prompt.is_empty() {
        payload.insert("prompt".to_string(), Value::String(request.prompt.clone()));
    }
    if let Some(size) = &request.size {
        payload.insert("image_size".to_string(), Value::String(size.clone()));
    }
    if let Some(model) = request.model.as_ref().filter(|m| !m.is_empty()) {
        payload.insert("model".to_string(), Value::String(model.clone()));
    } else {
        payload.insert("model".to_string(), Value::String(model.to_string()));
    }
    payload
}

fn video_payload(options_config: &Option<Value>, request: &VideoRequest) -> Map<String, Value> {
    let mut payload = merged_config(options_config, &request.provider_config);
    if !request.prompt.is_empty() {
        payload.insert("prompt".to_string(), Value::String(request.prompt.clone()));
    }
    if let Some(image_url) = &request.image_url {
        payload.insert("image_url".to_string(), Value::String(image_url.clone()));
    }
    payload
}

fn merged_config(
    options_config: &Option<Value>,
    request_config: &Option<Value>,
) -> Map<String, Value> {
    let mut payload = Map::new();
    merge_object(&mut payload, options_config);
    merge_object(&mut payload, request_config);
    payload
}

fn merge_object(payload: &mut Map<String, Value>, value: &Option<Value>) {
    if let Some(Value::Object(map)) = value {
        for (key, value) in map {
            payload.insert(key.clone(), value.clone());
        }
    }
}

fn extract_images(raw: &Value) -> Vec<ImageData> {
    let mut images = Vec::new();
    if let Some(items) = raw.get("images").and_then(Value::as_array) {
        images.extend(items.iter().filter_map(extract_image));
    }
    if let Some(image) = raw.get("image").and_then(extract_image) {
        images.push(image);
    }
    images
}

fn extract_image(value: &Value) -> Option<ImageData> {
    let url = value.get("url").and_then(Value::as_str);
    if let Some(url) = url {
        return Some(ImageData::Url {
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
        .unwrap_or("image/png");
    Some(ImageData::Base64 {
        mime_type: mime_type.to_string(),
        data: data.to_string(),
    })
}

fn extract_videos(raw: &Value) -> Vec<VideoData> {
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

fn video_job_from_submit(model: &str, raw: Value) -> Result<VideoJob, RathError> {
    let id = raw
        .get("request_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RathError::invalid("Fal queue submit response missing request_id", &raw))?
        .to_string();
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
    queue_base_url: &str,
    model: &str,
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
            let endpoint = raw
                .get("response_url")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| build_response_endpoint(queue_base_url, model, job_id));
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
        other => Ok(VideoJobStatus::Failed {
            message: format!("Fal video job returned unknown status '{other}'"),
            raw_metadata: Some(raw),
        }),
    }
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
fn completed_video(response_raw: Value, client: &FalClient) -> Result<VideoJobStatus, RathError> {
    let videos = extract_videos(&response_raw);
    if videos.is_empty() {
        return Err(
            RathError::invalid("missing video(s), video_url or video data", &response_raw)
                .with_context(Provider::Fal, "video result")
                .sanitized(&[&client.api_key]),
        );
    }
    Ok(VideoJobStatus::Succeeded {
        response: VideoResponse {
            videos,
            raw_metadata: Some(response_raw),
        },
    })
}

/// Recognizes failed-job envelopes without treating partial job output as a successful video.
fn video_failure(raw: &Value, client: &FalClient) -> Option<VideoJobStatus> {
    raw.get("error").filter(|error| !error.is_null())?;
    let error = provider_failure(Provider::Fal, "video result", raw, &[&client.api_key]);
    Some(VideoJobStatus::Failed {
        message: error.message().to_owned(),
        raw_metadata: Some(raw.clone()),
    })
}

#[cfg(test)]
fn video_status_from_pending(raw: Value) -> VideoJobStatus {
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

#[cfg(test)]
mod tests;
