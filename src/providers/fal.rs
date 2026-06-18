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
        let api_key = match &url.api_key {
            Some(key) => key.clone(),
            None => std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| {
                RathError::Validation(format!(
                    "set {DEFAULT_API_KEY_ENV} or pass api_key_env for Fal {capability} calls"
                ))
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

    async fn post(&self, endpoint: &str, payload: Value) -> Result<Value, RathError> {
        let response = self
            .http
            .post(endpoint)
            .header("Authorization", format!("Key {}", self.api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;
        if !status.is_success() {
            return Err(RathError::Provider(format!(
                "Fal request failed with status {status}: {body}"
            )));
        }
        serde_json::from_str(&body).map_err(|source| RathError::Deserialize { source, raw: body })
    }

    async fn get(&self, endpoint: &str) -> Result<Value, RathError> {
        let response = self
            .http
            .get(endpoint)
            .header("Authorization", format!("Key {}", self.api_key))
            .send()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| RathError::Provider(e.to_string()))?;
        if !status.is_success() {
            return Err(RathError::Provider(format!(
                "Fal request failed with status {status}: {body}"
            )));
        }
        serde_json::from_str(&body).map_err(|source| RathError::Deserialize { source, raw: body })
    }
}

#[async_trait]
impl ImageClient for FalClient {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        let payload = image_payload(&self.model, &self.provider_config, request);
        let raw = self.post(&self.endpoint, Value::Object(payload)).await?;
        Ok(ImageResponse {
            images: extract_images(&raw),
            raw_metadata: Some(raw),
        })
    }
}

#[async_trait]
impl VideoClient for FalClient {
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        let payload = video_payload(&self.provider_config, request);
        let endpoint = build_endpoint(&self.queue_base_url, &self.model);
        let raw = self.post(&endpoint, Value::Object(payload)).await?;
        video_job_from_submit(&self.model, raw)
    }

    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        let endpoint = build_status_endpoint(&self.queue_base_url, &self.model, job_id);
        let raw = self.get(&endpoint).await?;
        video_status_from_status(&self.queue_base_url, &self.model, job_id, raw, self).await
    }

    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        loop {
            match self.get_video(job_id).await? {
                VideoJobStatus::Queued { .. } | VideoJobStatus::Running { .. } => {
                    tokio::time::sleep(self.poll_interval).await;
                }
                VideoJobStatus::Succeeded { response } => return Ok(response),
                VideoJobStatus::Failed { message, .. } => return Err(RathError::Provider(message)),
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
        .ok_or_else(|| RathError::Provider("Fal queue submit response missing request_id".into()))?
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

async fn video_status_from_status(
    queue_base_url: &str,
    model: &str,
    job_id: &str,
    raw: Value,
    client: &FalClient,
) -> Result<VideoJobStatus, RathError> {
    let status = raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN");
    match status {
        "IN_QUEUE" => Ok(VideoJobStatus::Queued {
            queue_position: raw.get("queue_position").and_then(Value::as_u64),
            raw_metadata: Some(raw),
        }),
        "IN_PROGRESS" => Ok(VideoJobStatus::Running {
            raw_metadata: Some(raw),
        }),
        "COMPLETED" => {
            if let Some(error) = raw.get("error").and_then(Value::as_str) {
                return Ok(VideoJobStatus::Failed {
                    message: error.to_string(),
                    raw_metadata: Some(raw),
                });
            }
            let endpoint = raw
                .get("response_url")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| build_response_endpoint(queue_base_url, model, job_id));
            let response_raw = client.get(&endpoint).await?;
            Ok(VideoJobStatus::Succeeded {
                response: VideoResponse {
                    videos: extract_videos(&response_raw),
                    raw_metadata: Some(response_raw),
                },
            })
        }
        other => Ok(VideoJobStatus::Failed {
            message: format!("Fal video job returned unknown status '{other}'"),
            raw_metadata: Some(raw),
        }),
    }
}

#[cfg(test)]
fn video_status_from_pending(raw: Value) -> VideoJobStatus {
    let status = raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN");
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
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn builds_fal_endpoint() {
        assert_eq!(
            build_endpoint("https://fal.run", "fal-ai/flux/schnell"),
            "https://fal.run/fal-ai/flux/schnell"
        );
        assert_eq!(
            build_endpoint("https://queue.fal.run", "fal-ai/wan/text-to-video"),
            "https://queue.fal.run/fal-ai/wan/text-to-video"
        );
        assert_eq!(
            build_status_endpoint(
                "https://queue.fal.run",
                "fal-ai/wan/text-to-video",
                "abc123"
            ),
            "https://queue.fal.run/fal-ai/wan/text-to-video/requests/abc123/status"
        );
        assert_eq!(queue_base_url("https://fal.run"), "https://queue.fal.run");
    }

    #[test]
    fn builds_image_payload() {
        let options = Some(json!({"num_images": 2, "guidance_scale": 3.5}));
        let request = ImageRequest {
            prompt: "a brass astrolabe".to_string(),
            size: Some("landscape_4_3".to_string()),
            provider_config: Some(json!({"num_images": 1})),
            ..ImageRequest::default()
        };
        let payload = image_payload("fal-ai/flux/schnell", &options, &request);
        assert_eq!(payload["prompt"], "a brass astrolabe");
        assert_eq!(payload["image_size"], "landscape_4_3");
        assert_eq!(payload["model"], "fal-ai/flux/schnell");
        assert_eq!(payload["num_images"], 1);
        assert_eq!(payload["guidance_scale"], 3.5);
    }

    #[test]
    fn extracts_image_urls() {
        let raw = json!({
            "images": [
                {"url": "https://example.com/one.png", "content_type": "image/png"}
            ]
        });
        let images = extract_images(&raw);
        assert_eq!(images.len(), 1);
        assert!(
            matches!(&images[0], ImageData::Url { url } if url == "https://example.com/one.png")
        );
    }

    #[test]
    fn extracts_video_url() {
        let raw = json!({
            "video": {"url": "https://example.com/out.mp4", "content_type": "video/mp4"}
        });
        let videos = extract_videos(&raw);
        assert_eq!(videos.len(), 1);
        assert!(
            matches!(&videos[0], VideoData::Url { url } if url == "https://example.com/out.mp4")
        );
    }

    #[test]
    fn maps_queue_submit_to_video_job() {
        let raw = json!({
            "request_id": "abc123",
            "status_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/status",
            "response_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/response",
            "cancel_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/cancel"
        });
        let job = video_job_from_submit("fal-ai/wan", raw).unwrap();
        assert_eq!(job.id, "abc123");
        assert_eq!(job.provider, Provider::Fal);
        assert_eq!(job.provider_model.as_deref(), Some("fal-ai/wan"));
        assert!(job.status_url.as_deref().unwrap().ends_with("/status"));
        assert!(job.response_url.as_deref().unwrap().ends_with("/response"));
        assert!(job.cancel_url.as_deref().unwrap().ends_with("/cancel"));
    }

    #[test]
    fn maps_pending_statuses() {
        let queued = video_status_from_pending(json!({
            "status": "IN_QUEUE",
            "queue_position": 7
        }));
        assert!(matches!(
            queued,
            VideoJobStatus::Queued {
                queue_position: Some(7),
                ..
            }
        ));

        let running = video_status_from_pending(json!({"status": "IN_PROGRESS"}));
        assert!(matches!(running, VideoJobStatus::Running { .. }));
    }
}
