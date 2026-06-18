use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::core::ModelUrl;
use crate::images::{ImageClient, ImageData, ImageOptions, ImageRequest, ImageResponse};
use crate::llm::LlmError;
use crate::video::{VideoClient, VideoData, VideoOptions, VideoRequest, VideoResponse};

const DEFAULT_BASE_URL: &str = "https://fal.run";
const DEFAULT_API_KEY_ENV: &str = "FAL_KEY";

pub(crate) fn new_image_client(
    url: &ModelUrl,
    options: ImageOptions,
) -> Result<Box<dyn ImageClient>, LlmError> {
    Ok(Box::new(FalClient::new(
        url,
        options.provider_config,
        "image",
    )?))
}

pub(crate) fn new_video_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<Box<dyn VideoClient>, LlmError> {
    Ok(Box::new(FalClient::new(
        url,
        options.provider_config,
        "video",
    )?))
}

struct FalClient {
    http: reqwest::Client,
    api_key: String,
    endpoint: String,
    model: String,
    provider_config: Option<Value>,
}

impl FalClient {
    fn new(
        url: &ModelUrl,
        provider_config: Option<Value>,
        capability: &str,
    ) -> Result<Self, LlmError> {
        let api_key = match &url.api_key {
            Some(key) => key.clone(),
            None => std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| {
                LlmError::Validation(format!(
                    "set {DEFAULT_API_KEY_ENV} or pass api_key_env for Fal {capability} calls"
                ))
            })?,
        };
        let base_url = url
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let endpoint = build_endpoint(&base_url, &url.model);
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            endpoint,
            model: url.model.clone(),
            provider_config,
        })
    }

    async fn post(&self, payload: Value) -> Result<Value, LlmError> {
        let response = self
            .http
            .post(&self.endpoint)
            .header("Authorization", format!("Key {}", self.api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::Llm(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::Llm(e.to_string()))?;
        if !status.is_success() {
            return Err(LlmError::Llm(format!(
                "Fal request failed with status {status}: {body}"
            )));
        }
        serde_json::from_str(&body).map_err(|source| LlmError::Deserialize { source, raw: body })
    }
}

#[async_trait]
impl ImageClient for FalClient {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, LlmError> {
        let payload = image_payload(&self.model, &self.provider_config, request);
        let raw = self.post(Value::Object(payload)).await?;
        Ok(ImageResponse {
            images: extract_images(&raw),
            raw_metadata: Some(raw),
        })
    }
}

#[async_trait]
impl VideoClient for FalClient {
    async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, LlmError> {
        let payload = video_payload(&self.provider_config, request);
        let raw = self.post(Value::Object(payload)).await?;
        Ok(VideoResponse {
            videos: extract_videos(&raw),
            raw_metadata: Some(raw),
        })
    }
}

fn build_endpoint(base_url: &str, model: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        model.trim_start_matches('/')
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
}
