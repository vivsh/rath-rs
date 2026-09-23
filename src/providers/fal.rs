use crate::core::error::http;
mod audio;
mod queue;
mod video;
mod video_input;
mod video_media;
mod video_webhook;
mod voice;
pub(crate) use video::new_client as new_video_client;
#[cfg(test)]
use video::{
    build_status_endpoint, extract_videos, video_job_from_submit, video_status_from_pending,
};

pub(crate) use audio::{new_stt_client, new_tts_client};
pub(crate) use voice::{
    new_clone as new_voice_clone_client, new_design as new_voice_design_client,
};

use async_trait::async_trait;
use serde_json::{Map, Value};
use std::time::Duration;

use crate::core::RathError;
use crate::core::{ModelUrl, Provider};
use crate::images::{ImageClient, ImageData, ImageOptions, ImageRequest, ImageResponse};
#[cfg(test)]
use crate::video::{VideoClient, VideoData, VideoJobStatus};

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

#[cfg(test)]
mod tests;
