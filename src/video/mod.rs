use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::LlmError;

/// Options used when constructing a video client.
#[derive(Debug, Clone, Default)]
pub struct VideoOptions {
    pub provider_config: Option<Value>,
}

impl VideoOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn VideoClient>, LlmError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_video_client(&url, self)
    }
}

/// Video generation request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub image_url: Option<String>,
    pub provider_config: Option<Value>,
}

/// Video returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoData {
    Url { url: String },
    Base64 { mime_type: String, data: String },
}

/// Video generation response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoResponse {
    pub videos: Vec<VideoData>,
    pub raw_metadata: Option<Value>,
}

/// Provider-agnostic video client.
#[async_trait]
pub trait VideoClient: Send + Sync {
    async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, LlmError>;
}
