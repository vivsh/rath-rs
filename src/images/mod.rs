use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing an image client.
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub provider_config: Option<Value>,
}

impl ImageOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn ImageClient>, RathError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_image_client(&url, self)
    }
}

/// Image generation or editing request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub size: Option<String>,
    pub provider_config: Option<serde_json::Value>,
}

/// Image returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageData {
    Url { url: String },
    Base64 { mime_type: String, data: String },
}

/// Image generation or editing response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageResponse {
    pub images: Vec<ImageData>,
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic image client.
#[async_trait]
pub trait ImageClient: Send + Sync {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError>;
}
