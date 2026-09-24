use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing an image client.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ImageOptions {
    /// Native settings; object-shaped defaults or request overrides.
    pub provider_config: Option<Value>,
}

impl ImageOptions {
    /// Builds a provider client for the given model URL.
    pub async fn create(self, model_url: &str) -> Result<ImageClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .image(model_url, self)
            .await
    }
}

/// Image generation or editing request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct ImageRequest {
    /// Generation or editing instructions.
    pub prompt: String,
    /// Explicit provider-supported image size.
    pub size: Option<String>,
    /// Native settings; object-shaped defaults or request overrides.
    pub provider_config: Option<serde_json::Value>,
}

/// Image returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageData {
    /// A remote image location; Rath does not download it.
    Url {
        /// Provider-reported image URL.
        url: String,
    },
    /// Complete encoded image data.
    Base64 {
        /// Image MIME type.
        mime_type: String,
        /// Base64-encoded bytes.
        data: String,
    },
}

/// Image generation or editing response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct ImageResponse {
    /// Generated images in provider order.
    pub images: Vec<ImageData>,
    /// Native response evidence, potentially containing private data.
    pub raw_metadata: Option<serde_json::Value>,
}

use super::ImageClient;

impl ImageOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl ImageRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            size: None,
            provider_config: None,
        }
    }
    /// Sets `size` for this value.
    pub fn with_size(mut self, value: String) -> Self {
        self.size = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: serde_json::Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl ImageResponse {
    /// Constructs required values; optional controls start unset.
    pub fn new(images: Vec<ImageData>) -> Self {
        Self {
            images,
            raw_metadata: None,
        }
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: serde_json::Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
