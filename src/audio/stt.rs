use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing a speech-to-text client.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SttOptions {
    /// Native provider defaults; Fal requires a JSON object and request settings take precedence.
    pub provider_config: Option<Value>,
}

impl SttOptions {
    /// Builds a provider client for the given model URL.
    pub async fn create(self, model_url: &str) -> Result<SttClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .stt(model_url, self)
            .await
    }
}

/// Speech-to-text request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SttRequest {
    /// Input MIME type; Fal requires an `audio/*` type without parameters.
    pub mime_type: String,
    /// Complete encoded audio file; Fal sends a base64 data URI and rejects empty bytes.
    pub data: Vec<u8>,
    /// Native request settings, such as language or diarization, overriding client defaults.
    pub provider_config: Option<serde_json::Value>,
}

/// Speech-to-text response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct SttResponse {
    /// Transcription text; an empty transcript can be a valid provider result.
    pub text: String,
    /// Original provider result, including timestamps and speakers when supplied.
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic speech-to-text client.
#[path = "stt_client.rs"]
mod client;
pub use client::{SttBackend, SttClient};

impl SttOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl SttRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(mime_type: String, data: Vec<u8>) -> Self {
        Self {
            mime_type,
            data,
            provider_config: None,
        }
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: serde_json::Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl SttResponse {
    /// Constructs required values; optional controls start unset.
    pub fn new(text: String) -> Self {
        Self {
            text,
            raw_metadata: None,
        }
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: serde_json::Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
