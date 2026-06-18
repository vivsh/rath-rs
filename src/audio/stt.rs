use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::LlmError;

/// Options used when constructing a speech-to-text client.
#[derive(Debug, Clone, Default)]
pub struct SttOptions {
    pub provider_config: Option<Value>,
}

impl SttOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn SttClient>, LlmError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_stt_client(&url, self)
    }
}

/// Speech-to-text request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttRequest {
    pub mime_type: String,
    pub data: Vec<u8>,
    pub model: Option<String>,
    pub provider_config: Option<serde_json::Value>,
}

/// Speech-to-text response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SttResponse {
    pub text: String,
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic speech-to-text client.
#[async_trait]
pub trait SttClient: Send + Sync {
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, LlmError>;
}
