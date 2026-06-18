use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::llm::LlmError;

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
