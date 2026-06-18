use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::llm::LlmError;

/// Text-to-speech request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TtsRequest {
    pub input: String,
    pub voice: Option<String>,
    pub model: Option<String>,
    pub format: Option<String>,
    pub provider_config: Option<serde_json::Value>,
}

/// Text-to-speech response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResponse {
    pub mime_type: String,
    pub data: Vec<u8>,
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic text-to-speech client.
#[async_trait]
pub trait TtsClient: Send + Sync {
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, LlmError>;
}
