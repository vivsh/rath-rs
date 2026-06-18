use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::LlmError;

/// Options used when constructing a text-to-speech client.
#[derive(Debug, Clone, Default)]
pub struct TtsOptions {
    pub provider_config: Option<Value>,
}

impl TtsOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn TtsClient>, LlmError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_tts_client(&url, self)
    }
}

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
