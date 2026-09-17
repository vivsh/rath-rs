use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing a text-to-speech client.
#[derive(Debug, Clone, Default)]
pub struct TtsOptions {
    /// Native provider defaults; Fal requires a JSON object and request settings take precedence.
    pub provider_config: Option<Value>,
}

impl TtsOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn TtsClient>, RathError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_tts_client(&url, self)
    }
}

/// Text-to-speech request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TtsRequest {
    /// Text to synthesize; Fal rejects empty or whitespace-only input.
    pub input: String,
    /// Native voice identifier; omitted values use the provider's configured default.
    pub voice: Option<String>,
    /// Per-request model override; Fal accepts only supported synthesis endpoint slugs.
    pub model: Option<String>,
    /// Requested output format; the supported Fal endpoints reject explicit format selection.
    pub format: Option<String>,
    /// Native request settings, overriding client defaults; supplied typed fields take precedence.
    pub provider_config: Option<serde_json::Value>,
}

/// Text-to-speech response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResponse {
    /// MIME type describing the returned audio bytes.
    pub mime_type: String,
    /// Complete audio bytes; Fal downloads its generated file before returning.
    pub data: Vec<u8>,
    /// Original provider result when available, including model-specific metadata.
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic text-to-speech client.
#[async_trait]
pub trait TtsClient: Send + Sync {
    /// Generates audio or returns a validation/provider error. Fal polls every five seconds
    /// with a five-minute I/O deadline; timeout or future cancellation does not cancel the remote job.
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError>;
}
