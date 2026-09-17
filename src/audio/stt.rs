use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing a speech-to-text client.
#[derive(Debug, Clone, Default)]
pub struct SttOptions {
    /// Native provider defaults; Fal requires a JSON object and request settings take precedence.
    pub provider_config: Option<Value>,
}

impl SttOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn SttClient>, RathError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_stt_client(&url, self)
    }
}

/// Speech-to-text request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttRequest {
    /// Input MIME type; Fal requires an `audio/*` type without parameters.
    pub mime_type: String,
    /// Complete encoded audio file; Fal sends a base64 data URI and rejects empty bytes.
    pub data: Vec<u8>,
    /// Per-request model override; Fal accepts only supported transcription endpoint slugs.
    pub model: Option<String>,
    /// Native request settings, such as language or diarization, overriding client defaults.
    pub provider_config: Option<serde_json::Value>,
}

/// Speech-to-text response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SttResponse {
    /// Transcription text; an empty transcript can be a valid provider result.
    pub text: String,
    /// Original provider result, including timestamps and speakers when supplied.
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic speech-to-text client.
#[async_trait]
pub trait SttClient: Send + Sync {
    /// Transcribes audio or returns a validation/provider error. Fal polls every five seconds
    /// with a five-minute I/O deadline; timeout or future cancellation does not cancel the remote job.
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError>;
}
