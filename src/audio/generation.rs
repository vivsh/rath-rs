//! Caller-owned audio job receipts and result snapshots.
use crate::Provider;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Durable provider correlation; metadata can contain private information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AudioJob {
    /// Provider request identifier.
    pub id: String,
    /// Provider responsible for the job.
    pub provider: Provider,
    /// Model endpoint used for submission.
    pub provider_model: Option<String>,
    /// Provider queue status URL.
    pub status_url: Option<String>,
    /// Provider result URL.
    pub response_url: Option<String>,
    /// Sanitized provider receipt; treat as private.
    pub raw_metadata: Option<Value>,
}
impl AudioJob {
    /// Constructs a receipt with optional correlation unset.
    pub fn new(id: String, provider: Provider) -> Self {
        Self {
            id,
            provider,
            provider_model: None,
            status_url: None,
            response_url: None,
            raw_metadata: None,
        }
    }
    /// Sets the exact provider model used for submission.
    pub fn with_provider_model(mut self, value: String) -> Self {
        self.provider_model = Some(value);
        self
    }
    /// Sets the provider status URL.
    pub fn with_status_url(mut self, value: String) -> Self {
        self.status_url = Some(value);
        self
    }
    /// Sets the provider result URL.
    pub fn with_response_url(mut self, value: String) -> Self {
        self.response_url = Some(value);
        self
    }
    /// Sets private, credential-free native receipt metadata.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
/// Encoded audio or a location usable as input or caller-owned output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioData {
    /// HTTP(S) URL; downloads must not use provider credentials.
    Url {
        /// Media location, possibly temporary or private.
        url: String,
    },
    /// Complete standard-base64 encoded audio.
    Base64 {
        /// Parameter-free audio MIME type.
        mime_type: String,
        /// Standard base64 without a data URI.
        data: String,
    },
}
/// One usable generated audio result and optional private provider evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AudioResponse {
    /// Generated audio.
    pub audio: AudioData,
    /// Native response, potentially private.
    pub raw_metadata: Option<Value>,
}
impl AudioResponse {
    /// Constructs an audio response without provider metadata.
    pub fn new(audio: AudioData) -> Self {
        Self {
            audio,
            raw_metadata: None,
        }
    }
    /// Adds private native result metadata.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
/// Snapshot of a remote job; transport failures are returned separately.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioJobStatus {
    /// Accepted and waiting.
    Queued {
        /// Optional queue position.
        queue_position: Option<u64>,
        /// Private provider evidence.
        raw_metadata: Option<Value>,
    },
    /// Provider is executing the job.
    Running {
        /// Private provider evidence.
        raw_metadata: Option<Value>,
    },
    /// Provider returned usable audio.
    Succeeded {
        /// Audio and private provider evidence.
        response: AudioResponse,
    },
    /// Provider declared failure; metadata may be private.
    Failed {
        /// Sanitized diagnostic message.
        message: String,
        /// Sanitized native evidence.
        raw_metadata: Option<Value>,
    },
}
