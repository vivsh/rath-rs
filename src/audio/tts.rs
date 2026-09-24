use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::RathError;

/// Options used when constructing a text-to-speech client.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TtsOptions {
    /// Native provider defaults; must be an object and request settings take precedence.
    pub provider_config: Option<Value>,
}

impl TtsOptions {
    /// Builds a provider client for the given model URL.
    pub async fn create(self, model_url: &str) -> Result<TtsClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .tts(model_url, self)
            .await
    }
}

/// Text-to-speech request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct TtsRequest {
    /// Text to synthesize; Fal rejects empty or whitespace-only input.
    pub input: String,
    /// Scoped voice identity; omission uses a provider default only where supported.
    pub voice: Option<super::voice::Voice>,
    /// Requested output format; the supported Fal endpoints reject explicit format selection.
    pub format: Option<String>,
    /// Optional BCP-47 language tag; unsupported explicit selections fail.
    pub language: Option<String>,
    /// Per-utterance delivery instructions; unsupported explicit controls fail.
    pub instructions: Option<String>,
    /// Native request settings, overriding client defaults; supplied typed fields take precedence.
    pub provider_config: Option<serde_json::Value>,
}

/// Text-to-speech response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TtsResponse {
    /// MIME type describing the returned audio bytes.
    pub mime_type: String,
    /// Complete audio bytes; Fal downloads its generated file before returning.
    pub data: Vec<u8>,
    /// Original provider result when available, including model-specific metadata.
    pub raw_metadata: Option<serde_json::Value>,
}

/// Provider-agnostic text-to-speech client.
#[path = "tts_client.rs"]
mod client;
pub use client::{TtsBackend, TtsClient};

impl TtsOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl TtsRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(input: String) -> Self {
        Self {
            input,
            voice: None,
            format: None,
            language: None,
            instructions: None,
            provider_config: None,
        }
    }
    /// Sets `voice` for this value.
    pub fn with_voice(mut self, value: super::voice::Voice) -> Self {
        self.voice = Some(value);
        self
    }
    /// Sets `format` for this value.
    pub fn with_format(mut self, value: String) -> Self {
        self.format = Some(value);
        self
    }
    /// Sets `language` for this value.
    pub fn with_language(mut self, value: String) -> Self {
        self.language = Some(value);
        self
    }
    /// Sets `instructions` for this value.
    pub fn with_instructions(mut self, value: String) -> Self {
        self.instructions = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: serde_json::Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl TtsResponse {
    /// Constructs required values; optional controls start unset.
    pub fn new(mime_type: String, data: Vec<u8>) -> Self {
        Self {
            mime_type,
            data,
            raw_metadata: None,
        }
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: serde_json::Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
