//! Stateless voice-preview generation and explicit provider-side registration.
use super::voice::{Voice, VoiceSample};
use crate::core::{ErrorKind, RathError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a voice design client.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct VoiceDesignOptions {
    /// Native provider settings, overridden by request settings and typed inputs.
    pub provider_config: Option<Value>,
}

impl VoiceDesignOptions {
    /// Constructs an adapter or fails if the selected provider/model cannot design voices.
    pub async fn create(self, model_url: &str) -> Result<VoiceDesignClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .voice_design(model_url, self)
            .await
    }
}

/// Describes a voice and the exact words to use when auditioning it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VoiceDesignRequest {
    /// Vocal identity, such as timbre and accent; must not be blank.
    pub description: String,
    /// Exact sample words; adapters validate native length limits.
    pub text: String,
    /// Optional BCP-47 language tag; unsupported explicit selections fail.
    pub language: Option<String>,
    /// Native controls; typed fields take precedence.
    pub provider_config: Option<Value>,
}

/// Auditionable recording, not implicitly a registered or synthesis-ready voice.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VoicePreview {
    /// Generated recording and transcript.
    pub sample: VoiceSample,
    /// Adapter-defined namespace for validating native registration.
    pub scope: String,
    /// Opaque native preview identifier; may expire and is not a TTS voice ID.
    pub registration_token: Option<String>,
}

/// Generated previews in provider order, owned entirely by the caller.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VoiceDesignResponse {
    /// Nonempty list of auditionable candidates.
    pub previews: Vec<VoicePreview>,
    /// Original provider output; may contain private text and audio.
    pub raw_metadata: Option<Value>,
}

/// Voice design capability, independent of cloning and speech synthesis.
#[path = "voice_design_client.rs"]
mod client;
pub use client::{VoiceDesignBackend, VoiceDesignClient};

impl VoiceDesignOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl VoiceDesignRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(description: String, text: String) -> Self {
        Self {
            description,
            text,
            language: None,
            provider_config: None,
        }
    }
    /// Sets `language` for this value.
    pub fn with_language(mut self, value: String) -> Self {
        self.language = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl VoicePreview {
    /// Constructs required values; optional controls start unset.
    pub fn new(sample: VoiceSample, scope: String) -> Self {
        Self {
            sample,
            scope,
            registration_token: None,
        }
    }
    /// Sets `registration_token` for this value.
    pub fn with_registration_token(mut self, value: String) -> Self {
        self.registration_token = Some(value);
        self
    }
}

impl VoiceDesignResponse {
    /// Constructs required values; optional controls start unset.
    pub fn new(previews: Vec<VoicePreview>) -> Self {
        Self {
            previews,
            raw_metadata: None,
        }
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
