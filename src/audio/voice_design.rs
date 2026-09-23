//! Stateless voice-preview generation and explicit provider-side registration.
use super::voice::{Voice, VoiceSample};
use crate::core::{ErrorKind, ModelUrl, RathError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a voice design client.
#[derive(Clone, Debug, Default)]
pub struct VoiceDesignOptions {
    /// Native provider settings, overridden by request settings and typed inputs.
    pub provider_config: Option<Value>,
}

impl VoiceDesignOptions {
    /// Constructs an adapter or fails if the selected provider/model cannot design voices.
    pub fn create(self, model_url: &str) -> Result<Box<dyn VoiceDesignClient>, RathError> {
        crate::providers::create_voice_design_client(&ModelUrl::parse(model_url)?, self)
    }
}

/// Describes a voice and the exact words to use when auditioning it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
pub struct VoiceDesignResponse {
    /// Nonempty list of auditionable candidates.
    pub previews: Vec<VoicePreview>,
    /// Original provider output; may contain private text and audio.
    pub raw_metadata: Option<Value>,
}

/// Voice design capability, independent of cloning and speech synthesis.
#[async_trait]
pub trait VoiceDesignClient: Send + Sync {
    /// Generates previews without implicitly registering, cloning, or persisting them.
    async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError>;

    /// Explicitly creates a provider-owned voice from a preview, when supported.
    /// No local state is retained; remote resources survive cancellation or local data deletion.
    async fn register_voice(
        &self,
        _preview: &VoicePreview,
        _name: &str,
        _provider_config: Option<&Value>,
    ) -> Result<Voice, RathError> {
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "native voice registration is unsupported",
        ))
    }
}
