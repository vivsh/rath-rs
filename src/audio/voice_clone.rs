//! Stateless creation of reusable voice identities from reference audio.
use super::voice::{Voice, VoiceSample};
use crate::core::RathError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a cloning client.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct VoiceCloneOptions {
    /// Provider settings; typed input takes precedence.
    pub provider_config: Option<Value>,
}

impl VoiceCloneOptions {
    /// Constructs an adapter or reports an unsupported cloning capability.
    pub async fn create(self, model_url: &str) -> Result<VoiceCloneClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .voice_clone(model_url, self)
            .await
    }
}

/// Caller-owned reference recordings used to create a voice.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VoiceCloneRequest {
    /// Nonempty reference recordings; native count/transcript requirements are validated.
    pub samples: Vec<VoiceSample>,
    /// Name for providers that create a remote resource; never generated implicitly.
    pub name: Option<String>,
    /// Native controls; typed fields take precedence.
    pub provider_config: Option<Value>,
}

/// Independent cloning capability; callers are responsible for rights to reference recordings.
#[path = "voice_clone_client.rs"]
mod client;
pub use client::{VoiceCloneBackend, VoiceCloneClient};

impl VoiceCloneOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl VoiceCloneRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(samples: Vec<VoiceSample>) -> Self {
        Self {
            samples,
            name: None,
            provider_config: None,
        }
    }
    /// Sets `name` for this value.
    pub fn with_name(mut self, value: String) -> Self {
        self.name = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}
