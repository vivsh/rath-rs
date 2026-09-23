//! Stateless creation of reusable voice identities from reference audio.
use super::voice::{Voice, VoiceSample};
use crate::core::{ModelUrl, RathError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a cloning client.
#[derive(Clone, Debug, Default)]
pub struct VoiceCloneOptions {
    /// Provider settings; typed input takes precedence.
    pub provider_config: Option<Value>,
}

impl VoiceCloneOptions {
    /// Constructs an adapter or reports an unsupported cloning capability.
    pub fn create(self, model_url: &str) -> Result<Box<dyn VoiceCloneClient>, RathError> {
        crate::providers::create_voice_clone_client(&ModelUrl::parse(model_url)?, self)
    }
}

/// Caller-owned reference recordings used to create a voice.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoiceCloneRequest {
    /// Nonempty reference recordings; native count/transcript requirements are validated.
    pub samples: Vec<VoiceSample>,
    /// Name for providers that create a remote resource; never generated implicitly.
    pub name: Option<String>,
    /// Native controls; typed fields take precedence.
    pub provider_config: Option<Value>,
}

/// Independent cloning capability; callers are responsible for rights to reference recordings.
#[async_trait]
pub trait VoiceCloneClient: Send + Sync {
    /// Returns a ready voice or an error, without local storage or automatic retries.
    /// ID-based providers may create remote resources, even if local waiting is interrupted.
    async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError>;
}
