//! Model-bound sound generation.
use super::generation::{AudioJob, AudioJobStatus};
use crate::RathError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a sound client.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SoundOptions {
    /// Object-shaped provider defaults.
    pub provider_config: Option<Value>,
}
impl SoundOptions {
    /// Constructs a model-bound client without submitting generation.
    pub async fn create(self, model_url: &str) -> Result<SoundClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .sound(model_url, self)
            .await
    }
    /// Sets native provider defaults.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}
/// Caller-owned generation settings; typed fields override native settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SoundRequest {
    /// Nonblank description.
    pub prompt: String,
    /// Optional duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Optional looped control.
    pub looped: Option<bool>,
    /// Object-shaped native request overrides.
    pub provider_config: Option<Value>,
}
impl SoundRequest {
    /// Creates a request with all optional controls unset.
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            ..Self::default()
        }
    }
    /// Sets duration in seconds.
    pub fn with_duration_seconds(mut self, value: f64) -> Self {
        self.duration_seconds = Some(value);
        self
    }
    /// Sets the looped control.
    pub fn with_looped(mut self, value: bool) -> Self {
        self.looped = Some(value);
        self
    }
    /// Sets native request overrides.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}
#[path = "sound_client.rs"]
mod client;
pub use client::{SoundBackend, SoundClient};

#[cfg(test)]
#[path = "tests/sound.rs"]
mod tests;
