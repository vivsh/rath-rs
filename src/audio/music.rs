//! Model-bound music generation.
use super::generation::{AudioJob, AudioJobStatus};
use crate::RathError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Immutable native defaults for a music client.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MusicOptions {
    /// Object-shaped provider defaults.
    pub provider_config: Option<Value>,
}
impl MusicOptions {
    /// Constructs a model-bound client without submitting generation.
    pub async fn create(self, model_url: &str) -> Result<MusicClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .music(model_url, self)
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
pub struct MusicRequest {
    /// Nonblank description.
    pub prompt: String,
    /// Optional duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Optional instrumental control.
    pub instrumental: Option<bool>,
    /// Object-shaped native request overrides.
    pub provider_config: Option<Value>,
}
impl MusicRequest {
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
    /// Sets the instrumental control.
    pub fn with_instrumental(mut self, value: bool) -> Self {
        self.instrumental = Some(value);
        self
    }
    /// Sets native request overrides.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}
#[path = "music_client.rs"]
mod client;
pub use client::{MusicBackend, MusicClient};

#[cfg(test)]
#[path = "tests/music.rs"]
mod tests;
