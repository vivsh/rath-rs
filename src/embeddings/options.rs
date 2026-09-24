use super::{EmbeddingClient, FastEmbedOptions};
use crate::RathError;
use serde_json::Value;
/// Immutable construction settings; unsupported explicit configuration fails.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EmbeddingOptions {
    /// Provider-native inference defaults.
    pub provider_config: Option<Value>,
    /// Explicit local runtime preparation; rejected by remote adapters.
    pub fastembed: Option<FastEmbedOptions>,
}
impl EmbeddingOptions {
    /// Builds one initialized embedding client without performing inference.
    pub async fn create(self, model_url: &str) -> Result<EmbeddingClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .embeddings(model_url, self)
            .await
    }
    /// Supplies explicit local runtime setup.
    pub fn with_fastembed(mut self, options: FastEmbedOptions) -> Self {
        self.fastembed = Some(options);
        self
    }
    /// Sets native inference defaults, validated at construction.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}
