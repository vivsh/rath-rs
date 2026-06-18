use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::LlmError;

/// Options used when constructing an embedding client.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingOptions {
    pub provider_config: Option<Value>,
}

impl EmbeddingOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn EmbeddingClient>, LlmError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_embedding_client(&url, self)
    }
}

/// Task hint for embedding optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedTaskType {
    RetrievalDocument,
    RetrievalQuery,
    SemanticSimilarity,
    Classification,
    Clustering,
    QuestionAnswering,
    FactVerification,
    CodeRetrievalQuery,
}

/// Request to generate a text embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedRequest {
    /// Text to embed.
    pub input: String,
    /// Optional task hint for embedding quality optimization.
    pub task_type: Option<EmbedTaskType>,
    /// Document title hint, used by providers that support it.
    pub title: Option<String>,
    /// Truncate the output vector to this many dimensions.
    pub output_dimensionality: Option<i32>,
    /// Provider-specific options serialized as a JSON object.
    pub provider_config: Option<serde_json::Value>,
}

/// Embedding vector returned by an embedding provider.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    /// Embedding coefficients in the order returned by the provider.
    pub values: Vec<f32>,
}

/// Provider-agnostic embedding client.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, LlmError>;
}
