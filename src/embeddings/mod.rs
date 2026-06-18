use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::llm::LlmError;

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
