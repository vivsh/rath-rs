use crate::{Provider, TokenUsage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Task hint for embedding optimization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum EmbedTaskType {
    /// Retrieval corpus documents.
    RetrievalDocument,
    /// Queries used to search a retrieval corpus.
    RetrievalQuery,
    /// Pairwise semantic similarity.
    SemanticSimilarity,
    /// Inputs for classification.
    Classification,
    /// Inputs for clustering.
    Clustering,
    /// Questions used to retrieve answers.
    QuestionAnswering,
    /// Statements used to retrieve supporting evidence.
    FactVerification,
    /// Queries used to retrieve code.
    CodeRetrievalQuery,
}

/// Request to generate a text embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct EmbedRequest {
    /// Text to embed.
    pub input: Vec<String>,
    /// Optional task hint for embedding quality optimization.
    pub task_type: Option<EmbedTaskType>,
    /// Document title hint, used by providers that support it.
    pub title: Option<String>,
    /// Requests native output dimensionality; vectors are never sliced locally.
    pub output_dimensionality: Option<u32>,
    /// Model-specific retrieval-query instruction; unsupported combinations fail.
    pub instructions: Option<String>,
    /// Explicit input truncation policy, when supported.
    pub truncate: Option<bool>,
    /// Provider-specific options serialized as a JSON object.
    pub provider_config: Option<serde_json::Value>,
}

/// Embedding vector returned by an embedding provider.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmbedResponse {
    /// Embedding coefficients in the order returned by the provider.
    pub embeddings: Vec<Vec<f32>>,
    /// Aggregate provider-reported usage.
    pub usage: Option<TokenUsage>,
    /// Provider that produced the vectors.
    pub provider: Provider,
    /// Provider-echoed model identifier, when available.
    pub provider_model: Option<String>,
    /// Original response, potentially containing private data.
    pub raw_metadata: Option<Value>,
}

impl EmbedRequest {
    /// Builds a one-input request without explicit model controls.
    pub fn new(text: impl Into<String>) -> Self {
        Self::batch([text.into()])
    }
    /// Builds an ordered homogeneous batch; validated when executed.
    pub fn batch(inputs: impl IntoIterator<Item = String>) -> Self {
        Self {
            input: inputs.into_iter().collect(),
            ..Self::default()
        }
    }
}
impl EmbedResponse {
    /// Constructs a response; adapters must validate vector count and dimensions.
    pub fn new(provider: Provider, embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            provider,
            usage: None,
            provider_model: None,
            raw_metadata: None,
        }
    }
}

impl EmbedRequest {
    /// Sets `task_type` for this value.
    pub fn with_task_type(mut self, value: EmbedTaskType) -> Self {
        self.task_type = Some(value);
        self
    }
    /// Sets `title` for this value.
    pub fn with_title(mut self, value: String) -> Self {
        self.title = Some(value);
        self
    }
    /// Sets `output_dimensionality` for this value.
    pub fn with_output_dimensionality(mut self, value: u32) -> Self {
        self.output_dimensionality = Some(value);
        self
    }
    /// Sets `instructions` for this value.
    pub fn with_instructions(mut self, value: String) -> Self {
        self.instructions = Some(value);
        self
    }
    /// Sets `truncate` for this value.
    pub fn with_truncate(mut self, value: bool) -> Self {
        self.truncate = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: serde_json::Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl EmbedResponse {
    /// Sets `usage` for this value.
    pub fn with_usage(mut self, value: TokenUsage) -> Self {
        self.usage = Some(value);
        self
    }
    /// Sets `provider_model` for this value.
    pub fn with_provider_model(mut self, value: String) -> Self {
        self.provider_model = Some(value);
        self
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}
