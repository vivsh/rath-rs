use super::{EmbedRequest, EmbedResponse};
use crate::RathError;
use std::{future::Future, pin::Pin};

/// Implement dense embeddings without runtime erasure or procedural macros.
pub trait EmbeddingBackend: Send + Sync {
    /// Returns one vector per input, or an error without partial success.
    fn embed(
        &self,
        request: &EmbedRequest,
    ) -> impl Future<Output = Result<EmbedResponse, RathError>> + Send;
}
trait ErasedEmbeddingBackend: Send + Sync {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbedResponse, RathError>> + Send + 'a>>;
}
impl<B: EmbeddingBackend> ErasedEmbeddingBackend for B {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbedResponse, RathError>> + Send + 'a>> {
        Box::pin(EmbeddingBackend::embed(self, request))
    }
}
/// An owned model-bound embedding backend, with private runtime erasure.
pub struct EmbeddingClient {
    inner: Box<dyn ErasedEmbeddingBackend>,
}
impl EmbeddingClient {
    /// Wraps an initialized backend without I/O or additional state.
    pub fn from_backend<B: EmbeddingBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Embeds a homogeneous ordered batch.
    pub async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        self.inner.embed(request).await
    }
}
impl EmbeddingBackend for EmbeddingClient {
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        self.inner.embed(request).await
    }
}
