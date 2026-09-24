use super::*;
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait SttBackend: Send + Sync {
    /// Transcribes audio or returns a validation/provider error. Fal polls every five seconds
    /// with a five-minute I/O deadline; timeout or future cancellation does not cancel the remote job.
    fn transcribe_audio(
        &self,
        request: &SttRequest,
    ) -> impl std::future::Future<Output = Result<SttResponse, RathError>> + Send;
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedSttBackend: Send + Sync {
    fn transcribe_audio<'a>(
        &'a self,
        request: &'a SttRequest,
    ) -> ErasedFuture<'a, Result<SttResponse, RathError>>;
}
impl<B: SttBackend> ErasedSttBackend for B {
    fn transcribe_audio<'a>(
        &'a self,
        request: &'a SttRequest,
    ) -> ErasedFuture<'a, Result<SttResponse, RathError>> {
        Box::pin(SttBackend::transcribe_audio(self, request))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct SttClient {
    inner: Box<dyn ErasedSttBackend>,
}
impl SttClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: SttBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `transcribe_audio` to the owned backend.
    pub async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        self.inner.transcribe_audio(request).await
    }
}
impl SttBackend for SttClient {
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        SttClient::transcribe_audio(self, request).await
    }
}
