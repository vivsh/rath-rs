use super::*;
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait TtsBackend: Send + Sync {
    /// Generates audio or returns a validation/provider error. Fal polls every five seconds
    /// with a five-minute I/O deadline; timeout or future cancellation does not cancel the remote job.
    fn synthesize_speech(
        &self,
        request: &TtsRequest,
    ) -> impl std::future::Future<Output = Result<TtsResponse, RathError>> + Send;
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedTtsBackend: Send + Sync {
    fn synthesize_speech<'a>(
        &'a self,
        request: &'a TtsRequest,
    ) -> ErasedFuture<'a, Result<TtsResponse, RathError>>;
}
impl<B: TtsBackend> ErasedTtsBackend for B {
    fn synthesize_speech<'a>(
        &'a self,
        request: &'a TtsRequest,
    ) -> ErasedFuture<'a, Result<TtsResponse, RathError>> {
        Box::pin(TtsBackend::synthesize_speech(self, request))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct TtsClient {
    inner: Box<dyn ErasedTtsBackend>,
}
impl TtsClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: TtsBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `synthesize_speech` to the owned backend.
    pub async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        self.inner.synthesize_speech(request).await
    }
}
impl TtsBackend for TtsClient {
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        TtsClient::synthesize_speech(self, request).await
    }
}
