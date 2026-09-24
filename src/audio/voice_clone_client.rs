use super::*;
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait VoiceCloneBackend: Send + Sync {
    /// Returns a ready voice or an error, without local storage or automatic retries.
    /// ID-based providers may create remote resources, even if local waiting is interrupted.
    fn clone_voice(
        &self,
        request: &VoiceCloneRequest,
    ) -> impl std::future::Future<Output = Result<Voice, RathError>> + Send;
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedVoiceCloneBackend: Send + Sync {
    fn clone_voice<'a>(
        &'a self,
        request: &'a VoiceCloneRequest,
    ) -> ErasedFuture<'a, Result<Voice, RathError>>;
}
impl<B: VoiceCloneBackend> ErasedVoiceCloneBackend for B {
    fn clone_voice<'a>(
        &'a self,
        request: &'a VoiceCloneRequest,
    ) -> ErasedFuture<'a, Result<Voice, RathError>> {
        Box::pin(VoiceCloneBackend::clone_voice(self, request))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct VoiceCloneClient {
    inner: Box<dyn ErasedVoiceCloneBackend>,
}
impl VoiceCloneClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: VoiceCloneBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `clone_voice` to the owned backend.
    pub async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError> {
        self.inner.clone_voice(request).await
    }
}
impl VoiceCloneBackend for VoiceCloneClient {
    async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError> {
        VoiceCloneClient::clone_voice(self, request).await
    }
}
