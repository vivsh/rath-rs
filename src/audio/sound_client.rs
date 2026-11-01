use super::*;
use std::{future::Future, pin::Pin};
/// Asynchronous sound provider operations; polling must never resubmit.
pub trait SoundBackend: Send + Sync {
    /// Submits one job; a transport error may leave acceptance unknown.
    fn submit_sound(
        &self,
        request: &SoundRequest,
    ) -> impl Future<Output = Result<AudioJob, RathError>> + Send;
    /// Retrieves a snapshot using a caller-owned receipt.
    fn get_sound(
        &self,
        job: &AudioJob,
    ) -> impl Future<Output = Result<AudioJobStatus, RathError>> + Send;
}
type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedSoundBackend: Send + Sync {
    fn submit<'a>(
        &'a self,
        request: &'a SoundRequest,
    ) -> ErasedFuture<'a, Result<AudioJob, RathError>>;
    fn get<'a>(&'a self, job: &'a AudioJob) -> ErasedFuture<'a, Result<AudioJobStatus, RathError>>;
}
impl<B: SoundBackend> ErasedSoundBackend for B {
    fn submit<'a>(
        &'a self,
        request: &'a SoundRequest,
    ) -> ErasedFuture<'a, Result<AudioJob, RathError>> {
        Box::pin(SoundBackend::submit_sound(self, request))
    }
    fn get<'a>(&'a self, job: &'a AudioJob) -> ErasedFuture<'a, Result<AudioJobStatus, RathError>> {
        Box::pin(SoundBackend::get_sound(self, job))
    }
}
/// Owned, model-bound sound capability.
pub struct SoundClient {
    inner: Box<dyn ErasedSoundBackend>,
}
impl SoundClient {
    /// Wraps a backend without I/O.
    pub fn from_backend<B: SoundBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Submits exactly once.
    pub async fn submit_sound(&self, request: &SoundRequest) -> Result<AudioJob, RathError> {
        self.inner.submit(request).await
    }
    /// Retrieves the remote job without resubmitting.
    pub async fn get_sound(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        self.inner.get(job).await
    }
}
impl SoundBackend for SoundClient {
    async fn submit_sound(&self, request: &SoundRequest) -> Result<AudioJob, RathError> {
        SoundClient::submit_sound(self, request).await
    }
    async fn get_sound(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        SoundClient::get_sound(self, job).await
    }
}
