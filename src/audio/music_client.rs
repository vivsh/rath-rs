use super::*;
use std::{future::Future, pin::Pin};
/// Asynchronous music provider operations; polling must never resubmit.
pub trait MusicBackend: Send + Sync {
    /// Submits one job; a transport error may leave acceptance unknown.
    fn submit_music(
        &self,
        request: &MusicRequest,
    ) -> impl Future<Output = Result<AudioJob, RathError>> + Send;
    /// Retrieves a snapshot using a caller-owned receipt.
    fn get_music(
        &self,
        job: &AudioJob,
    ) -> impl Future<Output = Result<AudioJobStatus, RathError>> + Send;
}
type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedMusicBackend: Send + Sync {
    fn submit<'a>(
        &'a self,
        request: &'a MusicRequest,
    ) -> ErasedFuture<'a, Result<AudioJob, RathError>>;
    fn get<'a>(&'a self, job: &'a AudioJob) -> ErasedFuture<'a, Result<AudioJobStatus, RathError>>;
}
impl<B: MusicBackend> ErasedMusicBackend for B {
    fn submit<'a>(
        &'a self,
        request: &'a MusicRequest,
    ) -> ErasedFuture<'a, Result<AudioJob, RathError>> {
        Box::pin(MusicBackend::submit_music(self, request))
    }
    fn get<'a>(&'a self, job: &'a AudioJob) -> ErasedFuture<'a, Result<AudioJobStatus, RathError>> {
        Box::pin(MusicBackend::get_music(self, job))
    }
}
/// Owned, model-bound music capability.
pub struct MusicClient {
    inner: Box<dyn ErasedMusicBackend>,
}
impl MusicClient {
    /// Wraps a backend without I/O.
    pub fn from_backend<B: MusicBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Submits exactly once.
    pub async fn submit_music(&self, request: &MusicRequest) -> Result<AudioJob, RathError> {
        self.inner.submit(request).await
    }
    /// Retrieves the remote job without resubmitting.
    pub async fn get_music(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        self.inner.get(job).await
    }
}
impl MusicBackend for MusicClient {
    async fn submit_music(&self, request: &MusicRequest) -> Result<AudioJob, RathError> {
        MusicClient::submit_music(self, request).await
    }
    async fn get_music(&self, job: &AudioJob) -> Result<AudioJobStatus, RathError> {
        MusicClient::get_music(self, job).await
    }
}
