use super::*;
use crate::{ErrorKind, RathError};
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait VideoBackend: Send + Sync {
    /// Submits once or returns an error. A lost response does not prove no job was created.
    fn submit_video(
        &self,
        request: &VideoRequest,
    ) -> impl std::future::Future<Output = Result<VideoJob, RathError>> + Send;
    /// Reads a job using the same provider/model/account as its submission.
    fn get_video(
        &self,
        job_id: &str,
    ) -> impl std::future::Future<Output = Result<VideoJobStatus, RathError>> + Send;
    /// Polls without an overall deadline. Dropping the future does not cancel remote work.
    fn wait_video(
        &self,
        job_id: &str,
    ) -> impl std::future::Future<Output = Result<VideoResponse, RathError>> + Send;
    /// Submits once and waits; errors or cancellation never resubmit generation.
    fn generate_video(
        &self,
        request: &VideoRequest,
    ) -> impl std::future::Future<Output = Result<VideoResponse, RathError>> + Send {
        async move {
            let job = self.submit_video(request).await?;
            self.wait_video(&job.id).await
        }
    }
    /// Verifies original HTTP bytes before decoding, or fails explicitly when unsupported.
    /// Callers authorize the job, durably accept/deduplicate it, and acknowledge delivery.
    /// Verification may fetch provider keys, but never fetches the generated video.
    fn parse_webhook(
        &self,
        _headers: &http::HeaderMap,
        _body: &[u8],
    ) -> impl std::future::Future<Output = Result<VideoEvent, RathError>> + Send {
        async move {
            Err(RathError::new(
                ErrorKind::UnsupportedCapability,
                "video webhooks are unsupported",
            ))
        }
    }
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedVideoBackend: Send + Sync {
    fn submit_video<'a>(
        &'a self,
        request: &'a VideoRequest,
    ) -> ErasedFuture<'a, Result<VideoJob, RathError>>;
    fn get_video<'a>(
        &'a self,
        job_id: &'a str,
    ) -> ErasedFuture<'a, Result<VideoJobStatus, RathError>>;
    fn wait_video<'a>(
        &'a self,
        job_id: &'a str,
    ) -> ErasedFuture<'a, Result<VideoResponse, RathError>>;
    fn generate_video<'a>(
        &'a self,
        request: &'a VideoRequest,
    ) -> ErasedFuture<'a, Result<VideoResponse, RathError>>;
    fn parse_webhook<'a>(
        &'a self,
        headers: &'a http::HeaderMap,
        body: &'a [u8],
    ) -> ErasedFuture<'a, Result<VideoEvent, RathError>>;
}
impl<B: VideoBackend> ErasedVideoBackend for B {
    fn submit_video<'a>(
        &'a self,
        request: &'a VideoRequest,
    ) -> ErasedFuture<'a, Result<VideoJob, RathError>> {
        Box::pin(VideoBackend::submit_video(self, request))
    }
    fn get_video<'a>(
        &'a self,
        job_id: &'a str,
    ) -> ErasedFuture<'a, Result<VideoJobStatus, RathError>> {
        Box::pin(VideoBackend::get_video(self, job_id))
    }
    fn wait_video<'a>(
        &'a self,
        job_id: &'a str,
    ) -> ErasedFuture<'a, Result<VideoResponse, RathError>> {
        Box::pin(VideoBackend::wait_video(self, job_id))
    }
    fn generate_video<'a>(
        &'a self,
        request: &'a VideoRequest,
    ) -> ErasedFuture<'a, Result<VideoResponse, RathError>> {
        Box::pin(VideoBackend::generate_video(self, request))
    }
    fn parse_webhook<'a>(
        &'a self,
        headers: &'a http::HeaderMap,
        body: &'a [u8],
    ) -> ErasedFuture<'a, Result<VideoEvent, RathError>> {
        Box::pin(VideoBackend::parse_webhook(self, headers, body))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct VideoClient {
    inner: Box<dyn ErasedVideoBackend>,
}
impl VideoClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: VideoBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `submit_video` to the owned backend.
    pub async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        self.inner.submit_video(request).await
    }
    /// Forwards `get_video` to the owned backend.
    pub async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        self.inner.get_video(job_id).await
    }
    /// Forwards `wait_video` to the owned backend.
    pub async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        self.inner.wait_video(job_id).await
    }
    /// Forwards `generate_video` to the owned backend.
    pub async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, RathError> {
        self.inner.generate_video(request).await
    }
    /// Forwards `parse_webhook` to the owned backend.
    pub async fn parse_webhook(
        &self,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<VideoEvent, RathError> {
        self.inner.parse_webhook(headers, body).await
    }
}
impl VideoBackend for VideoClient {
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        VideoClient::submit_video(self, request).await
    }
    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        VideoClient::get_video(self, job_id).await
    }
    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        VideoClient::wait_video(self, job_id).await
    }
    async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, RathError> {
        VideoClient::generate_video(self, request).await
    }
    async fn parse_webhook(
        &self,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<VideoEvent, RathError> {
        VideoClient::parse_webhook(self, headers, body).await
    }
}
