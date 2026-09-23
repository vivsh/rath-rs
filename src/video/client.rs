use super::{VideoEvent, VideoJob, VideoJobStatus, VideoRequest, VideoResponse};
use crate::core::{ErrorKind, ModelUrl, RathError};
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

/// Immutable provider configuration; no jobs or callback state are retained.
#[derive(Debug, Clone)]
pub struct VideoOptions {
    /// Object-shaped native defaults, overridden by request settings and explicit inputs.
    pub provider_config: Option<Value>,
    /// Delay between status requests; defaults to five seconds.
    pub poll_interval: Duration,
}
impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            provider_config: None,
            poll_interval: Duration::from_secs(5),
        }
    }
}
impl VideoOptions {
    /// Selects provider and endpoint from a URL without generation.
    /// Invalid URLs/configuration or unsupported providers return an error.
    ///
    /// ```no_run
    /// use rath::images::ImageData;
    /// use rath::video::{VideoInput, VideoOptions, VideoRequest};
    /// # async fn example() -> Result<(), rath::RathError> {
    /// let client = VideoOptions::default()
    ///     .create("fal:///fal-ai/kling-video/v3/pro/image-to-video")?;
    /// let job = client.submit_video(&VideoRequest {
    ///     prompt: "The character waves.".into(),
    ///     input: VideoInput::ImageToVideo {
    ///         start_image: ImageData::Url { url: "https://example.com/shot.png".into() },
    ///         end_image: None,
    ///     },
    ///     ..Default::default()
    /// }).await?;
    /// let output = client.wait_video(&job.id).await?;
    /// # Ok(()) }
    /// ```
    pub fn create(self, model_url: &str) -> Result<Box<dyn VideoClient>, RathError> {
        crate::providers::create_video_client(&ModelUrl::parse(model_url)?, self)
    }
}

#[cfg(test)]
#[path = "tests/client.rs"]
mod tests;
/// Video submission, polling, and verified callbacks without caller-state ownership.
#[async_trait]
pub trait VideoClient: Send + Sync {
    /// Submits once or returns an error. A lost response does not prove no job was created.
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError>;
    /// Reads a job using the same provider/model/account as its submission.
    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError>;
    /// Polls without an overall deadline. Dropping the future does not cancel remote work.
    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError>;
    /// Submits once and waits; errors or cancellation never resubmit generation.
    async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, RathError> {
        let job = self.submit_video(request).await?;
        self.wait_video(&job.id).await
    }
    /// Verifies original HTTP bytes before decoding, or fails explicitly when unsupported.
    /// Callers authorize the job, durably accept/deduplicate it, and acknowledge delivery.
    /// Verification may fetch provider keys, but never fetches the generated video.
    async fn parse_webhook(
        &self,
        _headers: &http::HeaderMap,
        _body: &[u8],
    ) -> Result<VideoEvent, RathError> {
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "video webhooks are unsupported",
        ))
    }
}
