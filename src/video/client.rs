use crate::core::RathError;
use serde_json::Value;
use std::time::Duration;

/// Immutable provider configuration; no jobs or callback state are retained.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    ///     .create("fal:///fal-ai/kling-video/v3/pro/image-to-video").await?;
    /// let request = VideoRequest::new("The character waves.".into(), VideoInput::ImageToVideo {
    ///     start_image: ImageData::Url { url: "https://example.com/shot.png".into() },
    ///     end_image: None,
    /// });
    /// let job = client.submit_video(&request).await?;
    /// let output = client.wait_video(&job.id).await?;
    /// # Ok(()) }
    /// ```
    pub async fn create(self, model_url: &str) -> Result<VideoClient, RathError> {
        crate::registry::ProviderRegistry::with_builtins()
            .video(model_url, self)
            .await
    }
}

#[cfg(test)]
#[path = "tests/client.rs"]
mod tests;
/// Video submission, polling, and verified callbacks without caller-state ownership.
use super::VideoClient;

impl VideoOptions {
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl VideoOptions {
    /// Sets the interval between remote status polls.
    pub fn with_poll_interval(mut self, value: Duration) -> Self {
        self.poll_interval = value;
        self
    }
}
