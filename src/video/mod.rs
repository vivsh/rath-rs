use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

use crate::core::Provider;
use crate::core::RathError;

/// Options used when constructing a video client.
#[derive(Debug, Clone)]
pub struct VideoOptions {
    pub provider_config: Option<Value>,
    pub poll_interval: Duration,
}

impl VideoOptions {
    /// Builds a provider client for the given model URL.
    pub fn create(self, model_url: &str) -> Result<Box<dyn VideoClient>, RathError> {
        let url = crate::core::ModelUrl::parse(model_url)?;
        crate::providers::create_video_client(&url, self)
    }
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            provider_config: None,
            poll_interval: Duration::from_secs(5),
        }
    }
}

/// Video generation request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub image_url: Option<String>,
    pub provider_config: Option<Value>,
}

/// Video returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoData {
    Url { url: String },
    Base64 { mime_type: String, data: String },
}

/// Video generation response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoResponse {
    pub videos: Vec<VideoData>,
    pub raw_metadata: Option<Value>,
}

/// Submitted video job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoJob {
    pub id: String,
    pub provider: Provider,
    pub provider_model: Option<String>,
    pub status_url: Option<String>,
    pub response_url: Option<String>,
    pub cancel_url: Option<String>,
    pub raw_metadata: Option<Value>,
}

/// Current state of a submitted video job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VideoJobStatus {
    Queued {
        queue_position: Option<u64>,
        raw_metadata: Option<Value>,
    },
    Running {
        raw_metadata: Option<Value>,
    },
    Succeeded {
        response: VideoResponse,
    },
    Failed {
        message: String,
        raw_metadata: Option<Value>,
    },
}

/// Provider-agnostic video client.
#[async_trait]
pub trait VideoClient: Send + Sync {
    /// Submits a video job and returns immediately.
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError>;

    /// Returns the current state of a previously submitted video job.
    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError>;

    /// Polls a submitted job until it completes or fails.
    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError>;

    /// Convenience API for callers that want a single await.
    async fn generate_video(&self, request: &VideoRequest) -> Result<VideoResponse, RathError> {
        let job = self.submit_video(request).await?;
        self.wait_video(&job.id).await
    }
}
