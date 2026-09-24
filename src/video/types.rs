use crate::core::Provider;
use crate::images::ImageData;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Explicit operation inputs; typed variants require compatible endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VideoInput {
    /// Generate a shot from the required request prompt.
    #[default]
    TextToVideo,
    /// Animate a starting composition, optionally ending at a specified frame.
    ImageToVideo {
        /// Opening frame, not merely a character identity reference.
        start_image: ImageData,
        /// Closing frame when supported by the endpoint.
        end_image: Option<ImageData>,
    },
    /// Transfer a driving performance to a character image.
    MotionTransfer {
        /// Character appearance and composition.
        character_image: ImageData,
        /// Recording supplying movement/performance.
        driving_video: VideoData,
    },
    /// Native input without portable model validation or field mapping.
    Native {
        /// Object payload overriding client defaults and request configuration.
        payload: Value,
    },
}

/// Caller-owned generation request; Rath does not upload files or retain inputs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct VideoRequest {
    /// Required for typed T2V/I2V, optional for motion, and empty in Native mode.
    pub prompt: String,
    /// Generation operation and its media.
    pub input: VideoInput,
    /// Optional HTTPS completion callback, transported outside model input.
    pub webhook_url: Option<String>,
    /// Object-shaped native controls overriding client defaults.
    pub provider_config: Option<Value>,
}

/// Encoded video or remote location, usable as input or caller-owned output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VideoData {
    /// HTTP(S) location; input media is not downloaded by Rath.
    Url {
        /// Location of encoded video.
        url: String,
    },
    /// Complete encoded video in standard base64.
    Base64 {
        /// Parameter-free video MIME type.
        mime_type: String,
        /// Standard base64 without a data-URI prefix.
        data: String,
    },
}

/// Generated media; downloading and persistence belong to the caller.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct VideoResponse {
    /// Nonempty media for a successful adapter result.
    pub videos: Vec<VideoData>,
    /// Native result/envelope, potentially containing private data and signed URLs.
    pub raw_metadata: Option<Value>,
}

/// Caller-owned submission receipt, not a mutable registry entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VideoJob {
    /// Stable queue identifier.
    pub id: String,
    /// Provider owning the remote job.
    pub provider: Provider,
    /// Native submission endpoint.
    pub provider_model: Option<String>,
    /// Provider-supplied status location.
    pub status_url: Option<String>,
    /// Provider-supplied result location.
    pub response_url: Option<String>,
    /// Cancellation location; no cancellation API is implied.
    pub cancel_url: Option<String>,
    /// Native receipt, potentially containing private data.
    pub raw_metadata: Option<Value>,
}

/// Remote state snapshot; the caller owns transitions and reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VideoJobStatus {
    /// Accepted but not executing.
    Queued {
        /// Provider-reported position when available.
        queue_position: Option<u64>,
        /// Native queue metadata.
        raw_metadata: Option<Value>,
    },
    /// Execution in progress.
    Running {
        /// Native execution metadata.
        raw_metadata: Option<Value>,
    },
    /// Generation completed with usable output.
    Succeeded {
        /// Generated media and provider evidence.
        response: VideoResponse,
    },
    /// Provider-reported failure, not a local polling/verification error.
    Failed {
        /// Failure message, potentially containing private user text.
        message: String,
        /// Native failure evidence.
        raw_metadata: Option<Value>,
    },
}

/// Authenticated notification; authorization and deduplication remain caller-owned.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VideoEvent {
    /// Stable ID to match against the caller's provider/model/account record.
    pub job_id: String,
    /// Same status representation as polling.
    pub status: VideoJobStatus,
}

impl VideoRequest {
    /// Constructs required values; optional controls start unset.
    pub fn new(prompt: String, input: VideoInput) -> Self {
        Self {
            prompt,
            input,
            webhook_url: None,
            provider_config: None,
        }
    }
    /// Sets `webhook_url` for this value.
    pub fn with_webhook_url(mut self, value: String) -> Self {
        self.webhook_url = Some(value);
        self
    }
    /// Sets `provider_config` for this value.
    pub fn with_provider_config(mut self, value: Value) -> Self {
        self.provider_config = Some(value);
        self
    }
}

impl VideoResponse {
    /// Constructs required values; optional controls start unset.
    pub fn new(videos: Vec<VideoData>) -> Self {
        Self {
            videos,
            raw_metadata: None,
        }
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}

impl VideoJob {
    /// Constructs required values; optional controls start unset.
    pub fn new(id: String, provider: Provider) -> Self {
        Self {
            id,
            provider,
            provider_model: None,
            status_url: None,
            response_url: None,
            cancel_url: None,
            raw_metadata: None,
        }
    }
    /// Sets `provider_model` for this value.
    pub fn with_provider_model(mut self, value: String) -> Self {
        self.provider_model = Some(value);
        self
    }
    /// Sets `status_url` for this value.
    pub fn with_status_url(mut self, value: String) -> Self {
        self.status_url = Some(value);
        self
    }
    /// Sets `response_url` for this value.
    pub fn with_response_url(mut self, value: String) -> Self {
        self.response_url = Some(value);
        self
    }
    /// Sets `cancel_url` for this value.
    pub fn with_cancel_url(mut self, value: String) -> Self {
        self.cancel_url = Some(value);
        self
    }
    /// Sets `raw_metadata` for this value.
    pub fn with_raw_metadata(mut self, value: Value) -> Self {
        self.raw_metadata = Some(value);
        self
    }
}

impl VideoEvent {
    /// Constructs required values; optional controls start unset.
    pub fn new(job_id: String, status: VideoJobStatus) -> Self {
        Self { job_id, status }
    }
}
