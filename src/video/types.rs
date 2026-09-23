use crate::core::Provider;
use crate::images::ImageData;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Explicit operation inputs; typed variants require compatible endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
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
pub struct VideoResponse {
    /// Nonempty media for a successful adapter result.
    pub videos: Vec<VideoData>,
    /// Native result/envelope, potentially containing private data and signed URLs.
    pub raw_metadata: Option<Value>,
}

/// Caller-owned submission receipt, not a mutable registry entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct VideoEvent {
    /// Stable ID to match against the caller's provider/model/account record.
    pub job_id: String,
    /// Same status representation as polling.
    pub status: VideoJobStatus,
}
