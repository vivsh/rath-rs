//! URL-routed asynchronous video generation and authenticated completion events.
mod client;
mod types;
pub use client::{VideoClient, VideoOptions};
pub use types::{
    VideoData, VideoEvent, VideoInput, VideoJob, VideoJobStatus, VideoRequest, VideoResponse,
};
