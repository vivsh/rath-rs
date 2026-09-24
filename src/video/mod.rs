//! URL-routed asynchronous video generation and authenticated completion events.
mod client;
mod types;
pub use client::VideoOptions;
mod backend;
pub use backend::{VideoBackend, VideoClient};
pub use types::{
    VideoData, VideoEvent, VideoInput, VideoJob, VideoJobStatus, VideoRequest, VideoResponse,
};
