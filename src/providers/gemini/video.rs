//! Veo long-running operations with immutable client configuration and optional verified callbacks.
use super::media;
use crate::core::error::{http, provider_failure_message};
use crate::video::{
    VideoBackend, VideoClient, VideoEvent, VideoJob, VideoJobStatus, VideoOptions, VideoRequest,
    VideoResponse,
};
use crate::{ModelUrl, Provider, RathError};
mod request;
mod response;
mod webhook;

struct GeminiVideoBackend {
    http: reqwest::Client,
    url: ModelUrl,
    options: VideoOptions,
}

/// Prepares an immutable Veo adapter without generating video or registering callbacks.
pub(crate) fn new_video_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<VideoClient, RathError> {
    request::model(&url.model)?;
    request::config(options.provider_config.as_ref(), None)?;
    webhook::configured_env(&options)?;
    if options.poll_interval.is_zero() {
        return Err(media::invalid("video poll_interval must be positive"));
    }
    Ok(VideoClient::from_backend(GeminiVideoBackend {
        http: media::transport()?,
        url: media::resolved_url(url)?,
        options,
    }))
}

impl VideoBackend for GeminiVideoBackend {
    async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
        let key = self.url.api_key.as_deref().unwrap_or_default();
        let payload = request::payload(&self.url, &self.options, request).map_err(|error| {
            error
                .with_context(Provider::Gemini, "video submit")
                .sanitized(&[key])
        })?;
        http::mapped(
            self.http
                .post(request::submit_url(&self.url))
                .header("x-goog-api-key", key)
                .json(&payload),
            Provider::Gemini,
            "video submit",
            &[key],
            |raw| response::job(&self.url, raw),
        )
        .await
    }

    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        let key = self.url.api_key.as_deref().unwrap_or_default();
        let endpoint = response::operation_url(&self.url, job_id).map_err(|error| {
            error
                .with_context(Provider::Gemini, "video status")
                .sanitized(&[key])
        })?;
        let reply = http::send(
            self.http.get(endpoint).header("x-goog-api-key", key),
            Provider::Gemini,
            "video status",
            &[key],
        )
        .await?;
        http::mapped_response(reply, Provider::Gemini, "video status", &[key], |raw| {
            response::status(&self.url, job_id, raw)
        })
        .await
    }

    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        loop {
            match self.get_video(job_id).await? {
                VideoJobStatus::Succeeded { response } => return Ok(response),
                VideoJobStatus::Failed {
                    message,
                    raw_metadata,
                } => {
                    return Err(provider_failure_message(
                        Provider::Gemini,
                        "video status",
                        raw_metadata.as_ref().unwrap_or(&serde_json::Value::Null),
                        &[self.url.api_key.as_deref().unwrap_or_default()],
                        &message,
                    ));
                }
                VideoJobStatus::Queued { .. } | VideoJobStatus::Running { .. } => {
                    tokio::time::sleep(self.options.poll_interval).await;
                }
            }
        }
    }

    async fn parse_webhook(
        &self,
        headers: &::http::HeaderMap,
        body: &[u8],
    ) -> Result<VideoEvent, RathError> {
        webhook::parse(&self.url, &self.options, headers, body).map_err(|error| {
            error
                .with_context(Provider::Gemini, "video webhook")
                .sanitized(&[self.url.api_key.as_deref().unwrap_or_default()])
        })
    }
}

#[cfg(test)]
#[path = "tests/video.rs"]
mod tests;
