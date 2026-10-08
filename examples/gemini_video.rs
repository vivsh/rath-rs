//! Veo generation and optional registered-webhook parsing through existing Rath video contracts.
//! Running this example only constructs a client; generation requires an explicit function call.
use rath::RathError;
use rath::video::{
    VideoClient, VideoEvent, VideoInput, VideoJob, VideoOptions, VideoRequest, VideoResponse,
};
use serde_json::json;

/// Constructs the model-bound adapter without submitting paid work or registering a webhook.
#[tokio::main]
async fn main() -> Result<(), RathError> {
    let _client = configured_client().await?;
    println!("Constructed Gemini/Veo client; no generation requested.");
    Ok(())
}

/// Configures an optional caller-provisioned signing-secret variable and native generation defaults.
pub async fn configured_client() -> Result<VideoClient, RathError> {
    VideoOptions::default()
        .with_provider_config(json!({
            "parameters":{"durationSeconds":4,"resolution":"720p","aspectRatio":"16:9"},
            "webhook_signing_secret_env":"GEMINI_WEBHOOK_SIGNING_SECRET"
        }))
        .create("gemini:///veo-3.1-fast-generate-preview")
        .await
}

/// Submits one paid generation; callers persist the receipt before waiting or accepting callbacks.
pub async fn submit(client: &VideoClient, prompt: String) -> Result<VideoJob, RathError> {
    client
        .submit_video(&VideoRequest::new(prompt, VideoInput::TextToVideo))
        .await
}

/// Waits for a saved operation; returned Google URLs need authenticated, caller-owned downloads.
pub async fn wait(client: &VideoClient, saved_job_id: &str) -> Result<VideoResponse, RathError> {
    client.wait_video(saved_job_id).await
}

/// Verifies registered callbacks; the application durably authorizes/deduplicates before acknowledging.
pub async fn receive(
    client: &VideoClient,
    headers: &http::HeaderMap,
    original_body: &[u8],
) -> Result<VideoEvent, RathError> {
    client.parse_webhook(headers, original_body).await
}
