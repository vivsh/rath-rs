//! Single-speaker Gemini speech generation with client-owned immutable configuration.
use super::media;
use crate::audio::tts::{TtsBackend, TtsClient, TtsOptions, TtsRequest, TtsResponse};
use crate::core::error::http;
use crate::{ModelUrl, Provider, RathError};
mod request;
mod response;

struct GeminiTtsBackend {
    http: reqwest::Client,
    url: ModelUrl,
    options: TtsOptions,
}

/// Prepares a verified speech-model adapter without generating audio or creating remote resources.
pub(crate) fn new_tts_client(url: &ModelUrl, options: TtsOptions) -> Result<TtsClient, RathError> {
    request::model(&url.model)?;
    request::config(options.provider_config.as_ref(), None)?;
    Ok(TtsClient::from_backend(GeminiTtsBackend {
        http: media::transport()?,
        url: media::resolved_url(url)?,
        options,
    }))
}

impl TtsBackend for GeminiTtsBackend {
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let key = self.url.api_key.as_deref().unwrap_or_default();
        let payload = request::payload(&self.url, &self.options, request).map_err(|e| {
            e.with_context(Provider::Gemini, "text-to-speech")
                .sanitized(&[key])
        })?;
        http::mapped(
            self.http
                .post(media::endpoint(&self.url))
                .header("x-goog-api-key", key)
                .json(&payload),
            Provider::Gemini,
            "text-to-speech",
            &[key],
            |raw| response::decode(raw, request.format.as_deref() == Some("pcm")),
        )
        .await
    }
}

#[cfg(test)]
#[path = "tests/tts.rs"]
mod tests;
