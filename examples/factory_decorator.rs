//! Transparent generic factory composition: explicitly forward every capability.
use rath::audio::stt::{SttClient, SttOptions};
use rath::audio::tts::{TtsClient, TtsOptions};
use rath::audio::voice_clone::{VoiceCloneClient, VoiceCloneOptions};
use rath::audio::voice_design::{VoiceDesignClient, VoiceDesignOptions};
use rath::embeddings::{EmbeddingClient, EmbeddingOptions};
use rath::images::{ImageClient, ImageOptions};
use rath::llm::{LlmClient, LlmOptions};
use rath::registry::ProviderFactory;
use rath::video::{VideoClient, VideoOptions};
use rath::{ModelUrl, RathError};

/// Adds a diagnostic LLM name while preserving all other construction behavior.
pub struct NamedFactory<F> {
    inner: F,
}
impl<F> NamedFactory<F> {
    /// Takes ownership of the sole retained dependency.
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}
impl<F: ProviderFactory> ProviderFactory for NamedFactory<F> {
    async fn llm(&self, url: &ModelUrl, options: LlmOptions) -> Result<LlmClient, RathError> {
        let options = options.with_name("decorated");
        self.inner.llm(url, options).await
    }
    async fn embeddings(
        &self,
        url: &ModelUrl,
        options: EmbeddingOptions,
    ) -> Result<EmbeddingClient, RathError> {
        self.inner.embeddings(url, options).await
    }
    async fn image(&self, url: &ModelUrl, options: ImageOptions) -> Result<ImageClient, RathError> {
        self.inner.image(url, options).await
    }
    async fn video(&self, url: &ModelUrl, options: VideoOptions) -> Result<VideoClient, RathError> {
        self.inner.video(url, options).await
    }
    async fn tts(&self, url: &ModelUrl, options: TtsOptions) -> Result<TtsClient, RathError> {
        self.inner.tts(url, options).await
    }
    async fn stt(&self, url: &ModelUrl, options: SttOptions) -> Result<SttClient, RathError> {
        self.inner.stt(url, options).await
    }
    async fn voice_design(
        &self,
        url: &ModelUrl,
        options: VoiceDesignOptions,
    ) -> Result<VoiceDesignClient, RathError> {
        self.inner.voice_design(url, options).await
    }
    async fn voice_clone(
        &self,
        url: &ModelUrl,
        options: VoiceCloneOptions,
    ) -> Result<VoiceCloneClient, RathError> {
        self.inner.voice_clone(url, options).await
    }
}
fn main() {}
