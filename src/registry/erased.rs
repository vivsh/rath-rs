use super::ProviderFactory;
use crate::audio::stt::{SttClient, SttOptions};
use crate::audio::tts::{TtsClient, TtsOptions};
use crate::audio::voice_clone::{VoiceCloneClient, VoiceCloneOptions};
use crate::audio::voice_design::{VoiceDesignClient, VoiceDesignOptions};
use crate::embeddings::{EmbeddingClient, EmbeddingOptions};
use crate::images::{ImageClient, ImageOptions};
use crate::llm::{LlmClient, LlmOptions};
use crate::video::{VideoClient, VideoOptions};
use crate::{ModelUrl, RathError};
use std::future::Future;
use std::pin::Pin;
type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub(super) trait ErasedProviderFactory: Send + Sync {
    fn llm<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: LlmOptions,
    ) -> ErasedFuture<'a, Result<LlmClient, RathError>>;
    fn embeddings<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: EmbeddingOptions,
    ) -> ErasedFuture<'a, Result<EmbeddingClient, RathError>>;
    fn image<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: ImageOptions,
    ) -> ErasedFuture<'a, Result<ImageClient, RathError>>;
    fn video<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VideoOptions,
    ) -> ErasedFuture<'a, Result<VideoClient, RathError>>;
    fn tts<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: TtsOptions,
    ) -> ErasedFuture<'a, Result<TtsClient, RathError>>;
    fn stt<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: SttOptions,
    ) -> ErasedFuture<'a, Result<SttClient, RathError>>;
    fn voice_design<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VoiceDesignOptions,
    ) -> ErasedFuture<'a, Result<VoiceDesignClient, RathError>>;
    fn voice_clone<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VoiceCloneOptions,
    ) -> ErasedFuture<'a, Result<VoiceCloneClient, RathError>>;
}
impl<F: ProviderFactory> ErasedProviderFactory for F {
    fn llm<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: LlmOptions,
    ) -> ErasedFuture<'a, Result<LlmClient, RathError>> {
        Box::pin(ProviderFactory::llm(self, url, options))
    }
    fn embeddings<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: EmbeddingOptions,
    ) -> ErasedFuture<'a, Result<EmbeddingClient, RathError>> {
        Box::pin(ProviderFactory::embeddings(self, url, options))
    }
    fn image<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: ImageOptions,
    ) -> ErasedFuture<'a, Result<ImageClient, RathError>> {
        Box::pin(ProviderFactory::image(self, url, options))
    }
    fn video<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VideoOptions,
    ) -> ErasedFuture<'a, Result<VideoClient, RathError>> {
        Box::pin(ProviderFactory::video(self, url, options))
    }
    fn tts<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: TtsOptions,
    ) -> ErasedFuture<'a, Result<TtsClient, RathError>> {
        Box::pin(ProviderFactory::tts(self, url, options))
    }
    fn stt<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: SttOptions,
    ) -> ErasedFuture<'a, Result<SttClient, RathError>> {
        Box::pin(ProviderFactory::stt(self, url, options))
    }
    fn voice_design<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VoiceDesignOptions,
    ) -> ErasedFuture<'a, Result<VoiceDesignClient, RathError>> {
        Box::pin(ProviderFactory::voice_design(self, url, options))
    }
    fn voice_clone<'a>(
        &'a self,
        url: &'a ModelUrl,
        options: VoiceCloneOptions,
    ) -> ErasedFuture<'a, Result<VoiceCloneClient, RathError>> {
        Box::pin(ProviderFactory::voice_clone(self, url, options))
    }
}
