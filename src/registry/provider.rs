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

/// Constructs independent model-bound clients. Override only supported capabilities.
pub trait ProviderFactory: Send + Sync {
    /// Constructs llm or reports an unsupported capability.
    fn llm(
        &self,
        url: &ModelUrl,
        _options: LlmOptions,
    ) -> impl Future<Output = Result<LlmClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "llm")) }
    }
    /// Constructs embeddings or reports an unsupported capability.
    fn embeddings(
        &self,
        url: &ModelUrl,
        _options: EmbeddingOptions,
    ) -> impl Future<Output = Result<EmbeddingClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "embeddings")) }
    }
    /// Constructs image or reports an unsupported capability.
    fn image(
        &self,
        url: &ModelUrl,
        _options: ImageOptions,
    ) -> impl Future<Output = Result<ImageClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "image")) }
    }
    /// Constructs video or reports an unsupported capability.
    fn video(
        &self,
        url: &ModelUrl,
        _options: VideoOptions,
    ) -> impl Future<Output = Result<VideoClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "video")) }
    }
    /// Constructs tts or reports an unsupported capability.
    fn tts(
        &self,
        url: &ModelUrl,
        _options: TtsOptions,
    ) -> impl Future<Output = Result<TtsClient, RathError>> + Send {
        async move {
            Err(RathError::unsupported(
                url.provider.clone(),
                "text-to-speech",
            ))
        }
    }
    /// Constructs stt or reports an unsupported capability.
    fn stt(
        &self,
        url: &ModelUrl,
        _options: SttOptions,
    ) -> impl Future<Output = Result<SttClient, RathError>> + Send {
        async move {
            Err(RathError::unsupported(
                url.provider.clone(),
                "speech-to-text",
            ))
        }
    }
    /// Constructs voice design or reports an unsupported capability.
    fn voice_design(
        &self,
        url: &ModelUrl,
        _options: VoiceDesignOptions,
    ) -> impl Future<Output = Result<VoiceDesignClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "voice_design")) }
    }
    /// Constructs voice clone or reports an unsupported capability.
    fn voice_clone(
        &self,
        url: &ModelUrl,
        _options: VoiceCloneOptions,
    ) -> impl Future<Output = Result<VoiceCloneClient, RathError>> + Send {
        async move { Err(RathError::unsupported(url.provider.clone(), "voice_clone")) }
    }
}

/// Stateless routing to Rath's built-in adapters.
pub struct BuiltinProviderFactory;
impl ProviderFactory for BuiltinProviderFactory {
    async fn llm(&self, url: &ModelUrl, options: LlmOptions) -> Result<LlmClient, RathError> {
        let options = super::prepare::llm(url, options)?;
        crate::providers::create_llm_client(url, options)
    }
    async fn embeddings(
        &self,
        url: &ModelUrl,
        options: EmbeddingOptions,
    ) -> Result<EmbeddingClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        if let Some(setup) = &options.fastembed {
            setup.validate()?;
        }
        if url.provider == crate::Provider::FastEmbed {
            return crate::providers::local_embeddings(url, options).await;
        }
        crate::providers::create_embedding_client(url, options)
    }
    async fn image(&self, url: &ModelUrl, options: ImageOptions) -> Result<ImageClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_image_client(url, options)
    }
    async fn video(&self, url: &ModelUrl, options: VideoOptions) -> Result<VideoClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_video_client(url, options)
    }
    async fn tts(&self, url: &ModelUrl, options: TtsOptions) -> Result<TtsClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_tts_client(url, options)
    }
    async fn stt(&self, url: &ModelUrl, options: SttOptions) -> Result<SttClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_stt_client(url, options)
    }
    async fn voice_design(
        &self,
        url: &ModelUrl,
        options: VoiceDesignOptions,
    ) -> Result<VoiceDesignClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_voice_design_client(url, options)
    }
    async fn voice_clone(
        &self,
        url: &ModelUrl,
        options: VoiceCloneOptions,
    ) -> Result<VoiceCloneClient, RathError> {
        super::url::validate_controls(url, false)?;
        crate::core::url::validate_endpoint(url)?;
        super::prepare::native(options.provider_config.as_ref())?;
        crate::providers::create_voice_clone_client(url, options)
    }
}
