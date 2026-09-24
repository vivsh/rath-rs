use super::{BuiltinProviderFactory, ProviderFactory, erased::ErasedProviderFactory, url};
use crate::audio::stt::{SttClient, SttOptions};
use crate::audio::tts::{TtsClient, TtsOptions};
use crate::audio::voice_clone::{VoiceCloneClient, VoiceCloneOptions};
use crate::audio::voice_design::{VoiceDesignClient, VoiceDesignOptions};
use crate::core::builtin;
use crate::embeddings::{EmbeddingClient, EmbeddingOptions};
use crate::images::{ImageClient, ImageOptions};
use crate::llm::{LlmClient, LlmOptions};
use crate::video::{VideoClient, VideoOptions};
use crate::{ErrorKind, ModelUrl, RathError};
use std::collections::BTreeMap;
/// Application-owned factories. Returned clients own their execution dependencies.
pub struct ProviderRegistry {
    builtin_factory: Option<Box<dyn ErasedProviderFactory>>,
    providers: BTreeMap<String, Box<dyn ErasedProviderFactory>>,
}
impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl ProviderRegistry {
    /// Starts with external registrations only.
    pub fn new() -> Self {
        Self {
            builtin_factory: None,
            providers: BTreeMap::new(),
        }
    }
    /// Enables built-in routing alongside external registrations.
    pub fn with_builtins() -> Self {
        Self::with_builtin_factory(BuiltinProviderFactory)
    }
    /// Routes all built-in provider schemes through the supplied factory without I/O.
    ///
    /// URLs are validated and LLM URL controls take precedence over options before dispatch.
    /// The registry owns the factory; returned clients must own their dependencies.
    /// Unsupported capabilities are returned without falling back to Rath's built-ins.
    /// Decorators must explicitly forward capabilities they do not replace.
    /// External registrations and `Options::create` are unaffected.
    pub fn with_builtin_factory<P: ProviderFactory + 'static>(factory: P) -> Self {
        Self {
            builtin_factory: Some(Box::new(factory)),
            ..Self::new()
        }
    }
    /// Registers a factory without I/O; names are validated and never replaced.
    pub fn register<P: ProviderFactory + 'static>(
        mut self,
        scheme: &str,
        provider: P,
    ) -> Result<Self, RathError> {
        if !url::valid_scheme(scheme) {
            return Err(RathError::new(
                ErrorKind::Validation,
                "provider scheme must match [a-z][a-z0-9.-]*",
            ));
        }
        if builtin(scheme).is_some() || self.providers.contains_key(scheme) {
            return Err(RathError::new(
                ErrorKind::Validation,
                format!("provider scheme '{scheme}' is reserved or already registered"),
            ));
        }
        self.providers.insert(scheme.into(), Box::new(provider));
        Ok(self)
    }

    /// Constructs llm using exactly one selected factory.
    pub async fn llm(&self, raw: &str, options: LlmOptions) -> Result<LlmClient, RathError> {
        let url = url::parse_for(raw, true)?;
        let options = super::prepare::llm(&url, options)?;
        self.factory(&url)?
            .llm(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs embeddings using exactly one selected factory.
    pub async fn embeddings(
        &self,
        raw: &str,
        options: EmbeddingOptions,
    ) -> Result<EmbeddingClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .embeddings(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs image using exactly one selected factory.
    pub async fn image(&self, raw: &str, options: ImageOptions) -> Result<ImageClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .image(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs video using exactly one selected factory.
    pub async fn video(&self, raw: &str, options: VideoOptions) -> Result<VideoClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .video(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs tts using exactly one selected factory.
    pub async fn tts(&self, raw: &str, options: TtsOptions) -> Result<TtsClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .tts(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs stt using exactly one selected factory.
    pub async fn stt(&self, raw: &str, options: SttOptions) -> Result<SttClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .stt(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs voice design using exactly one selected factory.
    pub async fn voice_design(
        &self,
        raw: &str,
        options: VoiceDesignOptions,
    ) -> Result<VoiceDesignClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .voice_design(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    /// Constructs voice clone using exactly one selected factory.
    pub async fn voice_clone(
        &self,
        raw: &str,
        options: VoiceCloneOptions,
    ) -> Result<VoiceCloneClient, RathError> {
        let url = url::parse_for(raw, false)?;
        super::prepare::native(options.provider_config.as_ref())?;
        self.factory(&url)?
            .voice_clone(&url, options)
            .await
            .map_err(|e| construction_error(e, &url))
    }
    fn factory(&self, url: &ModelUrl) -> Result<&dyn ErasedProviderFactory, RathError> {
        let factory = if builtin(url.provider.as_str()).is_some() {
            self.builtin_factory.as_deref()
        } else {
            self.providers
                .get(url.provider.as_str())
                .map(|f| f.as_ref())
        };
        factory.ok_or_else(|| {
            RathError::new(
                ErrorKind::UnsupportedCapability,
                format!(
                    "provider '{}' is not enabled or registered",
                    url.provider.as_str()
                ),
            )
            .with_context(url.provider.clone(), "client construction")
        })
    }
}
fn construction_error(error: RathError, url: &ModelUrl) -> RathError {
    error
        .with_context(url.provider.clone(), "client construction")
        .sanitized(&url.api_key.as_deref().into_iter().collect::<Vec<_>>())
}
#[cfg(test)]
#[path = "tests/routing.rs"]
mod tests;
