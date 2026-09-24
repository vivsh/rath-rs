//! Decorators forward all capabilities without additional state or async macros.
#[allow(dead_code)]
#[path = "../examples/factory_decorator.rs"]
mod decorator;
use rath::audio::stt::{SttClient, SttOptions};
use rath::audio::tts::{TtsClient, TtsOptions};
use rath::audio::voice_clone::{VoiceCloneClient, VoiceCloneOptions};
use rath::audio::voice_design::{VoiceDesignClient, VoiceDesignOptions};
use rath::embeddings::{EmbeddingClient, EmbeddingOptions};
use rath::images::{ImageClient, ImageOptions};
use rath::llm::{LlmClient, LlmOptions};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::video::{VideoClient, VideoOptions};
use rath::{ErrorKind, ModelUrl, RathError};
struct Probe;
impl ProviderFactory for Probe {
    async fn llm(&self, url: &ModelUrl, options: LlmOptions) -> Result<LlmClient, RathError> {
        assert_eq!(options.name.as_deref(), Some("decorated"));
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded llm",
        ))
    }
    async fn embeddings(
        &self,
        url: &ModelUrl,
        options: EmbeddingOptions,
    ) -> Result<EmbeddingClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded embeddings",
        ))
    }
    async fn image(&self, url: &ModelUrl, options: ImageOptions) -> Result<ImageClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded image",
        ))
    }
    async fn video(&self, url: &ModelUrl, options: VideoOptions) -> Result<VideoClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded video",
        ))
    }
    async fn tts(&self, url: &ModelUrl, options: TtsOptions) -> Result<TtsClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded tts",
        ))
    }
    async fn stt(&self, url: &ModelUrl, options: SttOptions) -> Result<SttClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded stt",
        ))
    }
    async fn voice_design(
        &self,
        url: &ModelUrl,
        options: VoiceDesignOptions,
    ) -> Result<VoiceDesignClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded voice_design",
        ))
    }
    async fn voice_clone(
        &self,
        url: &ModelUrl,
        options: VoiceCloneOptions,
    ) -> Result<VoiceCloneClient, RathError> {
        assert_eq!(url.model(), "model");
        assert_eq!(
            options.provider_config,
            Some(serde_json::json!({"custom":1}))
        );
        Err(RathError::new(
            ErrorKind::UnsupportedCapability,
            "forwarded voice_clone",
        ))
    }
}
/// Registry dispatch reaches the decorated built-in factory for all eight capabilities.
#[tokio::test]
async fn forwards_all_capabilities() {
    let factory = ProviderRegistry::with_builtin_factory(decorator::NamedFactory::new(Probe));
    let url = "ollama:///model";
    let mut options = LlmOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.llm(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded llm");
    let mut options = EmbeddingOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.embeddings(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded embeddings");
    let mut options = ImageOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.image(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded image");
    let mut options = VideoOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.video(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded video");
    let mut options = TtsOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.tts(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded tts");
    let mut options = SttOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.stt(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded stt");
    let mut options = VoiceDesignOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.voice_design(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded voice_design");
    let mut options = VoiceCloneOptions::default();
    options.provider_config = Some(serde_json::json!({"custom":1}));
    let error = factory.voice_clone(url, options).await.err().unwrap();
    assert_eq!(error.message(), "forwarded voice_clone");
}

/// An injected decorator forwards to real built-in construction without changing Options::create.
#[tokio::test]
async fn builtin_decorator_is_local_to_its_registry() {
    let registry = ProviderRegistry::with_builtin_factory(decorator::NamedFactory::new(
        rath::registry::BuiltinProviderFactory,
    ));
    let raw = "ollama:///model?thinking=off&base_url=http://127.0.0.1:1";
    let client = registry.llm(raw, LlmOptions::default()).await.unwrap();
    assert_eq!(client.options().name.as_deref(), Some("decorated"));
    assert_eq!(
        client.options().thinking,
        Some(rath::llm::ThinkingLevel::Off)
    );
    let direct = LlmOptions::default().create(raw).await.unwrap();
    assert_eq!(direct.options().name, None);
    let _embeddings = registry
        .embeddings(
            "ollama:///model?base_url=http://127.0.0.1:1",
            EmbeddingOptions::default(),
        )
        .await
        .unwrap();
}
