use super::*;
use crate::Provider;

struct Empty;
impl ProviderFactory for Empty {}

/// Registration rejects duplicates, reserved aliases and malformed names without replacing factories.
#[test]
fn registration_names_are_unambiguous() {
    for name in [
        "",
        "Acme",
        "1acme",
        "a_b",
        "a+b",
        "a/b",
        " a",
        "a ",
        "openai",
        "openrouter",
        "ollama",
        "gemini",
        "fal",
        "anthropic",
        "claude",
        "elevenlabs",
    ] {
        assert_eq!(
            ProviderRegistry::new()
                .register(name, Empty)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Validation
        );
    }
    let registry = ProviderRegistry::new().register("acme", Empty).unwrap();
    assert_eq!(
        registry.register("acme", Empty).err().unwrap().kind(),
        ErrorKind::Validation
    );
    assert!(
        ProviderRegistry::default()
            .register("acme-2.test", Empty)
            .is_ok()
    );
}

/// Every missing external capability is a typed error identifying its provider and operation.
#[tokio::test]
async fn capability_defaults_report_unsupported() {
    let registry = ProviderRegistry::new().register("acme", Empty).unwrap();
    let errors = [
        registry
            .image("acme:///m", ImageOptions::default())
            .await
            .err()
            .unwrap(),
        registry
            .video("acme:///m", VideoOptions::default())
            .await
            .err()
            .unwrap(),
        registry
            .tts("acme:///m", TtsOptions::default())
            .await
            .err()
            .unwrap(),
        registry
            .stt("acme:///m", SttOptions::default())
            .await
            .err()
            .unwrap(),
        registry
            .voice_design("acme:///m", VoiceDesignOptions::default())
            .await
            .err()
            .unwrap(),
        registry
            .voice_clone("acme:///m", VoiceCloneOptions::default())
            .await
            .err()
            .unwrap(),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
        assert_eq!(error.provider(), Some(&Provider::External("acme".into())));
        assert!(error.to_string().contains("unsupported capability"));
    }
}

/// Missing/disabled routes never use a registered alternative or an implicit built-in.
#[tokio::test]
async fn routing_has_no_fallback() {
    let registry = ProviderRegistry::new().register("acme", Empty).unwrap();
    for raw in ["unknown:///m", "openai:///m", "claude:///m"] {
        let error = registry
            .tts(raw, TtsOptions::default())
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
        assert!(error.to_string().contains("not enabled or registered"));
    }
    let shared = std::sync::Arc::new(registry);
    fn send_sync<T: Send + Sync>(_: &T) {}
    send_sync(&shared);
}

/// Construction context wraps rather than flattens errors and redacts locator-resolved credentials.
#[tokio::test]
async fn construction_error_preserves_causes_and_redacts_credentials() {
    struct Broken;
    impl ProviderFactory for Broken {
        async fn tts(&self, url: &ModelUrl, _: TtsOptions) -> Result<TtsClient, RathError> {
            Err(RathError::new(
                ErrorKind::Validation,
                format!("bad setting {}", url.api_key.as_deref().unwrap()),
            )
            .with_context(url.provider.clone(), "model validation")
            .with_source(RathError::new(ErrorKind::Other, "unsupported format")))
        }
    }
    let registry = ProviderRegistry::new().register("acme", Broken).unwrap();
    let error = registry
        .tts("acme:///m?api_key_env=PATH", TtsOptions::default())
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.operation(), Some("client construction"));
    assert_eq!(
        error.source().unwrap().operation(),
        Some("model validation")
    );
    assert_eq!(
        error.source().unwrap().source().unwrap().message(),
        "unsupported format"
    );
    assert!(!format!("{error} {error:?}").contains(&std::env::var("PATH").unwrap()));
    assert!(error.to_string().contains("[REDACTED]"));
}

/// Built-in routing returns exactly the same typed failure as direct legacy construction.
#[tokio::test]
async fn builtin_errors_match_direct_construction() {
    let raw = "claude:///model";
    let direct = TtsOptions::default().create(raw).await.err().unwrap();
    let routed = ProviderRegistry::with_builtins()
        .tts(raw, TtsOptions::default())
        .await
        .err()
        .unwrap();
    assert_eq!(direct.kind(), routed.kind());
    assert_eq!(direct.provider(), routed.provider());
    assert_eq!(direct.to_string(), routed.to_string());
}

/// Passing external identity into internal built-in factories cannot dispatch or select credentials.
#[test]
fn legacy_factories_reject_external_values() {
    let mut url = ModelUrl::parse("fal:///m").unwrap();
    url.provider = Provider::External("acme".into());
    let errors = [
        crate::providers::create_tts_client(&url, TtsOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_image_client(&url, ImageOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_video_client(&url, VideoOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_stt_client(&url, SttOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_voice_clone_client(&url, VoiceCloneOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_voice_design_client(&url, VoiceDesignOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_llm_client(&url, crate::llm::LlmOptions::default())
            .err()
            .unwrap(),
        crate::providers::create_embedding_client(
            &url,
            crate::embeddings::EmbeddingOptions::default(),
        )
        .err()
        .unwrap(),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
        assert_eq!(error.provider(), Some(&url.provider));
    }
}
