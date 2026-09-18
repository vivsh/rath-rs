use super::super::*;

/// Parses a minimal gemini URL with empty authority; base_url is None.
#[test]
fn parse_gemini_empty_authority() {
    let url = ModelUrl::parse("gemini:///gemini-2.5-flash-lite").unwrap();
    assert_eq!(url.provider, Provider::Gemini);
    assert_eq!(url.model, "gemini-2.5-flash-lite");
    assert!(url.base_url.is_none());
    assert!(url.api_key.is_none());
    assert!(url.temperature.is_none());
    assert!(url.thinking.is_none());
}

/// Custom endpoints use base_url instead of authority.
#[test]
fn parse_ollama_with_base_url() {
    let url = ModelUrl::parse("ollama:///qwen3:8b?base_url=http://localhost:11434").unwrap();
    assert_eq!(url.provider, Provider::Ollama);
    assert_eq!(url.model, "qwen3:8b");
    assert_eq!(url.base_url.as_deref(), Some("http://localhost:11434"));
    assert!(url.api_key.is_none());
}

/// Slash-containing model identifiers are preserved for every provider.
#[test]
fn parse_openai_preserves_slash_model_id() {
    let url =
        ModelUrl::parse("openai:///models/gpt-4o?base_url=https://api.example.com/v1").unwrap();
    assert_eq!(url.provider, Provider::OpenAi);
    assert_eq!(url.model, "models/gpt-4o");
    assert_eq!(url.base_url.as_deref(), Some("https://api.example.com/v1"));
}

#[test]
fn parse_openrouter_preserves_prefixed_model_slug() {
    let url = ModelUrl::parse("openrouter:///openai/gpt-5.2").unwrap();
    assert_eq!(url.provider, Provider::OpenRouter);
    assert_eq!(url.model, "openai/gpt-5.2");
    assert!(url.base_url.is_none());
}

#[test]
fn parse_fal_preserves_model_path() {
    let url = ModelUrl::parse("fal:///fal-ai/flux/schnell").unwrap();
    assert_eq!(url.provider, Provider::Fal);
    assert_eq!(url.model, "fal-ai/flux/schnell");
    assert!(url.base_url.is_none());
}

/// Temperature and thinking are extracted from query params.
#[test]
fn parse_query_params() {
    let url =
        ModelUrl::parse("gemini:///gemini-2.5-flash?temperature=0.7&thinking=medium").unwrap();
    assert_eq!(url.temperature, Some(0.7));
    assert_eq!(url.thinking, Some(ThinkingLevel::Medium));
}

/// api_key_env is resolved from the environment at parse time.
#[test]
fn parse_api_key_env() {
    let expected = std::env::var("PATH").unwrap();
    let url = ModelUrl::parse("openai:///gpt-4o?api_key_env=PATH").unwrap();
    assert_eq!(url.api_key.as_deref(), Some(expected.as_str()));
}

/// anthropic and claude schemes both map to Provider::Anthropic.
#[test]
fn parse_anthropic_aliases() {
    let a = ModelUrl::parse("anthropic:///claude-sonnet-4-5").unwrap();
    let b = ModelUrl::parse("claude:///claude-sonnet-4-5").unwrap();
    assert_eq!(a.provider, Provider::Anthropic);
    assert_eq!(b.provider, Provider::Anthropic);
}

/// Explicit custom transport syntax is rejected; use base_url instead.
#[test]
fn reject_explicit_transport() {
    assert!(matches!(
        ModelUrl::parse("ollama+https:///llama3"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Non-empty authority is rejected to avoid host/path/model ambiguity.
#[test]
fn reject_non_empty_authority() {
    assert!(matches!(
        ModelUrl::parse("ollama://localhost:11434/qwen3:8b"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Inline credentials are rejected.
#[test]
fn reject_inline_credentials() {
    assert!(matches!(
        ModelUrl::parse("gemini://mykey@gemini-2.5-flash"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Fragment is rejected.
#[test]
fn reject_fragment() {
    assert!(matches!(
        ModelUrl::parse("gemini:///model#section"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// base_url is supported for custom provider endpoints.
#[test]
fn parse_base_url_query_param() {
    let url = ModelUrl::parse("openai:///gpt-4o?base_url=https://api.example.com/v1/").unwrap();
    assert_eq!(url.base_url.as_deref(), Some("https://api.example.com/v1"));
}

/// Unknown query parameters are rejected.
#[test]
fn reject_unknown_query_param() {
    assert!(matches!(
        ModelUrl::parse("openai:///gpt-4o?unknown=value"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Duplicate query parameters are rejected.
#[test]
fn reject_duplicate_query_param() {
    assert!(matches!(
        ModelUrl::parse("gemini:///model?temperature=0.5&temperature=0.7"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Temperature outside [0.0, 1.0] is rejected.
#[test]
fn reject_temperature_out_of_range() {
    assert!(matches!(
        ModelUrl::parse("gemini:///model?temperature=1.5"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Unknown provider scheme is rejected.
#[test]
fn reject_unknown_provider() {
    assert!(matches!(
        ModelUrl::parse("unknown:///model"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Missing model name (empty path) is rejected.
#[test]
fn reject_missing_model() {
    assert!(matches!(
        ModelUrl::parse("gemini:///"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Missing '://' is rejected.
#[test]
fn reject_missing_scheme_separator() {
    assert!(matches!(
        ModelUrl::parse("gemini-2.5-flash-lite"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Unset api_key_env variable is rejected at parse time.
#[test]
fn reject_missing_api_key_env() {
    assert!(matches!(
        ModelUrl::parse("openai:///gpt-4o?api_key_env=__PRAVAH_MISSING_ENV__"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// cache=5m parses to Ephemeral5m.
#[test]
fn parse_cache_5m() {
    let url = ModelUrl::parse("anthropic:///claude-sonnet-4-5?cache=5m").unwrap();
    assert_eq!(url.cache, Some(CacheControl::Ephemeral5m));
}

/// cache=1h parses to Ephemeral1h.
#[test]
fn parse_cache_1h() {
    let url = ModelUrl::parse("anthropic:///claude-sonnet-4-5?cache=1h").unwrap();
    assert_eq!(url.cache, Some(CacheControl::Ephemeral1h));
}

/// Unknown cache value is rejected.
#[test]
fn reject_unknown_cache_value() {
    assert!(matches!(
        ModelUrl::parse("anthropic:///claude-sonnet-4-5?cache=30s"),
        Err(error) if error.kind() == crate::core::ErrorKind::InvalidUrl
    ));
}

/// Absent cache param leaves cache as None.
#[test]
fn no_cache_param_is_none() {
    let url = ModelUrl::parse("anthropic:///claude-sonnet-4-5").unwrap();
    assert_eq!(url.cache, None);
}
