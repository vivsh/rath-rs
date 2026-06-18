use std::collections::HashSet;
use std::str::FromStr;

use super::{Provider, RathError};

/// Reasoning depth requested from the model.
///
/// Maps to a provider-specific token budget or reasoning flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingLevel {
    /// Disable thinking (budget = 0).
    Off,
    /// Minimal reasoning (budget ≈ 512 tokens).
    Low,
    /// Balanced reasoning (budget ≈ 4 096 tokens).
    Medium,
    /// Deep reasoning (budget ≈ 16 384 tokens).
    High,
    /// Maximum reasoning (budget = `i32::MAX`).
    XHigh,
}

impl FromStr for ThinkingLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for ThinkingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        };
        f.write_str(s)
    }
}

/// Prompt caching policy for providers that require explicit opt-in.
///
/// Currently only used by Anthropic. Other providers handle caching
/// automatically and ignore this setting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheControl {
    /// Ephemeral 5-minute cache (default Anthropic TTL).
    Ephemeral5m,
    /// Ephemeral 1-hour cache (billed at 2× base input token price).
    Ephemeral1h,
}

/// Parsed model URL.
///
/// Format: `provider:///provider-native-model-id[?params]`
///
/// - `provider`: `gemini`, `openai`, `openrouter`, `fal`, `anthropic`, `claude`, or `ollama`
/// - URL path: provider-native model id or endpoint slug
/// - Whitelisted query params: `temperature`, `thinking`, `api_key_env`, `cache`, `base_url`
///
/// Rejected: fragments, inline credentials, non-empty authority, unknown/duplicate query params.
#[derive(Debug, Clone)]
pub struct ModelUrl {
    pub provider: Provider,
    /// Provider-native model id or endpoint slug.
    pub model: String,
    /// API key resolved from `api_key_env` at parse time, or `None`.
    pub api_key: Option<String>,
    /// Custom endpoint URL. `None` means use the provider default.
    pub base_url: Option<String>,
    /// Sampling temperature in `[0.0, 1.0]`.
    pub temperature: Option<f32>,
    /// Reasoning depth. `None` means use the provider default.
    pub thinking: Option<ThinkingLevel>,
    /// Prompt caching policy. `None` means no explicit cache control.
    pub cache: Option<CacheControl>,
}

impl ModelUrl {
    /// Parses a model URL.
    ///
    /// Fails on unknown provider, inline credentials, fragment, missing model
    /// name, out-of-range temperature, unknown/duplicate query params, or
    /// unset `api_key_env` variable.
    pub fn parse(s: &str) -> Result<Self, RathError> {
        if s.contains('#') {
            return Err(RathError::InvalidUrl(
                "URL must not contain a fragment".into(),
            ));
        }

        let (scheme_part, rest) = s.split_once("://").ok_or_else(|| {
            RathError::InvalidUrl(format!(
                "missing '://' in '{s}'; expected e.g. gemini:///model-name"
            ))
        })?;

        // Reject inline credentials (user:pass@host) by checking the authority
        let authority_candidate = rest.split('/').next().unwrap_or("");
        if authority_candidate.contains('@') {
            return Err(RathError::InvalidUrl(
                "inline credentials are not allowed; use the api_key_env query parameter".into(),
            ));
        }

        let provider = parse_provider_scheme(scheme_part, s)?;

        let (path_authority, query_str) = match rest.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (rest, None),
        };

        let segments = parse_model_path(path_authority, s)?;

        if segments.is_empty() {
            return Err(RathError::InvalidUrl(format!(
                "'{s}' must contain a model name as the final path segment"
            )));
        }

        let model = segments.join("/");

        let (temperature, thinking, api_key, cache, base_url) = parse_query_str(query_str, s)?;

        Ok(ModelUrl {
            provider,
            model,
            api_key,
            base_url,
            temperature,
            thinking,
            cache,
        })
    }
}

/// Returns `true` when Gemini models below version 3.1 need an exit-tool workaround.
pub(crate) fn gemini_needs_exit_tool(model: &str) -> bool {
    let model = model.strip_prefix("models/").unwrap_or(model);
    let model = model.strip_prefix("gemini-").unwrap_or(model);
    let version = model.split('-').next().unwrap_or(model);
    let mut parts = version.split('.');
    let major: u32 = match parts.next().and_then(|s| s.parse().ok()) {
        Some(n) => n,
        None => return false,
    };
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor) < (3, 1)
}

impl ModelUrl {
    /// Returns `true` when this URL's provider uses an exit-tool strategy to
    /// collect structured output (Ollama always; Gemini before version 3.1).
    pub fn needs_exit_tool(&self) -> bool {
        match self.provider {
            Provider::Ollama => true,
            Provider::Gemini => gemini_needs_exit_tool(&self.model),
            _ => false,
        }
    }
}

fn parse_provider_scheme(scheme: &str, original: &str) -> Result<Provider, RathError> {
    if scheme.contains('+') {
        return Err(RathError::InvalidUrl(format!(
            "custom transports are not supported in '{original}'; use provider:///model?base_url=https://host/path"
        )));
    }

    let provider = match scheme {
        "gemini" => Provider::Gemini,
        "openai" => Provider::OpenAi,
        "openrouter" => Provider::OpenRouter,
        "fal" => Provider::Fal,
        "anthropic" | "claude" => Provider::Anthropic,
        "ollama" => Provider::Ollama,
        other => {
            return Err(RathError::InvalidUrl(format!(
                "unknown provider '{other}' in '{original}'; expected gemini, openai, openrouter, fal, anthropic, claude, or ollama"
            )));
        }
    };

    Ok(provider)
}

fn parse_model_path(path_authority: &str, original: &str) -> Result<Vec<String>, RathError> {
    if path_authority.starts_with('/') {
        let segments: Vec<String> = path_authority
            .split('/')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        return Ok(segments);
    }

    if path_authority.is_empty() {
        return Err(RathError::InvalidUrl(format!(
            "empty authority in '{original}'; use e.g. gemini:///model for no custom endpoint"
        )));
    }

    Err(RathError::InvalidUrl(format!(
        "non-empty authority is not supported in '{original}'; put the provider-native model id after /// and use base_url for custom endpoints"
    )))
}

type ParsedQuery = (
    Option<f32>,
    Option<ThinkingLevel>,
    Option<String>,
    Option<CacheControl>,
    Option<String>,
);

fn parse_query_str(query_str: Option<&str>, original: &str) -> Result<ParsedQuery, RathError> {
    let Some(query) = query_str else {
        return Ok((None, None, None, None, None));
    };

    let mut temperature: Option<f32> = None;
    let mut thinking: Option<ThinkingLevel> = None;
    let mut api_key: Option<String> = None;
    let mut cache: Option<CacheControl> = None;
    let mut base_url: Option<String> = None;
    let mut seen: HashSet<String> = HashSet::new();

    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').ok_or_else(|| {
            RathError::InvalidUrl(format!(
                "query parameter '{pair}' in '{original}' must be key=value"
            ))
        })?;

        if value.is_empty() {
            return Err(RathError::InvalidUrl(format!(
                "query parameter '{key}' must not be empty in '{original}'"
            )));
        }

        if !seen.insert(key.to_string()) {
            return Err(RathError::InvalidUrl(format!(
                "duplicate query parameter '{key}' in '{original}'"
            )));
        }

        match key {
            "temperature" => {
                let t: f32 = value.parse().map_err(|_| {
                    RathError::InvalidUrl(format!(
                        "temperature must be a number in '{original}', got '{value}'"
                    ))
                })?;
                if !(0.0..=1.0).contains(&t) {
                    return Err(RathError::InvalidUrl(format!(
                        "temperature must be 0.0–1.0, got {t} in '{original}'"
                    )));
                }
                temperature = Some(t);
            }
            "thinking" => {
                let level = ThinkingLevel::from_str(value).map_err(|_| {
                    RathError::InvalidUrl(format!(
                        "thinking must be off/low/medium/high/xhigh in '{original}', got '{value}'"
                    ))
                })?;
                thinking = Some(level);
            }
            "api_key_env" => {
                let resolved = std::env::var(value).map_err(|_| {
                    RathError::InvalidUrl(format!(
                        "environment variable '{value}' referenced by api_key_env is not set"
                    ))
                })?;
                api_key = Some(resolved);
            }
            "cache" => {
                cache = Some(match value {
                    "5m" => CacheControl::Ephemeral5m,
                    "1h" => CacheControl::Ephemeral1h,
                    other => {
                        return Err(RathError::InvalidUrl(format!(
                            "cache must be 5m or 1h in '{original}', got '{other}'"
                        )));
                    }
                });
            }
            "base_url" => {
                if !(value.starts_with("https://") || value.starts_with("http://")) {
                    return Err(RathError::InvalidUrl(format!(
                        "base_url must start with http:// or https:// in '{original}'"
                    )));
                }
                if value.contains('#') {
                    return Err(RathError::InvalidUrl(format!(
                        "base_url must not contain a fragment in '{original}'"
                    )));
                }
                base_url = Some(value.trim_end_matches('/').to_string());
            }
            other => {
                return Err(RathError::InvalidUrl(format!(
                    "unknown query parameter '{other}' in '{original}'; supported: temperature, thinking, api_key_env, cache, base_url"
                )));
            }
        }
    }

    Ok((temperature, thinking, api_key, cache, base_url))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Non-empty authority is rejected to avoid host/path/model ambiguity.
    #[test]
    fn reject_non_empty_authority() {
        assert!(matches!(
            ModelUrl::parse("ollama://localhost:11434/qwen3:8b"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Inline credentials are rejected.
    #[test]
    fn reject_inline_credentials() {
        assert!(matches!(
            ModelUrl::parse("gemini://mykey@gemini-2.5-flash"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Fragment is rejected.
    #[test]
    fn reject_fragment() {
        assert!(matches!(
            ModelUrl::parse("gemini:///model#section"),
            Err(RathError::InvalidUrl(_))
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
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Duplicate query parameters are rejected.
    #[test]
    fn reject_duplicate_query_param() {
        assert!(matches!(
            ModelUrl::parse("gemini:///model?temperature=0.5&temperature=0.7"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Temperature outside [0.0, 1.0] is rejected.
    #[test]
    fn reject_temperature_out_of_range() {
        assert!(matches!(
            ModelUrl::parse("gemini:///model?temperature=1.5"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Unknown provider scheme is rejected.
    #[test]
    fn reject_unknown_provider() {
        assert!(matches!(
            ModelUrl::parse("unknown:///model"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Missing model name (empty path) is rejected.
    #[test]
    fn reject_missing_model() {
        assert!(matches!(
            ModelUrl::parse("gemini:///"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Missing '://' is rejected.
    #[test]
    fn reject_missing_scheme_separator() {
        assert!(matches!(
            ModelUrl::parse("gemini-2.5-flash-lite"),
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Unset api_key_env variable is rejected at parse time.
    #[test]
    fn reject_missing_api_key_env() {
        assert!(matches!(
            ModelUrl::parse("openai:///gpt-4o?api_key_env=__PRAVAH_MISSING_ENV__"),
            Err(RathError::InvalidUrl(_))
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
            Err(RathError::InvalidUrl(_))
        ));
    }

    /// Absent cache param leaves cache as None.
    #[test]
    fn no_cache_param_is_none() {
        let url = ModelUrl::parse("anthropic:///claude-sonnet-4-5").unwrap();
        assert_eq!(url.cache, None);
    }
}
