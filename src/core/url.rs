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
            return Err(RathError::new(
                crate::core::ErrorKind::InvalidUrl,
                "URL must not contain a fragment",
            ));
        }
        let (scheme_part, rest) = s.split_once("://").ok_or_else(|| {
            RathError::new(
                crate::core::ErrorKind::InvalidUrl,
                format!("missing '://' in '{s}'; expected e.g. gemini:///model-name"),
            )
        })?;
        // Reject inline credentials (user:pass@host) by checking the authority
        let authority_candidate = rest.split('/').next().unwrap_or("");
        if authority_candidate.contains('@') {
            return Err(RathError::new(
                crate::core::ErrorKind::InvalidUrl,
                "inline credentials are not allowed; use the api_key_env query parameter",
            ));
        }
        let provider = parse_provider_scheme(scheme_part, s)?;
        let (path_authority, query_str) = match rest.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (rest, None),
        };
        let segments = parse_model_path(path_authority, s)?;
        if segments.is_empty() {
            return Err(RathError::new(
                crate::core::ErrorKind::InvalidUrl,
                format!("'{s}' must contain a model name as the final path segment"),
            ));
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
        return Err(RathError::new(
            crate::core::ErrorKind::InvalidUrl,
            format!(
                "custom transports are not supported in '{original}'; use provider:///model?base_url=https://host/path"
            ),
        ));
    }

    let provider = match scheme {
        "gemini" => Provider::Gemini,
        "openai" => Provider::OpenAi,
        "openrouter" => Provider::OpenRouter,
        "fal" => Provider::Fal,
        "anthropic" | "claude" => Provider::Anthropic,
        "ollama" => Provider::Ollama,
        other => {
            return Err(RathError::new(
                crate::core::ErrorKind::InvalidUrl,
                format!(
                    "unknown provider '{other}' in '{original}'; expected gemini, openai, openrouter, fal, anthropic, claude, or ollama"
                ),
            ));
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
        return Err(RathError::new(
            crate::core::ErrorKind::InvalidUrl,
            format!(
                "empty authority in '{original}'; use e.g. gemini:///model for no custom endpoint"
            ),
        ));
    }

    Err(RathError::new(
        crate::core::ErrorKind::InvalidUrl,
        format!(
            "non-empty authority is not supported in '{original}'; put the provider-native model id after /// and use base_url for custom endpoints"
        ),
    ))
}

type ParsedQuery = (
    Option<f32>,
    Option<ThinkingLevel>,
    Option<String>,
    Option<CacheControl>,
    Option<String>,
);

/// Validates query options and resolves credentials into operation-local parsing state.
fn parse_query_str(query_str: Option<&str>, original: &str) -> Result<ParsedQuery, RathError> {
    let Some(query) = query_str else {
        return Ok((None, None, None, None, None));
    };
    let (mut temperature, mut thinking, mut api_key, mut cache, mut base_url) =
        (None, None, None, None, None);
    let mut seen = HashSet::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = parse_pair(pair, original)?;
        if !seen.insert(key) {
            return Err(invalid_url(format!(
                "duplicate query parameter '{key}' in '{original}'"
            )));
        }
        match key {
            "temperature" => temperature = Some(parse_temperature(value, original)?),
            "thinking" => {
                thinking = Some(ThinkingLevel::from_str(value).map_err(|_| {
                    invalid_url(format!(
                        "thinking must be off/low/medium/high/xhigh in '{original}', got '{value}'"
                    ))
                })?)
            }
            "api_key_env" => {
                api_key = Some(std::env::var(value).map_err(|error| {
                    invalid_url(format!(
                        "environment variable '{value}' referenced by api_key_env is not set"
                    ))
                    .with_source(crate::core::error::credential_cause(&error))
                })?)
            }
            "cache" => cache = Some(parse_cache(value, original)?),
            "base_url" => base_url = Some(parse_base_url(value, original)?),
            other => {
                return Err(invalid_url(format!(
                    "unknown query parameter '{other}' in '{original}'; supported: temperature, thinking, api_key_env, cache, base_url"
                )));
            }
        }
    }
    Ok((temperature, thinking, api_key, cache, base_url))
}

/// Checks the syntax of one query pair without discarding the offending location.
fn parse_pair<'a>(pair: &'a str, original: &str) -> Result<(&'a str, &'a str), RathError> {
    let (key, value) = pair.split_once('=').ok_or_else(|| {
        invalid_url(format!(
            "query parameter '{pair}' in '{original}' must be key=value"
        ))
    })?;
    if value.is_empty() {
        return Err(invalid_url(format!(
            "query parameter '{key}' must not be empty in '{original}'"
        )));
    }
    Ok((key, value))
}

/// Keeps numeric parsing causes separate from the model-URL context.
fn parse_temperature(value: &str, original: &str) -> Result<f32, RathError> {
    let temperature = value.parse::<f32>().map_err(|error| {
        invalid_url(format!(
            "temperature must be a number in '{original}', got '{value}'"
        ))
        .with_source(RathError::from_error(crate::core::ErrorKind::Other, &error))
    })?;
    if !(0.0..=1.0).contains(&temperature) {
        return Err(invalid_url(format!(
            "temperature must be 0.0–1.0, got {temperature} in '{original}'"
        )));
    }
    Ok(temperature)
}

fn parse_cache(value: &str, original: &str) -> Result<CacheControl, RathError> {
    match value {
        "5m" => Ok(CacheControl::Ephemeral5m),
        "1h" => Ok(CacheControl::Ephemeral1h),
        other => Err(invalid_url(format!(
            "cache must be 5m or 1h in '{original}', got '{other}'"
        ))),
    }
}

/// Validates a deployment URL without retaining credential-bearing URL components in errors.
fn parse_base_url(value: &str, original: &str) -> Result<String, RathError> {
    if !(value.starts_with("https://") || value.starts_with("http://")) {
        return Err(invalid_url(format!(
            "base_url must start with http:// or https:// in '{original}'"
        )));
    }
    if value.contains('#') {
        return Err(invalid_url(format!(
            "base_url must not contain a fragment in '{original}'"
        )));
    }
    Ok(value.trim_end_matches('/').to_owned())
}

fn invalid_url(message: String) -> RathError {
    RathError::new(crate::core::ErrorKind::InvalidUrl, message)
}

#[cfg(test)]
mod tests;
