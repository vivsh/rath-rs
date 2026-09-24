use crate::core::{
    builtin,
    url::{parse_pair, parse_parts, split_locator},
};
use crate::{ErrorKind, ModelUrl, Provider, RathError};

/// Accepts lowercase registration schemes without URL transport composition.
pub(super) fn valid_scheme(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b".-".contains(&b))
}

/// Parses only media transport controls for external providers; legacy URLs keep their parser.
pub(super) fn parse_for(raw: &str, llm: bool) -> Result<ModelUrl, RathError> {
    let (scheme, rest) = split_locator(raw)?;
    if builtin(scheme).is_some() {
        let url = ModelUrl::parse(raw)?;
        validate_controls(&url, llm)?;
        crate::core::url::validate_endpoint(&url)?;
        return Ok(url);
    }
    if !valid_scheme(scheme) {
        return Err(invalid("invalid external provider scheme"));
    }
    if let Some((_, query)) = rest.split_once('?') {
        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            let (name, _) = parse_pair(pair, raw)?;
            if !matches!(name, "api_key_env" | "base_url")
                && !(llm && matches!(name, "temperature" | "thinking" | "cache"))
            {
                return Err(invalid(format!(
                    "unsupported external media URL parameter '{name}'; use api_key_env or base_url"
                )));
            }
        }
    }
    let url = parse_parts(Provider::External(scheme.into()), rest, raw)?;
    crate::core::url::validate_endpoint(&url).map_err(|error| {
        error.sanitized(&url.api_key.as_deref().into_iter().collect::<Vec<_>>())
    })?;
    Ok(url)
}

fn invalid(message: impl Into<String>) -> RathError {
    RathError::new(ErrorKind::InvalidUrl, message)
}

#[cfg(test)]
#[path = "tests/url.rs"]
mod tests;

/// Rejects language-model controls on other capabilities.
pub(crate) fn validate_controls(url: &ModelUrl, llm: bool) -> Result<(), RathError> {
    if !llm && (url.temperature.is_some() || url.thinking.is_some() || url.cache.is_some()) {
        return Err(invalid(
            "temperature, thinking, and cache URL controls apply only to LLM clients",
        ));
    }
    Ok(())
}
