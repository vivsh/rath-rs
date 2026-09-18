//! Stateless credential redaction; no credential material survives in an error.
use regex::{Captures, Regex};

const SECRET_FIELD: &str = r#"("(?:\\.|[^"\\])*")\s*:\s*"#;

/// Replaces complete JSON secret values, including nested objects, without rewriting other bytes.
fn secret_fields(input: &str) -> String {
    let Ok(regex) = Regex::new(SECRET_FIELD) else {
        return "[REDACTED: invalid redaction expression]".into();
    };
    let mut output = String::new();
    let mut consumed = 0;
    for captures in regex.captures_iter(input) {
        let Some(field) = captures.get(0) else {
            continue;
        };
        let name = serde_json::from_str::<String>(&captures[1]).unwrap_or_default();
        if !secret_name(&name) || field.start() < consumed {
            continue;
        }
        let tail = &input[field.end()..];
        let mut values = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
        let length = if values.next().is_some_and(|v| v.is_ok()) {
            values.byte_offset()
        } else {
            // An unterminated secret value cannot be safely retained.
            tail.len()
        };
        output.push_str(&input[consumed..field.end()]);
        output.push_str("\"[REDACTED]\"");
        consumed = field.end() + length;
    }
    output.push_str(&input[consumed..]);
    output
}

/// Recognizes explicitly credential-bearing JSON locations, including escaped field names.
fn secret_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase().replace(['_', '-'], "");
    matches!(
        name.as_str(),
        "authorization"
            | "proxyauthorization"
            | "cookie"
            | "setcookie"
            | "apikey"
            | "xapikey"
            | "xgoogapikey"
            | "accesstoken"
            | "refreshtoken"
            | "password"
            | "secret"
            | "token"
            | "clientsecret"
            | "apisecret"
            | "secretkey"
            | "privatekey"
            | "credentials"
    )
}

/// Covers raw, JSON-escaped and URL/form-encoded representations of known credentials.
fn redact_known(input: &str, secret: &str) -> String {
    let mut result = input.to_owned();
    if let Ok(escaped) = serde_json::to_string(secret) {
        result = result.replace(&escaped[1..escaped.len() - 1], "[REDACTED]");
    }
    result = result.replace(&secret.replace('/', "\\/"), "[REDACTED]");
    for encoded in [percent(secret), percent(secret).replace("%20", "+")] {
        let pattern = encoded
            .split('%')
            .enumerate()
            .map(|(i, part)| {
                if i == 0 {
                    regex::escape(part)
                } else {
                    format!("%(?i:{}){}", &part[..2], regex::escape(&part[2..]))
                }
            })
            .collect::<String>();
        result = replace(&result, &pattern, |_| "[REDACTED]".into());
    }
    result.replace(secret, "[REDACTED]")
}

/// Sanitizes known secrets, explicit secret fields and URL components in readable diagnostics.
pub(super) fn text(input: &str, secrets: &[&str]) -> String {
    let mut result = input.to_owned();
    let mut secrets = secrets.to_vec();
    secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
    for secret in secrets.iter().filter(|s| !s.is_empty()) {
        result = redact_known(&result, secret);
    }
    result = secret_fields(&result);
    result = replace(
        &result,
        r"(?im)(^\s*(?:authorization|proxy-authorization|cookie|set-cookie|x-api-key|x-goog-api-key)\s*:\s*)[^\r\n]+",
        |c| format!("{}[REDACTED]", &c[1]),
    );
    result = replace(
        &result,
        r#"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|token)\s*=\s*)[^\s&,;"']+"#,
        |c| format!("{}[REDACTED]", &c[1]),
    );
    result = replace(
        &result,
        r#"[A-Za-z][A-Za-z0-9+.-]*:\\/\\/[^ \t\r\n<>"']+"#,
        |c| url(&c[0].replace("\\/", "/")).replace('/', "\\/"),
    );
    replace(
        &result,
        r#"[A-Za-z][A-Za-z0-9+.-]*://[^ \t\r\n<>"'\\]+"#,
        |c| url(&c[0]),
    )
}

/// Applies fixed expressions fallibly, failing closed if an internal expression is invalid.
fn replace(input: &str, pattern: &str, replacement: impl Fn(&Captures<'_>) -> String) -> String {
    match Regex::new(pattern) {
        Ok(regex) => regex.replace_all(input, replacement).into_owned(),
        Err(_) => "[REDACTED: invalid redaction expression]".into(),
    }
}

/// Redacts query values and fragments while retaining useful URL origin/path and query names.
fn url(input: &str) -> String {
    let (scheme, rest) = input.split_once("://").unwrap_or(("", input));
    let boundary = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = rest[..boundary].rsplit('@').next().unwrap_or_default();
    let rest = format!("{scheme}://{authority}{}", &rest[boundary..]);
    let (head, fragment) = rest
        .split_once('#')
        .map_or((rest.as_str(), false), |(s, _)| (s, true));
    let mut result = if let Some((path, query)) = head.split_once('?') {
        let query = query
            .split('&')
            .map(|pair| {
                let key = pair.split('=').next().unwrap_or_default();
                format!("{key}=[REDACTED]")
            })
            .collect::<Vec<_>>()
            .join("&");
        format!("{path}?{query}")
    } else {
        head.to_owned()
    };
    if fragment {
        result.push_str("#[REDACTED]");
    }
    result
}

/// Encodes known credentials as RFC3986 query values without retaining them globally.
fn percent(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// Preserves binary bytes while redacting complete secret fields across invalid UTF-8 spans.
pub(super) fn bytes(input: &[u8], secrets: &[&str]) -> Vec<u8> {
    if let Ok(text_input) = std::str::from_utf8(input) {
        return text(text_input, secrets).into_bytes();
    }
    // A reversible byte-to-scalar view lets JSON field redaction consume a whole value,
    // including invalid UTF-8 inside it. No replacement decoding or bytes are discarded.
    let view: String = input.iter().map(|byte| char::from(*byte)).collect();
    let mapped_secrets: Vec<String> = secrets
        .iter()
        .map(|secret| secret.bytes().map(char::from).collect())
        .collect();
    let all_secrets: Vec<_> = secrets
        .iter()
        .copied()
        .chain(mapped_secrets.iter().map(String::as_str))
        .collect();
    // The view contains only U+0000..U+00FF; redaction inserts only ASCII markers.
    text(&view, &all_secrets)
        .chars()
        .map(|scalar| scalar as u8)
        .collect()
}
