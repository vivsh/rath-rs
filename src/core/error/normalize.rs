use super::{ErrorBody, ErrorKind, RathError, redact};
use crate::core::Provider;
use serde_json::Value;

impl RathError {
    /// Converts each native source level into a separately inspectable Rath diagnostic.
    pub(crate) fn from_error(kind: ErrorKind, error: &(dyn std::error::Error + 'static)) -> Self {
        let mut message = error.to_string();
        if let Some(source) = error.source() {
            let suffix = format!(": {source}");
            if let Some(context) = message.strip_suffix(&suffix).filter(|s| !s.is_empty()) {
                message = context.to_owned();
            }
        }
        let mut result = Self::new(kind, message);
        if let Some(source) = error.source() {
            result = result.with_source(Self::from_error(ErrorKind::Other, source));
        }
        result
    }

    /// Removes operation-local credentials from all retained diagnostic fields.
    pub(crate) fn sanitized(mut self, secrets: &[&str]) -> Self {
        self.message = redact::text(&self.message, secrets);
        for value in [
            &mut self.provider_code,
            &mut self.request_id,
            &mut self.retry_after,
        ]
        .into_iter()
        .flatten()
        {
            *value = redact::text(value, secrets);
        }
        if let Some(body) = &mut self.response_body {
            match body {
                ErrorBody::Complete(bytes) | ErrorBody::Incomplete(bytes) => {
                    *bytes = redact::bytes(bytes, secrets)
                }
            }
        }
        self.source = self
            .source
            .map(|source| Box::new(source.sanitized(secrets)));
        self
    }

    /// Retains explicit body bytes and their transfer completeness without formatting them.
    pub(crate) fn with_body(mut self, body: ErrorBody) -> Self {
        self.response_body = Some(body);
        self.sanitized(&[])
    }

    /// Attaches a decoded response when no original body is already available.
    pub(crate) fn with_response(mut self, response: &Value) -> Self {
        if self.response_body.is_none() {
            match serde_json::to_vec(response) {
                Ok(bytes) => self = self.with_body(ErrorBody::Complete(bytes)),
                Err(error) => {
                    self = self.with_source(Self::from_error(ErrorKind::Serialize, &error))
                }
            }
        }
        self
    }

    /// Reclassifies a failure without losing status, body or causes.
    pub(crate) fn classified(mut self, kind: ErrorKind) -> Self {
        self.kind = kind;
        self
    }

    /// Creates an unsupported-capability error with explicit provider context.
    pub(crate) fn unsupported(provider: Provider, capability: impl AsRef<str>) -> Self {
        Self::new(
            ErrorKind::UnsupportedCapability,
            format!("unsupported capability: {}", capability.as_ref()),
        )
        .with_context(provider, "capability validation")
    }

    /// Preserves parser details and explicitly accessible invalid output.
    pub(crate) fn deserialize(error: &serde_json::Error, raw: impl AsRef<[u8]>) -> Self {
        Self::from_error(ErrorKind::Deserialize, error)
            .with_body(ErrorBody::Complete(raw.as_ref().to_vec()))
    }

    /// Reports a required response field or shape and retains the available response.
    pub(crate) fn invalid(message: impl Into<String>, response: &Value) -> Self {
        Self::new(ErrorKind::InvalidResponse, message).with_response(response)
    }
}

/// Extracts native message/code/identifiers without rendering arbitrary response payloads.
pub(super) fn details(error: &mut RathError, value: &Value) {
    let native = value.get("error").unwrap_or(value);
    let message = native
        .as_str()
        .or_else(|| native.get("message").and_then(Value::as_str))
        .or_else(|| native.get("detail").and_then(Value::as_str));
    if let Some(message) = message {
        error.message = redact::text(message, &[]);
    } else if let Some(messages) = validation_messages(native) {
        error.message = redact::text(&messages, &[]);
    }
    error.provider_code = native
        .get("code")
        .or_else(|| native.get("type"))
        .or_else(|| native.get("status"))
        .and_then(scalar);
    if error.request_id.is_none() {
        error.request_id = value
            .get("request_id")
            .or_else(|| native.get("request_id"))
            .and_then(scalar);
    }
}

/// Extracts explicit validation messages without printing their input values or unknown fields.
fn validation_messages(value: &Value) -> Option<String> {
    let details = value.get("detail")?.as_array()?;
    let messages: Vec<_> = details
        .iter()
        .filter_map(|detail| {
            detail
                .get("msg")
                .or_else(|| detail.get("message"))
                .and_then(Value::as_str)
        })
        .collect();
    (!messages.is_empty()).then(|| messages.join("; "))
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

/// Retains a provider-reported failure returned within a successful JSON response.
pub(crate) fn provider_failure(
    provider: Provider,
    operation: &'static str,
    response: &Value,
    secrets: &[&str],
) -> RathError {
    provider_failure_message(
        provider,
        operation,
        response,
        secrets,
        "provider reported failure; response details available",
    )
}

/// Uses the operation's available failure message when no native envelope message is present.
pub(crate) fn provider_failure_message(
    provider: Provider,
    operation: &'static str,
    response: &Value,
    secrets: &[&str],
    message: &str,
) -> RathError {
    let mut error = RathError::new(ErrorKind::Provider, message)
        .with_context(provider, operation)
        .with_response(response);
    details(&mut error, response);
    error.sanitized(secrets)
}

/// Preserves HTTP diagnostics exposed by an SDK that does not expose the original response object.
pub(crate) fn sdk_http(status: u16, body: Option<&str>, secrets: &[&str]) -> RathError {
    let mut error = RathError::new(
        ErrorKind::Http,
        "request failed; response details available",
    );
    error.http_status = Some(status);
    if let Some(body) = body {
        match serde_json::from_str::<Value>(body) {
            Ok(value) => details(&mut error, &value),
            Err(cause) => {
                error = error.with_source(RathError::from_error(ErrorKind::Deserialize, &cause))
            }
        }
        error = error.with_body(ErrorBody::Complete(body.as_bytes().to_vec()));
    }
    error.sanitized(secrets)
}

/// Preserves environment failures while treating the contents of malformed credentials as secret.
pub(crate) fn credential_cause(error: &std::env::VarError) -> RathError {
    match error {
        std::env::VarError::NotPresent => RathError::from_error(ErrorKind::Other, error),
        std::env::VarError::NotUnicode(_) => RathError::new(
            ErrorKind::Other,
            "credential is not valid Unicode: [REDACTED]",
        ),
    }
}
