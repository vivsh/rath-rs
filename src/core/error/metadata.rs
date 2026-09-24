use super::{RathError, redact};

impl RathError {
    /// Attaches an HTTP status without changing the error classification or its causes.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.inner.http_status = Some(status);
        self
    }

    /// Attaches a provider code after removing known credentials and secret-bearing locations.
    pub fn with_provider_code(mut self, code: &str, secrets: &[&str]) -> Self {
        self.inner.provider_code = Some(redact::text(code, secrets));
        self.sanitized(secrets)
    }

    /// Attaches the provider's request identifier without retaining the supplied credentials.
    pub fn with_request_id(mut self, id: &str, secrets: &[&str]) -> Self {
        self.inner.request_id = Some(redact::text(id, secrets));
        self.sanitized(secrets)
    }

    /// Attaches the unparsed Retry-After value after credential sanitization; it schedules no retry.
    pub fn with_retry_after(mut self, value: &str, secrets: &[&str]) -> Self {
        self.inner.retry_after = Some(redact::text(value, secrets));
        self.sanitized(secrets)
    }
}
