use super::{ErrorDetails, RathError};
use std::fmt;

impl fmt::Display for ErrorDetails {
    /// Formats useful context and distinct cause messages, never response bytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut seen = Vec::new();
        let mut next = Some(self);
        let mut written = false;
        while let Some(error) = next {
            let distinct = !seen.contains(&error.message.as_str());
            let has_context = error.provider.is_some()
                || error.operation.is_some()
                || error.http_status.is_some()
                || error.provider_code.is_some()
                || error.request_id.is_some()
                || error.retry_after.is_some();
            if written && (distinct || has_context) {
                write!(f, ": ")?;
            }
            if let Some(provider) = &error.provider {
                write!(f, "{} ", provider.as_str())?;
            }
            if let Some(operation) = error.operation {
                write!(f, "{operation}: ")?;
            }
            if let Some(status) = error.http_status {
                write!(f, "HTTP {status} ")?;
            }
            if let Some(code) = &error.provider_code {
                write!(f, "[{code}] ")?;
            }
            if let Some(id) = &error.request_id {
                write!(f, "request {id}: ")?;
            }
            if let Some(retry) = &error.retry_after {
                write!(f, "retry-after {retry}: ")?;
            }
            if distinct {
                write!(f, "{}", error.message)?;
                seen.push(error.message.as_str());
            }
            written |= distinct || has_context;
            next = error.source.as_ref().map(|source| source.inner.as_ref());
        }
        Ok(())
    }
}

impl fmt::Debug for RathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RathError")
            .field("kind", &self.inner.kind)
            .field("provider", &self.inner.provider)
            .field("operation", &self.inner.operation)
            .field("message", &self.inner.message)
            .field("http_status", &self.inner.http_status)
            .field("provider_code", &self.inner.provider_code)
            .field("request_id", &self.inner.request_id)
            .field("retry_after", &self.inner.retry_after)
            .field("response_body", &self.inner.response_body)
            .field("source", &self.inner.source)
            .finish()
    }
}
