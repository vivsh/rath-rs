use super::*;
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait VoiceDesignBackend: Send + Sync {
    /// Generates previews without implicitly registering, cloning, or persisting them.
    fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> impl std::future::Future<Output = Result<VoiceDesignResponse, RathError>> + Send;

    /// Explicitly creates a provider-owned voice from a preview, when supported.
    /// No local state is retained; remote resources survive cancellation or local data deletion.
    fn register_voice(
        &self,
        _preview: &VoicePreview,
        _name: &str,
        _provider_config: Option<&Value>,
    ) -> impl std::future::Future<Output = Result<Voice, RathError>> + Send {
        async move {
            Err(RathError::new(
                ErrorKind::UnsupportedCapability,
                "native voice registration is unsupported",
            ))
        }
    }
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedVoiceDesignBackend: Send + Sync {
    fn design_voice<'a>(
        &'a self,
        request: &'a VoiceDesignRequest,
    ) -> ErasedFuture<'a, Result<VoiceDesignResponse, RathError>>;
    fn register_voice<'a>(
        &'a self,
        preview: &'a VoicePreview,
        name: &'a str,
        provider_config: Option<&'a Value>,
    ) -> ErasedFuture<'a, Result<Voice, RathError>>;
}
impl<B: VoiceDesignBackend> ErasedVoiceDesignBackend for B {
    fn design_voice<'a>(
        &'a self,
        request: &'a VoiceDesignRequest,
    ) -> ErasedFuture<'a, Result<VoiceDesignResponse, RathError>> {
        Box::pin(VoiceDesignBackend::design_voice(self, request))
    }
    fn register_voice<'a>(
        &'a self,
        preview: &'a VoicePreview,
        name: &'a str,
        provider_config: Option<&'a Value>,
    ) -> ErasedFuture<'a, Result<Voice, RathError>> {
        Box::pin(VoiceDesignBackend::register_voice(
            self,
            preview,
            name,
            provider_config,
        ))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct VoiceDesignClient {
    inner: Box<dyn ErasedVoiceDesignBackend>,
}
impl VoiceDesignClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: VoiceDesignBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `design_voice` to the owned backend.
    pub async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError> {
        self.inner.design_voice(request).await
    }
    /// Forwards `register_voice` to the owned backend.
    pub async fn register_voice(
        &self,
        preview: &VoicePreview,
        name: &str,
        provider_config: Option<&Value>,
    ) -> Result<Voice, RathError> {
        self.inner
            .register_voice(preview, name, provider_config)
            .await
    }
}
impl VoiceDesignBackend for VoiceDesignClient {
    async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError> {
        VoiceDesignClient::design_voice(self, request).await
    }
    async fn register_voice(
        &self,
        preview: &VoicePreview,
        name: &str,
        provider_config: Option<&Value>,
    ) -> Result<Voice, RathError> {
        VoiceDesignClient::register_voice(self, preview, name, provider_config).await
    }
}
