use super::api::*;
use crate::RathError;
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait ImageBackend: Send + Sync {
    /// Generates once using the model-bound request; errors retain provider diagnostics.
    fn generate_image(
        &self,
        request: &ImageRequest,
    ) -> impl std::future::Future<Output = Result<ImageResponse, RathError>> + Send;
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedImageBackend: Send + Sync {
    fn generate_image<'a>(
        &'a self,
        request: &'a ImageRequest,
    ) -> ErasedFuture<'a, Result<ImageResponse, RathError>>;
}
impl<B: ImageBackend> ErasedImageBackend for B {
    fn generate_image<'a>(
        &'a self,
        request: &'a ImageRequest,
    ) -> ErasedFuture<'a, Result<ImageResponse, RathError>> {
        Box::pin(ImageBackend::generate_image(self, request))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct ImageClient {
    inner: Box<dyn ErasedImageBackend>,
}
impl ImageClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: ImageBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `generate_image` to the owned backend.
    pub async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        self.inner.generate_image(request).await
    }
}
impl ImageBackend for ImageClient {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        ImageClient::generate_image(self, request).await
    }
}
