use crate::embeddings::{EmbeddingClient, EmbeddingOptions};
use crate::{ModelUrl, RathError};

#[cfg(feature = "fastembed")]
#[path = "fastembed/backend.rs"]
mod backend;
#[cfg(feature = "fastembed")]
#[path = "fastembed/loading.rs"]
mod loading;

/// Keeps the scheme reserved and configuration typed even without the native dependency.
pub(crate) async fn create(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<EmbeddingClient, RathError> {
    #[cfg(feature = "fastembed")]
    {
        backend::create(url, options).await
    }
    #[cfg(not(feature = "fastembed"))]
    {
        let _ = (url, options);
        Err(RathError::unsupported(
            crate::Provider::FastEmbed,
            "local embeddings require the Cargo feature 'fastembed'",
        ))
    }
}
