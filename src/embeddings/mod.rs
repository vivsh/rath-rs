//! Dense embedding requests and model-bound clients.
mod client;
pub(crate) mod contract;
mod fastembed_options;
mod options;
mod types;
pub use client::{EmbeddingBackend, EmbeddingClient};
pub use fastembed_options::FastEmbedOptions;
pub use options::EmbeddingOptions;
pub use types::*;
