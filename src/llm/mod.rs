//! Provider-neutral language-model requests, measurement, tools, and responses.
mod api;
pub(crate) mod counting;
/// JSON-schema normalization used by provider adapters.
pub(crate) mod schema;
mod stream;
mod tool;

pub use api::*;
pub use counting::{TokenCount, TokenCountSource};
pub use stream::{LlmEvent, LlmStream};

#[cfg(test)]
mod tests;

mod client;
pub use client::{LlmBackend, LlmClient};

pub(crate) mod output;
