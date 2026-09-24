//! Provider-agnostic AI APIs for Rust.
//!
//! Rath exposes capability-focused modules for LLM calls, embeddings, images,
//! video, and audio. Provider-specific adapters are selected through model URLs and
//! kept behind stable capability traits.

//! Compile-time examples of the intended pre-SemVer extension boundary.
//!
//! Extensible requests must use constructors rather than external literals:
//! ```compile_fail
//! let request = rath::embeddings::EmbedRequest {
//!     input: vec!["hello".into()], ..Default::default()
//! };
//! ```
//! Extensible identities require a fallback match arm:
//! ```compile_fail
//! fn classify(provider: rath::Provider) {
//!     match provider {
//!         rath::Provider::Gemini | rath::Provider::Ollama | rath::Provider::OpenAi |
//!         rath::Provider::OpenRouter | rath::Provider::Fal | rath::Provider::Anthropic |
//!         rath::Provider::ElevenLabs | rath::Provider::FastEmbed | rath::Provider::External(_) => ()
//!     }
//! }
//! ```
//! Closed token provenance remains exhaustively matchable:
//! ```
//! fn estimated(source: rath::llm::TokenCountSource) -> bool {
//!     match source {
//!         rath::llm::TokenCountSource::Estimated => true,
//!         rath::llm::TokenCountSource::ProviderReported => false,
//!     }
//! }
//! let count = rath::llm::TokenCount { input_tokens: 1, source: rath::llm::TokenCountSource::Estimated };
//! assert!(estimated(count.source));
//! ```

/// Speech synthesis, transcription and caller-owned voices.
pub mod audio;
/// Shared errors, locators, provider identity and token usage.
pub mod core;
pub mod embeddings;
pub mod images;
pub mod llm;
mod providers;
pub mod registry;
pub mod video;

pub use core::{ErrorBody, ErrorKind, ModelUrl, Provider, RathError, TokenUsage};
