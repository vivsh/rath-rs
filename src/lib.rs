//! Provider-agnostic AI APIs for Rust.
//!
//! Rath exposes capability-focused modules for LLM calls, embeddings, images,
//! video, and audio. Provider-specific adapters are selected through model URLs and
//! kept behind stable capability traits.

// The approved diagnostic snapshot is 160 bytes; keep the public Result<T, RathError>
// contract and avoid another allocation or wrapper solely to satisfy a size heuristic.
#![allow(clippy::result_large_err)]

pub mod audio;
pub mod core;
pub mod embeddings;
pub mod images;
pub mod llm;
mod providers;
pub mod video;

pub use core::{ErrorBody, ErrorKind, ModelUrl, Provider, RathError, TokenUsage};
