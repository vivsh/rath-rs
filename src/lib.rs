//! Provider-agnostic AI APIs for Rust.
//!
//! Rath exposes capability-focused modules for LLM calls, embeddings, images,
//! video, and audio. Provider-specific adapters are selected through model URLs and
//! kept behind stable capability traits.

pub mod audio;
pub mod core;
pub mod embeddings;
pub mod images;
pub mod llm;
mod providers;
pub mod video;

pub use core::{ModelUrl, Provider, RathError, TokenUsage};
