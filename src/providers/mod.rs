mod anthropic;
mod elevenlabs;
mod fal;
mod gemini;
mod ollama;
mod openai;
mod openrouter;
mod streaming;

mod factory;
pub(crate) use factory::*;

#[cfg(test)]
mod tests;

mod local_embeddings;
pub(crate) use local_embeddings::create as local_embeddings;
