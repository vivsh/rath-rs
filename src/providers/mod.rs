mod anthropic;
mod fal;
mod gemini;
mod ollama;
mod openai;
mod openrouter;

use crate::audio::stt::{SttClient, SttOptions};
use crate::audio::tts::{TtsClient, TtsOptions};
use crate::core::{ModelUrl, Provider};
use crate::embeddings::{EmbeddingClient, EmbeddingOptions};
use crate::images::{ImageClient, ImageOptions};
use crate::llm::{LlmClient, LlmError, LlmOptions};
use crate::video::{VideoClient, VideoOptions};

pub(crate) fn create_llm_client(
    url: &ModelUrl,
    options: LlmOptions,
) -> Result<Box<dyn LlmClient>, LlmError> {
    match url.provider {
        Provider::Gemini => gemini::new_client(url, options),
        Provider::OpenAi => openai::new_client(url, options),
        Provider::OpenRouter => openrouter::new_client(url, options),
        Provider::Anthropic => anthropic::new_client(url, options),
        Provider::Ollama => ollama::new_client(url, options),
        Provider::Fal => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "llm".to_string(),
        }),
    }
}

pub(crate) fn create_embedding_client(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<Box<dyn EmbeddingClient>, LlmError> {
    match url.provider {
        Provider::Gemini => gemini::new_embedding_client(url, options),
        Provider::OpenAi => openai::new_embedding_client(url, options),
        Provider::Ollama => ollama::new_embedding_client(url, options),
        _ => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "embeddings".to_string(),
        }),
    }
}

pub(crate) fn create_image_client(
    url: &ModelUrl,
    options: ImageOptions,
) -> Result<Box<dyn ImageClient>, LlmError> {
    match url.provider {
        Provider::Fal => fal::new_image_client(url, options),
        _ => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "image".to_string(),
        }),
    }
}

pub(crate) fn create_tts_client(
    url: &ModelUrl,
    options: TtsOptions,
) -> Result<Box<dyn TtsClient>, LlmError> {
    match url.provider {
        Provider::OpenAi => openai::new_tts_client(url, options),
        _ => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "text-to-speech".to_string(),
        }),
    }
}

pub(crate) fn create_stt_client(
    url: &ModelUrl,
    options: SttOptions,
) -> Result<Box<dyn SttClient>, LlmError> {
    match url.provider {
        Provider::OpenAi => openai::new_stt_client(url, options),
        _ => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "speech-to-text".to_string(),
        }),
    }
}

pub(crate) fn create_video_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<Box<dyn VideoClient>, LlmError> {
    match url.provider {
        Provider::Fal => fal::new_video_client(url, options),
        _ => Err(LlmError::UnsupportedCapability {
            provider: url.provider.clone(),
            capability: "video".to_string(),
        }),
    }
}
