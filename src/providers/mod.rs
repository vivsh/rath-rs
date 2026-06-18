mod anthropic;
mod fal;
mod gemini;
mod ollama;
mod openai;
mod openrouter;

use crate::core::{ModelUrl, Provider};
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
