use super::{anthropic, elevenlabs, fal, gemini, ollama, openai, openrouter};
use crate::audio::stt::{SttClient, SttOptions};
use crate::audio::tts::{TtsClient, TtsOptions};
use crate::audio::voice_clone::{VoiceCloneClient, VoiceCloneOptions};
use crate::audio::voice_design::{VoiceDesignClient, VoiceDesignOptions};
use crate::core::{ModelUrl, Provider};
use crate::embeddings::{EmbeddingClient, EmbeddingOptions};
use crate::images::{ImageClient, ImageOptions};
use crate::llm::{LlmClient, LlmOptions, RathError};
use crate::video::{VideoClient, VideoOptions};

/// Constructs an independent voice-design adapter without invoking generation.
pub(crate) fn create_voice_design_client(
    url: &ModelUrl,
    options: VoiceDesignOptions,
) -> Result<Box<dyn VoiceDesignClient>, RathError> {
    match url.provider {
        Provider::Fal => fal::new_voice_design_client(url, options),
        Provider::ElevenLabs => elevenlabs::new_design(url, options),
        _ => Err(RathError::unsupported(url.provider.clone(), "voice design")),
    }
    .map_err(|error| construction_error(error, url))
}

/// Constructs an independent cloning adapter without creating any voice resources.
pub(crate) fn create_voice_clone_client(
    url: &ModelUrl,
    options: VoiceCloneOptions,
) -> Result<Box<dyn VoiceCloneClient>, RathError> {
    match url.provider {
        Provider::Fal => fal::new_voice_clone_client(url, options),
        Provider::ElevenLabs => elevenlabs::new_clone(url, options),
        _ => Err(RathError::unsupported(
            url.provider.clone(),
            "voice cloning",
        )),
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported LLM adapter using the configured model URL and options.
pub(crate) fn create_llm_client(
    url: &ModelUrl,
    options: LlmOptions,
) -> Result<Box<dyn LlmClient>, RathError> {
    match url.provider {
        Provider::Gemini => gemini::new_client(url, options),
        Provider::OpenAi => openai::new_client(url, options),
        Provider::OpenRouter => openrouter::new_client(url, options),
        Provider::Anthropic => anthropic::new_client(url, options),
        Provider::Ollama => ollama::new_client(url, options),
        Provider::Fal | Provider::ElevenLabs => {
            Err(RathError::unsupported(url.provider.clone(), "llm"))
        }
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported embedding adapter or returns an unsupported capability error.
pub(crate) fn create_embedding_client(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<Box<dyn EmbeddingClient>, RathError> {
    match url.provider {
        Provider::Gemini => gemini::new_embedding_client(url, options),
        Provider::OpenAi => openai::new_embedding_client(url, options),
        Provider::Ollama => ollama::new_embedding_client(url, options),
        _ => Err(RathError::unsupported(url.provider.clone(), "embeddings")),
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported image adapter or returns an unsupported capability error.
pub(crate) fn create_image_client(
    url: &ModelUrl,
    options: ImageOptions,
) -> Result<Box<dyn ImageClient>, RathError> {
    match url.provider {
        Provider::Fal => fal::new_image_client(url, options),
        _ => Err(RathError::unsupported(url.provider.clone(), "image")),
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported speech synthesis adapter or returns an unsupported capability error.
pub(crate) fn create_tts_client(
    url: &ModelUrl,
    options: TtsOptions,
) -> Result<Box<dyn TtsClient>, RathError> {
    match url.provider {
        Provider::OpenAi => openai::new_tts_client(url, options),
        Provider::Fal => fal::new_tts_client(url, options),
        Provider::ElevenLabs => elevenlabs::new_tts(url, options),
        _ => Err(RathError::unsupported(
            url.provider.clone(),
            "text-to-speech",
        )),
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported speech transcription adapter or returns an unsupported capability error.
pub(crate) fn create_stt_client(
    url: &ModelUrl,
    options: SttOptions,
) -> Result<Box<dyn SttClient>, RathError> {
    match url.provider {
        Provider::OpenAi => openai::new_stt_client(url, options),
        Provider::Fal => fal::new_stt_client(url, options),
        _ => Err(RathError::unsupported(
            url.provider.clone(),
            "speech-to-text",
        )),
    }
    .map_err(|error| construction_error(error, url))
}

/// Selects a supported video adapter or returns an unsupported capability error.
pub(crate) fn create_video_client(
    url: &ModelUrl,
    options: VideoOptions,
) -> Result<Box<dyn VideoClient>, RathError> {
    match url.provider {
        Provider::Fal => fal::new_video_client(url, options),
        _ => Err(RathError::unsupported(url.provider.clone(), "video")),
    }
    .map_err(|error| construction_error(error, url))
}

/// Adds construction context and sanitizes operation-local credentials at the common factory boundary.
fn construction_error(error: RathError, url: &ModelUrl) -> RathError {
    let env = match url.provider {
        Provider::Gemini => "GEMINI_API_KEY",
        Provider::OpenAi => "OPENAI_API_KEY",
        Provider::OpenRouter => "OPENROUTER_API_KEY",
        Provider::Anthropic => "ANTHROPIC_API_KEY",
        Provider::Ollama => "OLLAMA_API_KEY",
        Provider::Fal => "FAL_KEY",
        Provider::ElevenLabs => "ELEVENLABS_API_KEY",
    };
    let fallback = std::env::var(env).ok();
    let secrets: Vec<_> = url
        .api_key
        .as_deref()
        .or(fallback.as_deref())
        .into_iter()
        .collect();
    error
        .with_context(url.provider.clone(), "client construction")
        .sanitized(&secrets)
}
