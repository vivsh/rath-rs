/// Supported provider backends.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Provider {
    /// Google Gemini.
    Gemini,
    /// Ollama local runtime.
    Ollama,
    /// OpenAI-compatible endpoint.
    OpenAi,
    /// OpenRouter gateway.
    OpenRouter,
    /// fal model APIs.
    Fal,
    /// Anthropic Claude.
    Anthropic,
    /// ElevenLabs native voice and speech APIs.
    ElevenLabs,
    /// Optional local CPU ONNX embeddings.
    FastEmbed,
    /// An application-registered provider; the owned name is its URL scheme.
    External(String),
}

impl Provider {
    /// Borrows the provider identifier. External names live as long as this value.
    pub fn as_str(&self) -> &str {
        match self {
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
            Provider::OpenRouter => "openrouter",
            Provider::Fal => "fal",
            Provider::Anthropic => "anthropic",
            Provider::ElevenLabs => "elevenlabs",
            Provider::FastEmbed => "fastembed",
            Provider::External(name) => name,
        }
    }
}

/// Resolves only built-in schemes and aliases, without accepting external registrations.
pub(crate) fn builtin(scheme: &str) -> Option<Provider> {
    match scheme {
        "fastembed" => Some(Provider::FastEmbed),
        "gemini" => Some(Provider::Gemini),
        "ollama" => Some(Provider::Ollama),
        "openai" => Some(Provider::OpenAi),
        "openrouter" => Some(Provider::OpenRouter),
        "fal" => Some(Provider::Fal),
        "anthropic" | "claude" => Some(Provider::Anthropic),
        "elevenlabs" => Some(Provider::ElevenLabs),
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests/provider.rs"]
mod tests;
