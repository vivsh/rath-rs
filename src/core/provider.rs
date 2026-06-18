/// Supported provider backends.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
}

impl Provider {
    /// Returns a lowercase string identifier for the provider.
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
            Provider::OpenRouter => "openrouter",
            Provider::Fal => "fal",
            Provider::Anthropic => "anthropic",
        }
    }
}
