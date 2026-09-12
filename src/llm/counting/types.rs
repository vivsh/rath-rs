/// Origin of an input-token measurement; neither variant promises exact billing usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenCountSource {
    /// Padded reference-tokenizer estimate; not a context-admission guarantee.
    Estimated,
    /// Count returned by the configured provider's counting endpoint.
    ProviderReported,
}

/// Input measurement, including request framing and excluding output reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCount {
    /// Estimated or provider-reported input tokens for this complete measurement.
    pub input_tokens: u64,
    /// Identifies whether the result came from a local estimate or the provider.
    pub source: TokenCountSource,
}
