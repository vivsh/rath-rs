use super::api::*;
use super::{TokenCount, counting};
use std::{future::Future, pin::Pin};

/// Adapter contract with borrowing, Send futures.
pub trait LlmBackend: Send + Sync {
    /// The parsed model URL used to construct this client.
    fn model_url(&self) -> &ModelUrl;

    /// The options this client was constructed with.
    ///
    /// Options remain immutable; adapter transformations are operation-local.
    fn options(&self) -> &LlmOptions;

    /// Estimates the full request locally, including adapter instructions and tools.
    /// This padded estimate supports monitoring, not strict context admission.
    /// Media or unsupported request components return an error.
    ///
    /// ```no_run
    /// use rath::llm::{LlmBackend, Message, RathError};
    /// fn monitor(client: &impl LlmBackend, messages: &[Message]) -> Result<u64, RathError> {
    ///     Ok(client.estimate_tokens(messages)?.input_tokens)
    /// }
    /// ```
    fn estimate_tokens(&self, _messages: &[Message]) -> Result<TokenCount, RathError> {
        Err(counting::unsupported(
            self.provider(),
            "local token estimation",
        ))
    }

    /// Requests the provider's full input count without generation or fallback.
    /// Provider counts may differ from subsequent generation usage.
    fn count_tokens(
        &self,
        _messages: &[Message],
    ) -> impl std::future::Future<Output = Result<TokenCount, RathError>> + Send {
        async move {
            Err(counting::unsupported(
                self.provider(),
                "provider token counting",
            ))
        }
    }

    /// Estimates one user message on this model, excluding client instructions/options.
    /// Includes minimal-request framing; compare directly to a summary budget.
    /// Never subtract padded request estimates to measure standalone content.
    fn estimate_content_tokens(&self, _content: &str) -> Result<TokenCount, RathError> {
        Err(counting::unsupported(
            self.provider(),
            "standalone token estimation",
        ))
    }

    /// Counts a minimal one-user-message request using this model and endpoint.
    /// Excludes client instructions/options but includes provider framing overhead.
    fn count_content_tokens(
        &self,
        _content: &str,
    ) -> impl std::future::Future<Output = Result<TokenCount, RathError>> + Send {
        async move {
            Err(counting::unsupported(
                self.provider(),
                "standalone provider counting",
            ))
        }
    }

    /// The provider backing this client instance.
    fn provider(&self) -> Provider {
        self.model_url().provider.clone()
    }

    /// Dispatches a validated request and rejects token-limited output before interpreting it.
    /// Generates once using fixed client options; token-limited output is an error.
    fn execute(
        &self,
        messages: &[Message],
    ) -> impl std::future::Future<Output = Result<LlmResponse, RathError>> + Send;
}

type ErasedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
trait ErasedLlmBackend: Send + Sync {
    fn model_url(&self) -> &ModelUrl;
    fn options(&self) -> &LlmOptions;
    fn provider(&self) -> Provider;
    fn estimate_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError>;
    fn estimate_content_tokens(&self, content: &str) -> Result<TokenCount, RathError>;
    fn count_tokens<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> ErasedFuture<'a, Result<TokenCount, RathError>>;
    fn count_content_tokens<'a>(
        &'a self,
        content: &'a str,
    ) -> ErasedFuture<'a, Result<TokenCount, RathError>>;
    fn execute<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> ErasedFuture<'a, Result<LlmResponse, RathError>>;
}
impl<B: LlmBackend> ErasedLlmBackend for B {
    fn model_url(&self) -> &ModelUrl {
        LlmBackend::model_url(self)
    }
    fn options(&self) -> &LlmOptions {
        LlmBackend::options(self)
    }
    fn provider(&self) -> Provider {
        LlmBackend::provider(self)
    }
    fn estimate_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError> {
        LlmBackend::estimate_tokens(self, messages)
    }
    fn estimate_content_tokens(&self, content: &str) -> Result<TokenCount, RathError> {
        LlmBackend::estimate_content_tokens(self, content)
    }
    fn count_tokens<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> ErasedFuture<'a, Result<TokenCount, RathError>> {
        Box::pin(LlmBackend::count_tokens(self, messages))
    }
    fn count_content_tokens<'a>(
        &'a self,
        content: &'a str,
    ) -> ErasedFuture<'a, Result<TokenCount, RathError>> {
        Box::pin(LlmBackend::count_content_tokens(self, content))
    }
    fn execute<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> ErasedFuture<'a, Result<LlmResponse, RathError>> {
        Box::pin(LlmBackend::execute(self, messages))
    }
}
/// Owned model-bound capability handle; share explicitly using `Arc`.
pub struct LlmClient {
    inner: Box<dyn ErasedLlmBackend>,
}
impl LlmClient {
    /// Wraps a prepared backend without I/O.
    pub fn from_backend<B: LlmBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }
    /// Forwards `model_url` to the owned backend.
    pub fn model_url(&self) -> &ModelUrl {
        self.inner.model_url()
    }
    /// Forwards `options` to the owned backend.
    pub fn options(&self) -> &LlmOptions {
        self.inner.options()
    }
    /// Forwards `provider` to the owned backend.
    pub fn provider(&self) -> Provider {
        self.inner.provider()
    }
    /// Forwards `estimate_tokens` to the owned backend.
    pub fn estimate_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError> {
        self.inner.estimate_tokens(messages)
    }
    /// Forwards `estimate_content_tokens` to the owned backend.
    pub fn estimate_content_tokens(&self, content: &str) -> Result<TokenCount, RathError> {
        self.inner.estimate_content_tokens(content)
    }
    /// Forwards `count_tokens` to the owned backend.
    pub async fn count_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError> {
        self.inner.count_tokens(messages).await
    }
    /// Forwards `count_content_tokens` to the owned backend.
    pub async fn count_content_tokens(&self, content: &str) -> Result<TokenCount, RathError> {
        self.inner.count_content_tokens(content).await
    }
    /// Forwards `execute` to the owned backend.
    pub async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        self.inner.execute(messages).await
    }
}
impl LlmBackend for LlmClient {
    fn model_url(&self) -> &ModelUrl {
        LlmClient::model_url(self)
    }
    fn options(&self) -> &LlmOptions {
        LlmClient::options(self)
    }
    fn provider(&self) -> Provider {
        LlmClient::provider(self)
    }
    fn estimate_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError> {
        LlmClient::estimate_tokens(self, messages)
    }
    fn estimate_content_tokens(&self, content: &str) -> Result<TokenCount, RathError> {
        LlmClient::estimate_content_tokens(self, content)
    }
    async fn count_tokens(&self, messages: &[Message]) -> Result<TokenCount, RathError> {
        LlmClient::count_tokens(self, messages).await
    }
    async fn count_content_tokens(&self, content: &str) -> Result<TokenCount, RathError> {
        LlmClient::count_content_tokens(self, content).await
    }
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        LlmClient::execute(self, messages).await
    }
}
