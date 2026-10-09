use crate::llm::*;
use futures_util::{StreamExt, stream};

struct Legacy {
    url: ModelUrl,
    options: LlmOptions,
}

impl LlmBackend for Legacy {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }
    fn options(&self) -> &LlmOptions {
        &self.options
    }
    async fn execute(&self, _messages: &[Message]) -> Result<LlmResponse, RathError> {
        Err(RathError::unsupported(
            Provider::OpenAi,
            "fixture generation",
        ))
    }
}

struct Streaming(Legacy);
impl LlmBackend for Streaming {
    fn model_url(&self) -> &ModelUrl {
        self.0.model_url()
    }
    fn options(&self) -> &LlmOptions {
        self.0.options()
    }
    async fn execute(&self, messages: &[Message]) -> Result<LlmResponse, RathError> {
        self.0.execute(messages).await
    }
    async fn execute_stream<'a>(
        &'a self,
        _messages: &[Message],
    ) -> Result<LlmStream<'a>, RathError> {
        Ok(Box::pin(stream::iter([
            Ok(LlmEvent::TextDelta {
                text: self.model_url().model().into(),
            }),
            Ok(LlmEvent::Completed {
                response: LlmResponse::new(Provider::OpenAi, LlmOutput::Output("done".into())),
            }),
        ])))
    }
}

/// Constructs the old backend contract without any new method or network activity.
fn legacy() -> Legacy {
    Legacy {
        url: ModelUrl::parse("openai:///test").unwrap(),
        options: LlmOptions::default(),
    }
}

/// Existing implementations remain source-compatible and never simulate streaming via execute.
#[tokio::test]
async fn old_backend_defaults_to_explicit_unsupported() {
    let client = LlmClient::from_backend(legacy());
    let error = client
        .execute_stream(&[Message::user("hello")])
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::UnsupportedCapability);
    assert!(error.to_string().contains("LLM response streaming"));
}

/// Private erasure preserves startup-only history borrowing and permanent terminal EOF.
#[tokio::test]
async fn handle_forwards_without_retaining_input_history() {
    let client = LlmClient::from_backend(Streaming(legacy()));
    let messages = vec![Message::user("hello")];
    let mut events = client.execute_stream(&messages).await.unwrap();
    drop(messages);
    assert!(
        matches!(events.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "test")
    );
    assert!(matches!(
        events.next().await.unwrap().unwrap(),
        LlmEvent::Completed { .. }
    ));
    assert!(events.next().await.is_none());
    assert!(events.next().await.is_none());
}

/// Generic backend calls and client forwarding retain Send futures and streams.
#[test]
fn streaming_remains_send() {
    fn send<T: Send>(_: T) {}
    let client = LlmClient::from_backend(Streaming(legacy()));
    let messages = [Message::user("hello")];
    send(LlmBackend::execute_stream(&client, &messages));
    send(async {
        let _ = client.execute_stream(&messages).await?.next().await;
        Ok::<_, RathError>(())
    });
}
