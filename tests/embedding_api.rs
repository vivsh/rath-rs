//! Public batch construction and execution contracts.
use rath::embeddings::{EmbedRequest, EmbeddingBackend, EmbeddingClient, EmbeddingOptions};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ModelUrl, RathError};

#[path = "media_provider_registry/http.rs"]
mod http;

/// Remote construction makes no request; execution preserves batch order using indices.
#[tokio::test]
async fn openai_batch_through_registry() {
    let (endpoint, requests) = http::serve(
        200,
        r#"{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"model":"echoed","usage":{"prompt_tokens":7}}"#,
    );
    let url = format!("openai:///test?api_key_env=USER&base_url={endpoint}");
    let registry = ProviderRegistry::with_builtins();
    let client = registry
        .embeddings(&url, EmbeddingOptions::default())
        .await
        .unwrap();
    assert!(requests.try_recv().is_err());
    drop(registry);
    let request = EmbedRequest::batch(["first".into(), "second".into()]);
    fn assert_send<T: Send>(_: &T) {}
    let future = client.embed(&request);
    assert_send(&future);
    let response = future.await.unwrap();
    assert_eq!(response.embeddings, vec![vec![1., 2.], vec![3., 4.]]);
    assert_eq!(response.provider_model.as_deref(), Some("echoed"));
    let wire = requests.recv().unwrap();
    assert!(wire.starts_with("POST /embeddings "));
    let body: serde_json::Value =
        serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["input"], serde_json::json!(["first", "second"]));
}

struct Factory;
struct Backend;
impl EmbeddingBackend for Backend {
    async fn embed(
        &self,
        request: &EmbedRequest,
    ) -> Result<rath::embeddings::EmbedResponse, RathError> {
        Ok(rath::embeddings::EmbedResponse::new(
            rath::Provider::External("acme".into()),
            request.input.iter().map(|_| vec![1.]).collect(),
        ))
    }
}
impl ProviderFactory for Factory {
    async fn embeddings(
        &self,
        _: &ModelUrl,
        _: EmbeddingOptions,
    ) -> Result<EmbeddingClient, RathError> {
        Ok(EmbeddingClient::from_backend(Backend))
    }
}
/// External extension authors need no async macro or crate-private API.
#[tokio::test]
async fn external_backend_survives_registry() {
    let registry = ProviderRegistry::new().register("acme", Factory).unwrap();
    let client = registry
        .embeddings("acme:///model", EmbeddingOptions::default())
        .await
        .unwrap();
    drop(registry);
    assert_eq!(
        client
            .embed(&EmbedRequest::new("hello"))
            .await
            .unwrap()
            .embeddings,
        vec![vec![1.]]
    );
}

/// Native defaults merge shallowly and typed dimensions reach one OpenAI batch request.
#[tokio::test]
async fn openai_native_defaults_and_dimensions() {
    let (endpoint, requests) = http::serve(200, r#"{"data":[{"index":0,"embedding":[1,2]}]}"#);
    let client = EmbeddingOptions::default()
        .with_provider_config(serde_json::json!({"user":"default"}))
        .create(&format!(
            "openai:///bound?api_key_env=USER&base_url={endpoint}"
        ))
        .await
        .unwrap();
    let request = EmbedRequest::new("hello")
        .with_output_dimensionality(2)
        .with_provider_config(serde_json::json!({"user":"request"}));
    client.embed(&request).await.unwrap();
    let wire = requests.recv().unwrap();
    let payload: serde_json::Value =
        serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["user"], "request");
    assert_eq!(payload["dimensions"], 2);
    assert_eq!(payload["model"], "bound");
}
