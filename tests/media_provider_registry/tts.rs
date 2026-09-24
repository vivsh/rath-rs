use super::http::serve;
use rath::audio::tts::TtsBackend;
use rath::audio::tts::{TtsClient, TtsOptions, TtsRequest, TtsResponse};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ErrorBody, ErrorKind, ModelUrl, Provider, RathError};
use serde_json::{Value, json};

struct SpeechProvider;
struct SpeechClient {
    http: reqwest::Client,
    url: ModelUrl,
}

impl ProviderFactory for SpeechProvider {
    async fn tts(&self, url: &ModelUrl, _options: TtsOptions) -> Result<TtsClient, RathError> {
        Ok(TtsClient::from_backend(SpeechClient {
            http: reqwest::Client::new(),
            url: url.clone(),
        }))
    }
}

impl TtsBackend for SpeechClient {
    /// Performs one synthetic provider request and normalizes errors using only the public API.
    async fn synthesize_speech(&self, request: &TtsRequest) -> Result<TtsResponse, RathError> {
        let secrets = self.url.api_key().into_iter().collect::<Vec<_>>();
        let response = self
            .http
            .post(self.url.base_url().unwrap())
            .json(&json!({"model":self.url.model(),"input":request.input}))
            .send()
            .await
            .map_err(|e| RathError::from_error(ErrorKind::Transport, &e, &secrets))?;
        let status = response.status().as_u16();
        let body = response.bytes().await.unwrap().to_vec();
        decode(status, &body, &secrets).map_err(|error| {
            error
                .with_context(self.url.provider().clone(), "speech synthesis")
                .sanitized(&secrets)
        })
    }
}

/// Preserves original wire bytes for HTTP and JSON failures without logging private audio/text.
fn decode(status: u16, body: &[u8], secrets: &[&str]) -> Result<TtsResponse, RathError> {
    let result = if status != 200 {
        Err(RathError::new(ErrorKind::Http, "voice is unavailable")
            .with_provider_code("unsupported_voice", secrets)
            .with_request_id("speech-request", secrets)
            .with_retry_after("10", secrets))
    } else {
        serde_json::from_slice::<Value>(body)
            .map(|raw| {
                let mut value = TtsResponse::new("audio/wav".into(), vec![1, 2, 3]);
                value.raw_metadata = Some(raw);
                value
            })
            .map_err(|cause| RathError::from_error(ErrorKind::Deserialize, &cause, secrets))
    };
    result.map_err(|error| {
        error
            .with_http_status(status)
            .with_body(ErrorBody::Complete(body.to_vec()), secrets)
    })
}

/// Registration and construction make no requests; the client survives registry disposal.
#[tokio::test]
async fn external_tts_executes_after_registry_is_dropped() {
    let (base, requests) = serve(200, r#"{"audio":"synthetic"}"#);
    let registry = ProviderRegistry::new()
        .register("acme", SpeechProvider)
        .unwrap();
    let client = registry
        .tts(
            &format!("acme:///nested/speech?base_url={base}"),
            TtsOptions::default(),
        )
        .await
        .unwrap();
    assert!(requests.try_recv().is_err());
    drop(registry);
    let response = client
        .synthesize_speech(&{
            let mut value = TtsRequest::default();
            value.input = "hello".into();
            value
        })
        .await
        .unwrap();
    assert_eq!(response.data, [1, 2, 3]);
    let wire = requests.try_recv().unwrap();
    assert!(wire.contains("nested/speech"));
    assert!(wire.contains("hello"));
    assert!(requests.try_recv().is_err());
}

/// HTTP rejection and malformed output retain typed evidence, with no fallback execution.
#[tokio::test]
async fn external_tts_preserves_failures() {
    for (status, body, kind) in [
        (422, "PRIVATE-AUDIO-INPUT", ErrorKind::Http),
        (200, "{", ErrorKind::Deserialize),
    ] {
        let (base, requests) = serve(status, body);
        let registry = ProviderRegistry::new()
            .register("acme", SpeechProvider)
            .unwrap();
        let client = registry
            .tts(
                &format!("acme:///speech?base_url={base}"),
                TtsOptions::default(),
            )
            .await
            .unwrap();
        let error = client
            .synthesize_speech(&TtsRequest::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(error.provider(), Some(&Provider::External("acme".into())));
        assert_eq!(error.http_status(), Some(status));
        assert_eq!(error.response_body().unwrap().bytes(), body.as_bytes());
        assert!(!format!("{error} {error:?}").contains("PRIVATE-AUDIO-INPUT"));
        if status == 422 {
            assert_eq!(error.provider_code(), Some("unsupported_voice"));
            assert_eq!(error.request_id(), Some("speech-request"));
            assert_eq!(error.retry_after(), Some("10"));
            assert!(error.to_string().contains("voice is unavailable"));
        }
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// A provider with no overridden constructors reports the unsupported capability and its identity.
#[tokio::test]
async fn unsupported_constructor_is_informative() {
    struct Unsupported;
    impl ProviderFactory for Unsupported {}
    let registry = ProviderRegistry::default()
        .register("acme", Unsupported)
        .unwrap();
    let error = registry
        .tts("acme:///speech", TtsOptions::default())
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(error.provider(), Some(&Provider::External("acme".into())));
    assert!(error.to_string().contains("text-to-speech"));
    assert!(error.source().is_some());
}
