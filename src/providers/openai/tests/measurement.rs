use super::super::*;

/// Constructs an isolated test client without contacting a provider.
fn client(options: LlmOptions) -> OpenAiClient {
    OpenAiClient {
        http: HttpClient::new(),
        api_key: "test".into(),
        base_url: "http://127.0.0.1:1".into(),
        model: "selected-model".into(),
        url: ModelUrl::parse("openai:///selected-model").unwrap(),
        provider_config: options.provider_config.clone(),
        options,
    }
}

/// Standalone measurement excludes persona/schema/cap and includes minimal framing.
#[test]
fn measures_whole_request_and_minimal_content() {
    let options = LlmOptions::default()
        .with_preamble("persona ".repeat(100))
        .with_output_schema(json!({"type": "object", "properties": {"answer": {"type":"string"}}}))
        .with_max_output_tokens(100);
    let configured = client(options);
    let bare = client(LlmOptions::default());
    let messages = [Message::user("summary")];
    assert!(
        configured.estimate_tokens(&messages).unwrap().input_tokens
            > bare.estimate_tokens(&messages).unwrap().input_tokens
    );
    assert_eq!(
        configured.estimate_content_tokens("summary").unwrap(),
        bare.estimate_tokens(&messages).unwrap()
    );
    assert_eq!(
        client(LlmOptions::default().with_max_output_tokens(1))
            .estimate_tokens(&messages)
            .unwrap(),
        bare.estimate_tokens(&messages).unwrap()
    );
}

/// Token-limit detection takes precedence over malformed JSON or partial tool parsing.
#[test]
fn cap_and_truncation_are_explicit() {
    let payload = build_payload(
        "m",
        &LlmOptions::default().with_max_output_tokens(7),
        &[Message::user("x")],
        false,
    );
    assert_eq!(payload["max_output_tokens"], 7);
    for output in [json!([]), json!([{"type":"function_call","arguments":"{"}])] {
        let response = json!({"status":"incomplete", "incomplete_details":{"reason":"max_output_tokens"}, "output":output});
        assert!(matches!(
            map_response(response, true),
            Err(RathError::OutputLimitReached { .. })
        ));
    }
}

/// Explicit counting sends the final prompt projection only to the configured endpoint.
#[tokio::test]
async fn native_count_uses_configured_endpoint_and_full_input() {
    let (base_url, requests) =
        crate::providers::tests::http::serve(200, json!({"input_tokens": 42}));
    let mut c = client(
        LlmOptions::default()
            .with_preamble("persona")
            .with_output_schema(json!({"type":"object"}))
            .with_max_output_tokens(20),
    );
    c.base_url = base_url;
    let result = c
        .count_tokens(&[Message::user("summary and current input")])
        .await
        .unwrap();
    assert_eq!(
        result,
        crate::llm::TokenCount {
            input_tokens: 42,
            source: crate::llm::TokenCountSource::ProviderReported
        }
    );
    let request = requests.recv().unwrap();
    assert!(request.starts_with("POST /responses/input_tokens "));
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["model"], "selected-model");
    assert_eq!(body["instructions"], "persona");
    assert!(body.get("text").is_some());
    assert!(body.get("max_output_tokens").is_none());
}

/// Endpoint absence and other failures remain explicit without generation fallback.
#[tokio::test]
async fn native_count_does_not_fallback() {
    for status in [404, 401, 429, 500] {
        let (base_url, requests) = crate::providers::tests::http::serve(status, json!({}));
        let mut c = client(LlmOptions::default());
        c.base_url = base_url;
        let error = c.count_content_tokens("summary").await.unwrap_err();
        if status == 404 {
            assert!(matches!(error, RathError::UnsupportedCapability { .. }));
        } else {
            assert!(matches!(error, RathError::Provider(_)));
        }
        assert!(
            requests
                .recv()
                .unwrap()
                .starts_with("POST /responses/input_tokens ")
        );
    }
}

/// A cap rejected by the deployment is not removed or retried.
#[tokio::test]
async fn rejected_cap_is_not_retried() {
    let (base_url, requests) =
        crate::providers::tests::http::serve(400, json!({"error":"unsupported cap"}));
    let mut c = client(LlmOptions::default().with_max_output_tokens(7));
    c.base_url = base_url;
    assert!(matches!(
        c.execute(&[Message::user("x")]).await,
        Err(RathError::Provider(_))
    ));
    let request = requests.recv().unwrap();
    assert!(request.starts_with("POST /responses "));
    assert!(request.contains("\"max_output_tokens\":7"));
}

/// Directly assigned options are revalidated immediately before execution.
#[tokio::test]
async fn invalid_execution_cap_never_dispatches() {
    let mut c = client(LlmOptions::default());
    c.options.max_output_tokens = Some(0);
    assert!(matches!(
        c.execute(&[Message::user("x")]).await,
        Err(RathError::Validation(_))
    ));
    assert!(matches!(
        c.estimate_tokens(&[Message::user("x")]),
        Err(RathError::Validation(_))
    ));
}

/// Minimal native requests contain only the selected model and standalone user input.
#[tokio::test]
async fn standalone_native_count_excludes_client_options() {
    let (url, rx) = crate::providers::tests::http::serve(200, json!({"input_tokens": 14}));
    let mut c = client(
        LlmOptions::default()
            .with_preamble("private persona")
            .with_output_schema(json!({"type":"object"}))
            .with_max_output_tokens(20),
    );
    c.base_url = url;
    c.count_content_tokens("summary").await.unwrap();
    let request = rx.recv().unwrap();
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({"model":"selected-model", "input":[{"role":"user", "content":"summary"}]})
    );
}

/// A missing model is a provider validation failure, not proof that counting is unsupported.
#[tokio::test]
async fn missing_model_is_not_endpoint_absence() {
    let (url, _requests) =
        crate::providers::tests::http::serve(404, json!({"error":{"code":"model_not_found"}}));
    let mut c = client(LlmOptions::default());
    c.base_url = url;
    assert!(matches!(
        c.count_content_tokens("summary").await,
        Err(RathError::Provider(_))
    ));
}

/// Application keys are absent from the shared generation and counting payload builder.
#[test]
fn message_keys_never_enter_provider_payloads() {
    let messages = [
        Message::user("hello"),
        Message::assistant("welcome"),
        Message::user("current"),
    ];
    let keyed: Vec<_> = messages
        .iter()
        .cloned()
        .map(|message| message.with_key("PRIVATE-APPLICATION-KEY"))
        .collect();
    let options = LlmOptions::default();
    assert_eq!(
        build_payload("model", &options, &messages, false),
        build_payload("model", &options, &keyed, false)
    );
}
