use super::super::*;

fn client(options: LlmOptions) -> AnthropicClient {
    AnthropicClient {
        http: HttpClient::new(),
        api_key: "test".into(),
        base_url: "http://127.0.0.1:1".into(),
        model: "selected-model".into(),
        url: ModelUrl::parse("anthropic:///selected-model").unwrap(),
        options,
    }
}

/// Whole-request estimation includes client instructions; standalone measurement excludes them.
#[test]
fn full_and_standalone_measurements_share_request_builders() {
    let configured = client(
        LlmOptions::default()
            .with_preamble("persona ".repeat(100))
            .with_output_schema(json!({"type":"object"}))
            .with_max_output_tokens(20),
    );
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

/// Explicit caps reach the provider and preserve absent-cap behavior.
#[test]
fn cap_reaches_payload() {
    let messages = [Message::user("x")];
    assert_eq!(
        build_payload(
            "m",
            &LlmOptions::default().with_max_output_tokens(7),
            &messages,
            false
        )["max_tokens"],
        7
    );
    assert_eq!(
        build_payload("m", &LlmOptions::default(), &messages, false)["max_tokens"],
        4096
    );
}

/// Native counting uses the same selected model and system/schema instructions.
#[tokio::test]
async fn native_count_preserves_system_and_schema() {
    let (url, rx) = crate::providers::tests::http::serve(200, json!({"input_tokens": 31}));
    let mut c = client(
        LlmOptions::default()
            .with_preamble("persona")
            .with_output_schema(json!({"type":"object"}))
            .with_max_output_tokens(20),
    );
    c.base_url = url;
    assert_eq!(
        c.count_tokens(&[Message::user("summary")])
            .await
            .unwrap()
            .input_tokens,
        31
    );
    let request = rx.recv().unwrap();
    assert!(request.starts_with("POST /messages/count_tokens "));
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["model"], "selected-model");
    assert!(body["system"].as_str().unwrap().contains("persona"));
    assert!(body["system"].as_str().unwrap().contains("JSON Schema"));
    assert!(body.get("max_tokens").is_none());
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

/// A partial JSON or tool response is rejected before decoding and is safely formatted.
#[test]
fn limited_output_is_not_success() {
    let response = json!({"stop_reason":"max_tokens", "content":[{"type":"text", "text":"PRIVATE-CONTENT-729{"}]});
    let error = map_response(response, true).unwrap_err();
    assert!(matches!(error, RathError::OutputLimitReached { .. }));
    assert!(!format!("{error:?} {error}").contains("PRIVATE-CONTENT-729"));
}

/// Minimal native requests preserve the model while excluding configured system, schema and cap.
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
    assert_eq!(body["model"], "selected-model");
    assert_eq!(
        body["messages"],
        json!([{"role":"user", "content":"summary"}])
    );
    assert!(body.get("system").is_none());
    assert!(body.get("tools").is_none());
    assert!(body.get("max_tokens").is_none());
}

/// Native failures and malformed counts never turn into estimates or generation calls.
#[tokio::test]
async fn native_failures_remain_errors() {
    for (status, response) in [
        (404, json!({})),
        (401, json!({})),
        (429, json!({})),
        (200, json!({"input_tokens": -1})),
        (200, json!({})),
    ] {
        let (url, rx) = crate::providers::tests::http::serve(status, response);
        let mut c = client(LlmOptions::default());
        c.base_url = url;
        assert!(c.count_content_tokens("summary").await.is_err());
        assert!(
            rx.recv()
                .unwrap()
                .starts_with("POST /messages/count_tokens ")
        );
    }
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
