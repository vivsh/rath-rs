use super::super::*;

fn client(options: LlmOptions) -> OpenRouterClient {
    OpenRouterClient {
        http: HttpClient::new(),
        api_key: "test".into(),
        base_url: "http://127.0.0.1:1".into(),
        model: "selected-model".into(),
        url: ModelUrl::parse("openrouter:///selected-model").unwrap(),
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
    assert!(
        build_payload("m", &LlmOptions::default(), &messages, false)
            .get("max_tokens")
            .is_none()
    );
}

/// Unsupported provider counting does not fall back to a generation request.
#[tokio::test]
async fn provider_count_is_explicitly_unsupported() {
    let c = client(LlmOptions::default());
    assert!(matches!(
        c.count_tokens(&[Message::user("x")]).await,
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
    assert!(matches!(
        c.count_content_tokens("x").await,
        Err(error) if error.kind() == crate::core::ErrorKind::UnsupportedCapability
    ));
}

/// Directly assigned options are revalidated immediately before execution.
#[tokio::test]
async fn invalid_execution_cap_never_dispatches() {
    let mut c = client(LlmOptions::default());
    c.options.max_output_tokens = Some(0);
    assert!(matches!(
        c.execute(&[Message::user("x")]).await,
        Err(error) if error.kind() == crate::core::ErrorKind::Validation
    ));
    assert!(matches!(
        c.estimate_tokens(&[Message::user("x")]),
        Err(error) if error.kind() == crate::core::ErrorKind::Validation
    ));
}

/// A partial JSON or tool response is rejected before decoding and is safely formatted.
#[test]
fn limited_output_is_not_success() {
    let response = json!({"choices":[{"finish_reason":"length", "message":{"content":"PRIVATE-CONTENT-729{", "tool_calls":[{"function":{"arguments":"{"}}]}}]});
    let error = map_response(response, true).unwrap_err();
    assert!(error.kind() == crate::core::ErrorKind::OutputLimitReached);
    assert!(!format!("{error:?} {error}").contains("PRIVATE-CONTENT-729"));
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
