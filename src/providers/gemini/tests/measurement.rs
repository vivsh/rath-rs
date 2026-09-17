use super::super::*;
use serde_json::json;

fn client(options: LlmOptions) -> GeminiClient {
    GeminiClient {
        client: Gemini::with_model("test", GeminiModel::Custom("models/selected-model".into()))
            .unwrap(),
        options,
        url: ModelUrl::parse("gemini:///selected-model").unwrap(),
        exit_tool_name: None,
    }
}

/// Supplied system messages join the persona and survive both generation and measurement.
#[test]
fn includes_system_instructions_and_excludes_options_for_content() {
    let c = client(
        LlmOptions::default()
            .with_preamble("persona")
            .with_output_schema(json!({"type":"object"}))
            .with_max_output_tokens(17),
    );
    let messages = [
        Message {
            key: None,
            role: Role::System,
            content: "retained summary".into(),
            attachments: vec![],
            usage: None,
        },
        Message::user("current"),
    ];
    let request = request::build_request(&c.client, &c.url.model, &c.options, &messages, false)
        .unwrap()
        .build();
    let payload = serde_json::to_value(request).unwrap();
    assert_eq!(
        payload["systemInstruction"]["parts"][0]["text"],
        "persona\n\nretained summary"
    );
    assert_eq!(payload["generationConfig"]["maxOutputTokens"], 17);
    assert!(
        c.estimate_tokens(&messages).unwrap().input_tokens
            > c.estimate_content_tokens("current").unwrap().input_tokens
    );
    assert_eq!(
        c.estimate_content_tokens("current").unwrap(),
        client(LlmOptions::default())
            .estimate_content_tokens("current")
            .unwrap()
    );
}

/// Minimal provider-count requests omit thinking and all configured instructions and caps.
#[tokio::test]
async fn provider_count_uses_full_request_envelope() {
    let (url, rx) = crate::providers::tests::http::serve(200, json!({"totalTokens": 27}));
    let mut c = client(
        LlmOptions::default()
            .with_preamble("secret persona")
            .with_max_output_tokens(42),
    );
    c.client = gemini_rust::client::GeminiBuilder::new("test")
        .with_model(GeminiModel::Custom("models/selected-model".into()))
        .with_base_url(url.parse().unwrap())
        .build()
        .unwrap();
    let count = c.count_content_tokens("summary").await.unwrap();
    assert_eq!(count.input_tokens, 27);
    let request = rx.recv().unwrap();
    assert!(request.lines().next().unwrap().contains(":countTokens"));
    let value: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    let body = &value["generateContentRequest"];
    assert_eq!(body["model"], "models/selected-model");
    assert!(body["systemInstruction"].is_null());
    assert!(body["generationConfig"]["thinkingConfig"].is_null());
    assert!(body["generationConfig"]["maxOutputTokens"].is_null());
    assert_eq!(body["contents"][0]["parts"][0]["text"], "summary");
}

/// Limit signals win over empty/invalid JSON and partial function-call handling.
#[test]
fn truncation_precedes_parsing() {
    let response: GenerationResponse = serde_json::from_value(json!({"candidates":[{"finishReason":"MAX_TOKENS","content":{"role":"model","parts":[{"text":"{"}]}}]})).unwrap();
    assert!(matches!(
        map_response(response, true),
        Err(RathError::OutputLimitReached { .. })
    ));
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

/// Native Gemini counting receives system instructions, sanitized schemas and tool declarations.
#[tokio::test]
async fn full_native_count_matches_generation_request() {
    let (url, rx) = crate::providers::tests::http::serve(200, json!({"totalTokens": 99}));
    let mut c = client(
        LlmOptions::default()
            .with_preamble("persona")
            .with_input_schema(json!({"type":"object"}))
            .with_output_schema(json!({"type":"object", "properties":{"answer":{"type":"string"}}}))
            .with_tools(vec![ToolDefinition {
                name: "lookup".into(),
                description: "Look up".into(),
                parameters: json!({"type":"object"}),
            }]),
    );
    c.client = gemini_rust::GeminiBuilder::new("test")
        .with_model(GeminiModel::Custom("models/selected-model".into()))
        .with_base_url(url.parse().unwrap())
        .build()
        .unwrap();
    let messages = [
        Message {
            key: None,
            role: Role::System,
            content: "summary".into(),
            attachments: vec![],
            usage: None,
        },
        Message::user("current"),
    ];
    let expected = serde_json::to_value(
        request::build_request(&c.client, &c.url.model, &c.options, &messages, false)
            .unwrap()
            .build(),
    )
    .unwrap();
    assert_eq!(c.count_tokens(&messages).await.unwrap().input_tokens, 99);
    let request = rx.recv().unwrap();
    let wire: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    for field in [
        "contents",
        "systemInstruction",
        "generationConfig",
        "tools",
        "toolConfig",
    ] {
        assert_eq!(wire["generateContentRequest"][field], expected[field]);
    }
}

/// Missing counting endpoints and provider failures remain errors without fallback generation.
#[tokio::test]
async fn native_count_failure_is_explicit() {
    for status in [404, 401, 429, 500] {
        let (url, rx) =
            crate::providers::tests::http::serve(status, json!({"error":"count failed"}));
        let mut c = client(LlmOptions::default());
        c.client = gemini_rust::GeminiBuilder::new("test")
            .with_model(GeminiModel::Custom("models/selected-model".into()))
            .with_base_url(url.parse().unwrap())
            .build()
            .unwrap();
        let error = c.count_content_tokens("summary").await.unwrap_err();
        if status == 404 {
            assert!(matches!(error, RathError::UnsupportedCapability { .. }));
        } else {
            assert!(matches!(error, RathError::Provider(_)));
        }
        assert!(
            rx.recv()
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .contains(":countTokens")
        );
    }
}

/// Application keys are absent from Gemini's shared generation and counting request.
#[test]
fn message_keys_never_enter_provider_payloads() {
    let c = client(LlmOptions::default());
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
    let plain = request::build_request(&c.client, &c.url.model, &c.options, &messages, false)
        .unwrap()
        .build();
    let keyed = request::build_request(&c.client, &c.url.model, &c.options, &keyed, false)
        .unwrap()
        .build();
    assert_eq!(
        serde_json::to_value(plain).unwrap(),
        serde_json::to_value(keyed).unwrap()
    );
}
