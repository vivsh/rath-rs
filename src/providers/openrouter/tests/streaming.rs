use super::*;
use futures_util::StreamExt;

/// Creates one compatible SSE frame without a named event.
fn frame(delta: Value, finish: Value, usage: Value) -> String {
    let value = json!({"id":"r", "model":"echo", "object":"chat.completion.chunk", "choices":[{"index":0,"delta":delta,"finish_reason":finish}], "usage":usage});
    format!("data: {value}\n\n")
}

/// Builds an isolated adapter with the normal immutable model/options fields.
fn client(base_url: String, options: LlmOptions) -> OpenRouterClient {
    OpenRouterClient {
        http: HttpClient::new(),
        api_key: "test".into(),
        base_url,
        model: "m".into(),
        options,
        url: ModelUrl::parse("openrouter:///m").unwrap(),
    }
}

/// Repeated finish metadata in the trailing usage chunk does not repeat text or complete early.
#[tokio::test]
async fn invalid_schema_output_waits_for_terminal_validation() {
    let body = format!(
        ": keepalive\n\n{}{}{}data: [DONE]\n\n",
        frame(json!({"content":"Hello"}), Value::Null, Value::Null),
        frame(json!({}), json!("stop"), Value::Null),
        frame(
            json!({"content":""}),
            json!("stop"),
            json!({"prompt_tokens":7,"completion_tokens":3})
        )
    );
    let (base, requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(
        base,
        LlmOptions::default().with_output_schema(json!({"type":"string"})),
    );
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "Hello")
    );
    // This fixture intentionally returns invalid JSON despite a schema request.
    assert_eq!(
        stream.next().await.unwrap().err().unwrap().kind(),
        crate::ErrorKind::Deserialize
    );
    assert!(stream.next().await.is_none());
    let request = requests.recv().unwrap();
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    crate::providers::tests::http::assert_transport_only(
        &request,
        build_payload("m", &client.options, &[Message::user("hello")], enabled),
    );
    assert!(request.starts_with("POST /chat/completions "));
    let payload: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["stream_options"]["include_usage"], true);
    assert!(payload.get("response_format").is_some());
}

/// Native errors after answer progress yield one error without a final response.
#[tokio::test]
async fn rejects_midstream_provider_error() {
    let body = format!(
        "{}data: {{\"error\":{{\"code\":\"overloaded\",\"message\":\"retry later\"}}}}\n\n",
        frame(json!({"content":"Hi"}), Value::Null, Value::Null)
    );
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Provider);
    assert_eq!(error.provider_code(), Some("overloaded"));
    assert!(stream.next().await.is_none());
}

/// A valid text result includes latest usage and the echoed model without duplicate deltas.
#[tokio::test]
async fn completes_text_with_reported_usage() {
    let body = format!(
        "{}{}data: [DONE]\n\n",
        frame(json!({"content":"Hello"}), json!("stop"), Value::Null),
        frame(
            json!({}),
            json!("stop"),
            json!({"prompt_tokens":7,"completion_tokens":3})
        )
    );
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("expected completed")
    };
    let expected = map_response(json!({"id":"r", "model":"echo", "object":"chat.completion.chunk", "choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}], "usage":{"prompt_tokens":7,"completion_tokens":3}}), false).unwrap();
    crate::providers::tests::http::assert_response_eq(&response, &expected);
    assert!(matches!(response.output, LlmOutput::Output(text) if text == "Hello"));
    assert_eq!(response.usage.unwrap().output, Some(3));
    assert_eq!(response.provider_model.as_deref(), Some("echo"));
    assert!(stream.next().await.is_none());
}

/// Invocation-owned stream controls cannot be overridden through native configuration.
#[tokio::test]
async fn rejects_reserved_stream_controls_before_io() {
    for key in ["stream", "stream_options"] {
        let options = LlmOptions::default().with_provider_config(json!({key:false}));
        let client = client("http://127.0.0.1:1".into(), options);
        let error = client
            .execute_stream(&[Message::user("hello")])
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), crate::ErrorKind::Validation);
    }
}

/// A choice-level stream failure cannot emit its partial content or lose provider diagnostics.
#[tokio::test]
async fn rejects_nested_choice_error_before_partial_content() {
    let error_chunk = json!({"choices":[{"index":0,"delta":{"content":"PRIVATE-OUTPUT"},"finish_reason":"error","error":{"code":502,"message":"upstream disconnected"}}]});
    let body = format!(
        "{}data: {error_chunk}\n\n",
        frame(json!({"content":"Hi"}), Value::Null, Value::Null)
    );
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body.clone());
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "Hi")
    );
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Provider);
    assert_eq!(error.http_status(), Some(200));
    assert_eq!(error.provider_code(), Some("502"));
    assert!(error.to_string().contains("upstream disconnected"));
    assert_eq!(error.response_body().unwrap().bytes(), body.as_bytes());
    assert!(!format!("{error:?}").contains("PRIVATE-OUTPUT"));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
}
