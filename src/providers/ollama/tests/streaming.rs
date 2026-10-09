use super::*;
use futures_util::StreamExt;

/// Creates a compatible native choice frame using the selected first candidate.
fn frame(delta: Value, finish: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    )
}

/// Builds an isolated adapter without changing the production endpoint or authentication pattern.
fn client(base_url: String, options: LlmOptions) -> OllamaClient {
    OllamaClient {
        http: HttpClient::new(),
        api_key: Some("test".into()),
        base_url,
        model: "m".into(),
        options,
        url: ModelUrl::parse("ollama:///m").unwrap(),
    }
}

/// Private final-output calls become structured completion without leaking tool progress.
#[tokio::test]
async fn converts_private_final_tool_after_terminal_checks() {
    let body = format!(
        "{}{}data: [DONE]\n\n",
        frame(
            json!({"tool_calls":[{"index":0,"id":"final","function":{"name":"__rath_final_output","arguments":"{\"answer\":"}}]}),
            Value::Null
        ),
        frame(
            json!({"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}),
            json!("tool_calls")
        )
    );
    let (base, requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_tools(vec![ToolDefinition::new(
            "lookup".into(),
            "lookup".into(),
            json!({"type":"object"}),
        )]);
    let client = client(base, options);
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("private tool escaped")
    };
    let prepared = crate::llm::output::prepare(Provider::Ollama, "m", &client.options).unwrap();
    let expected = map_response(json!({"choices":[{"message":{"content":null,"tool_calls":[{"id":"final","type":"function","function":{"name":"__rath_final_output","arguments":"{\"answer\":1}"}}]},"finish_reason":"tool_calls"}]}), &prepared).unwrap();
    let expected = crate::llm::output::finish(&client.options, expected).unwrap();
    crate::providers::tests::http::assert_response_eq(&response, &expected);
    assert!(matches!(response.output, LlmOutput::Output(value) if value == json!({"answer":1})));
    assert!(stream.next().await.is_none());
    let request = requests.recv().unwrap();
    let enabled = !prepared.tools.is_empty() && prepared.tool_choice != ToolChoice::Disabled;
    crate::providers::tests::http::assert_transport_only(
        &request,
        build_payload("m", &prepared, &[Message::user("hello")], enabled),
    );
    assert!(request.starts_with("POST /v1/chat/completions "));
    assert!(request.contains("__rath_final_output"));
    assert_eq!(client.options.tools.len(), 1);
}

/// Typed call progress is provisional until Ollama's existing offered-tool policy passes.
#[tokio::test]
async fn validates_offered_tools_before_completion() {
    for offered in [true, false] {
        let body = format!(
            "{}data: [DONE]\n\n",
            frame(
                json!({"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"n\":2}"}}]}),
                json!("tool_calls")
            )
        );
        let (base, _requests) =
            crate::providers::tests::http::serve_content(200, "text/event-stream", body);
        let options = if offered {
            LlmOptions::default().with_tools(vec![ToolDefinition::new(
                "lookup".into(),
                "lookup".into(),
                json!({"type":"object"}),
            )])
        } else {
            LlmOptions::default()
        };
        let client = client(base, options);
        let mut stream = client
            .execute_stream(&[Message::user("hello")])
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            LlmEvent::ToolCallDelta { index: 0, .. }
        ));
        let final_event = stream.next().await.unwrap();
        if offered {
            assert!(matches!(final_event.unwrap(), LlmEvent::Completed { .. }));
        } else {
            assert_eq!(
                final_event.err().unwrap().kind(),
                crate::ErrorKind::InvalidResponse
            );
        }
        assert!(stream.next().await.is_none());
    }
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
