use super::*;
use futures_util::StreamExt;

/// Encodes native Messages events including their matching SSE event names.
fn frame(value: &Value) -> String {
    format!(
        "event: {}\ndata: {value}\n\n",
        value["type"].as_str().unwrap()
    )
}

/// Builds an isolated adapter using the existing credential and model fields.
fn client(base_url: String, options: LlmOptions) -> AnthropicClient {
    AnthropicClient {
        http: HttpClient::new(),
        api_key: "test".into(),
        base_url,
        model: "m".into(),
        options,
        url: ModelUrl::parse("anthropic:///m").unwrap(),
    }
}

/// Supplies text and tool blocks, including fragmented JSON and cumulative final usage.
fn events(reason: &str, args: &str) -> Vec<Value> {
    vec![
        json!({"type":"message_start", "message":{"id":"msg", "type":"message", "role":"assistant", "model":"echo", "content":[], "stop_reason":null, "usage":{"input_tokens":9,"output_tokens":1}}}),
        json!({"type":"ping"}),
        json!({"type":"content_block_start", "index":0, "content_block":{"type":"text", "text":""}}),
        json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":"Checking"}}),
        json!({"type":"content_block_stop", "index":0}),
        json!({"type":"content_block_start", "index":1, "content_block":{"type":"tool_use", "id":"call", "name":"lookup", "input":{}}}),
        json!({"type":"content_block_delta", "index":1, "delta":{"type":"input_json_delta", "partial_json":args}}),
        json!({"type":"content_block_stop", "index":1}),
        json!({"type":"message_delta", "delta":{"stop_reason":reason,"stop_sequence":null}, "usage":{"output_tokens":4}}),
        json!({"type":"message_stop"}),
    ]
}

/// The native HTTP path streams sparse tool indices and preserves final text, calls and usage.
#[tokio::test]
async fn messages_stream_preserves_tools_and_usage() {
    let body: String = events("tool_use", "{\"city\":\"Delhi\"}")
        .iter()
        .map(frame)
        .collect();
    let (base, requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "Checking")
    );
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::ToolCallDelta { index:1, name_delta: Some(name), arguments_delta, .. } if name == "lookup" && arguments_delta.is_empty())
    );
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::ToolCallDelta { index:1, arguments_delta, .. } if arguments_delta == "{\"city\":\"Delhi\"}")
    );
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("expected completed")
    };
    let expected = map_response(json!({"id":"msg", "type":"message", "role":"assistant", "model":"echo", "content":[{"type":"text","text":"Checking"},{"type":"tool_use","id":"call","name":"lookup","input":{"city":"Delhi"}}], "stop_reason":"tool_use", "stop_sequence":null, "usage":{"input_tokens":9,"output_tokens":4}}), false).unwrap();
    crate::providers::tests::http::assert_response_eq(&response, &expected);
    let LlmOutput::ToolCalls { text, calls } = response.output else {
        panic!("expected tools")
    };
    assert_eq!(text.as_deref(), Some("Checking"));
    assert_eq!(calls[0].args["city"], "Delhi");
    let usage = response.usage.unwrap();
    assert_eq!(usage.input, Some(9));
    assert_eq!(usage.output, Some(4));
    assert!(stream.next().await.is_none());
    let request = requests.recv().unwrap();
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    crate::providers::tests::http::assert_transport_only(
        &request,
        build_payload("m", &client.options, &[Message::user("hello")], enabled),
    );
    assert!(request.starts_with("POST /messages "));
    assert!(request.contains("anthropic-version: 2023-06-01"));
    assert!(request.contains("\"stream\":true"));
}

/// A partial argument block is not parsed before later cap termination is known.
#[tokio::test]
async fn max_tokens_precedes_partial_argument_parsing() {
    let body: String = events("max_tokens", "{").iter().map(frame).collect();
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    for _ in 0..3 {
        stream.next().await.unwrap().unwrap();
    }
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::OutputLimitReached);
    assert!(stream.next().await.is_none());
}

/// Protocol block closure and malformed tool arguments remain failures rather than partial success.
#[test]
fn rejects_open_blocks_and_invalid_tool_json() {
    for skip_stop in [false, true] {
        let mut assembled = Value::Null;
        let events = events("tool_use", "{");
        let mut failed = false;
        for value in events {
            if skip_stop && value["type"] == "content_block_stop" {
                continue;
            }
            if decode(
                value["type"].as_str().unwrap(),
                &value.to_string(),
                &mut assembled,
                &LlmOptions::default(),
            )
            .is_err()
            {
                failed = true;
                break;
            }
        }
        assert!(failed);
    }
}

/// Active text blocks cannot emit fragments in an order that differs from final block order.
#[test]
fn rejects_interleaved_answer_blocks() {
    let mut values = events("end_turn", "{}");
    values.truncate(4);
    values.push(json!({"type":"content_block_start", "index":1, "content_block":{"type":"text", "text":"second"}}));
    values.push(json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta","text":"late first"}}));
    let mut assembled = Value::Null;
    assert!(values.into_iter().any(|v| {
        decode(
            v["type"].as_str().unwrap(),
            &v.to_string(),
            &mut assembled,
            &LlmOptions::default(),
        )
        .is_err()
    }));
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
