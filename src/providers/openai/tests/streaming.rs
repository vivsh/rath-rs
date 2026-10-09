use super::*;
use futures_util::StreamExt;
use std::time::Duration;

/// Encodes a named Responses event exactly as the HTTP stream carries it.
fn frame(value: &Value) -> String {
    format!(
        "event: {}\ndata: {value}\n\n",
        value["type"].as_str().unwrap()
    )
}

/// Constructs an isolated model-bound adapter with the normal credential owner.
fn client(base_url: String, options: LlmOptions) -> OpenAiClient {
    OpenAiClient {
        http: HttpClient::new(),
        api_key: "STREAM-SECRET".into(),
        base_url,
        model: "selected-model".into(),
        url: ModelUrl::parse("openai:///selected-model").unwrap(),
        provider_config: options.provider_config.clone(),
        options,
    }
}

/// Supplies the normal message lifecycle with two visible text fragments.
fn text_events() -> Vec<Value> {
    let item = json!({"id":"msg1", "type":"message", "content":[]});
    let completed = json!({"id":"msg1", "type":"message", "content":[{"type":"output_text", "text":"Hello🙂"}]});
    vec![
        json!({"type":"response.output_item.added", "output_index":0, "item":item}),
        json!({"type":"response.content_part.added", "item_id":"msg1", "output_index":0, "content_index":0, "part":{"type":"output_text", "text":""}}),
        json!({"type":"response.output_text.delta", "item_id":"msg1", "output_index":0, "content_index":0, "delta":"Hello"}),
        json!({"type":"response.output_text.delta", "item_id":"msg1", "output_index":0, "content_index":0, "delta":"🙂"}),
        json!({"type":"response.output_text.done", "item_id":"msg1", "output_index":0, "content_index":0, "text":"Hello🙂"}),
        json!({"type":"response.output_item.done", "output_index":0, "item":completed}),
        json!({"type":"response.completed", "response":{"id":"response1", "model":"echoed-model", "status":"completed", "output":[completed], "usage":{"input_tokens":5, "output_tokens":3}}}),
    ]
}

/// Live local delivery yields text before the server releases its final response.
#[tokio::test]
async fn streams_before_completion_and_preserves_final_response() {
    let events = text_events();
    let early: String = events[..3].iter().map(frame).collect();
    let late: String = events[3..].iter().map(frame).collect();
    let (base, requests) = crate::providers::tests::http::serve_stream(
        vec![(early, Duration::ZERO), (late, Duration::from_millis(800))],
        0,
    );
    let client = client(
        base,
        LlmOptions::default()
            .with_preamble("persona")
            .with_max_output_tokens(10),
    );
    let messages = vec![Message::user("hello").with_key("PRIVATE-KEY")];
    let mut stream = client.execute_stream(&messages).await.unwrap();
    drop(messages);
    let first = tokio::time::timeout(Duration::from_millis(300), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(first, LlmEvent::TextDelta { text } if text == "Hello"));
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "🙂")
    );
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("expected completion")
    };
    let expected = map_response(events.last().unwrap()["response"].clone(), false).unwrap();
    crate::providers::tests::http::assert_response_eq(&response, &expected);
    assert!(matches!(response.output, LlmOutput::Output(value) if value == "Hello🙂"));
    assert_eq!(response.provider_model.as_deref(), Some("echoed-model"));
    assert_eq!(response.usage.unwrap().output, Some(3));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
    check_text_request(&requests.recv().unwrap(), &client);
}

/// Verifies endpoint, request parity, persona, typed limit and application-key privacy.
fn check_text_request(request: &str, client: &OpenAiClient) {
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    crate::providers::tests::http::assert_transport_only(
        request,
        build_payload(
            "selected-model",
            &client.options,
            &[Message::user("hello").with_key("PRIVATE-KEY")],
            enabled,
        ),
    );
    assert!(request.starts_with("POST /responses "));
    let payload: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["instructions"], "persona");
    assert_eq!(payload["max_output_tokens"], 10);
    assert!(!request.contains("PRIVATE-KEY"));
}

/// Supplies two interleaved native call lifecycles for the correlation test.
fn tool_events() -> Vec<Value> {
    let mut events = Vec::new();
    for (index, name) in [(0, "lookup"), (1, "calculate")] {
        events.push(json!({"type":"response.output_item.added", "output_index":index, "item":{"type":"function_call", "id":format!("item{index}"), "call_id":format!("call{index}"), "name":name, "arguments":""}}));
    }
    for (index, args) in [
        (0, "{\"city\":"),
        (1, "{\"n\":"),
        (0, "\"Delhi\"}"),
        (1, "2}"),
    ] {
        events.push(json!({"type":"response.function_call_arguments.delta", "output_index":index, "item_id":format!("item{index}"), "delta":args}));
    }
    let output = json!([
        {"type":"function_call", "id":"item0", "call_id":"call0", "name":"lookup", "arguments":"{\"city\":\"Delhi\"}"},
        {"type":"function_call", "id":"item1", "call_id":"call1", "name":"calculate", "arguments":"{\"n\":2}"}
    ]);
    events.push(
        json!({"type":"response.completed", "response":{"status":"completed", "output":output}}),
    );
    events
}

/// Interleaved call argument fragments remain correlated and become parsed final calls only once.
#[tokio::test]
async fn streams_parallel_tool_calls_and_validated_completion() {
    let events = tool_events();
    let body: String = events.iter().map(frame).collect();
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("tools")])
        .await
        .unwrap();
    let mut args = [String::new(), String::new()];
    let mut names = [String::new(), String::new()];
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            LlmEvent::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                assert_eq!(id.as_deref(), Some(format!("call{index}").as_str()));
                names[index].push_str(name_delta.as_deref().unwrap_or_default());
                args[index].push_str(&arguments_delta);
            }
            LlmEvent::Completed { response } => {
                let LlmOutput::ToolCalls { calls, .. } = response.output else {
                    panic!("expected tools")
                };
                assert_eq!(calls[0].args["city"], "Delhi");
                assert_eq!(calls[1].args["n"], 2);
            }
            _ => panic!("unexpected event"),
        }
    }
    assert_eq!(names, ["lookup", "calculate"]);
    assert_eq!(args, ["{\"city\":\"Delhi\"}", "{\"n\":2}"]);
}

/// Missing terminal evidence and interrupted HTTP transfers yield one error and retain wire evidence.
#[tokio::test]
async fn rejects_eof_and_transport_interruption() {
    for missing in [0, 50] {
        let body: String = text_events()[..3].iter().map(frame).collect();
        let (base, _requests) = crate::providers::tests::http::serve_stream(
            vec![(body.clone(), Duration::ZERO)],
            missing,
        );
        let client = client(base, LlmOptions::default());
        let mut stream = client
            .execute_stream(&[Message::user("hello")])
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            LlmEvent::TextDelta { .. }
        ));
        let error = stream.next().await.unwrap().err().unwrap();
        assert_eq!(
            error.kind(),
            if missing == 0 {
                crate::ErrorKind::InvalidResponse
            } else {
                crate::ErrorKind::Transport
            }
        );
        assert_eq!(error.request_id(), Some("mock-stream"));
        assert_eq!(error.response_body().unwrap().bytes(), body.as_bytes());
        assert_eq!(error.response_body().unwrap().is_complete(), missing == 0);
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }
}

/// A terminal cap failure precedes invalid partial JSON parsing and never becomes completion.
#[tokio::test]
async fn output_limit_precedes_json_parsing() {
    let body = frame(
        &json!({"type":"response.incomplete", "response":{"status":"incomplete", "incomplete_details":{"reason":"max_output_tokens"}, "output":[{"type":"function_call", "arguments":"{"}]}}),
    );
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(
        base,
        LlmOptions::default().with_output_schema(json!({"type":"object"})),
    );
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::OutputLimitReached);
    assert!(stream.next().await.is_none());
}

/// Final payloads cannot silently rewrite already streamed answer content.
#[tokio::test]
async fn rejects_changed_terminal_answer() {
    let mut events = text_events();
    events.last_mut().unwrap()["response"]["output"][0]["content"][0]["text"] = json!("changed");
    let (base, _requests) = crate::providers::tests::http::serve_content(
        200,
        "text/event-stream",
        events.iter().map(frame).collect(),
    );
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    stream.next().await.unwrap().unwrap();
    assert_eq!(
        stream.next().await.unwrap().err().unwrap().kind(),
        crate::ErrorKind::InvalidResponse
    );
    assert!(stream.next().await.is_none());
}

/// Initial HTTP rejection happens during startup with no stream or fallback generation.
#[tokio::test]
async fn startup_rejection_preserves_http_details() {
    let (base, requests) = crate::providers::tests::http::serve(
        401,
        json!({"error":{"message":"invalid STREAM-SECRET"}, "private":"PRIVATE_STREAM_PAYLOAD"}),
    );
    let client = client(base, LlmOptions::default());
    let error = client
        .execute_stream(&[Message::user("hello")])
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Http);
    assert_eq!(error.http_status(), Some(401));
    assert!(!error.to_string().contains("STREAM-SECRET"));
    assert!(!format!("{error:?}").contains("PRIVATE_STREAM_PAYLOAD"));
    assert!(
        !String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("STREAM-SECRET")
    );
    assert!(requests.recv().unwrap().starts_with("POST /responses "));
}

/// Dropping a partially consumed stream releases its HTTP connection without a reader task.
#[tokio::test]
async fn dropping_stream_releases_connection() {
    let body = text_events()[..3].iter().map(frame).collect();
    let (base, disconnected) = crate::providers::tests::http::serve_until_disconnect(body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    drop(stream);
    assert!(
        tokio::task::spawn_blocking(move || disconnected
            .recv_timeout(Duration::from_secs(3))
            .unwrap())
        .await
        .unwrap()
    );
}

/// Output-item and text closure forbid later deltas, and completion cannot add undisclosed items.
#[test]
fn rejects_closed_output_and_new_terminal_items() {
    for insert_after in [4, 5, 6] {
        let mut values = text_events();
        if insert_after == 6 {
            let extra = values[6]["response"]["output"][0].clone();
            values[6]["response"]["output"]
                .as_array_mut()
                .unwrap()
                .push(extra);
        } else {
            values.insert(insert_after + 1, values[2].clone());
        }
        let mut assembled = json!({"output":[]});
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
}

/// Terminal JSON uses the same normalization as execute while partial fragments stay unparsed.
#[test]
fn structured_completion_matches_complete_mapper() {
    let mut values = text_events();
    for value in &mut values {
        if value["type"] == "response.output_text.delta" {
            value["delta"] = if value["delta"] == "Hello" {
                json!("{\"answer\":")
            } else {
                json!("1}")
            };
        }
        if value["type"] == "response.output_text.done" {
            value["text"] = json!("{\"answer\":1}");
        }
        if value["type"] == "response.output_item.done" {
            value["item"]["content"][0]["text"] = json!("{\"answer\":1}");
        }
        if value["type"] == "response.completed" {
            value["response"]["output"][0]["content"][0]["text"] = json!("{\"answer\":1}");
        }
    }
    let expected = map_response(values.last().unwrap()["response"].clone(), true).unwrap();
    let options = LlmOptions::default().with_output_schema(json!({"type":"object"}));
    let mut assembled = json!({"output":[]});
    let events: Vec<_> = values
        .iter()
        .flat_map(|v| {
            decode(
                v["type"].as_str().unwrap(),
                &v.to_string(),
                &mut assembled,
                &options,
            )
            .unwrap()
        })
        .collect();
    let LlmEvent::Completed { response } = events.last().unwrap() else {
        panic!("expected JSON completion")
    };
    crate::providers::tests::http::assert_response_eq(response, &expected);
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

/// A frame dispatched only after observed HTTP EOF retains a complete original body.
#[tokio::test]
async fn eof_frame_error_preserves_complete_body() {
    let body = "data: {\r\r".to_owned();
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body.clone());
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Deserialize);
    assert!(error.response_body().unwrap().is_complete());
    assert_eq!(error.response_body().unwrap().bytes(), body.as_bytes());
    assert!(stream.next().await.is_none());
}

/// Declared text parts must start empty rather than silently adding undisplayed final text.
#[test]
fn rejects_nonempty_initial_text_parts() {
    for text in [json!("unstreamed"), Value::Null] {
        let mut values = text_events();
        values[1]["part"]["text"] = text;
        let mut assembled = json!({"output":[]});
        decode(
            "",
            &values[0].to_string(),
            &mut assembled,
            &LlmOptions::default(),
        )
        .unwrap();
        assert!(
            decode(
                "",
                &values[1].to_string(),
                &mut assembled,
                &LlmOptions::default()
            )
            .is_err()
        );
    }
}
