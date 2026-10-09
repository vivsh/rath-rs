use super::*;
use futures_util::StreamExt;
use serde_json::json;
use std::time::Duration;

/// Uses the installed SDK's test-only endpoint builder without changing production URL policy.
fn client(base: String, options: LlmOptions) -> GeminiClient {
    let mut url = ModelUrl::parse("gemini:///gemini-3.8-flash").unwrap();
    url.api_key = Some("STREAM-SECRET".into());
    let client = gemini_rust::client::GeminiBuilder::new("STREAM-SECRET")
        .with_model(GeminiModel::Custom("models/gemini-3.8-flash".into()))
        .with_base_url(base.parse().unwrap())
        .build()
        .unwrap();
    GeminiClient {
        client,
        options,
        url,
    }
}

/// Encodes one generateContent response as an SDK-readable SSE data event.
fn frame(value: Value) -> String {
    format!("data: {value}\n\n")
}

/// Creates one selected candidate chunk, with native termination only when requested.
fn chunk(parts: Value, reason: Value) -> Value {
    let mut value = json!({"responseId":"r", "modelVersion":"echo", "candidates":[{"index":0,"content":{"role":"model","parts":parts}}]});
    if !reason.is_null() {
        value["candidates"][0]["finishReason"] = reason;
    }
    value
}

/// Builds the paired complete-response fixture with identical native parts and metadata.
fn visible_response() -> LlmResponse {
    let mut final_value = chunk(
        json!([{"text":"private","thought":true},{"text":"Hello"},{"text":"🙂"}]),
        json!("STOP"),
    );
    final_value["usageMetadata"] =
        json!({"promptTokenCount":8,"candidatesTokenCount":3,"totalTokenCount":11});
    map_response(serde_json::from_value(final_value).unwrap(), false).unwrap()
}

/// Real SDK transport yields multipart visible text early and waits for trailing reported usage.
#[tokio::test]
async fn sdk_stream_preserves_visible_parts_and_final_usage() {
    let early = frame(chunk(
        json!([{"text":"private", "thought":true},{"text":"Hello"}]),
        Value::Null,
    ));
    let late = format!(
        "{}{}",
        frame(chunk(json!([{"text":"🙂"}]), json!("STOP"))),
        frame(
            json!({"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":3,"totalTokenCount":11}})
        )
    );
    let (base, requests) = crate::providers::tests::http::serve_stream(
        vec![(early, Duration::ZERO), (late, Duration::from_millis(600))],
        0,
    );
    let client = client(base, LlmOptions::default().with_max_output_tokens(20));
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    let early = tokio::time::timeout(Duration::from_millis(300), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(early, LlmEvent::TextDelta { text } if text == "Hello"));
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), LlmEvent::TextDelta { text } if text == "🙂")
    );
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("expected completed")
    };
    let expected = visible_response();
    crate::providers::tests::http::assert_response_eq(&response, &expected);
    assert!(matches!(response.output, LlmOutput::Output(value) if value == "Hello🙂"));
    let usage = response.usage.unwrap();
    assert_eq!(usage.input, Some(8));
    assert_eq!(usage.output, Some(3));
    assert_eq!(response.provider_model.as_deref(), Some("echo"));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
    let request = requests.recv().unwrap();
    assert!(request.starts_with("POST /models/gemini-3.8-flash:streamGenerateContent?alt=sse "));
    assert!(request.contains("\"maxOutputTokens\":20"));
}

/// Complete native function parts surface as tool progress and retain continuation signatures.
#[tokio::test]
async fn sdk_stream_preserves_tool_calls_and_signatures() {
    let body = frame(chunk(
        json!([
            {"functionCall":{"name":"lookup","args":{"city":"Delhi"}},"thoughtSignature":"sig1"},
            {"functionCall":{"name":"lookup","args":{"city":"Mumbai"}},"thoughtSignature":"sig2"}
        ]),
        json!("STOP"),
    ));
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    for index in 0..2 {
        let LlmEvent::ToolCallDelta {
            index: got,
            id,
            name_delta,
            arguments_delta,
        } = stream.next().await.unwrap().unwrap()
        else {
            panic!("expected tool progress")
        };
        assert_eq!(got, index);
        assert_eq!(id.as_deref(), Some(format!("lookup_{index}").as_str()));
        assert_eq!(name_delta.as_deref(), Some("lookup"));
        assert!(
            serde_json::from_str::<Value>(&arguments_delta)
                .unwrap()
                .is_object()
        );
    }
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("expected completed")
    };
    let LlmOutput::ToolCalls { calls, .. } = response.output else {
        panic!("expected calls")
    };
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(
            call.thought_signatures,
            Some(vec![format!("sig{}", index + 1)])
        );
    }
    assert_eq!(calls[1].args["city"], "Mumbai");
    assert!(stream.next().await.is_none());
}

/// SDK EOF without stop evidence and token-limited JSON are failures, never final output.
#[tokio::test]
async fn sdk_stream_rejects_missing_stop_and_output_limit() {
    for reason in [Value::Null, json!("MAX_TOKENS")] {
        let body = frame(chunk(json!([{"text":"{"}]), reason.clone()));
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
        let error = if reason.is_null() {
            stream.next().await.unwrap().unwrap();
            stream.next().await.unwrap().err().unwrap()
        } else {
            stream.next().await.unwrap().err().unwrap()
        };
        assert_eq!(
            error.kind(),
            if reason.is_null() {
                crate::ErrorKind::InvalidResponse
            } else {
                crate::ErrorKind::OutputLimitReached
            }
        );
        assert!(stream.next().await.is_none());
    }
}

/// SDK transfer errors remain transport failures even after native stop evidence arrived.
#[tokio::test]
async fn sdk_stream_classifies_transfer_failure() {
    let body = frame(chunk(json!([{"text":"done"}]), json!("STOP")));
    let (base, _requests) =
        crate::providers::tests::http::serve_stream(vec![(body, Duration::ZERO)], 100);
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    stream.next().await.unwrap().unwrap();
    let error = stream.next().await.unwrap().err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Transport);
    assert!(error.source().is_some());
    assert!(stream.next().await.is_none());
}

/// Shared complete-response normalization includes every visible part and excludes thoughts.
#[test]
fn complete_mapping_uses_visible_text_parts() {
    let native: GenerationResponse = serde_json::from_value(chunk(
        json!([
            {"text":"private", "thought":true}, {"text":"Hello"}, {"text":"🙂"}
        ]),
        json!("STOP"),
    ))
    .unwrap();
    let response = map_response(native, false).unwrap();
    assert!(matches!(response.output, LlmOutput::Output(value) if value == "Hello🙂"));
}

/// Synthetic final-output progress stays private, while completed JSON uses the existing transform.
#[tokio::test]
async fn sdk_stream_hides_private_final_tool() {
    let body = frame(chunk(
        json!([{"functionCall":{"name":"__rath_final_output","args":{"answer":1}}}]),
        json!("STOP"),
    ));
    let (base, _requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", body);
    let options = LlmOptions::default().with_output_schema(json!({"type":"object"}));
    let client = client(base, options);
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    let LlmEvent::Completed { response } = stream.next().await.unwrap().unwrap() else {
        panic!("private tool escaped")
    };
    assert!(matches!(response.output, LlmOutput::Output(value) if value == json!({"answer":1})));
    assert!(stream.next().await.is_none());
}

/// Malformed SDK JSON is classified as decoding failure rather than empty completion.
#[tokio::test]
async fn sdk_stream_rejects_malformed_frame() {
    let (base, _requests) = crate::providers::tests::http::serve_content(
        200,
        "text/event-stream",
        "data: {\n\n".into(),
    );
    let client = client(base, LlmOptions::default());
    let mut stream = client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().err().unwrap().kind(),
        crate::ErrorKind::Deserialize
    );
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

/// The SDK uses identical semantic generation payloads for execute and execute_stream.
#[tokio::test]
async fn sdk_stream_request_matches_complete_request() {
    let options = LlmOptions::default()
        .with_preamble("persona")
        .with_max_output_tokens(20)
        .with_tools(vec![ToolDefinition::new(
            "lookup".into(),
            "Lookup".into(),
            json!({"type":"object"}),
        )]);
    let native = chunk(json!([{"text":"done"}]), json!("STOP"));
    let (base, complete_requests) = crate::providers::tests::http::serve(200, native.clone());
    let complete_client = client(base, options.clone());
    complete_client
        .execute(&[Message::user("hello")])
        .await
        .unwrap();
    let (base, stream_requests) =
        crate::providers::tests::http::serve_content(200, "text/event-stream", frame(native));
    let stream_client = client(base, options);
    let mut stream = stream_client
        .execute_stream(&[Message::user("hello")])
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    let complete_request = complete_requests.recv().unwrap();
    let stream_request = stream_requests.recv().unwrap();
    let body = |request: &str| {
        serde_json::from_str::<Value>(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    };
    assert_eq!(body(&complete_request), body(&stream_request));
    assert!(complete_request.starts_with("POST /models/gemini-3.8-flash:generateContent "));
    assert!(
        stream_request.starts_with("POST /models/gemini-3.8-flash:streamGenerateContent?alt=sse ")
    );
}

/// A private final-output call cannot be combined with an ordinary requested tool.
#[tokio::test]
async fn sdk_stream_rejects_mixed_final_tool() {
    let body = frame(chunk(
        json!([
            {"functionCall":{"name":"__rath_final_output","args":{"answer":1}}},
            {"functionCall":{"name":"lookup","args":{}}}
        ]),
        json!("STOP"),
    ));
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
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        LlmEvent::ToolCallDelta { index: 1, .. }
    ));
    assert_eq!(
        stream.next().await.unwrap().err().unwrap().kind(),
        crate::ErrorKind::InvalidResponse
    );
    assert!(stream.next().await.is_none());
}
