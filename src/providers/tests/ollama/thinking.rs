use super::super::*;
use crate::core::ErrorKind;
use crate::providers::tests::http::serve;

/// Exercises every enum value through public construction and the actual compatibility endpoint.
#[tokio::test]
async fn exact_thinking_wire_mapping() {
    for (level, expected) in [
        (None, None),
        (Some(ThinkingLevel::Off), Some("none")),
        (Some(ThinkingLevel::Low), Some("low")),
        (Some(ThinkingLevel::Medium), Some("medium")),
        (Some(ThinkingLevel::High), Some("high")),
        (Some(ThinkingLevel::XHigh), Some("max")),
    ] {
        let (base, requests) = serve(200, success());
        let client = LlmOptions::default()
            .with_thinking(level)
            .create(&format!("ollama:///qwen3:8b?base_url={base}"))
            .unwrap();
        let messages = [
            Message::user("  Original Ω\n"),
            Message::user("/no_think\nkeep this"),
        ];
        client.execute(&messages).await.unwrap();
        let request = requests.try_recv().unwrap();
        assert!(request.starts_with("POST /v1/chat/completions "));
        let payload: Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            payload.get("reasoning_effort").and_then(Value::as_str),
            expected
        );
        assert_eq!(
            payload.get("reasoning_effort").is_some(),
            expected.is_some()
        );
        assert!(payload.get("think").is_none());
        assert_eq!(payload["messages"][0]["content"], messages[0].content);
        assert_eq!(payload["messages"][1]["content"], messages[1].content);
        assert!(requests.try_recv().is_err());
    }
}

/// Explicit URL levels override programmatic levels; absent URL levels retain the options.
#[tokio::test]
async fn url_thinking_precedence_reaches_wire() {
    for (programmatic, query, expected) in [
        (Some(ThinkingLevel::High), "&thinking=off", "none"),
        (Some(ThinkingLevel::Off), "&thinking=high", "high"),
        (None, "&thinking=medium", "medium"),
        (Some(ThinkingLevel::Off), "", "none"),
        (Some(ThinkingLevel::XHigh), "", "max"),
    ] {
        let (base, requests) = serve(200, success());
        let client = LlmOptions::default()
            .with_thinking(programmatic)
            .create(&format!("ollama:///model?base_url={base}{query}"))
            .unwrap();
        client.execute(&[Message::user("hello")]).await.unwrap();
        let request = requests.try_recv().unwrap();
        let payload: Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(payload["reasoning_effort"], expected);
    }
}

/// Explicit Off combines API control, the existing JSON format, schema instructions and token cap.
#[tokio::test]
async fn off_with_structured_output() {
    let (base, requests) = serve(200, success());
    let client = LlmOptions::default()
        .with_thinking(Some(ThinkingLevel::Off))
        .with_output_schema(json!({"type":"object", "properties":{"name":{"type":"string"}}}))
        .with_max_output_tokens(256)
        .create(&format!("ollama:///granite4.2:3b?base_url={base}"))
        .unwrap();
    let response = client
        .execute(&[Message::user("Extract Ada's name.")])
        .await
        .unwrap();
    assert!(matches!(response.output, LlmOutput::Output(ref v) if v == &json!({"name":"Ada"})));
    let request = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["reasoning_effort"], "none");
    assert_eq!(payload["response_format"], json!({"type":"json_object"}));
    assert_eq!(payload["max_tokens"], 256);
    assert!(
        payload["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("schema")
    );
    assert_eq!(payload["messages"][1]["content"], "Extract Ada's name.");
}

/// Schema selection remains independent of assuming provider-default thinking is disabled.
#[test]
fn structured_output_preserves_default_and_enabled_selection() {
    for level in [
        None,
        Some(ThinkingLevel::Off),
        Some(ThinkingLevel::Low),
        Some(ThinkingLevel::Medium),
        Some(ThinkingLevel::High),
        Some(ThinkingLevel::XHigh),
    ] {
        let enabled = level.as_ref().is_some_and(|v| *v != ThinkingLevel::Off);
        let options = LlmOptions::default()
            .with_thinking(level)
            .with_output_schema(json!({"type":"object"}));
        let payload = build_payload("model", &options, &[Message::user("extract")], false);
        assert_eq!(payload.get("response_format").is_none(), enabled);
        assert!(
            payload["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("schema")
        );
    }
}

/// Model/server rejection keeps typed diagnostics and does not retry with a weaker effort.
#[tokio::test]
async fn unsupported_effort_keeps_provider_error() {
    for level in [ThinkingLevel::Off, ThinkingLevel::XHigh] {
        let body = json!({"error":{"message":"model does not support requested thinking", "code":"unsupported_thinking"}, "request_id":"reject-1"});
        let (base, requests) = serve(400, body.clone());
        let client = LlmOptions::default()
            .with_thinking(Some(level.clone()))
            .create(&format!("ollama:///model?base_url={base}"))
            .unwrap();
        let error = client.execute(&[Message::user("hello")]).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Http);
        assert_eq!(error.provider(), Some(&Provider::Ollama));
        assert_eq!(error.operation(), Some("generation"));
        assert_eq!(error.http_status(), Some(400));
        assert_eq!(error.provider_code(), Some("unsupported_thinking"));
        assert_eq!(error.request_id(), Some("reject-1"));
        assert!(error.message().contains("does not support"));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            body
        );
        let request = requests.try_recv().unwrap();
        let payload: Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(payload["reasoning_effort"], reasoning_effort(&level));
        assert!(requests.try_recv().is_err());
    }
}

/// Invalid URL levels remain typed validation failures before any provider dispatch.
#[test]
fn unknown_url_effort_is_rejected() {
    let error = LlmOptions::default()
        .create("ollama:///model?thinking=unsupported")
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidUrl);
    assert!(error.message().contains("thinking"));
}

/// Retained metadata distinguishes absent reasoning usage from explicitly reported zero.
#[test]
fn reasoning_metadata_preserves_reported_evidence() {
    for tokens in [None, Some(0), Some(12)] {
        let mut wire = success();
        if let Some(tokens) = tokens {
            wire["usage"]["completion_tokens_details"] = json!({"reasoning_tokens":tokens});
            wire["choices"][0]["message"]["reasoning"] = json!("reported reasoning");
        }
        let response = map_response(wire.clone(), true).unwrap();
        let metadata = response.raw_metadata.unwrap();
        assert_eq!(metadata["finish_reason"], "stop");
        assert_eq!(metadata["usage"], wire["usage"]);
        assert_eq!(
            metadata["reasoning"],
            wire["choices"][0]["message"]["reasoning"]
        );
        assert_eq!(
            metadata
                .pointer("/usage/completion_tokens_details/reasoning_tokens")
                .and_then(Value::as_u64),
            tokens
        );
    }
}

/// Provides synthetic structured content with a normal stop and provider-reported usage.
fn success() -> Value {
    json!({"id":"test", "model":"model", "choices":[{"finish_reason":"stop", "message":{"content":"{\"name\":\"Ada\"}"}}], "usage":{"prompt_tokens":12,"completion_tokens":6}})
}
