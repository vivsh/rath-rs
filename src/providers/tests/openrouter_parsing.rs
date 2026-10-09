use super::*;

/// Supplies a valid Chat Completions envelope with caller-selected generated content.
fn envelope(text: Value) -> Value {
    json!({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":text}}]})
}

/// Documented choice-level errors retain the native reason instead of parsing partial output.
#[test]
fn nested_choice_error_is_provider_failure() {
    for text in [Value::Null, json!("{"), json!("partial answer")] {
        let mut response = envelope(text);
        response["choices"][0]["error"] = json!({"code":502,"message":"upstream disconnected"});
        let error = map_response(response, false).err().unwrap();
        assert_eq!(error.kind(), crate::ErrorKind::Provider);
        assert_eq!(error.provider_code(), Some("502"));
        assert!(error.to_string().contains("upstream disconnected"));
    }
}

/// Native error and content-filter completion reasons never become a successful partial answer.
#[test]
fn failed_finish_reasons_are_not_successful_output() {
    for reason in ["error", "content_filter"] {
        let mut response = envelope(json!("partial answer"));
        response["choices"][0]["finish_reason"] = json!(reason);
        assert_eq!(
            map_response(response, false).err().unwrap().kind(),
            crate::ErrorKind::Provider
        );
    }
}

/// A refusal with no answer reports refusal instead of a generic missing-content error.
#[test]
fn refusal_is_provider_failure() {
    let mut response = envelope(Value::Null);
    response["choices"][0]["message"]["refusal"] = json!("Cannot answer this request");
    assert_eq!(
        map_response(response, true).err().unwrap().kind(),
        crate::ErrorKind::Provider
    );
}

/// Malformed tool containers cannot be ignored in favor of accompanying plain text.
#[test]
fn malformed_tool_container_is_rejected() {
    for tools in [json!({}), json!("invalid"), json!(true)] {
        let mut response = envelope(json!("partial answer"));
        response["choices"][0]["message"]["tool_calls"] = tools;
        assert_eq!(
            map_response(response, false).err().unwrap().kind(),
            crate::ErrorKind::InvalidResponse
        );
    }
}

/// Truncation takes priority over decoding partial output and any provider failure description.
#[test]
fn cap_failure_precedes_generated_json_parsing() {
    let mut response = envelope(json!("{"));
    response["choices"][0]["finish_reason"] = json!("length");
    response["choices"][0]["error"] = json!({"message":"truncated"});
    assert_eq!(
        map_response(response, true).err().unwrap().kind(),
        crate::ErrorKind::OutputLimitReached
    );
}

/// Returns an isolated public client using fixed credentials and a loopback provider endpoint.
fn client(base: String, options: LlmOptions) -> LlmClient {
    let mut url = ModelUrl::parse("openrouter:///model").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("ROUTER-SECRET".into());
    new_client(&url, options).unwrap()
}

/// Real HTTP mapping preserves choice error codes, original bytes and credential-safe diagnostics.
#[tokio::test]
async fn nested_http_error_preserves_evidence() {
    let mut native = envelope(json!("{"));
    native["choices"][0]["error"] = json!({"code":502,"message":"upstream disconnected ROUTER-SECRET", "request_id":"router-request"});
    native["private"] = json!("PRIVATE-OUTPUT");
    let (base, requests) = crate::providers::tests::http::serve(200, native.clone());
    let client = client(
        base,
        LlmOptions::default().with_output_schema(json!({"type":"object"})),
    );
    let error = client
        .execute(&[Message::user("hello")])
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Provider);
    assert_eq!(error.http_status(), Some(200));
    assert_eq!(error.provider_code(), Some("502"));
    assert_eq!(error.request_id(), Some("router-request"));
    assert!(error.to_string().contains("upstream disconnected"));
    assert!(!error.to_string().contains("ROUTER-SECRET"));
    assert!(!format!("{error:?}").contains("PRIVATE-OUTPUT"));
    let body = error.response_body().unwrap();
    assert!(body.is_complete());
    let retained: Value = serde_json::from_slice(body.bytes()).unwrap();
    assert_eq!(retained["private"], native["private"]);
    assert_eq!(retained["choices"][0]["message"]["content"], "{");
    assert!(!String::from_utf8_lossy(body.bytes()).contains("ROUTER-SECRET"));
    assert!(
        requests
            .recv()
            .unwrap()
            .starts_with("POST /chat/completions ")
    );
}

/// Malformed generated JSON remains a decoding failure, retaining the full HTTP response.
#[tokio::test]
async fn malformed_generated_json_retains_native_response() {
    let native = envelope(json!("```json\n{\"ok\":true}\n```"));
    let (base, _requests) = crate::providers::tests::http::serve(200, native.clone());
    let client = client(
        base,
        LlmOptions::default().with_output_schema(json!({"type":"object"})),
    );
    let error = client
        .execute(&[Message::user("hello")])
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Deserialize);
    assert!(error.source().is_some());
    assert_eq!(
        serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
        native
    );
}

/// Strict JSON parsing preserves every valid JSON value and never extracts JSON from arbitrary prose.
#[test]
fn json_parsing_preserves_valid_output_and_rejects_invalid_output() {
    for text in ["{\"ok\":true}", "[1,2]", "null", "true", "42", "\"hello\""] {
        let expected: Value = serde_json::from_str(text).unwrap();
        let response = map_response(envelope(json!(text)), true).unwrap();
        assert!(matches!(response.output, LlmOutput::Output(value) if value == expected));
    }
    for text in [
        "{",
        "{\"ok\":true,}",
        "Here's JSON: {\"ok\":true}",
        "```json\n{\"ok\":true}\n```",
    ] {
        assert_eq!(
            map_response(envelope(json!(text)), true)
                .err()
                .unwrap()
                .kind(),
            crate::ErrorKind::Deserialize
        );
        let response = map_response(envelope(json!(text)), false).unwrap();
        assert!(matches!(response.output, LlmOutput::Output(value) if value == text));
    }
}

/// Tool argument failures have distinct context and preserve the original JSON parser cause.
#[test]
fn invalid_tool_arguments_have_specific_context() {
    let mut response = envelope(Value::Null);
    response["choices"][0]["message"]["tool_calls"] =
        json!([{"id":"call","type":"function","function":{"name":"lookup","arguments":"{"}}]);
    let error = map_response(response, false).err().unwrap();
    assert_eq!(error.kind(), crate::ErrorKind::Deserialize);
    assert!(error.to_string().contains("invalid JSON tool arguments"));
    assert!(error.source().unwrap().to_string().contains("line 1"));
}

/// Endpoint support filtering and opt-in healing remain explicit caller-owned native controls.
#[test]
fn native_json_routing_controls_reach_request_unchanged() {
    let native = json!({"provider":{"require_parameters":true,"only":["example"]},"plugins":[{"id":"response-healing"}]});
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_provider_config(native.clone());
    let payload = build_payload("provider/model", &options, &[Message::user("hello")], false);
    assert_eq!(payload["provider"], native["provider"]);
    assert_eq!(payload["plugins"], native["plugins"]);
    assert_eq!(payload["response_format"]["type"], "json_schema");
    assert_eq!(payload["stream"], false);
    assert!(
        build_payload(
            "provider/model",
            &LlmOptions::default(),
            &[Message::user("hello")],
            false
        )
        .get("plugins")
        .is_none()
    );
}
