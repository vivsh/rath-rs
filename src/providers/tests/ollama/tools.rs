use super::super::*;
use super::fixtures::{call, options, response};
use crate::core::ErrorKind;
use crate::providers::tests::http::serve;

/// Required requests cannot succeed without a call, including an empty successful response.
#[tokio::test]
async fn required_calls_are_enforced_before_output_parsing() {
    for text in ["I cannot call tools.", "", "not JSON"] {
        let body = response(json!({"content":text}), "stop");
        let (base, requests) = serve(200, body.clone());
        let client = options(ToolChoice::Required)
            .with_response_format(crate::llm::ResponseFormat::Json)
            .create(&format!("ollama:///model?base_url={base}"))
            .unwrap();
        let error = client
            .execute(&[Message::user("lookup")])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert!(error.message().contains("Required"));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            body
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Impossible required requests fail construction and operation-time validation without dispatch.
#[tokio::test]
async fn required_without_tools_is_validation_error() {
    let opts = LlmOptions::default().with_tool_choice(ToolChoice::Required);
    let error = opts.clone().create("ollama:///test").err().unwrap();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("at least one tool"));
    let mut client = OllamaClient {
        http: HttpClient::new(),
        api_key: None,
        base_url: "http://127.0.0.1:1".into(),
        model: "test".into(),
        options: LlmOptions::default(),
        url: ModelUrl::parse("ollama:///test").unwrap(),
        exit_tool_name: None,
    };
    client.options = opts;
    assert_eq!(
        client
            .execute(&[Message::user("x")])
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        client
            .estimate_tokens(&[Message::user("x")])
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

/// Disabled and unconfigured tools preserve literal text, but reject protocol-level calls.
#[test]
fn disabled_tools_cannot_become_executable() {
    let text = "<function=lookup><parameter=q>x</parameter></function>";
    for opts in [options(ToolChoice::Disabled), LlmOptions::default()] {
        let result = map_response(response(json!({"content":text}), "stop"), &opts).unwrap();
        assert!(matches!(result.output, LlmOutput::Output(v) if v == text));
        let error = map_response(
            response(json!({"tool_calls":[call()]}), "tool_calls"),
            &opts,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert!(error.message().contains("disabled or unavailable"));
    }
}

/// Malformed protocol calls and inconsistent finish signals must never become successful text.
#[test]
fn invalid_protocol_calls_fail_explicitly() {
    for calls in [json!("bad"), json!({}), json!([]), json!(null), json!([{}])] {
        let error = map_response(
            response(json!({"content":"", "tool_calls":calls}), "tool_calls"),
            &options(ToolChoice::Auto),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert!(error.message().contains("Ollama"));
    }
}

/// Incomplete legacy blocks or parameters cannot become partial executable calls.
#[test]
fn incomplete_text_calls_fail_explicitly() {
    for text in [
        "<function=lookup",
        "<function=lookup><parameter=q>x",
        "<function=lookup><parameter=q>x</function>",
        "<function=></function>",
        "<function=lookup><parameter=q>x<parameter=z>y</parameter></function>",
        "<function=lookup>stray data</function>",
    ] {
        let error = map_response(
            response(json!({"content":text}), "stop"),
            &options(ToolChoice::Auto),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse, "{text}");
    }
}

/// Only offered tools with valid identities and object arguments can be returned to callers.
#[test]
fn tool_call_identity_and_arguments_are_checked() {
    for (path, value, kind) in [
        ("/id", json!(""), ErrorKind::InvalidResponse),
        (
            "/function/name",
            json!("unknown"),
            ErrorKind::InvalidResponse,
        ),
        (
            "/function/arguments",
            json!("[]"),
            ErrorKind::InvalidResponse,
        ),
        ("/function/arguments", json!("{"), ErrorKind::Deserialize),
    ] {
        let mut wire = call();
        *wire.pointer_mut(path).unwrap() = value;
        let error = map_response(
            response(json!({"tool_calls":[wire]}), "tool_calls"),
            &options(ToolChoice::Auto),
        )
        .unwrap_err();
        assert_eq!(error.kind(), kind);
    }
    assert!(
        map_response(
            response(json!({"tool_calls":[call(),call()]}), "tool_calls"),
            &options(ToolChoice::Auto)
        )
        .unwrap_err()
        .message()
        .contains("duplicate")
    );
}

/// Realistic unsupported-model envelopes retain their useful message, type, status and body.
#[tokio::test]
async fn unsupported_model_keeps_meaningful_diagnostics() {
    let body = json!({"error":{"code":null,"type":"invalid_request_error",
        "message":"model-without-tools does not support tools"},"private":"PRIVATE-RESPONSE"});
    let (base, requests) = serve(400, body.clone());
    let client = options(ToolChoice::Auto)
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let error = client
        .execute(&[Message::user("lookup")])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.http_status(), Some(400));
    assert_eq!(error.provider_code(), Some("invalid_request_error"));
    assert!(error.to_string().contains("does not support tools"));
    assert!(!format!("{error} {error:?}").contains("PRIVATE-RESPONSE"));
    assert_eq!(
        serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
        body
    );
    assert_eq!(requests.try_iter().count(), 1);
}

/// Tool responses retain IDs and arguments and replay correctly alongside tool-result messages.
#[tokio::test]
async fn protocol_calls_and_history_are_preserved() {
    let (base, requests) = serve(
        200,
        response(json!({"content":null,"tool_calls":[call()]}), "tool_calls"),
    );
    let client = options(ToolChoice::Required)
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let result = client.execute(&[Message::user("lookup")]).await.unwrap();
    let LlmOutput::ToolCalls { calls, .. } = result.output else {
        panic!("expected tool calls")
    };
    assert_eq!(calls[0].id, "call-1");
    assert_eq!(calls[0].args, json!({"q":"x"}));
    let mut assistant = Message::assistant("");
    assistant.role = Role::AssistantToolCalls { calls };
    let history = [assistant, Message::tool_output("call-1".into(), "result")];
    let payload = build_payload("model", &options(ToolChoice::Auto), &history, true);
    assert_eq!(
        payload["messages"][0]["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    assert_eq!(payload["messages"][1]["tool_call_id"], "call-1");
    assert_eq!(requests.try_iter().count(), 1);
}
