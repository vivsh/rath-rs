use super::super::*;
use super::fixtures::{call, options, response};
use crate::core::ErrorKind;
use crate::llm::ResponseFormat;
use crate::providers::tests::http::serve;

/// JSON input is serialized unchanged and schemas reach the wire; structured output is decoded.
#[tokio::test]
async fn json_input_and_output_round_trip() {
    let input = json!({"name":"Ada", "city":"Pune", "note":"Ω\n\"quoted\""});
    let output = json!({"city":"Pune","name":"Ada"});
    let (base, requests) = serve(200, response(json!({"content":output.to_string()}), "stop"));
    let client = LlmOptions::default()
        .with_thinking(Some(ThinkingLevel::Off))
        .with_input_schema(json!({"type":"object","properties":{"name":{"type":"string"}}}))
        .with_output_schema(json!({"type":"object","properties":{"city":{"type":"string"}}}))
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let input_message = Message::from_json(Role::User, &input).unwrap();
    let result = client
        .execute(std::slice::from_ref(&input_message))
        .await
        .unwrap();
    assert!(matches!(result.output,LlmOutput::Output(v) if v==output));
    let wire = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["messages"][1]["content"], input_message.content);
    assert_eq!(
        serde_json::from_str::<Value>(payload["messages"][1]["content"].as_str().unwrap()).unwrap(),
        input
    );
    let system = payload["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("The user message is JSON"));
    assert!(system.contains("Respond with JSON"));
    assert_eq!(payload["reasoning_effort"], "none");
    assert_eq!(payload["response_format"]["type"], "json_object");
}

/// Valid JSON literals are never interpreted as tool blocks, reasoning markers or decorated keys.
#[test]
fn valid_json_content_is_preserved() {
    let value = json!({"**literal**":"</think>","example":"<function=lookup><parameter=q>x</parameter></function>"});
    let opts = options(ToolChoice::Auto).with_response_format(ResponseFormat::Json);
    for text in [value.to_string(), format!("```json\n{value}\n```")] {
        let result = map_response(response(json!({"content":text}), "stop"), &opts).unwrap();
        assert!(matches!(result.output,LlmOutput::Output(v) if v==value));
    }
}

/// Malformed or empty structured output preserves decoder causes, wire bytes and provider context.
#[tokio::test]
async fn malformed_json_has_useful_private_diagnostics() {
    for text in ["PRIVATE-BAD-JSON", "", "{\"name\":"] {
        let wire = response(json!({"content":text}), "stop");
        let (base, requests) = serve(200, wire.clone());
        let client = LlmOptions::default()
            .with_response_format(ResponseFormat::Json)
            .create(&format!("ollama:///model?base_url={base}"))
            .unwrap();
        let error = client
            .execute(&[Message::user("extract")])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Deserialize);
        assert_eq!(error.provider(), Some(&Provider::Ollama));
        assert_eq!(error.operation(), Some("generation"));
        assert_eq!(error.http_status(), Some(200));
        assert!(error.message().contains("invalid JSON output"));
        assert!(error.source().unwrap().message().contains("line"));
        assert!(!format!("{error} {error:?}").contains("PRIVATE-BAD-JSON"));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            wire
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Invalid tool arguments retain JSON parsing details and the complete provider response.
#[tokio::test]
async fn malformed_tool_argument_json_is_an_error() {
    let mut tool = call();
    tool["function"]["arguments"] = json!("{\"PRIVATE-ARGUMENT\":");
    let wire = response(json!({"tool_calls":[tool]}), "tool_calls");
    let (base, _requests) = serve(200, wire.clone());
    let client = options(ToolChoice::Required)
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let error = client
        .execute(&[Message::user("lookup")])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Deserialize);
    assert!(error.message().contains("arguments are invalid JSON"));
    assert!(error.source().unwrap().message().contains("line"));
    assert!(!format!("{error} {error:?}").contains("PRIVATE-ARGUMENT"));
    assert_eq!(
        serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
        wire
    );
}

/// Rejected structured-output settings keep the deployment's message and are never retried relaxed.
#[tokio::test]
async fn unsupported_json_mode_keeps_provider_rejection() {
    let (base, requests) = serve(
        400,
        json!({"error":{"message":"response_format is unsupported for this model",
        "code":null,"type":"invalid_request_error"}}),
    );
    let client = LlmOptions::default()
        .with_response_format(ResponseFormat::Json)
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let error = client
        .execute(&[Message::user("extract")])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Http);
    assert_eq!(error.provider_code(), Some("invalid_request_error"));
    assert!(error.message().contains("response_format is unsupported"));
    let wire = requests.try_recv().unwrap();
    assert!(wire.contains("json_object"));
    assert!(requests.try_recv().is_err());
}

/// Output-limit signals take precedence over malformed JSON, tool-call policy and exit extraction.
#[test]
fn truncation_precedes_structured_parsing() {
    let opts = options(ToolChoice::Required).with_response_format(ResponseFormat::Json);
    for message in [json!({"content":"{"}), json!({"tool_calls":[{}]})] {
        assert_eq!(
            map_response(response(message, "length"), &opts)
                .unwrap_err()
                .kind(),
            ErrorKind::OutputLimitReached
        );
    }
}

/// Exit-tool arguments still produce structured output after the required-tool checks pass.
#[tokio::test]
async fn exit_tool_json_output_survives_validation() {
    let mut tool = call();
    tool["function"] = json!({"name":"final_answer","arguments":"{\"answer\":42}"});
    let (base, _requests) = serve(200, response(json!({"tool_calls":[tool]}), "tool_calls"));
    let mut opts = options(ToolChoice::Auto).with_output_schema(json!({"type":"object"}));
    opts.output_type_name = "final_answer".into();
    let client = opts
        .create(&format!("ollama:///model?base_url={base}"))
        .unwrap();
    let result = client.execute(&[Message::user("answer")]).await.unwrap();
    assert!(matches!(result.output,LlmOutput::Output(v) if v==json!({"answer":42})));
}

/// Caller serialization failures remain explicit, without producing a partial input message.
#[test]
fn input_serialization_failure_is_informative() {
    struct FailingInput;
    impl serde::Serialize for FailingInput {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(
                "synthetic input serialization failure",
            ))
        }
    }
    let error = Message::from_json(Role::User, &FailingInput).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("synthetic input serialization failure")
    );
}
