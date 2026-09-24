use super::*;
use crate::llm::ToolCall;
use serde_json::json;

/// Request preparation retains original schema/options and respects disabled user tools.
#[test]
fn transformations_are_operation_local() {
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_tools(vec![ToolDefinition::new(
            "lookup".into(),
            "Lookup".into(),
            json!({"type":"object"}),
        )]);
    let prepared = prepare(Provider::Ollama, "model", &options).unwrap();
    assert_eq!(options.tools.len(), 1);
    assert!(options.output_schema().is_some());
    assert_eq!(prepared.tools.len(), 2);
    assert_eq!(prepared.tools[1].name, FINAL_TOOL);
    let disabled = prepare(
        Provider::Ollama,
        "model",
        &options.clone().with_tool_choice(ToolChoice::Disabled),
    )
    .unwrap();
    assert_eq!(disabled.tool_choice, ToolChoice::Disabled);
    assert_eq!(disabled.tools.len(), 1);
    assert!(disabled.output_schema().is_some());
    let conflict = LlmOptions::default().with_tools(vec![ToolDefinition::new(
        FINAL_TOOL.into(),
        "invalid".into(),
        json!({}),
    )]);
    assert_eq!(
        prepare(Provider::Ollama, "model", &conflict)
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

/// Mixed ordinary/final calls are never partially interpreted or exposed as successful output.
#[test]
fn mixed_calls_retain_private_evidence() {
    let options = LlmOptions::default().with_output_schema(json!({"type":"object"}));
    let response = LlmResponse::new(
        Provider::Ollama,
        LlmOutput::ToolCalls {
            text: None,
            calls: vec![
                ToolCall::new("1".into(), FINAL_TOOL.into(), json!({"a":1})),
                ToolCall::new("2".into(), "lookup".into(), json!({})),
            ],
        },
    )
    .with_raw_metadata(Some(json!({"private":"SECRET-PAYLOAD"})));
    let error = finish(&options, response).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert!(
        String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("SECRET-PAYLOAD")
    );
    assert!(!format!("{error:?} {error}").contains("SECRET-PAYLOAD"));
}
