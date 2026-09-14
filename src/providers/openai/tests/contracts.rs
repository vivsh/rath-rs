use super::super::*;
use serde_json::json;

/// Verifies custom base url builds openai responses endpoint.
#[test]
fn custom_base_url_builds_openai_responses_endpoint() {
    assert_eq!(
        responses_endpoint("https://openrouter.ai/api/v1/"),
        "https://openrouter.ai/api/v1/responses"
    );
    assert_eq!(
        embeddings_endpoint("https://openrouter.ai/api/v1"),
        "https://openrouter.ai/api/v1/embeddings"
    );
}

/// Verifies responses payload uses schema and required tools.
#[test]
fn responses_payload_uses_schema_and_required_tools() {
    let options = LlmOptions::default()
        .with_tool_choice(ToolChoice::Required)
        .with_tools(vec![ToolDefinition {
            name: "lookup".into(),
            description: "Lookup a thing.".into(),
            parameters: json!({"type":"object","properties":{}}),
        }]);
    let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);
    assert_eq!(payload["model"], "custom-model");
    assert_eq!(payload["tool_choice"], "required");
    assert_eq!(payload["tools"][0]["name"], "lookup");
}

/// Verifies responses payload appends input schema to instructions.
#[test]
fn responses_payload_appends_input_schema_to_instructions() {
    let options = LlmOptions::default()
        .with_preamble("You are helpful.")
        .with_input_schema(json!({
            "type": "object",
            "properties": {
                "kind": { "type": "string" }
            },
            "required": ["kind"]
        }));

    let payload = build_payload("custom-model", &options, &[Message::user("hi")], false);

    let instructions = payload["instructions"]
        .as_str()
        .expect("instructions should be a string");
    assert!(instructions.contains("You are helpful."));
    assert!(instructions.contains("The user message is JSON."));
    assert!(instructions.contains("\"required\":[\"kind\"]"));
}

/// Verifies payload without input schema uses text mode.
#[test]
fn payload_without_input_schema_uses_text_mode() {
    let payload = build_payload(
        "custom-model",
        &LlmOptions::default(),
        &[Message::user("hi")],
        false,
    );
    assert!(payload.get("text").is_none());
}

/// Explicit JSON response mode uses the OpenAI json_object format when no schema is supplied.
#[test]
fn payload_with_json_response_and_no_output_schema_uses_json_object_mode() {
    let payload = build_payload(
        "custom-model",
        &LlmOptions::default().with_response_format(crate::llm::ResponseFormat::Json),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["text"]["format"]["type"], "json_object");
}

/// Non-image attachments are ignored on the OpenAI image-only wire path.
#[test]
fn non_image_user_attachments_are_dropped() {
    let payload = build_payload(
        "custom-model",
        &LlmOptions::default(),
        &[Message {
            key: None,
            role: Role::User,
            content: "describe this".into(),
            attachments: vec![Attachment::Inline {
                mime_type: "application/pdf".into(),
                data: "aGVsbG8=".into(),
            }],
            usage: None,
        }],
        false,
    );

    assert_eq!(payload["input"][0]["role"], "user");
    assert_eq!(payload["input"][0]["content"].as_array().unwrap().len(), 1);
    assert_eq!(payload["input"][0]["content"][0]["type"], "input_text");
}

/// User attachments are encoded as OpenAI `input_image` content items.
#[test]
fn user_attachments_use_content_array() {
    let payload = build_payload(
        "custom-model",
        &LlmOptions::default(),
        &[Message {
            key: None,
            role: Role::User,
            content: "describe this".into(),
            attachments: vec![Attachment::Inline {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            }],
            usage: None,
        }],
        false,
    );

    assert_eq!(payload["input"][0]["role"], "user");
    assert_eq!(payload["input"][0]["content"][0]["type"], "input_image");
    assert_eq!(payload["input"][0]["content"][1]["type"], "input_text");
}

/// Tool outputs remain unchanged when a reminder is appended as a later user turn.
#[test]
fn build_input_keeps_tool_output_and_reminder_separate() {
    let input = build_input(&[
        Message {
            key: None,
            role: Role::AssistantToolCalls {
                calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "lookup".into(),
                    args: json!({"q":"x"}),
                    thought_signatures: None,
                }],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
        Message::tool_output("call_1".into(), r#"{"ok":true}"#),
        Message::user("FINAL TURN: call final_answer"),
    ]);

    assert_eq!(input[0]["type"], "function_call");
    assert_eq!(input[1]["type"], "function_call_output");
    assert_eq!(input[1]["output"], r#"{"ok":true}"#);
    assert_eq!(input[2]["role"], "user");
    assert_eq!(input[2]["content"], "FINAL TURN: call final_answer");
}

/// Verifies schema and tools openai prefers tools over structured output.
#[test]
fn schema_and_tools_openai_prefers_tools_over_structured_output() {
    let options = LlmOptions::default()
        .with_tool_choice(ToolChoice::Required)
        .with_output_schema(json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        }))
        .with_tools(vec![ToolDefinition {
            name: "submit".into(),
            description: "Submit the final answer.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }),
        }]);

    let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);

    assert!(
        payload.get("text").is_some(),
        "text format should be set alongside tools"
    );
    assert_eq!(payload["tools"][0]["name"], "submit");
    assert_eq!(payload["tool_choice"], "required");
}

/// Verifies maps response usage and tool call.
#[test]
fn maps_response_usage_and_tool_call() {
    let response = json!({
        "id": "resp_1",
        "model": "gpt-x",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "output": [{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}]
    });
    let mapped = map_response(response, false).unwrap();
    assert_eq!(mapped.usage.unwrap().total(), Some(15));
    assert_eq!(mapped.provider_model.as_deref(), Some("gpt-x"));
    match mapped.output {
        LlmOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "call_1"),
        _ => panic!("expected tool calls"),
    }
}

/// Verifies map response without json mode returns string.
#[test]
fn map_response_without_json_mode_returns_string() {
    let response = json!({
        "id": "resp_1",
        "model": "gpt-x",
        "output_text": "plain text"
    });
    let mapped = map_response(response, false).unwrap();
    match mapped.output {
        LlmOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
        _ => panic!("expected string output"),
    }
}
