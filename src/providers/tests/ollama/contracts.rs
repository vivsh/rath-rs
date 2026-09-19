use super::super::*;
use serde_json::json;

/// Verifies custom base url builds ollama endpoints.
#[test]
fn custom_base_url_builds_ollama_endpoints() {
    assert_eq!(
        chat_completions_endpoint("https://ollama-proxy.example/"),
        "https://ollama-proxy.example/v1/chat/completions"
    );
    assert_eq!(
        embed_endpoint("https://ollama-proxy.example"),
        "https://ollama-proxy.example/api/embed"
    );
}

/// User attachments are emitted as OpenAI-compatible image_url parts.
#[test]
fn user_attachments_use_content_parts() {
    let messages = build_messages(
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
        None,
        None,
    );
    assert_eq!(messages[0]["content"][0]["type"], "image_url");
    assert_eq!(messages[0]["content"][1]["type"], "text");
    assert_eq!(messages[0]["content"][1]["text"], "describe this");
}

/// Tool-result attachments are replayed as synthetic user image turns.
#[test]
fn tool_attachments_become_synthetic_user_images() {
    let messages = build_messages(
        &[Message {
            key: None,
            role: Role::Tool {
                call_id: "call-1".into(),
            },
            content: "done".into(),
            attachments: vec![Attachment::Inline {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            }],
            usage: None,
        }],
        None,
        None,
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "image_url");
}

/// Tool results remain unchanged when a reminder is appended as a later user turn.
#[test]
fn build_messages_keep_tool_result_and_reminder_separate() {
    let messages = build_messages(
        &[
            Message {
                key: None,
                role: Role::AssistantToolCalls {
                    calls: vec![ToolCall {
                        id: "call-1".into(),
                        name: "lookup".into(),
                        args: json!({"q":"x"}),
                        thought_signatures: None,
                    }],
                },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
            Message::tool_output("call-1".into(), r#"{"ok":true}"#),
            Message::user("FINAL TURN: call final_answer"),
        ],
        None,
        None,
    );

    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["content"], r#"{"ok":true}"#);
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"], "FINAL TURN: call final_answer");
}

/// Verifies payload uses supplied model.
#[test]
fn payload_uses_supplied_model() {
    let payload = build_payload(
        "custom-local",
        &LlmOptions::default(),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["model"], "custom-local");
    assert!(payload.get("response_format").is_none());
}

/// Structured-output mode includes the provided output schema.
#[test]
fn payload_uses_output_schema_when_present() {
    let schema = json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"]
    });
    let payload = build_payload(
        "custom-local",
        &LlmOptions::default()
            .with_input_schema(json!({ "type": "object" }))
            .with_output_schema(schema.clone()),
        &[Message::user("hi")],
        false,
    );

    assert_eq!(payload["response_format"]["type"], "json_object");
    // schema is injected as a system message, not in response_format
    let system_msgs: Vec<_> = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "system")
        .collect();
    assert!(
        system_msgs
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("answer"))
    );
}

/// Explicit JSON response mode uses Ollama's json_object response format.
#[test]
fn payload_with_json_response_and_no_output_schema_uses_json_object_mode() {
    let payload = build_payload(
        "custom-local",
        &LlmOptions::default().with_response_format(crate::llm::ResponseFormat::Json),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["response_format"]["type"], "json_object");
}

/// Verifies payload prepends input schema to system message.
#[test]
fn payload_prepends_input_schema_to_system_message() {
    let payload = build_payload(
        "custom-local",
        &LlmOptions::default()
            .with_preamble("You are helpful.")
            .with_input_schema(json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" }
                },
                "required": ["kind"]
            })),
        &[Message::user("hi")],
        false,
    );

    let system = payload["messages"][0]["content"]
        .as_str()
        .expect("system message should be a string");
    assert!(system.contains("You are helpful."));
    assert!(system.contains("The user message is JSON."));
    assert!(system.contains("\"required\":[\"kind\"]"));
}

/// Verifies schema and tools ollama sends both tools and response format.
#[test]
fn schema_and_tools_ollama_sends_both_tools_and_response_format() {
    let payload = build_payload(
        "custom-local",
        &LlmOptions::default()
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
            }]),
        &[Message::user("hi")],
        true,
    );

    assert_eq!(
        payload["response_format"]["type"], "json_object",
        "response_format should be set alongside tools"
    );
    assert_eq!(payload["tool_choice"], "required");
    assert_eq!(payload["tools"][0]["function"]["name"], "submit");
}

/// Verifies map response without json mode returns string.
#[test]
fn map_response_without_json_mode_returns_string() {
    let response = json!({
        "model": "local-model",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "plain text"
            }
        }]
    });
    let mapped = map_response(response, &LlmOptions::default()).unwrap();
    match mapped.output {
        LlmOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
        _ => panic!("expected string output"),
    }
}

/// Markdown wrappers around bare JSON keys are repaired before deserialization.
#[test]
fn map_response_repairs_markdown_wrapped_keys_in_json_mode() {
    let response = json!({
        "model": "local-model",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": r#"{"inferences":[],**progress**:75,"queries":[]}"#
            }
        }]
    });

    let mapped = map_response(
        response,
        &LlmOptions::default().with_response_format(crate::llm::ResponseFormat::Json),
    )
    .unwrap();
    match mapped.output {
        LlmOutput::Output(Value::Object(output)) => {
            assert_eq!(output.get("progress"), Some(&json!(75)));
        }
        _ => panic!("expected JSON object output"),
    }
}

/// Fenced JSON and markdown-decorated keys are normalized before parsing.
#[test]
fn sanitize_json_markdown_strips_fences_and_bold_keys() {
    let sanitized = sanitize_json_markdown("```json\n{\"a\":1, **progress**: 75}\n```");
    assert_eq!(sanitized.as_ref(), "{\"a\":1, \"progress\": 75}");
}
