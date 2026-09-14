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

/// Verifies qwen no think is added to first user message.
#[test]
fn qwen_no_think_is_added_to_first_user_message() {
    let messages = build_messages(&[Message::user("do it")], None, "qwen3:8b", false, None);
    assert!(
        messages[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("/no_think")
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
        "qwen3-vl:8b",
        false,
        None,
    );
    assert_eq!(messages[0]["content"][0]["type"], "image_url");
    assert_eq!(messages[0]["content"][1]["type"], "text");
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
        "qwen3-vl:8b",
        false,
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
        "llama3.1",
        false,
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
    let mapped = map_response(response, false).unwrap();
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

    let mapped = map_response(response, true).unwrap();
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

/// Models that emit tool calls as XML-style text in the content field are parsed correctly.
#[test]
fn parse_content_tool_calls_extracts_function_and_params() {
    let content = "<function=file_search>\n<parameter=query>\nforgot password\n</parameter>\n<parameter=globs>\n[\"**/*auth*.js\"]\n</parameter>\n</function>";
    let calls = parse_content_tool_calls(content).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "file_search");
    assert_eq!(calls[0].args["query"], "forgot password");
    assert_eq!(calls[0].args["globs"][0], "**/*auth*.js");
}

/// Multiple tool calls in content are all extracted.
#[test]
fn parse_content_tool_calls_handles_multiple_functions() {
    let content = "<function=search><parameter=q>hello</parameter></function><function=fetch><parameter=url>http://x</parameter></function>";
    let calls = parse_content_tool_calls(content).unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[1].name, "fetch");
}

/// Content with no function tags yields an empty vec (not an error).
#[test]
fn parse_content_tool_calls_returns_empty_on_plain_text() {
    let calls = parse_content_tool_calls("Just a normal response.").unwrap();
    assert!(calls.is_empty());
}

/// collect_tool_calls falls back to content parsing when tool_calls field is absent.
#[test]
fn collect_tool_calls_falls_back_to_content_when_no_tool_calls_field() {
    let message = json!({
        "role": "assistant",
        "content": "<function=my_tool><parameter=x>42</parameter></function>"
    });
    let calls = collect_tool_calls(&message).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "my_tool");
    assert_eq!(calls[0].args["x"], 42);
}
