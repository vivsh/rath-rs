use super::super::*;
use serde_json::json;

/// Verifies custom base url builds anthropic messages endpoint.
#[test]
fn custom_base_url_builds_anthropic_messages_endpoint() {
    assert_eq!(
        messages_endpoint("https://anthropic-proxy.example/v1/"),
        "https://anthropic-proxy.example/v1/messages"
    );
}

/// Anthropic HTTP errors keep the API's structured message and type.
#[test]
fn formats_structured_http_errors() {
    let msg = format_anthropic_http_error(
        404,
        r#"{"type":"error","error":{"type":"not_found_error","message":"model claude-3-5-haiku-latest not found"}}"#,
    );
    assert!(msg.contains("HTTP 404"));
    assert!(msg.contains("not_found_error"));
    assert!(msg.contains("model claude-3-5-haiku-latest not found"));
}

/// Anthropic HTTP errors fall back to the raw body when it is not valid JSON.
#[test]
fn formats_unstructured_http_errors() {
    let msg = format_anthropic_http_error(500, "upstream unavailable");
    assert!(msg.contains("HTTP 500"));
    assert!(msg.contains("upstream unavailable"));
}

/// Verifies messages encode tool exchange.
#[test]
fn messages_encode_tool_exchange() {
    let msgs = build_messages(&[
        Message::user("hi"),
        Message {
            role: Role::AssistantToolCalls {
                calls: vec![ToolCall {
                    id: "toolu_1".into(),
                    name: "lookup".into(),
                    args: json!({"q":"x"}),
                    thought_signatures: None,
                }],
            },
            content: "checking".into(),
            attachments: Vec::new(),
            usage: None,
        },
        Message::tool_output("toolu_1".into(), r#"{"ok":true}"#),
    ]);
    assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
    assert_eq!(msgs[2]["content"][0]["tool_use_id"], "toolu_1");
}

/// Tool results remain unchanged when a reminder is appended as a later user turn.
#[test]
fn messages_keep_tool_result_and_reminder_separate() {
    let msgs = build_messages(&[
        Message {
            role: Role::AssistantToolCalls {
                calls: vec![ToolCall {
                    id: "toolu_1".into(),
                    name: "lookup".into(),
                    args: json!({"q":"x"}),
                    thought_signatures: None,
                }],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
        Message::tool_output("toolu_1".into(), r#"{"ok":true}"#),
        Message::user("<system-reminder><critical>call final_answer</critical></system-reminder>"),
    ]);

    assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
    assert_eq!(
        msgs[1]["content"][0]["content"][0]["text"],
        r#"{"ok":true}"#
    );
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(
        msgs[2]["content"],
        "<system-reminder><critical>call final_answer</critical></system-reminder>"
    );
}

/// Verifies maps tool call and usage.
#[test]
fn maps_tool_call_and_usage() {
    let response = json!({
        "id": "msg_1",
        "model": "claude-x",
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "content": [{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]
    });
    let mapped = map_response(response, false).unwrap();
    assert_eq!(mapped.usage.unwrap().total(), Some(10));
    match mapped.output {
        LlmOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "toolu_1"),
        _ => panic!("expected tool call"),
    }
}

/// Verifies payload appends input schema to system prompt.
#[test]
fn payload_appends_input_schema_to_system_prompt() {
    let options = LlmOptions::default()
        .with_preamble("You are helpful.")
        .with_input_schema(json!({
            "type": "object",
            "properties": {
                "kind": { "type": "string" }
            },
            "required": ["kind"]
        }));

    let payload = build_payload("claude", &options, &[Message::user("hi")], false);

    let system = payload["system"]
        .as_str()
        .expect("system prompt should be a string");
    assert!(system.contains("You are helpful."));
    assert!(system.contains("The user message is JSON."));
    assert!(system.contains("\"required\":[\"kind\"]"));
}

/// Verifies payload without input schema keeps text mode.
#[test]
fn payload_without_input_schema_keeps_text_mode() {
    let payload = build_payload(
        "claude",
        &LlmOptions::default().with_preamble("You are helpful."),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["system"], "You are helpful.");
}

/// Explicit JSON response mode appends a textual JSON-only instruction.
#[test]
fn payload_with_json_response_and_no_output_schema_requests_json_textually() {
    let payload = build_payload(
        "claude",
        &LlmOptions::default()
            .with_preamble("You are helpful.")
            .with_response_format(crate::llm::ResponseFormat::Json),
        &[Message::user("hi")],
        false,
    );
    let system = payload["system"]
        .as_str()
        .expect("system prompt should be present");
    assert!(system.contains("Return only valid JSON."));
}

/// User attachments are emitted as Anthropic image blocks ahead of text.
#[test]
fn user_attachments_use_image_blocks() {
    let msgs = build_messages(&[Message {
        role: Role::User,
        content: "describe this".into(),
        attachments: vec![Attachment::Inline {
            mime_type: "image/png".into(),
            data: "aGVsbG8=".into(),
        }],
        usage: None,
    }]);

    assert_eq!(msgs[0]["content"][0]["type"], "image");
    assert_eq!(msgs[0]["content"][1]["type"], "text");
}

/// Verifies schema and tools anthropic sends both tools and output hint.
#[test]
fn schema_and_tools_anthropic_sends_both_tools_and_output_hint() {
    let options = LlmOptions::default()
        .with_preamble("You are helpful.")
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

    let payload = build_payload("claude", &options, &[Message::user("hi")], true);

    let system = payload["system"]
        .as_str()
        .expect("system should be a string");
    assert!(
        system.contains("You are helpful."),
        "system should contain preamble"
    );
    assert!(
        system.contains("JSON Schema"),
        "system should contain JSON schema hint"
    );
    assert_eq!(payload["tool_choice"]["type"], "any");
    assert_eq!(payload["tools"][0]["name"], "submit");
}

/// Verifies map response without json mode returns string.
#[test]
fn map_response_without_json_mode_returns_string() {
    let response = json!({
        "id": "msg_1",
        "model": "claude-x",
        "content": [{"type":"text","text":"plain text"}]
    });
    let mapped = map_response(response, false).unwrap();
    match mapped.output {
        LlmOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
        _ => panic!("expected string output"),
    }
}

/// With cache=5m, system is an array with a cache_control block.
#[test]
fn payload_with_cache_5m_uses_ephemeral_cache_control() {
    let mut options = LlmOptions::default().with_preamble("Be concise.");
    options.cache = Some(CacheControl::Ephemeral5m);
    let payload = build_payload("claude", &options, &[Message::user("hi")], false);
    let system = payload["system"]
        .as_array()
        .expect("system should be an array with cache");
    assert_eq!(system[0]["type"], "text");
    assert_eq!(system[0]["text"], "Be concise.");
    assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
    assert!(system[0]["cache_control"].get("ttl").is_none());
}

/// With cache=1h, system includes ttl field.
#[test]
fn payload_with_cache_1h_includes_ttl() {
    let mut options = LlmOptions::default().with_preamble("Be concise.");
    options.cache = Some(CacheControl::Ephemeral1h);
    let payload = build_payload("claude", &options, &[Message::user("hi")], false);
    let system = payload["system"]
        .as_array()
        .expect("system should be an array with cache");
    assert_eq!(system[0]["cache_control"]["ttl"], "1h");
}

/// Without cache, system remains a plain string.
#[test]
fn payload_without_cache_system_is_string() {
    let options = LlmOptions::default().with_preamble("Be concise.");
    let payload = build_payload("claude", &options, &[Message::user("hi")], false);
    assert!(payload["system"].is_string());
}

/// cache=5m + json output: schema hint is included in the cached system text.
#[test]
fn payload_cache_with_json_output_includes_schema_hint() {
    let mut options = LlmOptions::default()
        .with_preamble("Be concise.")
        .with_response_format(crate::llm::ResponseFormat::Json);
    options.cache = Some(CacheControl::Ephemeral5m);
    let payload = build_payload("claude", &options, &[Message::user("hi")], false);
    let system = payload["system"]
        .as_array()
        .expect("system should be an array with cache");
    assert!(
        system[0]["text"]
            .as_str()
            .unwrap()
            .contains("Return only valid JSON.")
    );
}
