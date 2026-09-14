use super::super::*;
use serde_json::json;

fn make_call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args: json!({}),
        thought_signatures: None,
    }
}

/// A single user turn produces one provider message.
#[test]
fn build_messages_user_only() {
    let history = vec![Message::user(r#"{"text":"hi"}"#)];
    let msgs = build_gemini_messages(&history);
    assert_eq!(msgs.len(), 1);
}

/// User attachments are converted into Gemini inline or file parts.
#[test]
fn build_messages_user_with_attachment_adds_inline_part() {
    let history = vec![Message {
        key: None,
        role: Role::User,
        content: "describe this".into(),
        attachments: vec![Attachment::Inline {
            mime_type: "image/png".into(),
            data: "aGVsbG8=".into(),
        }],
        usage: None,
    }];
    let msgs = build_gemini_messages(&history);
    let parts = msgs[0]
        .content
        .parts
        .as_ref()
        .expect("user message parts should be present");
    assert!(matches!(parts.first(), Some(Part::InlineData { .. })));
    assert!(matches!(
        parts.last(),
        Some(Part::Text { text, .. }) if text == "describe this"
    ));
}

/// Preambles are not duplicated into history messages.
#[test]
fn build_messages_preamble_is_separate() {
    let history = vec![Message::user(r#"{"text":"hi"}"#)];
    let msgs = build_gemini_messages(&history);
    assert_eq!(msgs.len(), 1);
}

/// History order is preserved.
#[test]
fn build_messages_history_in_order() {
    let history = vec![
        Message::user("prev question"),
        Message::assistant("prev answer"),
        Message::user("next question"),
    ];
    let msgs = build_gemini_messages(&history);
    assert_eq!(msgs.len(), 3);
    let debug = format!("{msgs:?}");
    assert!(debug.contains("prev question"));
    assert!(debug.contains("prev answer"));
}

/// Tool responses are grouped into a function-response message.
#[test]
fn build_messages_tool_role_included() {
    let history = vec![
        Message {
            key: None,
            role: Role::AssistantToolCalls {
                calls: vec![make_call("call-42", "read_file")],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
        Message {
            key: None,
            role: Role::Tool {
                call_id: "call-42".into(),
            },
            content: r#"{"temp":22}"#.into(),
            attachments: Vec::new(),
            usage: None,
        },
    ];
    let msgs = build_gemini_messages(&history);
    assert_eq!(msgs.len(), 2);
    let debug = format!("{msgs:?}");
    assert!(debug.contains("read_file"));
}

/// Tool results keep the exchange length aligned with history.
#[test]
fn build_messages_continue_after_tool_result() {
    let history = vec![
        Message::user(r#"{"goal":"ship","known_context":[]}"#),
        Message {
            key: None,
            role: Role::AssistantToolCalls {
                calls: vec![make_call("c1", "project_outline")],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
        Message {
            key: None,
            role: Role::Tool {
                call_id: "c1".into(),
            },
            content: r#"{"files":[]}"}"#.into(),
            attachments: Vec::new(),
            usage: None,
        },
    ];
    let msgs = build_gemini_messages(&history);
    assert_eq!(msgs.len(), 3);
}

/// Tool responses remain structured when a reminder user turn follows them.
#[test]
fn build_messages_keeps_tool_response_and_reminder_separate() {
    let history = vec![
        Message {
            key: None,
            role: Role::AssistantToolCalls {
                calls: vec![make_call("c1", "project_outline")],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
        Message {
            key: None,
            role: Role::Tool {
                call_id: "c1".into(),
            },
            content: r#"{"result":"ok"}"#.into(),
            attachments: Vec::new(),
            usage: None,
        },
        Message::user("<system-reminder><critical>call final_answer</critical></system-reminder>"),
    ];

    let msgs = build_gemini_messages(&history);

    assert_eq!(msgs.len(), 3);
    assert!(matches!(msgs[1].role, GeminiRole::User));
    assert!(matches!(msgs[2].role, GeminiRole::User));

    let tool_parts = msgs[1]
        .content
        .parts
        .as_ref()
        .expect("tool response parts should be present");
    assert!(matches!(
        tool_parts.first(),
        Some(Part::FunctionResponse { .. })
    ));

    let tool_debug = format!("{:?}", tool_parts[0]);
    assert!(tool_debug.contains("result"));
    assert!(!tool_debug.contains("system-reminder"));

    let reminder_debug = format!("{:?}", msgs[2]);
    assert!(reminder_debug.contains("system-reminder"));
    assert!(!reminder_debug.contains("result\":\"ok"));
}

/// Explicit JSON response mode enables Gemini JSON output and response schema handling.
#[test]
fn response_mode_uses_explicit_json_setting() {
    let no_schema = LlmOptions::default();
    assert!(!wants_json_output(&no_schema));
    assert!(response_schema(&no_schema).is_none());

    let with_schema = LlmOptions::default()
        .with_response_format(crate::llm::ResponseFormat::Json)
        .with_output_schema(json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        }));
    assert!(wants_json_output(&with_schema));
    assert!(response_schema(&with_schema).is_some());
}

/// Input schema hints do not change Gemini response mode on their own.
#[test]
fn input_schema_alone_does_not_enable_json_output() {
    let with_input_schema = LlmOptions::default().with_input_schema(json!({ "type": "object" }));
    assert!(!wants_json_output(&with_input_schema));
    assert!(response_schema(&with_input_schema).is_none());
}

/// Verifies provider config parses safety settings.
#[test]
fn provider_config_parses_safety_settings() {
    let config = Some(json!({
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
    }));

    let settings = gemini_safety_settings_from_provider_config(&config)
        .expect("safety settings should parse")
        .expect("safety settings should be present");

    assert_eq!(settings.len(), 2);
}

/// Verifies provider config rejects malformed safety settings.
#[test]
fn provider_config_rejects_malformed_safety_settings() {
    let config = Some(json!({
        "safetySettings": [
            {
                "category": "not-a-category",
                "threshold": "BLOCK_NONE"
            }
        ]
    }));

    let error = gemini_safety_settings_from_provider_config(&config)
        .expect_err("invalid safety settings should fail before request execution");

    assert!(matches!(error, RathError::Validation(_)));
    assert!(error.to_string().contains("provider_config.safetySettings"));
}
