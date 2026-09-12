use super::super::*;

/// Verifies endpoint uses chat completions.
#[test]
fn endpoint_uses_chat_completions() {
    assert_eq!(
        chat_completions_endpoint("https://openrouter.ai/api/v1/"),
        "https://openrouter.ai/api/v1/chat/completions"
    );
}

/// Verifies payload uses openrouter model slug.
#[test]
fn payload_uses_openrouter_model_slug() {
    let payload = build_payload(
        "openai/gpt-5.2",
        &LlmOptions::default(),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["model"], "openai/gpt-5.2");
    assert_eq!(payload["messages"][0]["content"], "hi");
}

/// Verifies payload with schema uses openrouter json schema shape.
#[test]
fn payload_with_schema_uses_openrouter_json_schema_shape() {
    let payload = build_payload(
        "openai/gpt-5.2",
        &LlmOptions::default().with_output_schema(json!({
            "type": "object",
            "properties": {
                "ok": { "type": "boolean" }
            }
        })),
        &[Message::user("hi")],
        false,
    );
    assert_eq!(payload["response_format"]["type"], "json_schema");
    assert_eq!(
        payload["response_format"]["json_schema"]["schema"]["type"],
        "object"
    );
}

/// Verifies maps tool call response.
#[test]
fn maps_tool_call_response() {
    let response = json!({
        "id": "gen-1",
        "object": "chat.completion",
        "model": "openai/gpt-5.2",
        "usage": { "prompt_tokens": 10, "completion_tokens": 4 },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\"}"
                    }
                }]
            }
        }]
    });
    let mapped = map_response(response, false).unwrap();
    assert_eq!(mapped.provider, Provider::OpenRouter);
    assert_eq!(mapped.usage.unwrap().total(), Some(14));
    match mapped.output {
        LlmOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].name, "lookup"),
        _ => panic!("expected tool calls"),
    }
}
