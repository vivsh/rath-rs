//! Opt-in public-client tests for installed local models; no downloads or real tool side effects.
use rath::llm::{
    ErrorKind, LlmOptions, LlmOutput, LlmResponse, Message, Role, ThinkingLevel, ToolChoice,
    ToolDefinition,
};
use serde_json::json;
use std::time::{Duration, Instant};

/// Exercises JSON input, a real Granite tool call, result replay and JSON final output.
#[tokio::test]
#[ignore = "requires installed granite4.2:3b on local Ollama"]
async fn granite_json_tool_roundtrip() {
    roundtrip("granite4.2:3b", ToolChoice::Disabled).await;
}

/// Exercises JSON input, a real Qwen tool call, result replay and JSON final output.
#[tokio::test]
#[ignore = "requires installed qwen3:8b on local Ollama"]
async fn qwen_json_tool_roundtrip() {
    roundtrip("qwen3:8b", ToolChoice::Auto).await;
}

/// A small model that fails to call a required tool must return an informative error, not text.
#[tokio::test]
#[ignore = "requires installed qwen3:0.6b on local Ollama"]
async fn qwen_small_required_tool_has_explicit_outcome() {
    let client = options(ToolChoice::Required)
        .create(&url("qwen3:0.6b"))
        .unwrap();
    let result = tokio::time::timeout(Duration::from_secs(90),client.execute(&[
        Message::user("Use lookup_code with label alpha. After getting its result, reply with the code exactly.")
    ])).await.unwrap();
    match result {
        Ok(response) => {
            println!("qwen3:0.6b required call: {:?}", response.output);
            assert!(
                matches!(response.output,LlmOutput::ToolCalls { calls, .. } if !calls.is_empty())
            );
        }
        Err(error) => {
            println!("qwen3:0.6b required-call error: {error}");
            assert_eq!(error.kind(), ErrorKind::InvalidResponse);
            assert!(error.message().contains("Required"));
            assert!(error.response_body().is_some());
        }
    }
}

/// Supplies one harmless synthetic lookup tool, with no actual external execution.
fn options(choice: ToolChoice) -> LlmOptions {
    LlmOptions::default().with_temperature(0.0).with_thinking(Some(ThinkingLevel::Off))
        .with_max_output_tokens(512).with_tool_choice(choice).with_tools(vec![ToolDefinition {
            name:"lookup_code".into(),description:"Look up the code for a label. Always use this tool to look up codes.".into(),
            parameters:json!({"type":"object","properties":{"label":{"type":"string"}},"required":["label"]}),
        }])
}

fn url(model: &str) -> String {
    format!("ollama:///{model}?base_url=http://127.0.0.1:11434")
}

/// Verifies a real call and passes its matching result into a separately configured JSON-output turn.
async fn roundtrip(model: &str, final_choice: ToolChoice) {
    let first = options(ToolChoice::Required)
        .with_preamble("Use lookup_code to look up the label in the user's JSON.")
        .with_input_schema(json!({"type":"object","properties":{"label":{"type":"string"}}}))
        .create(&url(model))
        .unwrap();
    let mut history = vec![Message::from_json(Role::User, &json!({"label":"alpha"})).unwrap()];
    let start = Instant::now();
    let response = tokio::time::timeout(Duration::from_secs(90), first.execute(&history))
        .await
        .unwrap()
        .unwrap();
    report(model, "tool_call", &response, start);
    let LlmOutput::ToolCalls { calls, thought } = response.output else {
        panic!("expected tool calls")
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "lookup_code");
    assert_eq!(calls[0].args, json!({"label":"alpha"}));
    let id = calls[0].id.clone();
    let mut assistant = Message::assistant(thought.unwrap_or_default());
    assistant.role = Role::AssistantToolCalls { calls };
    history.push(assistant);
    history.push(Message::tool_output(id, r#"{"code":"KITE-742"}"#));
    let second = options(final_choice)
        .with_preamble("The lookup is complete. Return its result as JSON with the code field.")
        .with_output_schema(
            json!({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}),
        )
        .create(&url(model))
        .unwrap();
    let start = Instant::now();
    let response = tokio::time::timeout(Duration::from_secs(90), second.execute(&history))
        .await
        .unwrap()
        .unwrap();
    report(model, "json_output", &response, start);
    assert!(matches!(response.output,LlmOutput::Output(v) if v==json!({"code":"KITE-742"})));
}

/// Records synthetic results, token usage and finish status without assuming missing reasoning is zero.
fn report(model: &str, phase: &str, response: &LlmResponse, start: Instant) {
    let metadata = response.raw_metadata.as_ref().unwrap();
    println!(
        "{}",
        json!({"model":model,"phase":phase,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
        "output":format!("{:?}",response.output),"metadata":metadata})
    );
    let expected = if phase == "tool_call" {
        "tool_calls"
    } else {
        "stop"
    };
    assert_eq!(metadata["finish_reason"], expected);
}
