use crate::llm::{LlmOptions, ToolChoice, ToolDefinition};
use serde_json::{Value, json};

pub(super) fn options(choice: ToolChoice) -> LlmOptions {
    LlmOptions::default()
        .with_tool_choice(choice)
        .with_tools(vec![ToolDefinition {
            name: "lookup".into(),
            description: "Look up a value".into(),
            parameters: json!({"type":"object","properties":{"q":{"type":"string"}}}),
        }])
}

pub(super) fn response(message: Value, finish: &str) -> Value {
    json!({"model":"test", "choices":[{"finish_reason":finish,"message":message}],
        "usage":{"prompt_tokens":12,"completion_tokens":8}})
}

pub(super) fn call() -> Value {
    json!({"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}})
}
