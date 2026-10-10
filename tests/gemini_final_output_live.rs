//! Opt-in scalar-answer verification against Gemini; no business tools are executed.

use futures_util::StreamExt;
use rath::llm::{
    LlmEvent, LlmOptions, LlmOutput, LlmResponse, Message, RathError, Role, ToolChoice,
    ToolDefinition,
};
use serde_json::json;
use std::time::Duration;

/// Verifies the public client returns a string when Gemini requires object function arguments.
#[tokio::test]
#[ignore = "requires explicit live-provider approval and GEMINI_API_KEY"]
async fn gemini_scalar_final_output_live() -> Result<(), RathError> {
    let client = LlmOptions::default()
        .with_preamble("Return exactly OK using the final-output tool. Do not call business tools.")
        .with_output_schema(json!({"type":"string"}))
        .with_tools(tools())
        .with_max_output_tokens(1800)
        .create("gemini:///gemini-3-flash-preview")
        .await?;
    let response = tokio::time::timeout(
        Duration::from_secs(30),
        client.execute(&[Message::user("Return exactly OK.")]),
    )
    .await
    .expect("live Gemini request exceeded 30 seconds")?;
    assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
    let messages = [Message::user("Return exactly OK.")];
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut stream = client.execute_stream(&messages).await?;
        let LlmEvent::Completed { response } = stream.next().await.unwrap()? else {
            panic!("expected private final-output completion without tool progress");
        };
        assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
        assert!(stream.next().await.is_none());
        Ok::<(), RathError>(())
    })
    .await
    .expect("live Gemini stream exceeded 30 seconds")?;
    Ok(())
}

/// Supplies only a synthetic numeric-enum tool, never private application metadata.
fn tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition::new(
        "lookup".into(),
        "Synthetic lookup; do not call this tool.".into(),
        json!({"type":"object","properties":{"status":{"type":"integer","enum":[1,2,3]}}}),
    )]
}

/// Real Gemini calls a synthetic tool, accepts its array result, then returns a scalar answer.
#[tokio::test]
#[ignore = "requires explicit live-provider approval and GEMINI_API_KEY"]
async fn gemini_array_tool_result_roundtrip_live() -> Result<(), RathError> {
    let options = LlmOptions::default()
        .with_max_output_tokens(1800)
        .with_tools(vec![ToolDefinition::new(
            "lookup_code".into(),
            "Retrieve the synthetic test code.".into(),
            json!({"type":"object","properties":{}}),
        )]);
    let first = options
        .clone()
        .with_tool_choice(ToolChoice::Required)
        .create("gemini:///gemini-3-flash-preview")
        .await?;
    let mut history = vec![Message::user(
        "Use lookup_code, then return the code from its result.",
    )];
    let response = bounded(first.execute(&history)).await?;
    let LlmOutput::ToolCalls { calls, text } = response.output else {
        panic!("expected real Gemini tool call");
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "lookup_code");
    let id = calls[0].id.clone();
    let mut assistant = Message::assistant(text.unwrap_or_default());
    assistant.role = Role::AssistantToolCalls { calls };
    history.push(assistant);
    history.push(Message::tool_output(id, r#"[{"code":"KITE-742"}]"#));
    let final_client = options
        .with_output_schema(json!({"type":"string"}))
        .with_preamble("Lookup is complete. Use the final-output tool to return only the code.")
        .create("gemini:///gemini-3-flash-preview")
        .await?;
    let response = bounded(final_client.execute(&history)).await?;
    assert!(matches!(response.output, LlmOutput::Output(value) if value == "KITE-742"));
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut stream = final_client.execute_stream(&history).await?;
        let LlmEvent::Completed { response } = stream.next().await.unwrap()? else {
            panic!("expected private final-output completion");
        };
        assert!(matches!(response.output, LlmOutput::Output(value) if value == "KITE-742"));
        assert!(stream.next().await.is_none());
        Ok::<(), RathError>(())
    })
    .await
    .expect("live Gemini stream exceeded 30 seconds")?;
    Ok(())
}

/// Bounds live provider generation without adding retries or automatic tool dispatch.
async fn bounded(
    future: impl std::future::Future<Output = Result<LlmResponse, RathError>>,
) -> Result<LlmResponse, RathError> {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("live Gemini request exceeded 30 seconds")
}
