//! Streams text and provisional tool progress through the existing LLM client.
//! Running this example makes one generation request only when URL and prompt are supplied.

use futures_util::StreamExt;
use rath::RathError;
use rath::llm::{LlmEvent, LlmOptions, LlmOutput, LlmResponse, Message, ToolDefinition};
use serde_json::json;

/// Accepts an explicit model URL and prompt; without them, prints usage without generating.
#[tokio::main]
async fn main() -> Result<(), RathError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        println!("Usage: cargo run --example llm_streaming -- <model-url> <prompt>");
        return Ok(());
    }
    let client = LlmOptions::default()
        .with_tools(vec![ToolDefinition::new(
            "get_weather".into(),
            "Get the weather for a city".into(),
            json!({"type":"object", "properties":{"city":{"type":"string"}}, "required":["city"]}),
        )])
        .create(&args[0])
        .await?;
    let mut events = client.execute_stream(&[Message::user(&args[1])]).await?;
    while let Some(event) = events.next().await {
        match event? {
            LlmEvent::TextDelta { text } => print!("{text}"),
            LlmEvent::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                println!(
                    "\nTool progress {index} ({id:?}): name={name_delta:?}, args={arguments_delta}"
                );
            }
            LlmEvent::Completed { response } => show_completed(response),
            _ => {}
        }
    }
    Ok(())
}

/// Shows validated final results; the application decides whether and how to execute tools.
fn show_completed(response: LlmResponse) {
    println!("\nCompleted; reported usage: {:?}", response.usage);
    match response.output {
        LlmOutput::Output(value) => println!("Final output: {value}"),
        LlmOutput::ToolCalls { calls, .. } => {
            for call in calls {
                println!(
                    "Requested tool: {} ({}) with {}",
                    call.name, call.id, call.args
                );
            }
        }
    }
}
