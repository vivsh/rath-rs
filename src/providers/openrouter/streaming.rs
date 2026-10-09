//! OpenRouter streaming through the existing Chat Completions endpoint.

use super::*;
use crate::llm::{LlmEvent, LlmStream};
use crate::providers::streaming::{self as wire, compatible};

/// Sends one validated generation, preserving prompt construction and immutable options.
pub(super) async fn start<'a>(
    client: &'a OpenRouterClient,
    messages: &[Message],
) -> Result<LlmStream<'a>, RathError> {
    crate::llm::counting::validate_options(Provider::OpenRouter, &client.options)?;
    validate_history(messages)?;
    validate_tools(Provider::OpenRouter, &client.options.tools)?;
    wire::validate(&client.options)?;
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    let mut payload = build_payload(&client.model, &client.options, messages, enabled);
    payload["stream"] = json!(true);
    payload["stream_options"] = json!({"include_usage":true});
    wire::open(
        client
            .http
            .post(chat_completions_endpoint(&client.base_url))
            .bearer_auth(&client.api_key)
            .json(&payload),
        &client.url,
        &client.options,
        Some(&client.api_key),
        compatible::initial(),
        decode,
    )
    .await
}

/// Collects final usage before the terminal marker and existing completed-response mapping.
fn decode(
    event: &str,
    data: &str,
    assembled: &mut Value,
    options: &LlmOptions,
) -> Result<Vec<LlmEvent>, RathError> {
    if event == "ping" {
        return Ok(vec![]);
    }
    if let Some(events) = compatible::chunk(data, Provider::OpenRouter, assembled)? {
        return Ok(events);
    }
    let mut events = compatible::final_tools(assembled)?;
    let response = map_response(std::mem::take(assembled), options.wants_json_output())?;
    events.push(LlmEvent::Completed { response });
    Ok(events)
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
