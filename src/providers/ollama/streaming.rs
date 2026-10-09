//! Ollama streaming through its existing OpenAI-compatible endpoint.

use super::*;
use crate::llm::{LlmEvent, LlmStream};
use crate::providers::streaming::{self as wire, compatible};

/// Preserves current tools, thinking, schema transforms and authentication on one request.
pub(super) async fn start<'a>(
    client: &'a OllamaClient,
    messages: &[Message],
) -> Result<LlmStream<'a>, RathError> {
    crate::llm::counting::validate_options(Provider::Ollama, &client.options)?;
    validate_history(messages)?;
    validate_tool_request(&client.options)?;
    wire::validate(&client.options)?;
    let options = crate::llm::output::prepare(Provider::Ollama, &client.model, &client.options)?;
    let enabled = !options.tools.is_empty() && options.tool_choice != ToolChoice::Disabled;
    let mut payload = build_payload(&client.model, &options, messages, enabled);
    payload["stream"] = json!(true);
    payload["stream_options"] = json!({"include_usage":true});
    wire::open(
        with_bearer_auth(
            client
                .http
                .post(chat_completions_endpoint(&client.base_url)),
            client.api_key.as_deref(),
        )
        .json(&payload),
        &client.url,
        &client.options,
        client.api_key.as_deref(),
        compatible::initial(),
        decode,
    )
    .await
}

/// Runs terminal cap/tool checks and private final-output conversion before completion.
fn decode(
    event: &str,
    data: &str,
    assembled: &mut Value,
    options: &LlmOptions,
) -> Result<Vec<LlmEvent>, RathError> {
    if event == "ping" {
        return Ok(vec![]);
    }
    if let Some(events) = compatible::chunk(data, Provider::Ollama, assembled)? {
        return Ok(events);
    }
    let mut events = compatible::final_ollama_text(assembled)?;
    events.extend(compatible::final_tools(assembled)?);
    let prepared = crate::llm::output::prepare(Provider::Ollama, "", options)?;
    let response = map_response(std::mem::take(assembled), &prepared)?;
    let response = crate::llm::output::finish(options, response)?;
    events.push(LlmEvent::Completed { response });
    Ok(events)
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
