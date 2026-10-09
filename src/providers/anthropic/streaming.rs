//! Messages streaming with native block ordering, provisional tools and terminal validation.

use super::*;
use crate::llm::{LlmEvent, LlmStream};
use crate::providers::streaming as wire;

/// Sends the same validated Messages request with streaming enabled.
pub(super) async fn start<'a>(
    client: &'a AnthropicClient,
    messages: &[Message],
) -> Result<LlmStream<'a>, RathError> {
    crate::llm::counting::validate_options(Provider::Anthropic, &client.options)?;
    validate_history(messages)?;
    validate_tools(Provider::Anthropic, &client.options.tools)?;
    wire::validate(&client.options)?;
    if matches!(&client.options.thinking, Some(t) if *t != ThinkingLevel::Off) {
        return Err(RathError::unsupported(
            Provider::Anthropic,
            "thinking is not exposed by the Anthropic adapter yet",
        ));
    }
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    let mut payload = build_payload(&client.model, &client.options, messages, enabled);
    payload["stream"] = json!(true);
    wire::open(
        client
            .http
            .post(messages_endpoint(&client.base_url))
            .header("x-api-key", &client.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload),
        &client.url,
        &client.options,
        Some(&client.api_key),
        Value::Null,
        decode,
    )
    .await
}

/// Advances one Messages event; no tool arguments are parsed before terminal cap checks.
fn decode(
    event: &str,
    data: &str,
    assembled: &mut Value,
    options: &LlmOptions,
) -> Result<Vec<LlmEvent>, RathError> {
    let value = wire::json(data, Provider::Anthropic)?;
    let kind = wire::string(&value, "type")?;
    if !event.is_empty() && event != kind {
        return Err(wire::invalid("SSE event type disagrees with JSON type"));
    }
    if kind == "ping" {
        return Ok(vec![]);
    }
    if kind == "message_start" {
        message_start(&value, assembled)?;
        return Ok(vec![]);
    }
    if !assembled.is_object() {
        return Err(wire::invalid("event arrived before message_start"));
    }
    if !assembled["stop_reason"].is_null() && !matches!(kind, "message_delta" | "message_stop") {
        return Err(wire::invalid("content event after message termination"));
    }
    match kind {
        "content_block_start" => block_start(&value, assembled),
        "content_block_delta" => block_delta(&value, assembled),
        "content_block_stop" => block_stop(&value, assembled),
        "message_delta" => {
            message_delta(&value, assembled)?;
            Ok(vec![])
        }
        "message_stop" => complete(assembled, options),
        _ => Err(wire::invalid("unknown Messages streaming event")),
    }
}

/// Establishes one native message without accepting duplicate message starts.
fn message_start(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    if !assembled.is_null() {
        return Err(wire::invalid("duplicate message_start"));
    }
    let message = value
        .get("message")
        .filter(|v| v.is_object())
        .ok_or_else(|| wire::invalid("missing initial message"))?;
    if message["type"] != "message" || message["role"] != "assistant" {
        return Err(wire::invalid("invalid initial message type or role"));
    }
    wire::string(message, "id")?;
    if !message["content"].as_array().is_some_and(Vec::is_empty) {
        return Err(wire::invalid("initial message must have empty content"));
    }
    *assembled = message.clone();
    Ok(())
}

/// Starts an indexed content block and makes initial tool/text progress available.
fn block_start(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "index")?;
    let mut block = value
        .get("content_block")
        .filter(|v| v.is_object())
        .cloned()
        .ok_or_else(|| wire::invalid("missing content block"))?;
    let kind = wire::string(&block, "type")?;
    let events = match kind {
        "text" => {
            let text = wire::string(&block, "text")?;
            (!text.is_empty())
                .then(|| LlmEvent::TextDelta { text: text.into() })
                .into_iter()
                .collect()
        }
        "tool_use" => start_tool(index, &mut block)?,
        "thinking" | "redacted_thinking" => vec![],
        _ => return Err(wire::invalid("unsupported content block type")),
    };
    let blocks = assembled["content"]
        .as_array_mut()
        .ok_or_else(|| wire::invalid("missing message content"))?;
    if index != blocks.len() {
        return Err(wire::invalid("out-of-order or duplicate content block"));
    }
    block["status"] = json!("open");
    blocks.push(block);
    Ok(events)
}

/// Holds native tool input as a JSON string until whole-message termination is checked.
fn start_tool(index: usize, block: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let input = block
        .get("input")
        .filter(|v| v.is_object())
        .ok_or_else(|| wire::invalid("tool input must be an object"))?;
    let args = if input.as_object().is_some_and(|v| v.is_empty()) {
        String::new()
    } else {
        input.to_string()
    };
    block["input"] = Value::String(args);
    Ok(wire::tool_delta(
        index,
        Some(wire::string(block, "id")?),
        wire::string(block, "name")?,
        wire::string(block, "input")?,
        (0, 0, false),
        false,
    )
    .into_iter()
    .collect())
}

/// Borrows an active block and rejects deltas after its native stop event.
fn block<'a>(value: &Value, assembled: &'a mut Value) -> Result<&'a mut Value, RathError> {
    let index = wire::index(value, "index")?;
    let block = assembled["content"]
        .get_mut(index)
        .ok_or_else(|| wire::invalid("unknown content block index"))?;
    if block["status"] != "open" {
        return Err(wire::invalid("content block is already stopped"));
    }
    Ok(block)
}

/// Adds typed content fragments without exposing reasoning as answer text.
fn block_delta(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "index")?;
    let delta = value
        .get("delta")
        .ok_or_else(|| wire::invalid("missing content delta"))?;
    if delta["type"] == "text_delta" {
        check_text_order(assembled, index)?;
    }
    let block = block(value, assembled)?;
    match wire::string(delta, "type")? {
        "text_delta" if block["type"] == "text" => {
            let text = wire::string(delta, "text")?;
            wire::append(&mut block["text"], text)?;
            Ok((!text.is_empty())
                .then(|| LlmEvent::TextDelta { text: text.into() })
                .into_iter()
                .collect())
        }
        "input_json_delta" if block["type"] == "tool_use" => tool_delta(index, block, delta),
        "thinking_delta" if block["type"] == "thinking" => {
            wire::append(&mut block["thinking"], wire::string(delta, "thinking")?)?;
            Ok(vec![])
        }
        "signature_delta" if block["type"] == "thinking" => {
            wire::append(&mut block["signature"], wire::string(delta, "signature")?)?;
            Ok(vec![])
        }
        _ => Err(wire::invalid("invalid content delta for block type")),
    }
}

/// Rejects text interleaving whose delivery order disagrees with the final block order.
fn check_text_order(assembled: &Value, index: usize) -> Result<(), RathError> {
    if assembled["content"]
        .as_array()
        .into_iter()
        .flatten()
        .skip(index + 1)
        .any(|block| {
            block["type"] == "text" && block["text"].as_str().is_some_and(|text| !text.is_empty())
        })
    {
        return Err(wire::invalid(
            "interleaved answer text blocks are unsupported",
        ));
    }
    Ok(())
}

/// Streams argument strings while the native input field remains the sole assembly owner.
fn tool_delta(index: usize, block: &mut Value, delta: &Value) -> Result<Vec<LlmEvent>, RathError> {
    let name = wire::string(block, "name")?;
    let before = (
        name.len(),
        wire::string(block, "input")?.len(),
        wire::visible_name(name),
    );
    wire::append(&mut block["input"], wire::string(delta, "partial_json")?)?;
    Ok(wire::tool_delta(
        index,
        Some(wire::string(block, "id")?),
        wire::string(block, "name")?,
        wire::string(block, "input")?,
        before,
        false,
    )
    .into_iter()
    .collect())
}

/// Records native block termination once, without parsing incomplete argument JSON.
fn block_stop(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "index")?;
    let block = block(value, assembled)?;
    block["status"] = json!("stopped");
    if block["type"] != "tool_use" {
        return Ok(vec![]);
    }
    let name = wire::string(block, "name")?;
    let args = wire::string(block, "input")?;
    Ok(wire::tool_delta(
        index,
        Some(wire::string(block, "id")?),
        name,
        args,
        (name.len(), args.len(), wire::visible_name(name)),
        true,
    )
    .into_iter()
    .collect())
}

/// Merges stop fields and reported usage without summing cumulative token updates.
fn message_delta(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    let delta = value
        .get("delta")
        .filter(|v| v.is_object())
        .ok_or_else(|| wire::invalid("missing message delta"))?;
    for key in ["stop_reason", "stop_sequence"] {
        if let Some(value) = delta.get(key).filter(|v| !v.is_null()) {
            if !assembled[key].is_null() && assembled[key] != *value {
                return Err(wire::invalid("message termination changed"));
            }
            assembled[key] = value.clone();
        }
    }
    if let Some(usage) = value.get("usage") {
        let usage = usage
            .as_object()
            .ok_or_else(|| wire::invalid("invalid message usage"))?;
        if assembled["usage"].is_null() {
            assembled["usage"] = json!({});
        }
        for (key, value) in usage {
            assembled["usage"][key] = value.clone();
        }
    }
    crate::llm::counting::check_output_limit(Provider::Anthropic, assembled)
}

/// Validates native termination, then parses tool input and reuses the existing response mapper.
fn complete(assembled: &mut Value, options: &LlmOptions) -> Result<Vec<LlmEvent>, RathError> {
    crate::llm::counting::check_output_limit(Provider::Anthropic, assembled)?;
    let reason = wire::string(assembled, "stop_reason")?;
    if !matches!(reason, "end_turn" | "tool_use" | "stop_sequence") {
        return Err(wire::invalid("unsupported message stop reason"));
    }
    let blocks = assembled["content"]
        .as_array_mut()
        .ok_or_else(|| wire::invalid("missing content blocks"))?;
    for block in blocks {
        if block["status"] != "stopped" {
            return Err(wire::invalid("message stopped with an open content block"));
        }
        block
            .as_object_mut()
            .ok_or_else(|| wire::invalid("invalid content block"))?
            .remove("status");
        if block["type"] == "tool_use" {
            let input = wire::string(block, "input")?;
            block["input"] = if input.is_empty() {
                json!({})
            } else {
                serde_json::from_str(input)
                    .map_err(|e| RathError::from_native_error(crate::ErrorKind::Deserialize, &e))?
            };
            if !block["input"].is_object() {
                return Err(wire::invalid("tool arguments must be an object"));
            }
        }
    }
    let response = map_response(std::mem::take(assembled), options.wants_json_output())?;
    Ok(vec![LlmEvent::Completed { response }])
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
