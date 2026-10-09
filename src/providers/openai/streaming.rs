//! Responses API streaming using the existing request and terminal response mapping.

use super::*;
use crate::llm::{LlmEvent, LlmStream};
use crate::providers::streaming as wire;

/// Validates and dispatches once without retaining caller messages in the response stream.
pub(super) async fn start<'a>(
    client: &'a OpenAiClient,
    messages: &[Message],
) -> Result<LlmStream<'a>, RathError> {
    crate::llm::counting::validate_options(Provider::OpenAi, &client.options)?;
    validate_history(messages)?;
    validate_tools(Provider::OpenAi, &client.options.tools)?;
    wire::validate(&client.options)?;
    let enabled =
        !client.options.tools.is_empty() && client.options.tool_choice != ToolChoice::Disabled;
    let mut payload = build_payload(&client.model, &client.options, messages, enabled);
    payload["stream"] = json!(true);
    wire::open(
        client
            .http
            .post(responses_endpoint(&client.base_url))
            .bearer_auth(&client.api_key)
            .json(&payload),
        &client.url,
        &client.options,
        Some(&client.api_key),
        json!({"output":[]}),
        decode,
    )
    .await
}

/// Interprets one Responses event, keeping the provider envelope as the sole assembly owner.
fn decode(
    event: &str,
    data: &str,
    assembled: &mut Value,
    options: &LlmOptions,
) -> Result<Vec<LlmEvent>, RathError> {
    let value = wire::json(data, Provider::OpenAi)?;
    let kind = wire::string(&value, "type")?;
    if !event.is_empty() && event != kind {
        return Err(wire::invalid("SSE event type disagrees with JSON type"));
    }
    match kind {
        "response.output_item.added" => added(&value, assembled),
        "response.content_part.added" => {
            part_added(&value, assembled)?;
            Ok(vec![])
        }
        "response.output_text.delta" => text_delta(&value, assembled),
        "response.function_call_arguments.delta" => arguments_delta(&value, assembled),
        "response.function_call_arguments.done" => {
            arguments_done(&value, assembled)?;
            Ok(vec![])
        }
        "response.output_item.done" => item_done(&value, assembled),
        "response.output_text.done" => {
            text_done(&value, assembled)?;
            Ok(vec![])
        }
        "response.completed" => complete(&value, assembled, options),
        "response.failed" | "response.incomplete" => failed(&value),
        "error" => Err(crate::core::error::provider_failure(
            Provider::OpenAi,
            "streaming generation",
            &json!({"error":value}),
            &[],
        )),
        kind if kind.starts_with("response.") => Ok(vec![]),
        "ping" => Ok(vec![]),
        _ => Err(wire::invalid("unknown Responses streaming event")),
    }
}

/// Adds one indexed item while rejecting duplicate starts or sparse allocation attacks.
fn added(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "output_index")?;
    let item = value
        .get("item")
        .filter(|v| v.is_object())
        .ok_or_else(|| wire::invalid("missing output item"))?;
    wire::string(item, "id")?;
    let output = assembled["output"]
        .as_array_mut()
        .ok_or_else(|| wire::invalid("invalid output assembly"))?;
    if index != output.len() {
        return Err(wire::invalid("out-of-order or duplicate output item"));
    }
    if item["type"] == "message"
        && item["content"]
            .as_array()
            .is_some_and(|parts| !parts.is_empty())
    {
        return Err(wire::invalid("message item starts with undeclared content"));
    }
    output.push(item.clone());
    if item["type"] == "function_call" {
        let name = wire::string(item, "name")?;
        let args = wire::string(item, "arguments")?;
        let id = wire::string(item, "call_id")?;
        Ok(
            wire::tool_delta(index, Some(id), name, args, (0, 0, false), false)
                .into_iter()
                .collect(),
        )
    } else {
        Ok(vec![])
    }
}

/// Borrows the indexed native item, checking the provider's item correlation ID.
fn item<'a>(value: &Value, assembled: &'a mut Value) -> Result<&'a mut Value, RathError> {
    let index = wire::index(value, "output_index")?;
    let item = assembled["output"]
        .get_mut(index)
        .ok_or_else(|| wire::invalid("delta references unknown output item"))?;
    if wire::string(value, "item_id")? != wire::string(item, "id")? {
        return Err(wire::invalid("output item ID changed"));
    }
    if item["status"] == "completed" {
        return Err(wire::invalid("event references a completed output item"));
    }
    Ok(item)
}

/// Creates a text part before deltas arrive, retaining its native block ordering.
fn part_added(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    let index = wire::index(value, "content_index")?;
    let part = value
        .get("part")
        .filter(|v| v.is_object())
        .ok_or_else(|| wire::invalid("missing content part"))?;
    if part["type"] == "output_text" && !wire::string(part, "text")?.is_empty() {
        return Err(wire::invalid("text part starts with unstreamed content"));
    }
    let item = item(value, assembled)?;
    if item["type"] != "message" {
        return Err(wire::invalid("content part on a non-message item"));
    }
    let parts = item["content"]
        .as_array_mut()
        .ok_or_else(|| wire::invalid("missing message content"))?;
    if index != parts.len() {
        return Err(wire::invalid("out-of-order or duplicate content part"));
    }
    parts.push(part.clone());
    Ok(())
}

/// Borrows a declared text part; no missing output fields are fabricated.
fn part<'a>(value: &Value, assembled: &'a mut Value) -> Result<&'a mut Value, RathError> {
    let index = wire::index(value, "content_index")?;
    let part = item(value, assembled)?["content"]
        .get_mut(index)
        .ok_or_else(|| wire::invalid("delta references unknown content part"))?;
    if part["type"] != "output_text" {
        return Err(wire::invalid("text delta references non-text content"));
    }
    if part["status"] == "completed" {
        return Err(wire::invalid("event references completed text"));
    }
    Ok(part)
}

/// Appends visible text exactly once and produces provisional display progress.
fn text_delta(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "output_index")?;
    let content = wire::index(value, "content_index")?;
    check_text_order(assembled, index, content)?;
    let text = wire::string(value, "delta")?;
    wire::append(&mut part(value, assembled)?["text"], text)?;
    Ok((!text.is_empty())
        .then(|| LlmEvent::TextDelta { text: text.into() })
        .into_iter()
        .collect())
}

/// Rejects answer-block interleaving that a flat text stream cannot represent faithfully.
fn check_text_order(assembled: &Value, index: usize, content: usize) -> Result<(), RathError> {
    for (i, item) in assembled["output"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        for (j, part) in item["content"].as_array().into_iter().flatten().enumerate() {
            if (i, j) > (index, content) && part["text"].as_str().is_some_and(|s| !s.is_empty()) {
                return Err(wire::invalid(
                    "interleaved answer text blocks are unsupported",
                ));
            }
        }
    }
    Ok(())
}

/// Adds argument fragments while preserving IDs and previously emitted name progress.
fn arguments_delta(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "output_index")?;
    let call = item(value, assembled)?;
    if call["type"] != "function_call" {
        return Err(wire::invalid("argument delta references a non-tool item"));
    }
    if call["arguments_status"] == "completed" {
        return Err(wire::invalid("argument delta after argument termination"));
    }
    let name = wire::string(call, "name")?;
    let before = (
        name.len(),
        wire::string(call, "arguments")?.len(),
        wire::visible_name(name),
    );
    wire::append(&mut call["arguments"], wire::string(value, "delta")?)?;
    Ok(wire::tool_delta(
        index,
        Some(wire::string(call, "call_id")?),
        wire::string(call, "name")?,
        wire::string(call, "arguments")?,
        before,
        false,
    )
    .into_iter()
    .collect())
}

/// Ensures complete-argument events do not revise or duplicate provisional fragments.
fn arguments_done(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    let call = item(value, assembled)?;
    if call["type"] != "function_call"
        || wire::string(call, "arguments")? != wire::string(value, "arguments")?
    {
        return Err(wire::invalid(
            "completed arguments disagree with streamed arguments",
        ));
    }
    if call["arguments_status"] == "completed" {
        return Err(wire::invalid("duplicate completed arguments"));
    }
    call["arguments_status"] = json!("completed");
    Ok(())
}

/// Checks final text against already delivered fragments, without yielding it a second time.
fn text_done(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    let part = part(value, assembled)?;
    if wire::string(part, "text")? != wire::string(value, "text")? {
        return Err(wire::invalid("completed text disagrees with streamed text"));
    }
    part["status"] = json!("completed");
    Ok(())
}

/// Validates the completed native item and flushes an ambiguous ordinary tool name once.
fn item_done(value: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = wire::index(value, "output_index")?;
    let final_item = value
        .get("item")
        .ok_or_else(|| wire::invalid("missing completed item"))?;
    let current = assembled["output"]
        .get_mut(index)
        .ok_or_else(|| wire::invalid("completed unknown output item"))?;
    if current["status"] == "completed" {
        return Err(wire::invalid("duplicate completed output item"));
    }
    compare_item(current, final_item)?;
    current["status"] = json!("completed");
    if current["type"] == "function_call" {
        let name = wire::string(current, "name")?;
        let args = wire::string(current, "arguments")?;
        return Ok(wire::tool_delta(
            index,
            Some(wire::string(current, "call_id")?),
            name,
            args,
            (name.len(), args.len(), wire::visible_name(name)),
            true,
        )
        .into_iter()
        .collect());
    }
    Ok(vec![])
}

/// Preserves item identity and rejects terminal output that revises delivered content.
fn compare_item(current: &Value, final_item: &Value) -> Result<(), RathError> {
    if current["id"] != final_item["id"] || current["type"] != final_item["type"] {
        return Err(wire::invalid("completed output item changed identity"));
    }
    if current["type"] == "function_call" {
        for key in ["call_id", "name", "arguments"] {
            if current[key] != final_item[key] {
                return Err(wire::invalid("completed tool call disagrees with progress"));
            }
        }
    } else if current["type"] == "message" {
        let current = collect_text(&json!({"output":[current]}));
        let final_text = collect_text(&json!({"output":[final_item]}));
        if current != final_text {
            return Err(wire::invalid("completed message disagrees with progress"));
        }
    }
    Ok(())
}

/// Uses the provider's authoritative final response only after consistency and cap checks.
fn complete(
    value: &Value,
    assembled: &Value,
    options: &LlmOptions,
) -> Result<Vec<LlmEvent>, RathError> {
    let response = value
        .get("response")
        .ok_or_else(|| wire::invalid("missing completed response"))?;
    if response["status"] != "completed" {
        return Err(wire::invalid("invalid completed response status"));
    }
    crate::llm::counting::check_output_limit(Provider::OpenAi, response)?;
    if response["output"].as_array().map(Vec::len) != assembled["output"].as_array().map(Vec::len) {
        return Err(wire::invalid("completed response changed its output items"));
    }
    for (index, current) in assembled["output"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let final_item = response["output"]
            .get(index)
            .ok_or_else(|| wire::invalid("completed response omitted an output item"))?;
        compare_item(current, final_item)?;
    }
    let response = map_response(response.clone(), options.wants_json_output())?;
    Ok(vec![LlmEvent::Completed { response }])
}

/// Rejects failed/incomplete lifecycle events before parsing partial tool/JSON output.
fn failed(value: &Value) -> Result<Vec<LlmEvent>, RathError> {
    let response = value
        .get("response")
        .ok_or_else(|| wire::invalid("missing failed response"))?;
    crate::llm::counting::check_output_limit(Provider::OpenAi, response)?;
    if response.get("error").is_some_and(|e| !e.is_null()) {
        return Err(crate::core::error::provider_failure(
            Provider::OpenAi,
            "streaming generation",
            response,
            &[],
        ));
    }
    Err(RathError::new(
        crate::ErrorKind::Provider,
        "generation failed or was incomplete",
    ))
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
