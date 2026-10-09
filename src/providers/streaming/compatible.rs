//! Shared Chat Completions chunk assembly for the existing compatible endpoints.

use super::{append, index, invalid, json, string, tool_delta, visible_name};
use crate::llm::LlmEvent;
use crate::{Provider, RathError};
use serde_json::{Value, json};

/// Initializes one native compatible response; no model or usage value is fabricated.
pub(in crate::providers) fn initial() -> Value {
    json!({"choices":[{"index":0, "message":{"role":"assistant", "content":null}, "finish_reason":null}]})
}

/// Returns progress, or None only after both finish evidence and the provider's final marker.
pub(in crate::providers) fn chunk(
    data: &str,
    provider: Provider,
    assembled: &mut Value,
) -> Result<Option<Vec<LlmEvent>>, RathError> {
    if data == "[DONE]" {
        crate::llm::counting::check_output_limit(provider, assembled)?;
        let reason = string(&assembled["choices"][0], "finish_reason")?;
        if !matches!(reason, "stop" | "tool_calls" | "function_call") {
            return Err(invalid("unsupported choice finish reason"));
        }
        return Ok(None);
    }
    let value = json(data, provider.clone())?;
    crate::llm::counting::check_output_limit(provider.clone(), &value)?;
    if provider == Provider::OpenRouter {
        crate::providers::openrouter::check_response_failure(&value, "streaming generation")?;
    }
    update_metadata(&value, assembled)?;
    let Some(choice) = selected_choice(&value)? else {
        return Ok(Some(vec![]));
    };
    let delta = choice
        .get("delta")
        .filter(|v| v.is_object())
        .ok_or_else(|| invalid("missing choice delta"))?;
    let terminal = !assembled["choices"][0]["finish_reason"].is_null();
    let events = deltas(delta, provider.clone(), assembled, terminal)?;
    if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
        if !reason.is_string() {
            return Err(invalid("invalid finish reason"));
        }
        if terminal && assembled["choices"][0]["finish_reason"] != *reason {
            return Err(invalid("choice finish reason changed"));
        }
        assembled["choices"][0]["finish_reason"] = reason.clone();
    }
    if let Some(reason) = choice.get("native_finish_reason") {
        assembled["choices"][0]["native_finish_reason"] = reason.clone();
    }
    crate::llm::counting::check_output_limit(provider, assembled)?;
    Ok(Some(events))
}

/// Selects one native choice, allowing a usage-only accounting frame without output.
fn selected_choice(value: &Value) -> Result<Option<&Value>, RathError> {
    let choices = value["choices"]
        .as_array()
        .ok_or_else(|| invalid("missing streaming choices"))?;
    if choices.is_empty() {
        if value.get("usage").is_none() {
            return Err(invalid("empty choices without usage"));
        }
        return Ok(None);
    }
    if choices.len() != 1 || index(&choices[0], "index")? != 0 {
        return Err(invalid("multiple or alternate choices are unsupported"));
    }
    Ok(Some(&choices[0]))
}

/// Records echoed identity and latest reported usage, rejecting contradictory identities.
fn update_metadata(value: &Value, assembled: &mut Value) -> Result<(), RathError> {
    for key in ["id", "model"] {
        if let Some(value) = value.get(key).filter(|v| !v.is_null()) {
            if !value.is_string() {
                return Err(invalid("invalid response identity"));
            }
            if !assembled[key].is_null() && assembled[key] != *value {
                return Err(invalid("response identity changed"));
            }
            assembled[key] = value.clone();
        }
    }
    if let Some(object) = value.get("object") {
        assembled["object"] = object.clone();
    }
    if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
        if !usage.is_object() {
            return Err(invalid("invalid streaming usage"));
        }
        assembled["usage"] = usage.clone();
    }
    Ok(())
}

/// Applies answer, reasoning and interleaved tool fragments to one native message.
fn deltas(
    delta: &Value,
    provider: Provider,
    assembled: &mut Value,
    terminal: bool,
) -> Result<Vec<LlmEvent>, RathError> {
    if let Some(role) = delta.get("role")
        && role != "assistant"
    {
        return Err(invalid("invalid streamed message role"));
    }
    let message = &mut assembled["choices"][0]["message"];
    let mut events = Vec::new();
    if let Some(text) = delta.get("content").filter(|v| !v.is_null()) {
        let text = text.as_str().ok_or_else(|| invalid("invalid text delta"))?;
        if terminal && !text.is_empty() {
            return Err(invalid("text delta after choice termination"));
        }
        let before = visible_text(&message["content"], provider == Provider::Ollama).len();
        append(&mut message["content"], text)?;
        let text = visible_text(&message["content"], provider == Provider::Ollama);
        let fragment = text
            .get(before..)
            .ok_or_else(|| invalid("answer text was revised"))?;
        if !fragment.is_empty() {
            events.push(LlmEvent::TextDelta {
                text: fragment.into(),
            });
        }
    }
    reasoning(delta, message)?;
    if let Some(calls) = delta.get("tool_calls").filter(|v| !v.is_null()) {
        let calls = calls
            .as_array()
            .ok_or_else(|| invalid("invalid tool call deltas"))?;
        if terminal && !calls.is_empty() {
            return Err(invalid("tool delta after choice termination"));
        }
        for call in calls {
            events.extend(call_delta(call, message)?);
        }
    }
    Ok(events)
}

/// Retains private reasoning independently from visible answer text.
fn reasoning(delta: &Value, message: &mut Value) -> Result<(), RathError> {
    for key in ["reasoning", "reasoning_content"] {
        if let Some(text) = delta.get(key).filter(|v| !v.is_null()) {
            let text = text
                .as_str()
                .ok_or_else(|| invalid("invalid reasoning delta"))?;
            append(&mut message["reasoning"], text)?;
        }
    }
    Ok(())
}

/// Defers incomplete leading thinking tags and preserves the existing Ollama answer projection.
fn visible_text(content: &Value, ollama: bool) -> &str {
    let text = content.as_str().unwrap_or_default();
    if !ollama {
        return text;
    }
    let trimmed = text.trim_start();
    if "<think>".starts_with(trimmed) {
        return "";
    }
    if trimmed.starts_with("<think>") {
        return text
            .split_once("</think>")
            .map(|(_, answer)| answer.trim_start())
            .unwrap_or_default();
    }
    text
}

/// Appends one call's fragments using its stable native index and immutable known ID.
fn call_delta(value: &Value, message: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let index = index(value, "index")?;
    if message["tool_calls"].is_null() {
        message["tool_calls"] = json!([]);
    }
    let calls = message["tool_calls"]
        .as_array_mut()
        .ok_or_else(|| invalid("invalid tool assembly"))?;
    if index > calls.len() {
        return Err(invalid("unknown sparse tool index"));
    }
    if index == calls.len() {
        calls.push(json!({"type":"function", "function":{"name":"", "arguments":""}}));
    }
    let call = &mut calls[index];
    let name = string(&call["function"], "name")?;
    let before = (
        name.len(),
        string(&call["function"], "arguments")?.len(),
        visible_name(name),
    );
    let new_id = call["id"].is_null() && value.get("id").is_some_and(|id| !id.is_null());
    update_call_identity(value, call)?;
    append_call_fragments(value, call)?;
    let name = string(&call["function"], "name")?;
    let event = tool_delta(
        index,
        call["id"].as_str(),
        name,
        string(&call["function"], "arguments")?,
        before,
        false,
    )
    .or_else(|| {
        (new_id && visible_name(name)).then(|| LlmEvent::ToolCallDelta {
            index,
            id: call["id"].as_str().map(str::to_owned),
            name_delta: None,
            arguments_delta: String::new(),
        })
    });
    Ok(event.into_iter().collect())
}

/// Appends only typed function name and argument string fragments.
fn append_call_fragments(value: &Value, call: &mut Value) -> Result<(), RathError> {
    if let Some(function) = value.get("function") {
        if !function.is_object() {
            return Err(invalid("invalid function delta"));
        }
        for key in ["name", "arguments"] {
            if let Some(fragment) = function.get(key) {
                append(
                    &mut call["function"][key],
                    fragment
                        .as_str()
                        .ok_or_else(|| invalid("invalid tool string fragment"))?,
                )?;
            }
        }
    }
    Ok(())
}

/// Validates a complete known ID and the function-call kind before fragment assembly.
fn update_call_identity(value: &Value, call: &mut Value) -> Result<(), RathError> {
    if let Some(id) = value.get("id").filter(|v| !v.is_null()) {
        if !id.is_string() || (!call["id"].is_null() && call["id"] != *id) {
            return Err(invalid("tool call ID changed or is invalid"));
        }
        call["id"] = id.clone();
    }
    if let Some(kind) = value.get("type")
        && kind != "function"
    {
        return Err(invalid("unsupported tool type"));
    }
    Ok(())
}

/// Flushes an undecided literal prefix while rejecting an unclosed leading Ollama thought block.
pub(in crate::providers) fn final_ollama_text(
    assembled: &Value,
) -> Result<Vec<LlmEvent>, RathError> {
    let content = &assembled["choices"][0]["message"]["content"];
    let text = content.as_str().unwrap_or_default();
    if text.trim_start().starts_with("<think>") && !text.contains("</think>") {
        return Err(invalid("incomplete leading thinking block"));
    }
    let held = !text.is_empty() && "<think>".starts_with(text.trim_start());
    Ok(held
        .then(|| LlmEvent::TextDelta { text: text.into() })
        .into_iter()
        .collect())
}

/// Flushes only an ambiguous ordinary name at termination; private final-tool progress stays hidden.
pub(in crate::providers) fn final_tools(assembled: &Value) -> Result<Vec<LlmEvent>, RathError> {
    let mut events = Vec::new();
    for (index, call) in assembled["choices"][0]["message"]["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let name = string(&call["function"], "name")?;
        let args = string(&call["function"], "arguments")?;
        events.extend(tool_delta(
            index,
            call["id"].as_str(),
            name,
            args,
            (name.len(), args.len(), visible_name(name)),
            true,
        ));
    }
    Ok(events)
}

#[cfg(test)]
#[path = "tests/compatible.rs"]
mod tests;
