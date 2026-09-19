use super::*;
use crate::core::ErrorKind;

/// Validates offered tools and rejects an impossible required-tool request before dispatch.
pub(super) fn validate_tool_request(options: &LlmOptions) -> Result<(), RathError> {
    validate_tools(Provider::Ollama, &options.tools)?;
    if options.tool_choice == ToolChoice::Required && options.tools.is_empty() {
        return Err(RathError::new(
            ErrorKind::Validation,
            "Ollama tool_choice Required needs at least one tool",
        ));
    }
    Ok(())
}

/// Enforces tool policy before a response can become executable calls or an exit-tool result.
pub(super) fn validate_call_policy(
    calls: &[ToolCall],
    options: &LlmOptions,
    finish: Option<&Value>,
) -> Result<(), RathError> {
    if calls.is_empty() {
        if options.tool_choice == ToolChoice::Required {
            return Err(invalid(
                "Ollama returned no tool calls although tool_choice is Required",
            ));
        }
        if finish.and_then(Value::as_str) == Some("tool_calls") {
            return Err(invalid(
                "Ollama finish_reason is tool_calls but no tool calls were returned",
            ));
        }
        return Ok(());
    }
    if options.tool_choice == ToolChoice::Disabled || options.tools.is_empty() {
        return Err(invalid(
            "Ollama returned tool calls when tools are disabled or unavailable",
        ));
    }
    for call in calls {
        if !options.tools.iter().any(|tool| tool.name == call.name) {
            return Err(invalid("Ollama returned a tool name that was not offered"));
        }
    }
    Ok(())
}

/// Validates the protocol field; text fallback is allowed only for enabled tools and call-like text.
pub(super) fn collect_tool_calls(
    message: &Value,
    allow_text: bool,
) -> Result<Vec<ToolCall>, RathError> {
    if let Some(value) = message.get("tool_calls").filter(|v| !v.is_null()) {
        let items = value
            .as_array()
            .ok_or_else(|| invalid("Ollama message.tool_calls must be an array"))?;
        return parse_json_tool_calls(items);
    }
    if allow_text && let Some(content) = message.get("content").and_then(Value::as_str) {
        let content = strip_thinking(content).trim_start();
        if content.starts_with("<function=") {
            return parse_content_tool_calls(content);
        }
    }
    Ok(Vec::new())
}

/// Parses compatible API tool calls without accepting empty identifiers or non-object arguments.
fn parse_json_tool_calls(items: &[Value]) -> Result<Vec<ToolCall>, RathError> {
    let mut calls: Vec<ToolCall> = Vec::with_capacity(items.len());
    for item in items {
        let id = required_string(item, "/id", "tool call id")?;
        let name = required_string(item, "/function/name", "tool call function.name")?;
        let raw = required_string(item, "/function/arguments", "tool call function.arguments")?;
        let args: Value = serde_json::from_str(raw).map_err(|cause| {
            RathError::new(
                ErrorKind::Deserialize,
                "Ollama tool call arguments are invalid JSON",
            )
            .with_source(RathError::deserialize(&cause, raw))
        })?;
        if !args.is_object() {
            return Err(invalid("Ollama tool call arguments must be a JSON object"));
        }
        if calls.iter().any(|call| call.id == id) {
            return Err(invalid("Ollama returned duplicate tool call ids"));
        }
        calls.push(ToolCall {
            id: id.into(),
            name: name.into(),
            args,
            thought_signatures: None,
        });
    }
    Ok(calls)
}

fn required_string<'a>(value: &'a Value, path: &str, field: &str) -> Result<&'a str, RathError> {
    value
        .pointer(path)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid(format!("Ollama {field} must be a nonempty string")))
}

/// Accepts complete legacy function blocks only; incomplete blocks never become executable calls.
fn parse_content_tool_calls(content: &str) -> Result<Vec<ToolCall>, RathError> {
    let mut calls = Vec::new();
    let mut remaining = content;
    while let Some(tag_start) = remaining.find("<function=") {
        let after_tag = &remaining[tag_start + "<function=".len()..];
        let (name, body) = after_tag
            .split_once('>')
            .ok_or_else(|| invalid("Ollama tool call is missing >"))?;
        if name.trim().is_empty() {
            return Err(invalid("Ollama tool call function name is empty"));
        }
        let (body, tail) = body
            .split_once("</function>")
            .ok_or_else(|| invalid("Ollama tool call is missing </function>"))?;
        let args = parse_function_params(body)?;
        calls.push(ToolCall {
            id: uuid::Uuid::now_v7().to_string(),
            name: name.trim().into(),
            args,
            thought_signatures: None,
        });
        remaining = tail;
    }
    Ok(calls)
}

/// Decodes complete parameter blocks while rejecting missing delimiters, duplicates and stray data.
fn parse_function_params(text: &str) -> Result<Value, RathError> {
    let mut map = serde_json::Map::new();
    let mut remaining = text.trim();
    while !remaining.is_empty() {
        let after_tag = remaining
            .strip_prefix("<parameter=")
            .ok_or_else(|| invalid("Ollama tool call contains invalid parameter content"))?;
        let (key, value) = after_tag
            .split_once('>')
            .ok_or_else(|| invalid("Ollama tool parameter is missing >"))?;
        let key = key.trim();
        if key.is_empty() || map.contains_key(key) {
            return Err(invalid("Ollama tool parameter name is empty or duplicated"));
        }
        let (raw, tail) = value
            .split_once("</parameter>")
            .ok_or_else(|| invalid("Ollama tool parameter is missing </parameter>"))?;
        if raw.contains("<parameter=") || raw.contains("<function=") {
            return Err(invalid(
                "Ollama tool parameter contains an unclosed or nested tool block",
            ));
        }
        let raw = raw.trim();
        map.insert(
            key.into(),
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into())),
        );
        remaining = tail.trim();
    }
    Ok(Value::Object(map))
}

fn invalid(message: impl Into<String>) -> RathError {
    RathError::new(ErrorKind::InvalidResponse, message)
}

#[cfg(test)]
#[path = "tests/tool_calls.rs"]
mod tests;
