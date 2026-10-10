use super::{LlmOptions, LlmOutput, LlmResponse, ResponseFormat, ToolChoice, ToolDefinition};
use crate::{ErrorKind, Provider, RathError};
use serde_json::{Value, json};

const FINAL_TOOL: &str = "__rath_final_output";

/// Derives adapter workarounds without changing the client's immutable options.
pub(crate) fn prepare(
    provider: Provider,
    model: &str,
    options: &LlmOptions,
) -> Result<LlmOptions, RathError> {
    if options.tools.iter().any(|tool| tool.name == FINAL_TOOL) {
        return Err(RathError::new(
            ErrorKind::Validation,
            "tool name __rath_final_output is reserved",
        ));
    }
    let legacy = provider == Provider::Ollama
        || (provider == Provider::Gemini && crate::core::url::gemini_needs_exit_tool(model));
    let mut prepared = options.clone();
    if !legacy || options.tools.is_empty() || options.tool_choice == ToolChoice::Disabled {
        return Ok(prepared);
    }
    let Some(schema) = options.output_schema() else {
        if options.wants_json_output() {
            return Err(RathError::unsupported(
                provider,
                "JSON with enabled tools requires a JSON schema on this model",
            ));
        }
        return Ok(prepared);
    };
    prepared.tools.push(ToolDefinition {
        name: FINAL_TOOL.into(),
        description: "Submit your final answer.".into(),
        parameters: final_tool_schema(schema),
    });
    prepared.tool_choice = ToolChoice::Required;
    prepared.response_format = ResponseFormat::Text;
    Ok(prepared)
}

/// Converts only an exclusive final-output call and preserves diagnostic evidence on conflicts.
pub(crate) fn finish(
    options: &LlmOptions,
    mut response: LlmResponse,
) -> Result<LlmResponse, RathError> {
    let Some(schema) = options.output_schema() else {
        return Ok(response);
    };
    if let LlmOutput::ToolCalls { calls, .. } = &response.output
        && let Some(call) = calls.iter().find(|call| call.name == FINAL_TOOL)
    {
        if calls.len() != 1 {
            let error = RathError::new(
                ErrorKind::InvalidResponse,
                "final output cannot be mixed with ordinary tool calls",
            );
            return Err(match &response.raw_metadata {
                Some(raw) => error.with_response(raw),
                None => error,
            });
        }
        let output =
            final_value(schema, &call.args).map_err(|error| match &response.raw_metadata {
                Some(raw) => error.with_response(raw),
                None => error,
            })?;
        response.output = LlmOutput::Output(output);
    }
    Ok(response)
}

/// Function arguments must be objects; other answer types use a private value envelope.
/// Definitions remain at the root so local references retain their original targets.
fn final_tool_schema(schema: &Value) -> Value {
    if is_object_schema(schema) {
        return schema.clone();
    }
    let mut value_schema = schema.clone();
    let mut envelope = json!({
        "type":"object", "properties":{}, "required":["value"],
        "additionalProperties":false
    });
    if let Some(fields) = value_schema.as_object_mut() {
        for key in ["$defs", "definitions"] {
            if let Some(definitions) = fields.remove(key) {
                envelope[key] = definitions;
            }
        }
    }
    envelope["properties"]["value"] = value_schema;
    envelope
}

/// Only an explicitly object-only root can be used as function arguments without wrapping.
fn is_object_schema(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("object")
}

/// Restores the authored answer type and rejects malformed private transport envelopes.
fn final_value(schema: &Value, args: &Value) -> Result<Value, RathError> {
    if is_object_schema(schema) {
        return Ok(args.clone());
    }
    args.as_object()
        .filter(|fields| fields.len() == 1)
        .and_then(|fields| fields.get("value"))
        .cloned()
        .ok_or_else(|| {
            RathError::new(
                ErrorKind::InvalidResponse,
                "final output requires an object containing only the value field",
            )
        })
}

#[cfg(test)]
#[path = "tests/output.rs"]
mod tests;
