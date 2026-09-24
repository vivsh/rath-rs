use super::{LlmOptions, LlmOutput, LlmResponse, ResponseFormat, ToolChoice, ToolDefinition};
use crate::{ErrorKind, Provider, RathError};

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
        parameters: schema.clone(),
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
    if options.output_schema().is_none() {
        return Ok(response);
    }
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
        response.output = LlmOutput::Output(call.args.clone());
    }
    Ok(response)
}

#[cfg(test)]
#[path = "tests/output.rs"]
mod tests;
