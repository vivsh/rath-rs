use super::*;
use crate::llm::counting;
use gemini_rust::ContentBuilder;

/// Builds the same provider request for generation, estimation, and native counting.
pub(super) fn build_request(
    client: &Gemini,
    model: &str,
    options: &LlmOptions,
    messages: &[Message],
    minimal: bool,
) -> Result<ContentBuilder, RathError> {
    validate_request(options, messages)?;
    let mut builder = generation_options(client.generate_content(), model, options, minimal);
    let mut system = options.effective_preamble().into_iter().collect::<Vec<_>>();
    system.extend(
        messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.content.clone()),
    );
    if !system.is_empty() {
        builder = builder.with_system_prompt(system.join("\n\n"));
    }
    if let Some(settings) = gemini_safety_settings_from_provider_config(&options.provider_config)? {
        builder = builder.with_safety_settings(settings);
    }
    builder = builder.with_messages(build_gemini_messages(messages));
    if options.tool_choice != ToolChoice::Disabled
        && let Some(tools) = build_tools_spec(&options.tools)?
    {
        let mode = if options.tool_choice == ToolChoice::Required {
            FunctionCallingMode::Any
        } else {
            FunctionCallingMode::Auto
        };
        builder = builder.with_tool(tools).with_function_calling_mode(mode);
    }
    if wants_json_output(options) {
        builder = builder.with_response_mime_type("application/json");
        if let Some(schema) = response_schema(options) {
            builder = builder.with_response_schema(schema);
        }
    }
    Ok(builder)
}

/// Retains generation defaults while omitting thinking entirely for standalone content.
fn generation_options(
    mut builder: ContentBuilder,
    model: &str,
    options: &LlmOptions,
    minimal: bool,
) -> ContentBuilder {
    if !minimal
        && model
            .strip_prefix("models/")
            .unwrap_or(model)
            .starts_with("gemma-4-")
    {
        let level = match options.thinking {
            None | Some(ThinkingLevel::Off) => gemini_rust::ThinkingLevel::Minimal,
            Some(_) => gemini_rust::ThinkingLevel::High,
        };
        builder = builder.with_thinking_level(level);
    } else if !minimal {
        let budget = match options.thinking {
            None | Some(ThinkingLevel::Off) => 0,
            Some(ThinkingLevel::Low) => 512,
            Some(ThinkingLevel::Medium) => 4096,
            Some(ThinkingLevel::High) => 16384,
            Some(ThinkingLevel::XHigh) => i32::MAX,
        };
        builder = builder.with_thinking_budget(budget);
    }
    if let Some(t) = options.temperature {
        builder = builder.with_temperature(t);
    }
    if let Some(cap) = options.max_output_tokens {
        builder = builder.with_max_output_tokens(cap as i32);
    }
    builder
}

/// Validates the same options and message boundary before all Gemini request paths.
fn validate_request(options: &LlmOptions, messages: &[Message]) -> Result<(), RathError> {
    counting::validate_options(Provider::Gemini, options)?;
    validate_tools(Provider::Gemini, &options.tools)?;
    if messages.is_empty()
        || matches!(
            messages.last().map(|m| &m.role),
            Some(Role::AssistantToolCalls { .. })
        )
    {
        return Err(RathError::Validation(
            "messages must be nonempty and end without unresolved tool calls".into(),
        ));
    }
    Ok(())
}
