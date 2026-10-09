//! Pull-driven generation through the installed SDK, preserving full function-call parts.

use super::*;
use crate::llm::{LlmEvent, LlmStream};
use crate::providers::streaming as wire;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;

/// Starts the same generateContent request using the SDK's streaming dispatch.
pub(super) async fn start<'a>(
    client: &'a GeminiClient,
    messages: &[Message],
) -> Result<LlmStream<'a>, RathError> {
    wire::validate(&client.options)?;
    let stream = request::build_request(
        &client.client,
        &client.url.model,
        &client.options,
        messages,
        false,
    )?
    .execute_stream()
    .await
    .map_err(|e| errors::normalize(&e, "streaming generation", client.url.api_key.as_deref()))?;
    Ok(run(stream, &client.options, client.url.api_key.as_deref()))
}

/// Owns SDK consumption and one native assembly until final validation or drop.
fn run<'a>(
    mut stream: gemini_rust::client::GenerationStream,
    options: &'a LlmOptions,
    credential: Option<&'a str>,
) -> LlmStream<'a> {
    Box::pin(async_stream::try_stream! {
        let mut assembled = json!({"candidates":[{"content":{"role":"model", "parts":[]}}]});
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| errors::normalize(&e, "streaming generation", credential).with_response(&assembled))?;
            let chunk = serde_json::to_value(chunk).map_err(|e| {
                errors::context(RathError::from_native_error(crate::ErrorKind::Serialize, &e), "streaming generation", credential)
            })?;
            let events = merge(chunk, &mut assembled).map_err(|e| {
                errors::context(e.with_response(&assembled), "streaming generation", credential)
            })?;
            for event in events { yield event; }
        }
        let response = complete(&assembled, options).map_err(|e| {
            errors::context(e.with_response(&assembled), "streaming generation", credential)
        })?;
        yield LlmEvent::Completed { response };
    })
}

/// Merges one typed SDK chunk while rejecting alternate candidates and output after termination.
fn merge(mut chunk: Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    crate::llm::counting::check_output_limit(Provider::Gemini, &chunk)?;
    if chunk
        .pointer("/promptFeedback/blockReason")
        .is_some_and(|v| !v.is_null() && v != "BLOCK_REASON_UNSPECIFIED")
    {
        return Err(RathError::new(
            crate::ErrorKind::Provider,
            "generation was blocked by prompt feedback",
        ));
    }
    merge_metadata(&mut chunk, assembled)?;
    let mut candidates = chunk["candidates"].take();
    if candidates.is_null() {
        return Ok(vec![]);
    }
    let candidates = candidates
        .as_array_mut()
        .ok_or_else(|| wire::invalid("invalid streamed candidates"))?;
    if candidates.is_empty() {
        return Ok(vec![]);
    }
    if candidates.len() != 1 {
        return Err(wire::invalid(
            "multiple streamed candidates are unsupported",
        ));
    }
    let mut candidate = candidates.remove(0);
    if candidate
        .get("index")
        .is_some_and(|v| !v.is_null() && v != 0)
    {
        return Err(wire::invalid("alternate streamed candidate is unsupported"));
    }
    let parts = candidate["content"]["parts"].take();
    let mut events = Vec::new();
    if let Some(parts) = parts.as_array() {
        if !parts.is_empty() && !assembled["candidates"][0]["finishReason"].is_null() {
            return Err(wire::invalid("candidate output after termination"));
        }
        for part in parts {
            events.extend(merge_part(part, assembled)?);
        }
    } else if !parts.is_null() {
        return Err(wire::invalid("invalid candidate parts"));
    }
    merge_candidate_metadata(&candidate, &mut assembled["candidates"][0])?;
    Ok(events)
}

/// Retains reported identity and cumulative usage updates without fabricating counters.
fn merge_metadata(chunk: &mut Value, assembled: &mut Value) -> Result<(), RathError> {
    for key in ["modelVersion", "responseId"] {
        if let Some(value) = chunk.get(key).filter(|v| !v.is_null()) {
            if !assembled[key].is_null() && assembled[key] != *value {
                return Err(wire::invalid("generation identity changed"));
            }
            assembled[key] = value.clone();
        }
    }
    if let Some(usage) = chunk.get("usageMetadata").filter(|v| !v.is_null()) {
        let usage = usage
            .as_object()
            .ok_or_else(|| wire::invalid("invalid usage metadata"))?;
        if assembled["usageMetadata"].is_null() {
            assembled["usageMetadata"] = json!({});
        }
        for (key, value) in usage {
            if key.ends_with("TokenCount") && value.as_i64().is_some_and(|n| n < 0) {
                return Err(wire::invalid("negative reported token count"));
            }
            assembled["usageMetadata"][key] = value.clone();
        }
    }
    if let Some(feedback) = chunk.get("promptFeedback") {
        assembled["promptFeedback"] = feedback.clone();
    }
    Ok(())
}

/// Keeps candidate terminal facts in the native envelope, independently of content assembly.
fn merge_candidate_metadata(candidate: &Value, assembled: &mut Value) -> Result<(), RathError> {
    let fields = candidate
        .as_object()
        .ok_or_else(|| wire::invalid("invalid streamed candidate"))?;
    for (key, value) in fields {
        if key == "content" || value.is_null() {
            continue;
        }
        if key == "finishReason" && !assembled[key].is_null() && assembled[key] != *value {
            return Err(wire::invalid("candidate termination changed"));
        }
        assembled[key] = value.clone();
    }
    Ok(())
}

/// Emits visible text or a full SDK function-call part while preserving continuation data.
fn merge_part(part: &Value, assembled: &mut Value) -> Result<Vec<LlmEvent>, RathError> {
    let parts = assembled["candidates"][0]["content"]["parts"]
        .as_array_mut()
        .ok_or_else(|| wire::invalid("invalid candidate assembly"))?;
    let event = if part.get("text").is_some() {
        let text = wire::string(part, "text")?;
        if part.get("thought").is_some_and(|v| v == true) || text.is_empty() {
            None
        } else {
            Some(LlmEvent::TextDelta { text: text.into() })
        }
    } else if let Some(call) = part.get("functionCall") {
        let index = parts
            .iter()
            .filter(|p| p.get("functionCall").is_some())
            .count();
        let name = wire::string(call, "name")?;
        let args = call
            .get("args")
            .filter(|v| v.is_object())
            .ok_or_else(|| wire::invalid("function call args must be an object"))?;
        let id = format!("{name}_{index}");
        wire::tool_delta(
            index,
            Some(&id),
            name,
            &args.to_string(),
            (0, 0, false),
            true,
        )
    } else {
        return Err(wire::invalid("unsupported generated candidate part"));
    };
    parts.push(part.clone());
    Ok(event.into_iter().collect())
}

/// Requires native stop evidence before reusing complete-response parsing and exit-tool handling.
fn complete(assembled: &Value, options: &LlmOptions) -> Result<LlmResponse, RathError> {
    crate::llm::counting::check_output_limit(Provider::Gemini, assembled)?;
    let reason = assembled
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
        .ok_or_else(|| wire::invalid("stream ended without candidate termination"))?;
    if reason != "STOP" {
        return Err(RathError::new(
            crate::ErrorKind::Provider,
            "candidate did not complete successfully",
        ));
    }
    let native = GenerationResponse::deserialize(assembled)
        .map_err(|e| RathError::from_native_error(crate::ErrorKind::Deserialize, &e))?;
    let response = map_response(native, options.wants_json_output())?;
    crate::llm::output::finish(options, response)
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
