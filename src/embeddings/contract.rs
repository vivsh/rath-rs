//! Stateless embedding validation shared by HTTP and local backends.
use super::{EmbedRequest, EmbedResponse, EmbedTaskType, EmbeddingOptions};
use crate::{ErrorKind, Provider, RathError, TokenUsage};
use serde_json::{Map, Value};

pub(crate) fn invalid(message: impl Into<String>) -> RathError {
    RathError::new(ErrorKind::Validation, message)
}

/// Validates input shape without changing user text or making provider calls.
pub(crate) fn validate(request: &EmbedRequest) -> Result<(), RathError> {
    if request.input.is_empty() || request.input.iter().any(String::is_empty) {
        return Err(invalid(
            "embeddings require a nonempty batch of nonempty strings",
        ));
    }
    if request.output_dimensionality == Some(0) {
        return Err(invalid("output_dimensionality must be positive"));
    }
    if request.title.is_some() && request.task_type != Some(EmbedTaskType::RetrievalDocument) {
        return Err(invalid("title requires RetrievalDocument"));
    }
    if request
        .instructions
        .as_ref()
        .is_some_and(|s| s.trim().is_empty())
    {
        return Err(invalid("instructions must not be blank"));
    }
    if request.instructions.is_some() && request.task_type != Some(EmbedTaskType::RetrievalQuery) {
        return Err(invalid("instructions require RetrievalQuery"));
    }
    Ok(())
}

/// Resolves native defaults with explicit request overrides, rejecting unhandled fields.
pub(crate) fn config(
    defaults: Option<&Value>,
    request: Option<&Value>,
    allowed: &[&str],
) -> Result<Map<String, Value>, RathError> {
    let mut merged = Map::new();
    for value in [defaults, request].into_iter().flatten() {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("provider_config must be an object"))?;
        for (key, value) in object {
            if !allowed.contains(&key.as_str()) {
                return Err(invalid(format!(
                    "unsupported embedding provider_config.{key}; use typed controls where available"
                )));
            }
            merged.insert(key.clone(), value.clone());
        }
    }
    if merged.get("user").is_some_and(|v| !v.is_string())
        || merged.get("options").is_some_and(|v| !v.is_object())
    {
        return Err(invalid(
            "embedding user must be a string and options must be an object",
        ));
    }
    Ok(merged)
}

pub(crate) fn remote_options(
    options: &EmbeddingOptions,
    allowed: &[&str],
) -> Result<(), RathError> {
    if options.fastembed.is_some() {
        return Err(invalid(
            "fastembed setup is only supported by the FastEmbed backend",
        ));
    }
    config(options.provider_config.as_ref(), None, allowed)?;
    Ok(())
}

/// Enforces supported controls; absence leaves provider defaults untouched.
pub(crate) fn controls(
    provider: Provider,
    request: &EmbedRequest,
    tasks: bool,
    truncation: bool,
) -> Result<(), RathError> {
    validate(request)?;
    if (!tasks
        && (request.task_type.is_some()
            || request.title.is_some()
            || request.instructions.is_some()))
        || (!truncation && request.truncate.is_some())
    {
        return Err(RathError::unsupported(
            provider,
            "requested embedding task, title, instructions or truncation",
        ));
    }
    Ok(())
}

/// Prepares explicit Qwen retrieval queries; arbitrary model aliases receive no inferred prefix.
pub(crate) fn ollama_input(model: &str, request: &EmbedRequest) -> Result<Vec<String>, RathError> {
    validate(request)?;
    let qwen = model.split(':').next() == Some("qwen3-embedding");
    match &request.task_type {
        None => Ok(request.input.clone()),
        Some(EmbedTaskType::RetrievalDocument) if qwen && request.title.is_none() => {
            Ok(request.input.clone())
        }
        Some(EmbedTaskType::RetrievalQuery) if qwen => {
            let instruction = request
                .instructions
                .as_ref()
                .ok_or_else(|| invalid("Qwen retrieval queries require explicit instructions"))?;
            Ok(request
                .input
                .iter()
                .map(|text| format!("Instruct: {instruction}\nQuery: {text}"))
                .collect())
        }
        _ => Err(RathError::unsupported(
            Provider::Ollama,
            "embedding task/title for this model",
        )),
    }
}

/// Validates vectors before returning an all-or-nothing batch, retaining native evidence.
pub(crate) fn response(
    provider: Provider,
    raw: Value,
    vectors: Vec<Vec<f32>>,
    request: &EmbedRequest,
    usage: Option<TokenUsage>,
) -> Result<EmbedResponse, RathError> {
    let dimension = vectors.first().map_or(0, Vec::len);
    if vectors.len() != request.input.len()
        || dimension == 0
        || vectors
            .iter()
            .any(|v| v.len() != dimension || v.iter().any(|x| !x.is_finite()))
        || request
            .output_dimensionality
            .is_some_and(|d| d as usize != dimension)
    {
        return Err(RathError::invalid(
            "invalid embedding vector count, dimensions or nonfinite value",
            &raw,
        ));
    }
    let mut result = EmbedResponse::new(provider, vectors);
    result.provider_model = raw.get("model").and_then(Value::as_str).map(str::to_owned);
    result.usage = usage;
    result.raw_metadata = Some(raw);
    Ok(result)
}

/// Parses a numeric vector without losing response evidence on type errors.
pub(crate) fn vector(value: &Value, raw: &Value) -> Result<Vec<f32>, RathError> {
    value
        .as_array()
        .ok_or_else(|| RathError::invalid("missing embedding vector", raw))?
        .iter()
        .map(|n| {
            n.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| RathError::invalid("embedding contains a non-number", raw))
        })
        .collect()
}
