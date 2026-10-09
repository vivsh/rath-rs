//! Private framing and operation-local response streaming for direct HTTP adapters.

use crate::core::error::http;
use crate::llm::{LlmEvent, LlmOptions, LlmStream, ModelUrl};
use crate::{ErrorBody, ErrorKind, RathError};
use reqwest::{RequestBuilder, Response};
use serde_json::Value;

type Frame = (Option<String>, String);
type Decoder = fn(&str, &str, &mut Value, &LlmOptions) -> Result<Vec<LlmEvent>, RathError>;
pub(super) mod compatible;

/// Rejects native transport overrides before one model-bound streaming request.
pub(super) fn validate(options: &LlmOptions) -> Result<(), RathError> {
    if let Some(config) = &options.provider_config {
        for key in ["stream", "stream_options"] {
            if config.get(key).is_some() {
                return Err(RathError::new(
                    ErrorKind::Validation,
                    format!("provider_config.{key} is reserved; use execute_stream"),
                ));
            }
        }
    }
    Ok(())
}

/// Sends once and retains the established response for pull-driven decoding.
pub(super) async fn open<'a>(
    request: RequestBuilder,
    url: &'a ModelUrl,
    options: &'a LlmOptions,
    credential: Option<&'a str>,
    initial: Value,
    decode: Decoder,
) -> Result<LlmStream<'a>, RathError> {
    let secrets: Vec<_> = credential.into_iter().collect();
    let response = http::send(
        request,
        url.provider().clone(),
        "streaming generation",
        &secrets,
    )
    .await?;
    let response = validate_response(response, url, &secrets).await?;
    Ok(run(response, url, options, credential, initial, decode))
}

/// Checks startup HTTP status and MIME type without consuming a successful SSE response.
async fn validate_response(
    response: Response,
    url: &ModelUrl,
    secrets: &[&str],
) -> Result<Response, RathError> {
    if !response.status().is_success() {
        http::read(
            response,
            url.provider().clone(),
            "streaming generation",
            secrets,
        )
        .await?;
        return Err(invalid("unsuccessful streaming response"));
    }
    let mime = response
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok());
    if !mime.is_some_and(|mime| {
        mime.split(';')
            .next()
            .is_some_and(|m| m.trim().eq_ignore_ascii_case("text/event-stream"))
    }) {
        return http::mapped_response(
            response,
            url.provider().clone(),
            "streaming generation",
            secrets,
            |body| {
                if body.get("error").is_some_and(|e| !e.is_null()) {
                    Err(crate::core::error::provider_failure(
                        url.provider().clone(),
                        "streaming generation",
                        &body,
                        secrets,
                    ))
                } else {
                    Err(invalid("expected text/event-stream response"))
                }
            },
        )
        .await;
    }
    Ok(response)
}

/// Owns one HTTP operation's wire evidence, framing buffers and native output envelope.
fn run<'a>(
    mut response: Response,
    url: &'a ModelUrl,
    options: &'a LlmOptions,
    credential: Option<&'a str>,
    mut assembled: Value,
    decode: Decoder,
) -> LlmStream<'a> {
    Box::pin(async_stream::try_stream! {
        let meta = http::metadata(&response);
        let secrets: Vec<_> = credential.into_iter().collect();
        let mut wire = Vec::new();
        let mut pending = Vec::new();
        let mut name = None;
        let mut data = String::new();
        loop {
            let frame = next_frame(&mut response, &mut wire, &mut pending, &mut name, &mut data).await;
            let (frame, complete) = match frame {
                Ok((Some(frame), complete)) => (Ok(frame), complete),
                Ok((None, _)) => (Err(invalid("stream ended without completion evidence")), true),
                Err((error, complete)) => (Err(error), complete),
            };
            let frame = frame.map_err(|error| {
                failure(error, url, &secrets, meta.clone(), std::mem::take(&mut wire), complete)
            })?;
            let events = decode(frame.0.as_deref().unwrap_or(""), &frame.1, &mut assembled, options)
                .map_err(|error| failure(error, url, &secrets, meta.clone(), std::mem::take(&mut wire), complete))?;
            for event in events {
                let done = matches!(event, LlmEvent::Completed { .. });
                yield event;
                if done { return; }
            }
        }
    })
}

/// Enriches an error with the original complete body or the received undrained prefix.
fn failure(
    error: RathError,
    url: &ModelUrl,
    secrets: &[&str],
    meta: (
        u16,
        Option<reqwest::header::HeaderValue>,
        Option<reqwest::header::HeaderValue>,
    ),
    wire: Vec<u8>,
    complete: bool,
) -> RathError {
    let body = if complete {
        ErrorBody::Complete(wire)
    } else {
        ErrorBody::Incomplete(wire)
    };
    http::enrich(
        error.with_context(url.provider().clone(), "streaming generation"),
        meta,
        body,
        secrets,
    )
}

/// Reads exactly enough for one SSE frame, retaining split lines and original bytes.
async fn next_frame(
    response: &mut Response,
    wire: &mut Vec<u8>,
    pending: &mut Vec<u8>,
    name: &mut Option<String>,
    data: &mut String,
) -> Result<(Option<Frame>, bool), (RathError, bool)> {
    loop {
        let first = wire.len() == pending.len();
        if let Some(line) = take_line(pending, false, first).map_err(|e| (e, false))? {
            if let Some(frame) = feed_line(&line, name, data) {
                return Ok((Some(frame), false));
            }
            continue;
        }
        match response.chunk().await {
            Ok(Some(chunk)) => {
                wire.extend_from_slice(&chunk);
                pending.extend_from_slice(&chunk);
            }
            Ok(None) => {
                while let Some(line) =
                    take_line(pending, true, wire.len() == pending.len()).map_err(|e| (e, true))?
                {
                    if let Some(frame) = feed_line(&line, name, data) {
                        return Ok((Some(frame), true));
                    }
                }
                if !pending.is_empty() || !data.is_empty() || name.is_some() {
                    return Err((invalid("incomplete SSE frame at end of response"), true));
                }
                return Ok((None, true));
            }
            Err(cause) => {
                let kind = if cause.is_timeout() {
                    ErrorKind::Timeout
                } else {
                    ErrorKind::Transport
                };
                return Err((RathError::from_native_error(kind, &cause), false));
            }
        }
    }
}

/// Removes one complete UTF-8 line, respecting CRLF split between network reads.
fn take_line(pending: &mut Vec<u8>, eof: bool, first: bool) -> Result<Option<String>, RathError> {
    let Some(end) = pending.iter().position(|b| matches!(b, b'\r' | b'\n')) else {
        return Ok(None);
    };
    if pending[end] == b'\r' && end + 1 == pending.len() && !eof {
        return Ok(None);
    }
    let skip = 1 + usize::from(pending[end] == b'\r' && pending.get(end + 1) == Some(&b'\n'));
    let text = std::str::from_utf8(&pending[..end])
        .map_err(|e| RathError::from_native_error(ErrorKind::Deserialize, &e))?;
    let text = if first {
        text.strip_prefix('\u{feff}').unwrap_or(text)
    } else {
        text
    };
    let line = text.to_owned();
    pending.drain(..end + skip);
    Ok(Some(line))
}

/// Processes SSE fields without reconnecting or interpreting provider payloads.
fn feed_line(line: &str, name: &mut Option<String>, data: &mut String) -> Option<Frame> {
    if line.is_empty() {
        if data.is_empty() {
            *name = None;
            return None;
        }
        data.pop();
        return Some((name.take(), std::mem::take(data)));
    }
    if line.starts_with(':') {
        return None;
    }
    let (key, value) = line.split_once(':').unwrap_or((line, ""));
    let value = value.strip_prefix(' ').unwrap_or(value);
    match key {
        "event" => *name = Some(value.into()),
        "data" => {
            data.push_str(value);
            data.push('\n');
        }
        _ => {}
    }
    None
}

/// Decodes event JSON and rejects explicit provider errors before adapter interpretation.
pub(super) fn json(data: &str, provider: crate::Provider) -> Result<Value, RathError> {
    let value: Value = serde_json::from_str(data)
        .map_err(|e| RathError::from_native_error(ErrorKind::Deserialize, &e))?;
    if value.get("error").is_some_and(|e| !e.is_null()) {
        return Err(crate::core::error::provider_failure(
            provider,
            "streaming generation",
            &value,
            &[],
        ));
    }
    Ok(value)
}

/// Reads a required nonnegative index without allocating unbounded sparse arrays.
pub(super) fn index(value: &Value, key: &str) -> Result<usize, RathError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(format!("missing or invalid {key}")))
}

/// Borrows a required string while retaining provider framing errors at the outer boundary.
pub(super) fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, RathError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("missing or invalid {key}")))
}

/// Appends a string fragment while rejecting conflicting protocol shapes.
pub(super) fn append(target: &mut Value, fragment: &str) -> Result<(), RathError> {
    if target.is_null() {
        *target = Value::String(String::new());
    }
    let Value::String(text) = target else {
        return Err(invalid("expected string output assembly"));
    };
    text.push_str(fragment);
    Ok(())
}

/// Creates a provisional tool event using event-local previous lengths, not retained watermarks.
pub(super) fn tool_delta(
    index: usize,
    id: Option<&str>,
    name: &str,
    args: &str,
    before: (usize, usize, bool),
    final_name: bool,
) -> Option<LlmEvent> {
    if name == "__rath_final_output" || (!visible_name(name) && !final_name) {
        return None;
    }
    let (names, arguments) = if before.2 {
        (before.0, before.1)
    } else {
        (0, 0)
    };
    let name_delta = name
        .get(names..)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let arguments_delta = args.get(arguments..).unwrap_or_default().to_owned();
    if name_delta.is_none() && arguments_delta.is_empty() {
        return None;
    }
    Some(LlmEvent::ToolCallDelta {
        index,
        id: id.map(str::to_owned),
        name_delta,
        arguments_delta,
    })
}

/// Identifies names whose progress cannot be confused with the private final-output tool.
pub(super) fn visible_name(name: &str) -> bool {
    !name.is_empty() && !"__rath_final_output".starts_with(name)
}

/// Creates an output-shape failure without formatting private payloads.
pub(super) fn invalid(message: impl Into<String>) -> RathError {
    RathError::new(ErrorKind::InvalidResponse, message)
}

#[cfg(test)]
#[path = "tests/streaming.rs"]
mod tests;
