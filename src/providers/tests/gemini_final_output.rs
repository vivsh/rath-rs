use super::*;
use crate::llm::LlmEvent;
use futures_util::StreamExt;
use serde_json::json;

/// Routes the legacy Gemini model through a bounded local SDK transport fixture.
fn client(base: &str) -> GeminiClient {
    GeminiClient {
        client: gemini_rust::GeminiBuilder::new("test")
            .with_model(GeminiModel::Custom("models/gemini-3-flash-preview".into()))
            .with_base_url(base.parse().unwrap())
            .build()
            .unwrap(),
        options: LlmOptions::default()
            .with_output_schema(json!({"type":"string"}))
            .with_tools(vec![ToolDefinition::new(
                "lookup".into(),
                "Synthetic lookup".into(),
                json!({"type":"object"}),
            )]),
        url: ModelUrl::parse("gemini:///gemini-3-flash-preview").unwrap(),
    }
}

/// Checks the actual SDK wire schema, not a parallel reconstruction of the request.
fn assert_envelope(payload: &Value) {
    let declaration = &payload["tools"][0]["function_declarations"][1];
    assert_eq!(declaration["name"], "__rath_final_output");
    assert!(declaration.get("parameters").is_none());
    assert_eq!(
        declaration["parametersJsonSchema"],
        json!({
            "type":"object","properties":{"value":{"type":"string"}},"required":["value"]
        })
    );
    assert_eq!(
        payload["toolConfig"]["function_calling_config"]["mode"],
        "ANY"
    );
}

/// Generation and SSE both send object parameters and return a caller-visible string.
#[tokio::test]
async fn scalar_final_output_survives_generation_and_streaming_wire() -> Result<(), RathError> {
    for streaming in [false, true] {
        let response = json!({"candidates":[{"finishReason":"STOP", "content":{
        "role":"model", "parts":[{"functionCall":{
            "name":"__rath_final_output","args":{"value":"OK"}
        }}]}}]});
        let (mime, body) = if streaming {
            ("text/event-stream", format!("data: {response}\n\n"))
        } else {
            ("application/json", response.to_string())
        };
        let (base, requests) = crate::providers::tests::http::serve_content(200, mime, body);
        let client = client(&base);
        let messages = [Message::user("Return OK")];
        if streaming {
            let mut stream = client.execute_stream(&messages).await?;
            let LlmEvent::Completed { response } = stream.next().await.unwrap()? else {
                panic!("private final-output tool must not expose progress");
            };
            assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
            assert!(stream.next().await.is_none());
        } else {
            let response = client.execute(&messages).await?;
            assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
        }
        let wire = requests.try_recv().unwrap();
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_envelope(&payload);
    }
    Ok(())
}

/// Native token counting uses the same final-output envelope as generation and estimation.
#[tokio::test]
async fn scalar_final_output_survives_native_count_wire() -> Result<(), RathError> {
    let (base, requests) = crate::providers::tests::http::serve(200, json!({"totalTokens":7}));
    let client = client(&base);
    assert_eq!(
        client
            .count_tokens(&[Message::user("Return OK")])
            .await?
            .input_tokens,
        7
    );
    assert!(
        client
            .estimate_tokens(&[Message::user("Return OK")])?
            .input_tokens
            > 0
    );
    let wire = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_envelope(&payload["generateContentRequest"]);
    Ok(())
}

/// Invalid final envelopes and mixed calls preserve response evidence across both transports.
#[tokio::test]
async fn final_output_failures_preserve_evidence_and_redact_credentials() {
    for parts in invalid_final_parts() {
        for streaming in [false, true] {
            let native = json!({"responseId":"failure-id", "candidates":[{
                "finishReason":"STOP", "content":{"role":"model", "parts":parts}
            }]});
            let (mime, body) = if streaming {
                ("text/event-stream", format!("data: {native}\n\n"))
            } else {
                ("application/json", native.to_string())
            };
            let (base, requests) = crate::providers::tests::http::serve_content(200, mime, body);
            let mut client = client(&base);
            client.url.api_key = Some("KNOWN-KEY".into());
            let messages = [Message::user("Return OK")];
            let error = final_failure(&client, &messages, streaming).await;
            let operation = if streaming {
                "streaming generation"
            } else {
                "generation"
            };
            assert_final_failure(&error, operation);
            assert_eq!(requests.try_iter().count(), 1);
        }
    }
}

/// Consumes failed generation without accepting completion or repeated stream failures.
async fn final_failure(client: &GeminiClient, messages: &[Message], streaming: bool) -> RathError {
    if !streaming {
        return client.execute(messages).await.unwrap_err();
    }
    let mut stream = client.execute_stream(messages).await.unwrap();
    while let Some(event) = stream.next().await {
        match event {
            Err(error) => {
                assert!(stream.next().await.is_none());
                return error;
            }
            Ok(LlmEvent::Completed { .. }) => panic!("invalid final output completed"),
            Ok(_) => {}
        }
    }
    panic!("invalid final output must fail");
}

/// Supplies missing-value, extra-field and mixed-call failures carrying private diagnostics.
fn invalid_final_parts() -> Vec<Value> {
    let final_part = |args| json!({"functionCall":{"name":"__rath_final_output", "args":args}});
    let marker = "PRIVATE-ARGUMENT KNOWN-KEY";
    vec![
        json!([final_part(json!({"extra":marker}))]),
        json!([final_part(json!({"value":"OK", "extra":marker}))]),
        json!([
            final_part(json!({"value":"OK"})),
            {"functionCall":{"name":"lookup", "args":{"private":marker}}}
        ]),
    ]
}

/// Explicit body access retains malformed calls while formatted diagnostics remain private.
fn assert_final_failure(error: &RathError, operation: &'static str) {
    assert_eq!(error.kind(), crate::ErrorKind::InvalidResponse);
    assert_eq!(error.provider(), Some(&Provider::Gemini));
    assert_eq!(error.operation(), Some(operation));
    let body = error.response_body().unwrap();
    assert!(body.is_complete());
    let raw: Value = serde_json::from_slice(body.bytes()).unwrap();
    assert_eq!(raw["responseId"], "failure-id");
    let parts = raw
        .pointer("/candidates/0/content/parts")
        .unwrap()
        .to_string();
    assert!(parts.contains("__rath_final_output"));
    assert!(parts.contains("PRIVATE-ARGUMENT"));
    assert!(!parts.contains("KNOWN-KEY"));
    assert!(parts.contains("[REDACTED]"));
    let formatted = format!("{error:?} {error}");
    assert!(!formatted.contains("PRIVATE-ARGUMENT"));
    assert!(!formatted.contains("KNOWN-KEY"));
}
