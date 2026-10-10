use super::super::*;
use crate::llm::LlmEvent;
use futures_util::StreamExt;
use serde_json::json;

/// Reproduces nested integer choices while covering other non-string enum values.
fn tool() -> ToolDefinition {
    ToolDefinition::new(
        "productivity".into(),
        "Read synthetic productivity".into(),
        json!({
            "type":"object", "properties":{"query":{"type":"object", "properties":{
                "status":{"$ref":"#/$defs/Status"},
                "technique":{"type":"integer", "enum":[1,2,3]},
                "ratio":{"type":"number", "enum":[1.5,2.5]},
                "active":{"type":"boolean", "enum":[true]},
                "name":{"type":"string", "enum":["example"]}
            }}}, "$defs":{"Status":{"type":"integer", "enum":[1,2,3], "minimum":1}}
        }),
    )
}

/// Creates an SDK-backed client routed only to the supplied loopback fixture.
fn client(base: &str) -> GeminiClient {
    GeminiClient {
        client: gemini_rust::GeminiBuilder::new("test")
            .with_model(GeminiModel::Custom("models/selected-model".into()))
            .with_base_url(base.parse().unwrap())
            .build()
            .unwrap(),
        options: LlmOptions::default().with_tools(vec![tool()]),
        url: ModelUrl::parse("gemini:///selected-model").unwrap(),
    }
}

/// Checks the mutually exclusive JSON Schema field and preserves numeric/boolean types.
fn assert_declaration(value: &Value) {
    assert!(value.get("parameters").is_none());
    let fields = &value["parametersJsonSchema"]["properties"]["query"]["properties"];
    assert_eq!(fields["status"]["type"], "integer");
    assert_eq!(fields["status"]["enum"], json!([1, 2, 3]));
    assert_eq!(fields["status"]["minimum"], 1);
    assert_eq!(fields["technique"]["enum"], json!([1, 2, 3]));
    assert_eq!(fields["ratio"]["enum"], json!([1.5, 2.5]));
    assert_eq!(fields["active"]["enum"], json!([true]));
    assert_eq!(fields["name"]["enum"], json!(["example"]));
}

/// Integer choices use JSON Schema rather than Gemini's string-only native enum field.
#[test]
fn numeric_choices_use_parameters_json_schema() -> Result<(), RathError> {
    let definition = tool();
    let original = definition.parameters.clone();
    assert_declaration(&serde_json::to_value(build_fn_decl(&definition)?).unwrap());
    assert_eq!(definition.parameters, original);
    Ok(())
}

/// Generation and SSE dispatch send identical JSON Schema declarations to the SDK transport.
#[tokio::test]
async fn numeric_choices_survive_generation_and_streaming_wire() -> Result<(), RathError> {
    for streaming in [false, true] {
        let response = json!({"candidates":[{"finishReason":"STOP",
            "content":{"role":"model", "parts":[{"text":"ok"}]}}]});
        let (mime, body) = if streaming {
            ("text/event-stream", format!("data: {response}\n\n"))
        } else {
            ("application/json", response.to_string())
        };
        let (base, requests) = crate::providers::tests::http::serve_content(200, mime, body);
        let client = client(&base);
        let messages = [Message::user("hello")];
        if streaming {
            let mut stream = client.execute_stream(&messages).await?;
            let mut completed = false;
            while let Some(event) = stream.next().await {
                completed |= matches!(event?, LlmEvent::Completed { .. });
            }
            assert!(completed);
        } else {
            client.execute(&messages).await?;
        }
        let wire = requests.try_recv().unwrap();
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_declaration(&payload["tools"][0]["function_declarations"][0]);
    }
    Ok(())
}

/// Native counting receives the same repaired declaration without invoking generation.
#[tokio::test]
async fn numeric_choices_survive_native_count_wire() -> Result<(), RathError> {
    let (base, requests) = crate::providers::tests::http::serve(200, json!({"totalTokens":7}));
    let client = client(&base);
    assert_eq!(
        client
            .count_tokens(&[Message::user("hello")])
            .await?
            .input_tokens,
        7
    );
    let wire = requests.try_recv().unwrap();
    assert!(wire.lines().next().unwrap().contains(":countTokens"));
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_declaration(&payload["generateContentRequest"]["tools"][0]["function_declarations"][0]);
    Ok(())
}
