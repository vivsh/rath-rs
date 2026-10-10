use super::*;
use crate::llm::LlmEvent;
use futures_util::StreamExt;
use serde_json::json;

/// Retains a correlated tool exchange with the caller's original serialized result.
fn history(result: &str) -> Vec<Message> {
    let mut assistant = Message::assistant("");
    assistant.role = Role::AssistantToolCalls {
        calls: vec![ToolCall::new("call-one".into(), "lookup".into(), json!({}))],
    };
    vec![
        Message::user("Lookup the result"),
        assistant,
        Message::tool_output("call-one".into(), result),
    ]
}

/// Builds the production SDK-backed adapter using only the local transport fixture.
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

/// Gemini tool results always carry objects without changing the original history or JSON types.
#[test]
fn tool_result_json_types_use_object_transport() {
    for value in [
        json!([{"id":1}]),
        json!([]),
        json!("text"),
        json!(42),
        json!(1.5),
        json!(true),
        json!(null),
        json!({"output":[1,2]}),
        json!({}),
    ] {
        let original = value.to_string();
        let history = history(&original);
        let messages = build_gemini_messages(&history);
        let part = &messages[2].content.parts.as_ref().unwrap()[0];
        let Part::FunctionResponse { function_response } = part else {
            panic!("expected function response");
        };
        let expected = if value.is_object() {
            value
        } else {
            json!({"output":value})
        };
        assert_eq!(function_response.name, "lookup");
        assert_eq!(function_response.response.as_ref(), Some(&expected));
        assert_eq!(history[2].content, original);
    }
}

/// Plain text and malformed JSON remain complete strings inside the transport object.
#[test]
fn non_json_tool_results_preserve_text() {
    for result in ["ordinary text", "{\"incomplete\":", ""] {
        let messages = build_gemini_messages(&history(result));
        let Part::FunctionResponse { function_response } =
            &messages[2].content.parts.as_ref().unwrap()[0]
        else {
            panic!("expected function response");
        };
        assert_eq!(function_response.response, Some(json!({"output":result})));
    }
}

/// SDK generation and streaming send an array result inside an object and finish normally.
#[tokio::test]
async fn array_tool_result_survives_generation_and_streaming_wire() -> Result<(), RathError> {
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
        let history = history(r#"[{"id":1}]"#);
        if streaming {
            let mut stream = client.execute_stream(&history).await?;
            let LlmEvent::Completed { response } = stream.next().await.unwrap()? else {
                panic!("expected final output");
            };
            assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
            assert!(stream.next().await.is_none());
        } else {
            let response = client.execute(&history).await?;
            assert!(matches!(response.output, LlmOutput::Output(value) if value == "OK"));
        }
        let wire = requests.try_recv().unwrap();
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            payload["contents"][2]["parts"][0]["functionResponse"],
            json!({"name":"lookup","response":{"output":[{"id":1}]}})
        );
        assert_eq!(history[2].content, r#"[{"id":1}]"#);
    }
    Ok(())
}

/// Native counting sends the same array-result envelope as actual provider generation.
#[tokio::test]
async fn array_tool_result_survives_native_count_wire() -> Result<(), RathError> {
    let (base, requests) = crate::providers::tests::http::serve(200, json!({"totalTokens":7}));
    assert_eq!(
        client(&base)
            .count_tokens(&history("[1,2]"))
            .await?
            .input_tokens,
        7
    );
    let wire = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        payload["generateContentRequest"]["contents"][2]["parts"][0]["functionResponse"],
        json!({"name":"lookup","response":{"output":[1,2]}})
    );
    Ok(())
}
