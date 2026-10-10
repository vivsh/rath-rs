use super::*;
use crate::llm::ToolCall;
use serde_json::json;

/// Supplies one ordinary tool so legacy providers must use the private final-output tool.
fn options(schema: serde_json::Value) -> LlmOptions {
    LlmOptions::default()
        .with_output_schema(schema)
        .with_tools(vec![ToolDefinition::new(
            "lookup".into(),
            "Synthetic lookup".into(),
            json!({"type":"object"}),
        )])
}

/// Builds a private final call without involving provider transport or tool execution.
fn final_response(args: serde_json::Value) -> LlmResponse {
    LlmResponse::new(
        Provider::Gemini,
        LlmOutput::ToolCalls {
            text: None,
            calls: vec![ToolCall::new("1".into(), FINAL_TOOL.into(), args)],
        },
    )
}

/// Scalar and array answers acquire an object envelope only in derived tool parameters.
#[test]
fn non_object_schemas_use_an_operation_local_envelope() {
    for schema in [
        json!({"type":"string"}),
        json!({"type":"integer","enum":[1,2]}),
        json!({"type":"number"}),
        json!({"type":"boolean"}),
        json!({"type":"null"}),
        json!({"type":"array","items":{"type":"string"}}),
        json!({"type":["object","null"]}),
        json!({"anyOf":[{"type":"string"},{"type":"integer"}]}),
    ] {
        let original = options(schema.clone());
        for provider in [Provider::Gemini, Provider::Ollama] {
            let prepared = prepare(provider, "gemini-3-flash-preview", &original).unwrap();
            assert_eq!(
                prepared.tools[1].parameters,
                json!({
                    "type":"object", "properties":{"value":schema},
                    "required":["value"], "additionalProperties":false
                })
            );
            assert_eq!(original.output_schema(), Some(&schema));
            assert_eq!(original.tools.len(), 1);
        }
    }
}

/// Returning a wrapped answer preserves its original JSON type, including a literal null.
#[test]
fn final_envelopes_restore_original_json_values() {
    for (schema, value) in [
        (json!({"type":"string"}), json!("OK")),
        (json!({"type":"integer"}), json!(42)),
        (json!({"type":"number"}), json!(1.5)),
        (json!({"type":"boolean"}), json!(true)),
        (json!({"type":"null"}), json!(null)),
        (json!({"type":"array"}), json!(["first", "second"])),
        (json!({"type":["object","null"]}), json!({"answer":1})),
    ] {
        let response = finish(&options(schema), final_response(json!({"value":value}))).unwrap();
        assert!(matches!(response.output, LlmOutput::Output(output) if output == value));
    }
}

/// Authored object schemas and payloads, including a field named value, remain unchanged.
#[test]
fn object_output_contract_is_unchanged() {
    let schema = json!({"type":"object","properties":{"value":{"type":"string"}},
        "required":["value"]});
    let original = options(schema.clone());
    let prepared = prepare(Provider::Gemini, "gemini-3-flash-preview", &original).unwrap();
    assert_eq!(prepared.tools[1].parameters, schema);
    let payload = json!({"value":"keep this field"});
    let response = finish(&original, final_response(payload.clone())).unwrap();
    assert!(matches!(response.output, LlmOutput::Output(output) if output == payload));
}

/// Missing, non-object or ambiguous envelopes fail with private provider evidence retained.
#[test]
fn malformed_final_envelopes_are_rejected() {
    for args in [
        json!("OK"),
        json!(null),
        json!({}),
        json!({"value":"OK","extra":1}),
    ] {
        let response =
            final_response(args).with_raw_metadata(Some(json!({"private":"SECRET-PAYLOAD"})));
        let error = finish(&options(json!({"type":"string"})), response).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert!(
            String::from_utf8_lossy(error.response_body().unwrap().bytes())
                .contains("SECRET-PAYLOAD")
        );
        assert!(!format!("{error:?} {error}").contains("SECRET-PAYLOAD"));
    }
}

/// Relocated local definitions still resolve inside the final-output parameter envelope.
#[test]
fn wrapped_schema_preserves_local_references() {
    for keyword in ["$defs", "definitions"] {
        let mut schema = json!({"type":"array","items":{"$ref":format!("#/{keyword}/Item")}});
        schema[keyword] = json!({"Item":{"type":"integer","enum":[1,2]}});
        let original = options(schema.clone());
        let prepared = prepare(Provider::Gemini, "gemini-3-flash-preview", &original).unwrap();
        let normalized =
            super::super::schema::sanitize_strict(prepared.tools[1].parameters.clone()).unwrap();
        assert_eq!(
            normalized["properties"]["value"]["items"],
            json!({"type":"integer","enum":[1,2]})
        );
        assert_eq!(original.output_schema(), Some(&schema));
    }
}

/// A root reference resolves after wrapping and restores the referenced answer unchanged.
#[test]
fn wrapped_root_reference_retains_its_contract() {
    let schema = json!({"$ref":"#/$defs/Answer", "$defs":{
        "Answer":{"type":"object","properties":{"value":{"type":"string"}}}
    }});
    let original = options(schema.clone());
    let prepared = prepare(Provider::Gemini, "gemini-3-flash-preview", &original).unwrap();
    let normalized =
        super::super::schema::sanitize_strict(prepared.tools[1].parameters.clone()).unwrap();
    assert_eq!(normalized["properties"]["value"], schema["$defs"]["Answer"]);
    let payload = json!({"value":"original object property"});
    let response = finish(&original, final_response(json!({"value":payload}))).unwrap();
    assert!(matches!(response.output, LlmOutput::Output(output) if output == payload));
    assert_eq!(original.output_schema(), Some(&schema));
}

/// Ordinary tool calls and already-completed output bypass private-envelope extraction.
#[test]
fn non_final_responses_are_unchanged() {
    let original = options(json!({"type":"string"}));
    let call = ToolCall::new("1".into(), "lookup".into(), json!({"query":"test"}));
    let response = LlmResponse::new(
        Provider::Gemini,
        LlmOutput::ToolCalls {
            text: None,
            calls: vec![call],
        },
    );
    let response = finish(&original, response).unwrap();
    assert!(matches!(response.output, LlmOutput::ToolCalls { calls, .. }
        if calls[0].name == "lookup" && calls[0].args == json!({"query":"test"})));
    let response = LlmResponse::new(Provider::Gemini, LlmOutput::Output(json!("OK")));
    let response = finish(&original, response).unwrap();
    assert!(matches!(response.output, LlmOutput::Output(output) if output == "OK"));
}

/// Providers using native structured output do not acquire a synthetic final-output envelope.
#[test]
fn native_structured_output_remains_unchanged() {
    let original = options(json!({"type":"string"}));
    for provider in [Provider::Gemini, Provider::OpenAi] {
        let prepared = prepare(provider, "gemini-3.1-flash", &original).unwrap();
        assert_eq!(prepared.tools.len(), 1);
        assert_eq!(prepared.output_schema(), original.output_schema());
    }
}

/// Request preparation retains original schema/options and respects disabled user tools.
#[test]
fn transformations_are_operation_local() {
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_tools(vec![ToolDefinition::new(
            "lookup".into(),
            "Lookup".into(),
            json!({"type":"object"}),
        )]);
    let prepared = prepare(Provider::Ollama, "model", &options).unwrap();
    assert_eq!(options.tools.len(), 1);
    assert!(options.output_schema().is_some());
    assert_eq!(prepared.tools.len(), 2);
    assert_eq!(prepared.tools[1].name, FINAL_TOOL);
    let disabled = prepare(
        Provider::Ollama,
        "model",
        &options.clone().with_tool_choice(ToolChoice::Disabled),
    )
    .unwrap();
    assert_eq!(disabled.tool_choice, ToolChoice::Disabled);
    assert_eq!(disabled.tools.len(), 1);
    assert!(disabled.output_schema().is_some());
    let conflict = LlmOptions::default().with_tools(vec![ToolDefinition::new(
        FINAL_TOOL.into(),
        "invalid".into(),
        json!({}),
    )]);
    assert_eq!(
        prepare(Provider::Ollama, "model", &conflict)
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

/// Mixed ordinary/final calls are never partially interpreted or exposed as successful output.
#[test]
fn mixed_calls_retain_private_evidence() {
    let options = LlmOptions::default().with_output_schema(json!({"type":"object"}));
    let response = LlmResponse::new(
        Provider::Ollama,
        LlmOutput::ToolCalls {
            text: None,
            calls: vec![
                ToolCall::new("1".into(), FINAL_TOOL.into(), json!({"a":1})),
                ToolCall::new("2".into(), "lookup".into(), json!({})),
            ],
        },
    )
    .with_raw_metadata(Some(json!({"private":"SECRET-PAYLOAD"})));
    let error = finish(&options, response).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert!(
        String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("SECRET-PAYLOAD")
    );
    assert!(!format!("{error:?} {error}").contains("SECRET-PAYLOAD"));
}
