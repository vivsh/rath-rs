use super::*;

/// Constructs one compatible chunk without fabricating final metadata or usage.
fn packet(delta: Value, finish: Value) -> String {
    json!({"id":"response", "model":"model", "choices":[{"index":0, "delta":delta, "finish_reason":finish}]}).to_string()
}

/// Interleaved function fragments assemble names/arguments and retain stable call IDs.
#[test]
fn tool_fragments_correlate_across_calls() {
    let mut assembled = initial();
    let mut names = [String::new(), String::new()];
    let mut args = [String::new(), String::new()];
    let packets = [
        json!({"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"look","arguments":"{"}}]}),
        json!({"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"calculate","arguments":"{"}}]}),
        json!({"tool_calls":[{"index":0,"function":{"name":"up","arguments":"\"city\":\"Delhi\"}"}}]}),
        json!({"tool_calls":[{"index":1,"function":{"arguments":"\"n\":2}"}}]}),
    ];
    for packet in packets {
        for event in chunk(
            &self::packet(packet, Value::Null),
            Provider::OpenRouter,
            &mut assembled,
        )
        .unwrap()
        .unwrap()
        {
            let LlmEvent::ToolCallDelta {
                index,
                name_delta,
                arguments_delta,
                ..
            } = event
            else {
                panic!("expected tool")
            };
            names[index].push_str(name_delta.as_deref().unwrap_or_default());
            args[index].push_str(&arguments_delta);
        }
    }
    chunk(
        &packet(json!({}), json!("tool_calls")),
        Provider::OpenRouter,
        &mut assembled,
    )
    .unwrap();
    assert!(
        chunk("[DONE]", Provider::OpenRouter, &mut assembled)
            .unwrap()
            .is_none()
    );
    assert_eq!(names, ["lookup", "calculate"]);
    assert_eq!(args, ["{\"city\":\"Delhi\"}", "{\"n\":2}"]);
}

/// Split private-tool names never escape as progress while ordinary prefixes flush once.
#[test]
fn private_exit_tool_progress_is_hidden() {
    let mut assembled = initial();
    for (name, args) in [("__rath", "{"), ("_final_output", "\"answer\":1}")] {
        let value = packet(
            json!({"tool_calls":[{"index":0,"id":"private","function":{"name":name,"arguments":args}}]}),
            Value::Null,
        );
        assert!(
            chunk(&value, Provider::Ollama, &mut assembled)
                .unwrap()
                .unwrap()
                .is_empty()
        );
    }
    assert!(final_tools(&assembled).unwrap().is_empty());
}

/// Contradictory IDs, sparse indices and content after termination are protocol failures.
#[test]
fn invalid_call_identity_and_lifecycle_are_rejected() {
    let mut assembled = initial();
    chunk(&packet(json!({"tool_calls":[{"index":0,"id":"a","function":{"name":"lookup","arguments":"{"}}]}), Value::Null), Provider::OpenRouter, &mut assembled).unwrap();
    assert!(
        chunk(
            &packet(json!({"tool_calls":[{"index":0,"id":"b"}]}), Value::Null),
            Provider::OpenRouter,
            &mut assembled
        )
        .is_err()
    );
    assert!(
        chunk(
            &packet(json!({"tool_calls":[{"index":usize::MAX}]}), Value::Null),
            Provider::OpenRouter,
            &mut initial()
        )
        .is_err()
    );
    chunk(
        &packet(json!({}), json!("stop")),
        Provider::OpenRouter,
        &mut assembled,
    )
    .unwrap();
    assert!(
        chunk(
            &packet(json!({"content":"late"}), Value::Null),
            Provider::OpenRouter,
            &mut assembled
        )
        .is_err()
    );
}

/// Repeated OpenRouter finish metadata and usage-only compatible frames preserve latest totals.
#[test]
fn trailing_usage_is_not_double_counted() {
    let mut assembled = initial();
    chunk(
        &packet(json!({"content":"done"}), json!("stop")),
        Provider::OpenRouter,
        &mut assembled,
    )
    .unwrap();
    for choices in [
        json!([]),
        json!([{"index":0,"delta":{"content":""},"finish_reason":"stop"}]),
    ] {
        let update = json!({"choices":choices, "usage":{"prompt_tokens":5,"completion_tokens":3}});
        assert!(
            chunk(&update.to_string(), Provider::OpenRouter, &mut assembled)
                .unwrap()
                .unwrap()
                .is_empty()
        );
    }
    assert_eq!(assembled["usage"]["completion_tokens"], 3);
    assert!(
        chunk("[DONE]", Provider::OpenRouter, &mut assembled)
            .unwrap()
            .is_none()
    );
}

/// A partial leading Ollama thought tag is held until it can be excluded from answer progress.
#[test]
fn thinking_text_is_not_answer_progress() {
    let mut assembled = initial();
    let mut visible = String::new();
    for text in [
        " ",
        "<th",
        "ink>",
        "private thought",
        "</thi",
        "nk> ",
        "Hello",
        "🙂",
    ] {
        for event in chunk(
            &packet(json!({"content":text}), Value::Null),
            Provider::Ollama,
            &mut assembled,
        )
        .unwrap()
        .unwrap()
        {
            let LlmEvent::TextDelta { text } = event else {
                panic!("expected text")
            };
            visible.push_str(&text);
        }
    }
    assert_eq!(visible, "Hello🙂");
}

/// Cap termination and absent finish reasons cannot become a successful final marker.
#[test]
fn terminal_validation_precedes_completion() {
    for provider in [Provider::OpenRouter, Provider::Ollama] {
        assert!(chunk("[DONE]", provider.clone(), &mut initial()).is_err());
        let error = chunk(
            &packet(json!({"content":"{"}), json!("length")),
            provider,
            &mut initial(),
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), crate::ErrorKind::OutputLimitReached);
    }
}

/// IDs first supplied after names are emitted remain available even in an ID-only chunk.
#[test]
fn late_complete_id_is_delivered_once() {
    let mut assembled = initial();
    chunk(
        &packet(
            json!({"tool_calls":[{"index":0,"function":{"name":"lookup","arguments":"{}"}}]}),
            Value::Null,
        ),
        Provider::OpenRouter,
        &mut assembled,
    )
    .unwrap();
    let value = packet(json!({"tool_calls":[{"index":0,"id":"late"}]}), Value::Null);
    let events = chunk(&value, Provider::OpenRouter, &mut assembled)
        .unwrap()
        .unwrap();
    assert!(
        matches!(&events[0], LlmEvent::ToolCallDelta { id:Some(id), name_delta:None, arguments_delta, .. } if id == "late" && arguments_delta.is_empty())
    );
    assert!(
        chunk(&value, Provider::OpenRouter, &mut assembled)
            .unwrap()
            .unwrap()
            .is_empty()
    );
}

/// Literal partial thought-tag prefixes flush at termination; actual unclosed thoughts fail.
#[test]
fn terminal_ollama_text_does_not_lose_held_prefixes() {
    for text in [" ", "<th", " <think"] {
        let mut assembled = initial();
        assert!(
            chunk(
                &packet(json!({"content":text}), json!("stop")),
                Provider::Ollama,
                &mut assembled
            )
            .unwrap()
            .unwrap()
            .is_empty()
        );
        assert!(
            matches!(&final_ollama_text(&assembled).unwrap()[0], LlmEvent::TextDelta { text:fragment } if fragment == text)
        );
    }
    let mut assembled = initial();
    chunk(
        &packet(
            json!({"content":"<think>unfinished private thought"}),
            json!("stop"),
        ),
        Provider::Ollama,
        &mut assembled,
    )
    .unwrap();
    assert!(final_ollama_text(&assembled).is_err());
}

/// A cap-bearing frame is rejected before malformed generated content can mask truncation.
#[test]
fn cap_frame_precedes_malformed_delta_parsing() {
    for provider in [Provider::OpenRouter, Provider::Ollama] {
        let value = packet(json!({"content":{}}), json!("length"));
        assert_eq!(
            chunk(&value, provider, &mut initial())
                .err()
                .unwrap()
                .kind(),
            crate::ErrorKind::OutputLimitReached
        );
    }
}
