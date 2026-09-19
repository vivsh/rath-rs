//! Opt-in extraction checks against explicit, already installed local models; never pulls models.
use rath::llm::{LlmOptions, LlmOutput, LlmResponse, Message, RathError};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

/// Verifies explicit Off on the installed small Qwen model through Rath's public client.
#[tokio::test]
#[ignore = "requires local Ollama with qwen3:0.6b already installed"]
async fn qwen_off() {
    extraction("qwen3:0.6b").await;
}

/// Verifies explicit Off on the installed Granite model through Rath's public client.
#[tokio::test]
#[ignore = "requires local Ollama with granite4.2:3b already installed"]
async fn granite_off() {
    extraction("granite4.2:3b").await;
}

/// Verifies explicit Off on the installed LFM thinking model through Rath's public client.
#[tokio::test]
#[ignore = "requires local Ollama with lfm2.5-thinking:1.2b already installed"]
async fn lfm_off() {
    extraction("lfm2.5-thinking:1.2b").await;
}

/// Records three bounded synthetic extractions, preserving unavailable reasoning usage as null.
async fn extraction(model: &str) {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let version: Value = http
        .get("http://127.0.0.1:11434/api/version")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let client = LlmOptions::default()
        .with_temperature(0.0)
        .with_max_output_tokens(256)
        .with_output_schema(json!({"type":"object", "properties":{
            "name":{"type":"string"}, "city":{"type":"string"}},
            "required":["name","city"], "additionalProperties":false}))
        .create(&format!(
            "ollama:///{model}?thinking=off&base_url=http://127.0.0.1:11434"
        ))
        .unwrap();
    for sample in 1..=3 {
        let start = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(90),
            client.execute(&[Message::user(
                "Extract the person's name and city as JSON. Text: Ada lives in Pune.",
            )]),
        )
        .await
        .expect("local extraction exceeded 90 seconds");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let response = result.unwrap_or_else(|error| failed(model, sample, elapsed_ms, error));
        report(model, &version, sample, elapsed_ms, &response);
    }
}

/// Emits only synthetic-task failure diagnostics; partial responses never count as success.
fn failed(model: &str, sample: u32, elapsed_ms: f64, error: RathError) -> ! {
    let body = error
        .response_body()
        .and_then(|b| serde_json::from_slice::<Value>(b.bytes()).ok());
    eprintln!(
        "{}",
        json!({"model":model,"sample":sample,"elapsed_ms":elapsed_ms,
        "error_kind":format!("{:?}",error.kind()),"diagnostics":error.to_string(),
        "finish_reason":body.as_ref().and_then(|b| b.pointer("/choices/0/finish_reason")),
        "usage":body.as_ref().and_then(|b| b.get("usage"))})
    );
    panic!("extraction failed: {error}");
}

/// Reports completion evidence without treating absent reasoning fields as zero reasoning tokens.
fn report(model: &str, version: &Value, sample: u32, elapsed_ms: f64, response: &LlmResponse) {
    let metadata = response.raw_metadata.as_ref().unwrap();
    let usable = matches!(&response.output, LlmOutput::Output(v) if v == &json!({"name":"Ada","city":"Pune"}));
    println!(
        "{}",
        json!({"model":model,"server_version":version["version"],"sample":sample,
            "elapsed_ms":elapsed_ms,"usable":usable,"output":format!("{:?}",response.output),
            "finish_reason":metadata["finish_reason"],"reasoning":metadata["reasoning"],
            "reasoning_tokens":metadata.pointer("/usage/completion_tokens_details/reasoning_tokens"),
            "reasoning_usage_available":metadata.pointer("/usage/completion_tokens_details/reasoning_tokens").is_some(),
            "usage":metadata["usage"],"normalized_usage":format!("{:?}",response.usage)})
    );
    assert!(
        usable,
        "synthetic extraction was not accurate structured output"
    );
    assert_eq!(metadata["finish_reason"], "stop");
}
