use super::*;
use crate::core::ErrorKind;
use crate::providers::tests::http::serve;

/// Every public thinking value reaches the configured endpoint without changing user content.
#[tokio::test]
async fn thinking_levels_reach_wire() {
    for (level, expected) in [
        (None, None),
        (Some(ThinkingLevel::Off), Some("none")),
        (Some(ThinkingLevel::Low), Some("low")),
        (Some(ThinkingLevel::Medium), Some("medium")),
        (Some(ThinkingLevel::High), Some("high")),
        (Some(ThinkingLevel::XHigh), Some("xhigh")),
    ] {
        let (base, requests) = serve(200, success());
        let client = LlmOptions::default()
            .with_thinking(level)
            .create(&url(&base, ""))
            .unwrap();
        let message = Message::user("  Original Ω\n");
        client
            .execute(std::slice::from_ref(&message))
            .await
            .unwrap();
        let wire = requests.try_recv().unwrap();
        assert!(wire.starts_with("POST /chat/completions "));
        let payload = payload(&wire);
        assert_eq!(
            payload.get("reasoning"),
            expected.map(|effort| json!({"effort":effort})).as_ref()
        );
        assert!(payload.get("reasoning_effort").is_none());
        assert_eq!(payload["messages"][0]["content"], message.content);
        assert!(requests.try_recv().is_err());
    }
}

/// URL thinking overrides the builder, while an absent URL value preserves the builder setting.
#[tokio::test]
async fn url_precedence_reaches_wire() {
    for (level, query, expected) in [
        (Some(ThinkingLevel::High), "&thinking=off", "none"),
        (Some(ThinkingLevel::Off), "&thinking=high", "high"),
        (None, "&thinking=xhigh", "xhigh"),
        (Some(ThinkingLevel::Medium), "", "medium"),
    ] {
        let (base, requests) = serve(200, success());
        let client = LlmOptions::default()
            .with_thinking(level)
            .create(&url(&base, query))
            .unwrap();
        client.execute(&[Message::user("hello")]).await.unwrap();
        assert_eq!(
            payload(&requests.try_recv().unwrap())["reasoning"]["effort"],
            expected
        );
    }
}

/// Thinking remains independent of structured output, tool declarations and the output budget.
#[test]
fn thinking_preserves_other_payload_fields() {
    let options = LlmOptions::default()
        .with_output_schema(json!({"type":"object"}))
        .with_max_output_tokens(512)
        .with_tools(vec![ToolDefinition {
            name: "lookup".into(),
            description: "Look up a value".into(),
            parameters: json!({"type":"object"}),
        }]);
    let messages = [Message::user("extract")];
    let baseline = build_payload("model", &options, &messages, true);
    for level in [
        ThinkingLevel::Off,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::XHigh,
    ] {
        let mut wire = build_payload(
            "model",
            &options.clone().with_thinking(Some(level)),
            &messages,
            true,
        );
        wire.as_object_mut().unwrap().remove("reasoning");
        assert_eq!(wire, baseline);
    }
}

/// Rejected efforts preserve provider diagnostics and do not trigger a relaxed retry.
#[tokio::test]
async fn rejected_effort_preserves_error() {
    for level in [ThinkingLevel::Off, ThinkingLevel::XHigh] {
        let body = json!({"error":{"code":400,"message":"model does not support requested reasoning effort"},"request_id":"rejected-effort"});
        let (base, requests) = serve(400, body.clone());
        let client = LlmOptions::default()
            .with_thinking(Some(level.clone()))
            .create(&url(&base, ""))
            .unwrap();
        let error = client.execute(&[Message::user("hello")]).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Http);
        assert_eq!(error.provider(), Some(&Provider::OpenRouter));
        assert_eq!(error.operation(), Some("generation"));
        assert_eq!(error.http_status(), Some(400));
        assert_eq!(error.provider_code(), Some("400"));
        assert_eq!(error.request_id(), Some("rejected-effort"));
        assert!(
            error
                .message()
                .contains("does not support requested reasoning effort")
        );
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            body
        );
        assert_eq!(
            payload(&requests.try_recv().unwrap())["reasoning"]["effort"],
            reasoning_effort(&level)
        );
        assert!(requests.try_recv().is_err());
    }
}

/// Reasoning control is excluded from prompt estimates, and standalone counts retain minimal framing.
#[test]
fn thinking_does_not_inflate_prompt_estimates() {
    let messages = [Message::user("summary")];
    let mut url = ModelUrl::parse("openrouter:///selected-model").unwrap();
    url.api_key = Some("test".into());
    let baseline = new_client(&url, LlmOptions::default()).unwrap();
    for level in [ThinkingLevel::Off, ThinkingLevel::XHigh] {
        let client = new_client(&url, LlmOptions::default().with_thinking(Some(level))).unwrap();
        assert_eq!(
            client.estimate_tokens(&messages).unwrap(),
            baseline.estimate_tokens(&messages).unwrap()
        );
        assert_eq!(
            client.estimate_content_tokens("summary").unwrap(),
            baseline.estimate_content_tokens("summary").unwrap()
        );
    }
}

/// Uses an existing non-secret environment value solely as a local mock credential, without mutation.
fn url(base: &str, query: &str) -> String {
    format!("openrouter:///vendor/model?api_key_env=PATH&base_url={base}{query}")
}

fn payload(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}

fn success() -> Value {
    json!({"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]})
}
