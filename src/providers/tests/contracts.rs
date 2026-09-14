use crate::core::{ModelUrl, RathError};
use crate::llm::{LlmClient, LlmOptions, Message, Role, ToolCall, ToolDefinition};
use serde_json::json;

fn client(provider: &str, options: LlmOptions) -> Result<Box<dyn LlmClient>, RathError> {
    let mut url = ModelUrl::parse(&format!("{provider}:///test-model"))?;
    url.api_key = Some("test-key".into());
    super::super::create_llm_client(&url, options)
}

fn options() -> LlmOptions {
    LlmOptions::default()
        .with_preamble("persona")
        .with_input_schema(json!({"type":"object", "properties":{"prompt":{"type":"string"}}}))
        .with_output_schema(json!({"type":"object", "properties":{"answer":{"type":"string"}}}))
        .with_tools(vec![ToolDefinition {
            name: "lookup".into(),
            description: "Look up a term".into(),
            parameters: json!({"type":"object"}),
        }])
}

/// All built-in adapters reject direct invalid caps and reserved aliases at construction.
#[test]
fn constructors_validate_caps_before_credentials_or_dispatch() {
    for provider in ["openai", "gemini", "anthropic", "ollama", "openrouter"] {
        for options in [
            LlmOptions {
                max_output_tokens: Some(0),
                ..Default::default()
            },
            LlmOptions::default().with_provider_config(json!({"max_tokens": 20})),
            LlmOptions::default()
                .with_max_output_tokens(20)
                .with_provider_config(json!({"max_tokens":20})),
        ] {
            assert!(
                matches!(client(provider, options), Err(RathError::Validation(_))),
                "{provider}"
            );
        }
    }
}

/// Every adapter measures retained messages, tools, schemas, reminders and assistant tool text.
#[test]
fn all_adapters_measure_complete_history() {
    for provider in ["openai", "gemini", "anthropic", "ollama", "openrouter"] {
        let c = client(provider, options()).unwrap();
        let base = c
            .estimate_tokens(&[Message::user("current")])
            .unwrap()
            .input_tokens;
        let marker = "preserved content ".repeat(300);
        for role in [
            Role::System,
            Role::User,
            Role::Assistant,
            Role::AssistantToolCalls {
                calls: vec![ToolCall {
                    id: "1".into(),
                    name: "lookup".into(),
                    args: json!({}),
                    thought_signatures: None,
                }],
            },
        ] {
            let messages = [
                Message {
                    key: None,
                    role,
                    content: marker.clone(),
                    attachments: vec![],
                    usage: None,
                },
                Message::tool_output("1".into(), "retained result"),
                Message::user("reminder and current"),
            ];
            assert!(
                c.estimate_tokens(&messages).unwrap().input_tokens > base,
                "{provider}"
            );
        }
        let richer = client(provider, options().with_preamble(marker.clone())).unwrap();
        assert!(
            richer
                .estimate_tokens(&[Message::user("current")])
                .unwrap()
                .input_tokens
                > base
        );
        assert_eq!(
            richer.estimate_content_tokens("summary").unwrap(),
            c.estimate_content_tokens("summary").unwrap()
        );
    }
}

/// Effective options expose injected exit tools, and their schemas are included in estimates.
#[test]
fn injected_exit_tool_is_measured_once_as_effective_request_state() {
    for model_url in ["ollama:///qwen3:8b", "gemini:///gemini-2.5-flash"] {
        let mut url = ModelUrl::parse(model_url).unwrap();
        url.api_key = Some("test".into());
        let mut opts = options();
        opts.output_type_name = "final_answer".into();
        opts.output_schema =
            Some(json!({"type":"object", "description":"output schema ".repeat(300)}));
        let c = super::super::create_llm_client(&url, opts).unwrap();
        assert!(c.options().output_schema.is_none());
        assert_eq!(
            c.options()
                .tools
                .iter()
                .filter(|t| t.name == "final_answer")
                .count(),
            1
        );
        let full = c
            .estimate_tokens(&[Message::user("x")])
            .unwrap()
            .input_tokens;
        assert!(full > c.estimate_content_tokens("x").unwrap().input_tokens + 100);
    }
}

/// Application keys have no influence on any adapter's prompt projection or estimate.
#[test]
fn message_keys_do_not_affect_token_estimates() {
    let messages = [
        Message::user("hello"),
        Message::assistant("welcome"),
        Message::user("current"),
    ];
    let keyed: Vec<_> = messages
        .iter()
        .cloned()
        .map(|message| message.with_key("private application key ".repeat(1000)))
        .collect();
    for provider in ["openai", "gemini", "anthropic", "ollama", "openrouter"] {
        let c = client(provider, options()).unwrap();
        assert_eq!(
            c.estimate_tokens(&messages).unwrap(),
            c.estimate_tokens(&keyed).unwrap(),
            "{provider}"
        );
    }
}
