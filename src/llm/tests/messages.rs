use crate::llm::{Message, Role, TokenUsage};
use serde_json::{Value, json};

/// Constructors omit the optional key, preserving the previous storage representation.
#[test]
fn constructors_default_to_no_key() {
    let messages = [
        Message::user("hello"),
        Message::assistant("hello"),
        Message::tool_output("call-1".into(), "result"),
        Message::from_json(Role::User, &json!({"hello":true})).unwrap(),
    ];
    for message in messages {
        assert!(message.key.is_none());
        assert!(serde_json::to_value(&message).unwrap().get("key").is_none());
    }
}

/// Old records and explicit null keys deserialize without a migration.
#[test]
fn legacy_and_null_keys_remain_readable() {
    let mut record = serde_json::to_value(Message::user("hello")).unwrap();
    assert!(
        serde_json::from_value::<Message>(record.clone())
            .unwrap()
            .key
            .is_none()
    );
    record["key"] = Value::Null;
    assert!(
        serde_json::from_value::<Message>(record)
            .unwrap()
            .key
            .is_none()
    );
}

/// Keys round-trip unchanged and survive cloning and subsequent builder calls.
#[test]
fn key_survives_storage_and_builders() {
    let message = Message::user("hello")
        .with_key("message:42/नमस्ते")
        .with_usage(TokenUsage {
            input: Some(3),
            output: None,
        })
        .with_url("image/png", "https://example.com/image.png");
    let encoded = serde_json::to_value(message.clone()).unwrap();
    assert_eq!(encoded["key"], "message:42/नमस्ते");
    let decoded: Message = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.key, message.key);
    assert_eq!(decoded.content, "hello");
    assert_eq!(decoded.attachments.len(), 1);
    assert_eq!(decoded.usage.unwrap().input, Some(3));
    assert_eq!(
        message.with_key(String::from("replacement")).key.as_deref(),
        Some("replacement")
    );
}
