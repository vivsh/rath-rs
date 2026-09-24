use super::*;

/// Existing variants retain their serialized names; external identities own and round-trip theirs.
#[test]
fn serialized_identities_remain_distinct() {
    for (provider, wire) in [
        (Provider::OpenAi, serde_json::json!("OpenAi")),
        (Provider::Fal, serde_json::json!("Fal")),
        (
            Provider::External("acme".into()),
            serde_json::json!({"External":"acme"}),
        ),
    ] {
        assert_eq!(serde_json::to_value(&provider).unwrap(), wire);
        assert_eq!(serde_json::from_value::<Provider>(wire).unwrap(), provider);
    }
    assert_eq!(Provider::External("acme".into()).as_str(), "acme");
    assert_eq!(builtin("claude"), Some(Provider::Anthropic));
    assert_eq!(builtin("acme"), None);
}
