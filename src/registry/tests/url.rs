use super::*;

/// External locators retain nested model paths and use the existing lexical escaping convention.
#[test]
fn nested_model_and_transport_options() {
    let url = parse_for(
        "acme-1.test:///vendor/model%2Fversion?base_url=http://localhost:1234/api/",
        false,
    )
    .unwrap();
    assert_eq!(url.provider, Provider::External("acme-1.test".into()));
    assert_eq!(url.model, "vendor/model%2Fversion");
    assert_eq!(url.base_url.as_deref(), Some("http://localhost:1234/api"));
    assert!(url.thinking.is_none() && url.temperature.is_none() && url.cache.is_none());
    assert_eq!(
        ModelUrl::parse("acme:///model").unwrap_err().kind(),
        ErrorKind::InvalidUrl
    );
}

/// Malformed URLs and unsupported external controls fail before constructing a provider client.
#[test]
fn invalid_external_urls_are_rejected() {
    for raw in [
        "acme",
        "Acme:///m",
        "1acme:///m",
        "acme+http:///m",
        "acme://host/m",
        "acme://user:PRIVATE@host/m",
        "acme:///",
        "acme:///m#PRIVATE",
        "acme:///m?thinking=off",
        "acme:///m?temperature=0.1",
        "acme:///m?cache=5m",
        "acme:///m?unknown=PRIVATE",
        "acme:///m?base_url=",
        "acme:///m?base_url",
        "acme:///m?base_url=http://localhost&base_url=http://localhost",
        "acme:///m?base_url=https://user:PRIVATE@host",
        "acme:///m?base_url=ftp://host",
        "acme:///m?api_key_env=__RATH_REGISTRY_MISSING_CREDENTIAL__",
    ] {
        let error = parse_for(raw, false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidUrl, "{raw}");
        assert!(!format!("{error} {error:?}").contains("PRIVATE"));
    }
}

/// Credential variables are resolved per construction without changing the environment.
#[test]
fn credentials_are_operation_local() {
    let expected = std::env::var("PATH").unwrap();
    let url = parse_for("acme:///m?api_key_env=PATH", false).unwrap();
    assert_eq!(url.api_key.as_deref(), Some(expected.as_str()));
    assert!(parse_for("acme:///m", false).unwrap().api_key.is_none());
}

/// Built-in aliases and all their existing URL controls are delegated to the legacy parser.
#[test]
fn builtin_url_parity() {
    let raw = "claude:///model?thinking=high&temperature=0.3&cache=5m";
    let old = ModelUrl::parse(raw).unwrap();
    let new = parse_for(raw, true).unwrap();
    assert_eq!(old.provider, new.provider);
    assert_eq!(old.model, new.model);
    assert_eq!(old.thinking, new.thinking);
    assert_eq!(old.temperature, new.temperature);
    assert_eq!(old.cache, new.cache);
}
