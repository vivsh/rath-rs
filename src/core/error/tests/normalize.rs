use super::*;
use serde_json::json;

/// Null or non-scalar codes must not mask useful type/status metadata from the provider.
#[test]
fn code_falls_back_to_useful_type_or_status() {
    for (native, expected) in [
        (
            json!({"code":null,"type":"invalid_request_error"}),
            "invalid_request_error",
        ),
        (
            json!({"code":{},"type":null,"status":"UNSUPPORTED"}),
            "UNSUPPORTED",
        ),
        (json!({"code":429,"type":"rate_limit"}), "429"),
        (json!({"code":"specific","type":"generic"}), "specific"),
    ] {
        let mut error = RathError::new(ErrorKind::Http, "rejected");
        details(&mut error, &json!({"error":native}));
        assert_eq!(error.provider_code(), Some(expected));
    }
}
