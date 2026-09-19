use super::*;

/// Models that emit tool calls as XML-style text in the content field are parsed correctly.
#[test]
fn parse_content_tool_calls_extracts_function_and_params() {
    let content = "<function=file_search>\n<parameter=query>\nforgot password\n</parameter>\n<parameter=globs>\n[\"**/*auth*.js\"]\n</parameter>\n</function>";
    let calls = parse_content_tool_calls(content).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "file_search");
    assert_eq!(calls[0].args["query"], "forgot password");
    assert_eq!(calls[0].args["globs"][0], "**/*auth*.js");
}

/// Multiple tool calls in content are all extracted.
#[test]
fn parse_content_tool_calls_handles_multiple_functions() {
    let content = "<function=search><parameter=q>hello</parameter></function><function=fetch><parameter=url>http://x</parameter></function>";
    let calls = parse_content_tool_calls(content).unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[1].name, "fetch");
}

/// Content with no function tags yields an empty vec (not an error).
#[test]
fn parse_content_tool_calls_returns_empty_on_plain_text() {
    let calls = parse_content_tool_calls("Just a normal response.").unwrap();
    assert!(calls.is_empty());
}

/// collect_tool_calls falls back to content parsing when tool_calls field is absent.
#[test]
fn collect_tool_calls_falls_back_to_content_when_no_tool_calls_field() {
    let message = json!({
        "role": "assistant",
        "content": "<function=my_tool><parameter=x>42</parameter></function>"
    });
    let calls = collect_tool_calls(&message, true).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "my_tool");
    assert_eq!(calls[0].args["x"], 42);
}
