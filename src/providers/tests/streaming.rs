use super::*;

/// SSE fields preserve multiline data and ignore comments/reconnect hints.
#[test]
fn parses_sse_fields_without_interpreting_payloads() {
    let mut name = None;
    let mut data = String::new();
    assert!(feed_line(": keepalive", &mut name, &mut data).is_none());
    feed_line("id: event1", &mut name, &mut data);
    feed_line("retry: 100", &mut name, &mut data);
    feed_line("event: update", &mut name, &mut data);
    feed_line("data: first", &mut name, &mut data);
    feed_line("data:second", &mut name, &mut data);
    assert_eq!(
        feed_line("", &mut name, &mut data),
        Some((Some("update".into()), "first\nsecond".into()))
    );
    assert!(data.is_empty());
    assert!(name.is_none());
}

/// Network boundaries may split BOM, multibyte text or either byte of CRLF.
#[test]
fn split_utf8_and_line_endings_remain_intact() {
    for separator in ["\n", "\r\n", "\r"] {
        let bytes = format!("\u{feff}data: नमस्ते🙂{separator}{separator}").into_bytes();
        let mut pending = Vec::new();
        let mut consumed = 0;
        let mut lines = Vec::new();
        for byte in bytes {
            pending.push(byte);
            while let Some(line) = take_line(&mut pending, false, consumed == 0).unwrap() {
                consumed += 1;
                lines.push(line);
            }
        }
        while let Some(line) = take_line(&mut pending, true, consumed == 0).unwrap() {
            consumed += 1;
            lines.push(line);
        }
        assert_eq!(lines, ["data: नमस्ते🙂", ""]);
        assert!(pending.is_empty());
    }
}

/// Malformed UTF-8 is a decoding failure, never replaced or dropped silently.
#[test]
fn invalid_utf8_is_rejected() {
    let mut pending = vec![b'd', b'a', b't', b'a', b':', 255, b'\n'];
    assert_eq!(
        take_line(&mut pending, false, true).unwrap_err().kind(),
        ErrorKind::Deserialize
    );
}

/// Ambiguous private-tool name prefixes defer progress without retaining emitted state.
#[test]
fn tool_progress_suppresses_private_calls_and_flushes_ordinary_names() {
    assert!(tool_delta(1, Some("id"), "__rath", "{", (0, 0, false), false).is_none());
    assert!(
        tool_delta(
            1,
            Some("id"),
            "__rath_final_output",
            "{}",
            (0, 0, false),
            true
        )
        .is_none()
    );
    let event = tool_delta(1, Some("id"), "__rath_user", "{}", (6, 1, false), false).unwrap();
    assert!(
        matches!(event, LlmEvent::ToolCallDelta { index: 1, name_delta: Some(name), arguments_delta, .. } if name == "__rath_user" && arguments_delta == "{}")
    );
    assert!(tool_delta(1, Some("id"), "lookup", "{}", (6, 2, true), true).is_none());
}
