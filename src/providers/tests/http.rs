use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::{Duration, Instant};

/// Serves exactly one request, captures its wire bytes, and bounds failures by timeout.
pub(in crate::providers) fn serve(
    status: u16,
    body: serde_json::Value,
) -> (String, Receiver<String>) {
    serve_content(status, "application/json", body.to_string())
}

/// Serves a bounded binary or text response for media adapters without contacting a provider.
pub(in crate::providers) fn serve_content(
    status: u16,
    mime: &'static str,
    body: String,
) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = channel();
    thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let request = read_request(&mut stream);
                tx.send(request).unwrap();
                write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });
    (address, rx)
}

/// Reads the complete JSON request using its Content-Length header.
pub(in crate::providers) fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 4096];
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "request ended before Content-Length");
        bytes.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&bytes);
        if let Some((headers, body)) = text.split_once("\r\n\r\n") {
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap();
            if body.len() >= length {
                return text.into_owned();
            }
        }
    }
}

/// Serves one SSE response in independently delayed writes, or closes its body early.
pub(in crate::providers) fn serve_stream(
    parts: Vec<(String, Duration)>,
    missing_bytes: usize,
) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = channel();
    thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                tx.send(read_request(&mut stream)).unwrap();
                let length = parts.iter().map(|(s, _)| s.len()).sum::<usize>() + missing_bytes;
                write!(stream, "HTTP/1.1 200 Test\r\nContent-Type: text/event-stream\r\nx-request-id: mock-stream\r\nRetry-After: 7\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n").unwrap();
                for (part, delay) in parts {
                    thread::sleep(delay);
                    if stream.write_all(part.as_bytes()).is_err() {
                        return;
                    }
                }
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });
    (address, rx)
}

/// Compares all normalized output, continuation, metadata and usage fields across execution paths.
pub(in crate::providers) fn assert_response_eq(
    actual: &crate::llm::LlmResponse,
    expected: &crate::llm::LlmResponse,
) {
    use crate::llm::LlmOutput;
    let output = |response: &crate::llm::LlmResponse| match &response.output {
        LlmOutput::Output(value) => serde_json::json!({"output":value}),
        LlmOutput::ToolCalls { text, calls } => serde_json::json!({"text":text, "calls":calls}),
    };
    assert_eq!(output(actual), output(expected));
    assert_eq!(actual.provider, expected.provider);
    assert_eq!(actual.provider_model, expected.provider_model);
    assert_eq!(actual.raw_metadata, expected.raw_metadata);
    assert_eq!(
        actual.usage.as_ref().map(|u| (u.input, u.output)),
        expected.usage.as_ref().map(|u| (u.input, u.output))
    );
}

/// Holds an unfinished SSE response open and reports whether its client disconnects.
pub(in crate::providers) fn serve_until_disconnect(body: String) -> (String, Receiver<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = channel();
    thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok((mut stream, _)) = listener.accept() {
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                read_request(&mut stream);
                write!(stream, "HTTP/1.1 200 Test\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len() + 100).unwrap();
                let result = stream.read(&mut [0; 1]);
                let closed = matches!(result, Ok(0))
                    || result.is_err_and(|e| e.kind() == std::io::ErrorKind::ConnectionReset);
                tx.send(closed).unwrap();
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });
    (address, rx)
}

/// Asserts that streaming changes transport fields only, preserving the normal request builder.
pub(in crate::providers) fn assert_transport_only(request: &str, mut expected: serde_json::Value) {
    let mut actual: serde_json::Value =
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(actual["stream"], true);
    for field in ["stream", "stream_options"] {
        actual.as_object_mut().unwrap().remove(field);
        expected.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(actual, expected);
}
