use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::{Duration, Instant};

/// Serves a bounded script of responses and captures requests, replacing BASE with its URL.
pub(in crate::providers::fal) fn serve(
    responses: Vec<(u16, &str, String)>,
) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let address = base.clone();
    let (tx, rx) = channel();
    let responses: Vec<_> = responses
        .into_iter()
        .map(|(s, m, b)| (s, m.to_string(), b))
        .collect();
    thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        for (status, mime, body) in responses {
            loop {
                if Instant::now() > deadline {
                    return;
                }
                if let Ok((mut stream, _)) = listener.accept() {
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(3)))
                        .unwrap();
                    let request = read_request(&mut stream);
                    if tx.send(request).is_err() {
                        return;
                    }
                    let body = body.replace("BASE", &address);
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status} Test\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
    });
    (base, rx)
}

/// Reads a complete request, including GET requests without a Content-Length header.
fn read_request(stream: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 4096];
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0);
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
                .unwrap_or(0);
            if body.len() >= length {
                return text.into_owned();
            }
        }
    }
}

/// Produces a JSON response for a scripted queue server.
pub(in crate::providers::fal) fn json(body: serde_json::Value) -> (u16, &'static str, String) {
    (200, "application/json", body.to_string())
}
