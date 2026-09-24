use super::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// Models hf-hub's metadata-range and content-range requests using only a localhost fixture.
fn hub(count: usize, fail: bool) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = requests.clone();
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while captured.lock().unwrap().len() < count && std::time::Instant::now() < deadline {
            let Ok((mut stream, _)) = listener.accept() else {
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut byte = [0];
                if stream.read(&mut byte).unwrap() == 0 {
                    break;
                }
                bytes.push(byte[0]);
                if bytes.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(bytes).unwrap();
            let metadata = request.to_lowercase().contains("range: bytes=0-0");
            captured.lock().unwrap().push(request);
            if fail {
                write!(
                    stream,
                    "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .unwrap();
            } else {
                let body = if metadata { "d" } else { "data" };
                write!(stream,"HTTP/1.1 206 Partial Content\r\nX-Repo-Commit: pinned\r\nETag: fixture-blob\r\nContent-Range: bytes 0-3/4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            }
        }
    });
    (endpoint, requests)
}

/// Downloads only the missing artifact, pins the revision, and transmits only explicit credentials.
#[test]
fn download_pins_revision_and_reuses_cache() {
    let directory = tempfile::tempdir().unwrap();
    let (base, requests) = hub(3, false);
    let mut url = ModelUrl::parse("fastembed:///BGESmallENV15").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("explicit-only".into());
    let setup = FastEmbedOptions::new(directory.path())
        .with_download(true)
        .with_revision("branch");
    let repo = Repo::with_revision("org/model".into(), RepoType::Model, "branch".into());
    let first = artifacts(&repo, &["config.json"], &setup, &url).unwrap();
    assert_eq!(std::fs::read(&first["config.json"]).unwrap(), b"data");
    assert_eq!(
        artifacts(&repo, &["config.json"], &setup, &url).unwrap(),
        first
    );
    let wires = requests.lock().unwrap();
    assert_eq!(wires.len(), 3);
    assert!(wires[0].contains("/resolve/branch/config.json"));
    assert!(
        wires[1..]
            .iter()
            .all(|s| s.contains("/resolve/pinned/config.json"))
    );
    assert!(wires.iter().all(|s| {
        s.to_lowercase()
            .contains("authorization: bearer explicit-only")
    }));
}

/// Explicit offline loading and upstream retrieval failures never trigger retries.
#[test]
fn offline_and_rejection_do_not_retry() {
    let directory = tempfile::tempdir().unwrap();
    let (base, requests) = hub(1, true);
    let mut url = ModelUrl::parse("fastembed:///BGESmallENV15").unwrap();
    url.base_url = Some(base);
    let setup = FastEmbedOptions::new(directory.path());
    let repo = Repo::new("org/model".into(), RepoType::Model);
    assert!(artifacts(&repo, &["config.json"], &setup, &url).is_err());
    assert!(requests.lock().unwrap().is_empty());
    assert!(artifacts(&repo, &["config.json"], &setup.with_download(true), &url).is_err());
    assert_eq!(requests.lock().unwrap().len(), 1);
}
