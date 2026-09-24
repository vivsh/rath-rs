use super::*;
use crate::embeddings::FastEmbedOptions;
use std::path::Path;

/// Writes an explicit application-owned cache revision containing the tiny licensed fixture.
fn fixture(root: &Path) {
    let repo = root.join("models--Xenova--bge-small-en-v1.5");
    let snapshot = repo.join("snapshots/fixture");
    std::fs::create_dir_all(snapshot.join("onnx")).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/test-revision"), "fixture").unwrap();
    for (name, bytes) in [
        (
            "onnx/model.onnx",
            include_bytes!("fixtures/model.onnx").as_slice(),
        ),
        (
            "tokenizer.json",
            include_bytes!("fixtures/tokenizer.json").as_slice(),
        ),
        (
            "config.json",
            include_bytes!("fixtures/config.json").as_slice(),
        ),
        (
            "tokenizer_config.json",
            include_bytes!("fixtures/tokenizer_config.json").as_slice(),
        ),
        (
            "special_tokens_map.json",
            include_bytes!("fixtures/special_tokens_map.json").as_slice(),
        ),
    ] {
        std::fs::write(snapshot.join(name), bytes).unwrap();
    }
}

/// Loads only local fixture bytes and exercises batching, finite dimensions and truncation.
#[tokio::test]
async fn offline_fixture_inference_and_truncation() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path());
    let client = EmbeddingOptions::default()
        .with_fastembed(
            FastEmbedOptions::new(directory.path())
                .with_revision("test-revision")
                .with_max_length(3)
                .with_batch_size(1),
        )
        .create("fastembed:///BGESmallENV15?base_url=http://127.0.0.1:1")
        .await
        .unwrap();
    let request = EmbedRequest::batch(["hello world".into(), "world".into()]);
    let response = client.embed(&request).await.unwrap();
    assert_eq!(response.embeddings.len(), 2);
    assert!(
        response
            .embeddings
            .iter()
            .all(|v| v.len() == 384 && v.iter().all(|x| x.is_finite()))
    );
    assert!(response.usage.is_none() && response.provider_model.is_none());
    let request = EmbedRequest::new("hello world hello world");
    assert_eq!(
        client.embed(&request).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert!(client.embed(&request.with_truncate(true)).await.is_ok());
}

/// Missing files and revisions fail before any network operation or implicit model download.
#[tokio::test]
async fn offline_missing_artifacts_never_contact_endpoint() {
    let directory = tempfile::tempdir().unwrap();
    for revision in ["missing", "main"] {
        let error = EmbeddingOptions::default()
            .with_fastembed(FastEmbedOptions::new(directory.path()).with_revision(revision))
            .create("fastembed:///BGESmallENV15?base_url=http://127.0.0.1:1")
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("offline FastEmbed cache is missing")
        );
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

/// Retrieval hints are exact, dimensions native, and unsupported explicit controls fail.
#[test]
fn retrieval_and_settings_matrix() {
    let model = EmbeddingModel::BGESmallENV15;
    let raw = EmbedRequest::new("hello");
    assert_eq!(prepare(&model, &raw).unwrap(), vec!["hello"]);
    let query = raw.clone().with_task_type(EmbedTaskType::RetrievalQuery);
    assert_eq!(
        prepare(&model, &query).unwrap(),
        vec!["Represent this sentence for searching relevant passages: hello"]
    );
    assert_eq!(
        prepare(&model, &query.with_instructions("Find:".into())).unwrap(),
        vec!["Find: hello"]
    );
    assert_eq!(
        prepare(
            &model,
            &raw.clone().with_task_type(EmbedTaskType::RetrievalDocument)
        )
        .unwrap(),
        vec!["hello"]
    );
    for request in [
        raw.clone().with_task_type(EmbedTaskType::Clustering),
        raw.clone().with_output_dimensionality(2),
        raw.clone()
            .with_task_type(EmbedTaskType::RetrievalDocument)
            .with_title("title".into()),
        raw.clone()
            .with_provider_config(serde_json::json!({"batch_size":1})),
    ] {
        assert!(prepare(&model, &request).is_err());
    }
    assert!(prepare(&model, &raw.clone().with_output_dimensionality(384)).is_ok());
    assert!(
        prepare(
            &EmbeddingModel::AllMiniLML6V2,
            &raw.with_task_type(EmbedTaskType::RetrievalQuery)
        )
        .is_err()
    );
}

/// Dropping a waiting future leaves the lock authoritative; subsequent calls remain usable.
#[tokio::test]
async fn concurrency_and_cancellation_keep_session_usable() {
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path());
    let client = Arc::new(
        EmbeddingOptions::default()
            .with_fastembed(FastEmbedOptions::new(directory.path()).with_revision("test-revision"))
            .create("fastembed:///BGESmallENV15")
            .await
            .unwrap(),
    );
    let first = client.clone();
    let handle = tokio::spawn(async move {
        first
            .embed(&EmbedRequest::batch(vec!["hello world".into(); 2048]))
            .await
    });
    tokio::task::yield_now().await;
    handle.abort();
    let request = EmbedRequest::new("hello");
    let (a, b, _) = tokio::join!(
        client.embed(&request),
        client.embed(&request),
        tokio::time::sleep(std::time::Duration::from_millis(1))
    );
    assert_eq!(a.unwrap().embeddings.len(), 1);
    assert_eq!(b.unwrap().embeddings.len(), 1);
}

/// Missing Tokio context is a structured validation error, not a panic.
#[test]
fn missing_runtime_is_explicit() {
    assert_eq!(runtime().unwrap_err().kind(), ErrorKind::Validation);
}

/// CPU inference runs on blocking workers while a single-thread async executor continues ticking.
#[tokio::test(flavor = "current_thread")]
async fn inference_keeps_executor_responsive() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let directory = tempfile::tempdir().unwrap();
    fixture(directory.path());
    let client = EmbeddingOptions::default()
        .with_fastembed(FastEmbedOptions::new(directory.path()).with_revision("test-revision"))
        .create("fastembed:///BGESmallENV15")
        .await
        .unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let complete = done.clone();
    let task = tokio::spawn(async move {
        let result = client
            .embed(&EmbedRequest::batch(vec!["hello world".into(); 32768]))
            .await;
        complete.store(true, Ordering::SeqCst);
        result
    });
    let ticks = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        let mut ticks = 0;
        while !done.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            ticks += 1;
        }
        ticks
    })
    .await
    .unwrap();
    assert_eq!(task.await.unwrap().unwrap().embeddings.len(), 32768);
    assert!(
        ticks > 1,
        "inference must permit executor progress, observed {ticks} ticks"
    );
}
