//! Offline positive and explicit-rejection coverage for remote batch controls.
use rath::ErrorKind;
use rath::embeddings::{EmbedRequest, EmbedTaskType, EmbeddingOptions, FastEmbedOptions};
use serde_json::{Value, json};
#[path = "media_provider_registry/http.rs"]
mod http;

/// Every Gemini task is encoded per batch item with dimensions, truncation and aggregate usage.
#[tokio::test]
async fn gemini_batch_controls() {
    for task in [
        EmbedTaskType::RetrievalDocument,
        EmbedTaskType::RetrievalQuery,
        EmbedTaskType::SemanticSimilarity,
        EmbedTaskType::Classification,
        EmbedTaskType::Clustering,
        EmbedTaskType::QuestionAnswering,
        EmbedTaskType::FactVerification,
        EmbedTaskType::CodeRetrievalQuery,
    ] {
        let (base, wire) = http::serve(
            200,
            r#"{"embeddings":[{"values":[1,2]},{"values":[3,4]}],"usageMetadata":{"promptTokenCount":9}}"#,
        );
        let client = EmbeddingOptions::default()
            .create(&format!("gemini:///embed?api_key_env=USER&base_url={base}"))
            .await
            .unwrap();
        let mut request = EmbedRequest::batch(["one".into(), "two".into()]);
        request.task_type = Some(task.clone());
        request.output_dimensionality = Some(2);
        request.truncate = Some(false);
        if task == EmbedTaskType::RetrievalDocument {
            request.title = Some("title".into());
        }
        let response = client.embed(&request).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(response.usage.unwrap().input, Some(9));
        assert!(response.provider_model.is_none());
        let wire = wire.recv().unwrap();
        assert!(wire.starts_with("POST /models/embed:batchEmbedContents "));
        let body: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["requests"].as_array().unwrap().len(), 2);
        let config = &body["requests"][0]["embedContentConfig"];
        assert_eq!(
            config["taskType"],
            serde_json::to_value(task)
                .unwrap()
                .as_str()
                .unwrap()
                .to_uppercase()
        );
        assert_eq!(config["autoTruncate"], false);
        assert_eq!(config["outputDimensionality"], 2);
    }
}

/// Ollama preserves raw inputs and explicitly prepares only recognized Qwen retrieval queries.
#[tokio::test]
async fn ollama_batch_controls() {
    for task in [
        None,
        Some(EmbedTaskType::RetrievalQuery),
        Some(EmbedTaskType::RetrievalDocument),
    ] {
        let (base, wire) =
            http::serve(200, r#"{"embeddings":[[1,2],[3,4]],"prompt_eval_count":4}"#);
        let client = EmbeddingOptions::default()
            .with_provider_config(json!({"keep_alive":"1m","options":{"num_ctx":128}}))
            .create(&format!("ollama:///qwen3-embedding:0.6b?base_url={base}"))
            .await
            .unwrap();
        let mut request = EmbedRequest::batch(["one".into(), "two".into()]);
        request.task_type = task.clone();
        request.output_dimensionality = Some(2);
        request.truncate = Some(true);
        request.provider_config = Some(json!({"keep_alive":"2m","options":{"num_ctx":256}}));
        if task == Some(EmbedTaskType::RetrievalQuery) {
            request.instructions = Some("Find documents".into());
        }
        let response = client.embed(&request).await.unwrap();
        assert_eq!(response.usage.unwrap().input, Some(4));
        let wire = wire.recv().unwrap();
        assert!(wire.starts_with("POST /api/embed "));
        let body: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["dimensions"], 2);
        assert_eq!(body["truncate"], true);
        assert_eq!(body["keep_alive"], "2m");
        assert_eq!(body["options"], json!({"num_ctx":256}));
        let expected = if task == Some(EmbedTaskType::RetrievalQuery) {
            "Instruct: Find documents\nQuery: one"
        } else {
            "one"
        };
        assert_eq!(body["input"][0], expected);
    }
}

/// Invalid batches and unsupported controls fail locally without making a request.
#[tokio::test]
async fn explicit_controls_are_never_silently_ignored() {
    for scheme in ["openai", "gemini", "ollama"] {
        let client = EmbeddingOptions::default()
            .create(&format!(
                "{scheme}:///alias?api_key_env=USER&base_url=http://127.0.0.1:1"
            ))
            .await
            .unwrap();
        let mut cases = vec![
            EmbedRequest::batch([]),
            EmbedRequest::new(""),
            EmbedRequest::new("a").with_output_dimensionality(0),
            EmbedRequest::new("a").with_instructions(" ".into()),
            EmbedRequest::new("a").with_title("title".into()),
            EmbedRequest::new("a").with_provider_config(json!({"dimensions":2})),
        ];
        if scheme != "gemini" {
            cases.push(EmbedRequest::new("a").with_task_type(EmbedTaskType::Classification));
        }
        if scheme == "openai" {
            for truncate in [true, false] {
                cases.push(EmbedRequest::new("a").with_truncate(truncate));
            }
        }
        if scheme == "gemini" {
            cases.push(
                EmbedRequest::new("a")
                    .with_task_type(EmbedTaskType::RetrievalQuery)
                    .with_instructions("find".into()),
            );
        }
        for request in cases {
            let error = client.embed(&request).await.unwrap_err();
            assert!(
                matches!(
                    error.kind(),
                    ErrorKind::Validation | ErrorKind::UnsupportedCapability
                ),
                "{error}"
            );
        }
        let error = EmbeddingOptions::default()
            .with_fastembed(FastEmbedOptions::new("models"))
            .create(&format!("{scheme}:///model"))
            .await
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}

/// Duplicate/out-of-range indices, partial batches, empty vectors and inconsistent dimensions fail.
#[tokio::test]
async fn invalid_remote_batches_keep_raw_evidence() {
    for data in [
        json!([]),
        json!([{"index":0,"embedding":[1,2]},{"index":0,"embedding":[3,4]}]),
        json!([{"index":2,"embedding":[1,2]}]),
        json!([{"index":0,"embedding":[]}]),
        json!([{"index":0,"embedding":[1]},{"index":1,"embedding":[2,3]}]),
    ] {
        let raw = json!({"data":data,"private":"batch-marker"});
        let (base, wire) = http::serve(200, &raw.to_string());
        let client = EmbeddingOptions::default()
            .create(&format!("openai:///m?api_key_env=USER&base_url={base}"))
            .await
            .unwrap();
        let error = client
            .embed(&EmbedRequest::batch(["one".into(), "two".into()]))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert!(
            String::from_utf8_lossy(error.response_body().unwrap().bytes())
                .contains("batch-marker")
        );
        assert!(!format!("{error} {error:?}").contains("batch-marker"));
        assert_eq!(wire.try_iter().count(), 1);
    }
}
