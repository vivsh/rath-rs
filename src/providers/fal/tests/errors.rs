use super::super::*;
use super::http::{json as reply, serve};
use crate::core::ErrorKind;
use serde_json::json;

/// Uses explicit credentials and isolated queue endpoints for failed-job checks.
fn client(base: String) -> FalClient {
    let mut url = ModelUrl::parse("fal:///fal-ai/video-model").unwrap();
    url.base_url = Some(base);
    url.api_key = Some("SECRET-KEY".into());
    FalClient::new(&url, None, "video").unwrap()
}

/// Failed queue metadata and useful messages survive wait_video, including unknown statuses.
#[tokio::test]
async fn video_wait_preserves_failure_and_never_fetches_failed_result() {
    for body in [
        json!({"status":"COMPLETED", "error":{"code":"bad_video", "message":"unsupported duration SECRET-KEY"}, "request_id":"job-42", "private":"PRIVATE-PROSE"}),
        json!({"status":"ALIEN", "request_id":"job-42", "private":"PRIVATE-PROSE"}),
    ] {
        let (base, requests) = serve(vec![reply(body.clone())]);
        let error = client(base).wait_video("job-42").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Provider);
        assert_eq!(error.http_status(), Some(200));
        assert_eq!(error.request_id(), Some("job-42"));
        let text = format!("{error} {error:?}");
        assert!(text.contains(if body["status"] == "ALIEN" {
            "ALIEN"
        } else {
            "unsupported duration"
        }));
        assert!(!text.contains("SECRET-KEY"));
        assert!(!text.contains("PRIVATE-PROSE"));
        let raw = String::from_utf8_lossy(error.response_body().unwrap().bytes());
        assert!(raw.contains("PRIVATE-PROSE"));
        assert!(!raw.contains("SECRET-KEY"));
        assert_eq!(requests.try_iter().count(), 1);
    }
}

/// Failed result envelopes stay Failed in get_video and cannot become empty successful output.
#[tokio::test]
async fn failed_video_result_keeps_public_shape() {
    let (base, requests) = serve(vec![
        reply(json!({"status":"COMPLETED", "response_url":"BASE/result"})),
        reply(json!({"error":"generation failed", "private":"PRIVATE-PROSE"})),
    ]);
    let result = client(base).get_video("job").await.unwrap();
    let VideoJobStatus::Failed {
        message,
        raw_metadata,
    } = result
    else {
        panic!("expected failed job");
    };
    assert_eq!(message, "generation failed");
    assert_eq!(raw_metadata.unwrap()["private"], "PRIVATE-PROSE");
    assert_eq!(requests.try_iter().count(), 2);
}

/// Invalid image shapes retain provider JSON and never become an empty successful image list.
#[tokio::test]
async fn image_missing_required_output_is_invalid_response() {
    let (base, requests) = serve(vec![reply(json!({"other":"PRIVATE-PROSE"}))]);
    let error = client(base)
        .generate_image(&ImageRequest::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidResponse);
    assert_eq!(error.http_status(), Some(200));
    assert!(
        String::from_utf8_lossy(error.response_body().unwrap().bytes()).contains("PRIVATE-PROSE")
    );
    assert!(!error.to_string().contains("PRIVATE-PROSE"));
    assert_eq!(requests.try_iter().count(), 1);
}
