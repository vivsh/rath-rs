use serde_json::json;

use super::super::*;

/// Preserves endpoint construction for existing image and video clients.
#[test]
fn builds_fal_endpoint() {
    assert_eq!(
        build_endpoint("https://fal.run", "fal-ai/flux/schnell"),
        "https://fal.run/fal-ai/flux/schnell"
    );
    assert_eq!(
        build_endpoint("https://queue.fal.run", "fal-ai/wan/text-to-video"),
        "https://queue.fal.run/fal-ai/wan/text-to-video"
    );
    assert_eq!(
        build_status_endpoint(
            "https://queue.fal.run",
            "fal-ai/wan/text-to-video",
            "abc123"
        ),
        "https://queue.fal.run/fal-ai/wan/text-to-video/requests/abc123/status"
    );
    assert_eq!(queue_base_url("https://fal.run"), "https://queue.fal.run");
}

/// Preserves image options and typed-field precedence.
#[test]
fn builds_image_payload() {
    let options = Some(json!({"num_images": 2, "guidance_scale": 3.5}));
    let request = ImageRequest {
        prompt: "a brass astrolabe".to_string(),
        size: Some("landscape_4_3".to_string()),
        provider_config: Some(json!({"num_images": 1})),
        ..ImageRequest::default()
    };
    let payload = image_payload("fal-ai/flux/schnell", &options, &request);
    assert_eq!(payload["prompt"], "a brass astrolabe");
    assert_eq!(payload["image_size"], "landscape_4_3");
    assert_eq!(payload["model"], "fal-ai/flux/schnell");
    assert_eq!(payload["num_images"], 1);
    assert_eq!(payload["guidance_scale"], 3.5);
}

/// Preserves image URL extraction.
#[test]
fn extracts_image_urls() {
    let raw = json!({
        "images": [
            {"url": "https://example.com/one.png", "content_type": "image/png"}
        ]
    });
    let images = extract_images(&raw);
    assert_eq!(images.len(), 1);
    assert!(matches!(&images[0], ImageData::Url { url } if url == "https://example.com/one.png"));
}

/// Preserves video URL extraction.
#[test]
fn extracts_video_url() {
    let raw = json!({
        "video": {"url": "https://example.com/out.mp4", "content_type": "video/mp4"}
    });
    let videos = extract_videos(&raw);
    assert_eq!(videos.len(), 1);
    assert!(matches!(&videos[0], VideoData::Url { url } if url == "https://example.com/out.mp4"));
}

/// Preserves the video job metadata contract.
#[test]
fn maps_queue_submit_to_video_job() {
    let raw = json!({
        "request_id": "abc123",
        "status_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/status",
        "response_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/response",
        "cancel_url": "https://queue.fal.run/fal-ai/wan/requests/abc123/cancel"
    });
    let job = video_job_from_submit("fal-ai/wan", raw).unwrap();
    assert_eq!(job.id, "abc123");
    assert_eq!(job.provider, Provider::Fal);
    assert_eq!(job.provider_model.as_deref(), Some("fal-ai/wan"));
    assert!(job.status_url.as_deref().unwrap().ends_with("/status"));
    assert!(job.response_url.as_deref().unwrap().ends_with("/response"));
    assert!(job.cancel_url.as_deref().unwrap().ends_with("/cancel"));
}

/// Preserves queued and running video status mapping.
#[test]
fn maps_pending_statuses() {
    let queued = video_status_from_pending(json!({
        "status": "IN_QUEUE",
        "queue_position": 7
    }));
    assert!(matches!(
        queued,
        VideoJobStatus::Queued {
            queue_position: Some(7),
            ..
        }
    ));

    let running = video_status_from_pending(json!({"status": "IN_PROGRESS"}));
    assert!(matches!(running, VideoJobStatus::Running { .. }));
}
