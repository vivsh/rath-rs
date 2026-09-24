use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::video::VideoBackend;
use rath::video::*;
use rath::{ErrorBody, ErrorKind, ModelUrl, Provider, RathError};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

struct VideoFixture {
    submissions: Arc<AtomicUsize>,
}
struct Client {
    url: ModelUrl,
    submissions: Arc<AtomicUsize>,
    interval: Duration,
}

impl ProviderFactory for VideoFixture {
    async fn video(&self, url: &ModelUrl, options: VideoOptions) -> Result<VideoClient, RathError> {
        Ok(VideoClient::from_backend(Client {
            url: url.clone(),
            submissions: self.submissions.clone(),
            interval: options.poll_interval,
        }))
    }
}

impl VideoBackend for Client {
    async fn submit_video(&self, _: &VideoRequest) -> Result<VideoJob, RathError> {
        self.submissions.fetch_add(1, Ordering::SeqCst);
        Ok({
            let mut value = VideoJob::new("job-1".into(), self.url.provider().clone());
            value.provider_model = Some(self.url.model().to_owned());
            value.status_url = None;
            value.response_url = None;
            value.cancel_url = None;
            value.raw_metadata = None;
            value
        })
    }
    async fn get_video(&self, job_id: &str) -> Result<VideoJobStatus, RathError> {
        assert_eq!(job_id, "job-1");
        match self.url.model() {
            "pending" => Ok(VideoJobStatus::Running { raw_metadata: None }),
            "failed" => Ok(VideoJobStatus::Failed {
                message: "render rejected".into(),
                raw_metadata: Some(json!({"job":job_id,"detail":"PRIVATE-VIDEO"})),
            }),
            _ => Ok(VideoJobStatus::Succeeded {
                response: {
                    let mut value = VideoResponse::new(vec![VideoData::Url {
                        url: "https://example.test/clip.mp4".into(),
                    }]);
                    value.raw_metadata = None;
                    value
                },
            }),
        }
    }
    /// Retains failed-job evidence and polls without taking ownership of application job state.
    async fn wait_video(&self, job_id: &str) -> Result<VideoResponse, RathError> {
        loop {
            match self.get_video(job_id).await? {
                VideoJobStatus::Succeeded { response } => return Ok(response),
                VideoJobStatus::Failed {
                    message,
                    raw_metadata,
                } => {
                    return Err(RathError::new(ErrorKind::Provider, message)
                        .with_context(self.url.provider().clone(), "video wait")
                        .with_body(
                            ErrorBody::Complete(serde_json::to_vec(&raw_metadata).unwrap()),
                            &[],
                        ));
                }
                _ => tokio::time::sleep(self.interval).await,
            }
        }
    }
}

fn registry(count: &Arc<AtomicUsize>) -> ProviderRegistry {
    ProviderRegistry::new()
        .register(
            "acme",
            VideoFixture {
                submissions: count.clone(),
            },
        )
        .unwrap()
}

/// Submission receipts round-trip external identity; polling and webhook defaults require no wrapper.
#[tokio::test]
async fn external_video_identity_and_lifecycle() {
    let count = Arc::new(AtomicUsize::new(0));
    let registry = registry(&count);
    let client = registry
        .video("acme:///clip", VideoOptions::default())
        .await
        .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(registry);
    let job = client.submit_video(&VideoRequest::default()).await.unwrap();
    assert_eq!(job.provider, Provider::External("acme".into()));
    let value = serde_json::to_value(&job).unwrap();
    assert_eq!(value["provider"], json!({"External":"acme"}));
    let restored: VideoJob = serde_json::from_value(value).unwrap();
    assert_eq!(restored.provider, job.provider);
    assert!(matches!(
        client.get_video(&restored.id).await.unwrap(),
        VideoJobStatus::Succeeded { .. }
    ));
    assert_eq!(
        client.wait_video(&restored.id).await.unwrap().videos.len(),
        1
    );
    let error = client
        .parse_webhook(&http::HeaderMap::new(), b"unverified")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

/// Generate submits once for success, provider failure and cancelled local waiting.
#[tokio::test]
async fn video_generation_never_resubmits() {
    for model in ["clip", "failed", "pending"] {
        let count = Arc::new(AtomicUsize::new(0));
        let client = registry(&count)
            .video(&format!("acme:///{model}"), {
                let mut value = VideoOptions::default();
                value.poll_interval = Duration::from_millis(1);
                value
            })
            .await
            .unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(30),
            client.generate_video(&VideoRequest::default()),
        )
        .await;
        match model {
            "pending" => assert!(result.is_err()),
            "failed" => {
                let error = result.unwrap().unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Provider);
                assert!(error.message().contains("render rejected"));
                assert!(
                    String::from_utf8_lossy(error.response_body().unwrap().bytes())
                        .contains("PRIVATE-VIDEO")
                );
                assert!(!format!("{error} {error:?}").contains("PRIVATE-VIDEO"));
            }
            _ => assert_eq!(result.unwrap().unwrap().videos.len(), 1),
        }
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
