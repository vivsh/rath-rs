use super::*;
use crate::ErrorKind;
use crate::video::{VideoBackend, VideoJob, VideoJobStatus, VideoRequest, VideoResponse};

/// A downstream client implementing only the original required methods.
struct OriginalClient;

impl VideoBackend for OriginalClient {
    async fn submit_video(&self, _: &VideoRequest) -> Result<VideoJob, RathError> {
        Err(RathError::new(ErrorKind::UnsupportedCapability, "test"))
    }
    async fn get_video(&self, _: &str) -> Result<VideoJobStatus, RathError> {
        Err(RathError::new(ErrorKind::UnsupportedCapability, "test"))
    }
    async fn wait_video(&self, _: &str) -> Result<VideoResponse, RathError> {
        Err(RathError::new(ErrorKind::UnsupportedCapability, "test"))
    }
}

/// New webhook support does not force existing third-party clients to implement a parser.
#[tokio::test]
async fn default_webhook_method_is_unsupported() {
    let error = OriginalClient
        .parse_webhook(&http::HeaderMap::new(), b"{}")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
}

/// Existing prompt-only default construction retains the five-second polling default.
#[test]
fn defaults_remain_text_to_video() {
    assert!(matches!(
        VideoRequest::default().input,
        super::super::VideoInput::TextToVideo
    ));
    assert_eq!(
        VideoOptions::default().poll_interval,
        Duration::from_secs(5)
    );
}

/// A fake backend observes the typed lip-sync request and returns an ordinary video job.
#[tokio::test]
async fn lip_sync_uses_existing_backend() {
    use crate::Provider;
    use crate::audio::generation::AudioData;
    use crate::video::{LipSyncMode, LipSyncModel, VideoClient, VideoData, VideoInput};
    struct Fake;
    impl VideoBackend for Fake {
        async fn submit_video(&self, request: &VideoRequest) -> Result<VideoJob, RathError> {
            assert!(matches!(
                &request.input,
                VideoInput::LipSync {
                    sync_mode: LipSyncMode::Remap,
                    ..
                }
            ));
            Ok(VideoJob::new("job".into(), Provider::Fal))
        }
        async fn get_video(&self, _: &str) -> Result<VideoJobStatus, RathError> {
            Ok(VideoJobStatus::Running { raw_metadata: None })
        }
        async fn wait_video(&self, _: &str) -> Result<VideoResponse, RathError> {
            Err(RathError::new(ErrorKind::UnsupportedCapability, "test"))
        }
    }
    let client = VideoClient::from_backend(Fake);
    let request = VideoRequest::new(
        String::new(),
        VideoInput::LipSync {
            video: VideoData::Url {
                url: "https://example.com/video".into(),
            },
            audio: AudioData::Url {
                url: "https://example.com/audio".into(),
            },
            model: LipSyncModel::Pro,
            sync_mode: LipSyncMode::Remap,
        },
    );
    let job = client.submit_video(&request).await.unwrap();
    assert!(matches!(
        client.get_video(&job.id).await.unwrap(),
        VideoJobStatus::Running { .. }
    ));
}
