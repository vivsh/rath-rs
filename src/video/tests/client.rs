use super::*;

/// A downstream client implementing only the original required methods.
struct OriginalClient;

#[async_trait]
impl VideoClient for OriginalClient {
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
