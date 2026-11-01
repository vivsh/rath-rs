use super::*;
use crate::Provider;

struct Fake;
impl MusicBackend for Fake {
    async fn submit_music(&self, _: &MusicRequest) -> Result<AudioJob, RathError> {
        Ok(AudioJob::new("fake".into(), Provider::Fal))
    }
    async fn get_music(&self, _: &AudioJob) -> Result<AudioJobStatus, RathError> {
        Ok(AudioJobStatus::Running { raw_metadata: None })
    }
}

/// Concrete clients dispatch to independent injected backends.
#[tokio::test]
async fn injected_music_backend() {
    let client = MusicClient::from_backend(Fake);
    let receipt = client
        .submit_music(&MusicRequest::new("rain".into()))
        .await
        .unwrap();
    assert!(matches!(
        client.get_music(&receipt).await.unwrap(),
        AudioJobStatus::Running { .. }
    ));
}
