use super::*;
use crate::Provider;

struct Fake;
impl SoundBackend for Fake {
    async fn submit_sound(&self, _: &SoundRequest) -> Result<AudioJob, RathError> {
        Ok(AudioJob::new("fake".into(), Provider::Fal))
    }
    async fn get_sound(&self, _: &AudioJob) -> Result<AudioJobStatus, RathError> {
        Ok(AudioJobStatus::Running { raw_metadata: None })
    }
}

/// Concrete clients dispatch to independent injected backends.
#[tokio::test]
async fn injected_sound_backend() {
    let client = SoundClient::from_backend(Fake);
    let receipt = client
        .submit_sound(&SoundRequest::new("rain".into()))
        .await
        .unwrap();
    assert!(matches!(
        client.get_sound(&receipt).await.unwrap(),
        AudioJobStatus::Running { .. }
    ));
}
