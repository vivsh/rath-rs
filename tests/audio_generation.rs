use rath::audio::generation::{AudioJob, AudioJobStatus};
use rath::audio::music::{MusicBackend, MusicClient, MusicOptions, MusicRequest};
use rath::audio::sound::{SoundBackend, SoundClient, SoundOptions, SoundRequest};
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ModelUrl, Provider, RathError};

struct Fake;
impl SoundBackend for Fake {
    async fn submit_sound(&self, _: &SoundRequest) -> Result<AudioJob, RathError> {
        Ok(
            AudioJob::new("sound".into(), Provider::External("audio-test".into()))
                .with_provider_model("sound-model".into()),
        )
    }
    async fn get_sound(&self, _: &AudioJob) -> Result<AudioJobStatus, RathError> {
        Ok(AudioJobStatus::Running { raw_metadata: None })
    }
}
impl MusicBackend for Fake {
    async fn submit_music(&self, _: &MusicRequest) -> Result<AudioJob, RathError> {
        Ok(
            AudioJob::new("music".into(), Provider::External("audio-test".into()))
                .with_provider_model("music-model".into()),
        )
    }
    async fn get_music(&self, _: &AudioJob) -> Result<AudioJobStatus, RathError> {
        Ok(AudioJobStatus::Running { raw_metadata: None })
    }
}
impl ProviderFactory for Fake {
    async fn sound(&self, _: &ModelUrl, _: SoundOptions) -> Result<SoundClient, RathError> {
        Ok(SoundClient::from_backend(Fake))
    }
    async fn music(&self, _: &ModelUrl, _: MusicOptions) -> Result<MusicClient, RathError> {
        Ok(MusicClient::from_backend(Fake))
    }
}

/// External factories route both capabilities and clients outlive their registry.
#[tokio::test]
async fn external_audio_factory() {
    let registry = ProviderRegistry::new()
        .register("audio-test", Fake)
        .unwrap();
    let sound = registry
        .sound("audio-test:///sound-model", SoundOptions::default())
        .await
        .unwrap();
    let music = registry
        .music("audio-test:///music-model", MusicOptions::default())
        .await
        .unwrap();
    drop(registry);
    let s = sound
        .submit_sound(&SoundRequest::new("rain".into()))
        .await
        .unwrap();
    let m = music
        .submit_music(&MusicRequest::new("jazz".into()))
        .await
        .unwrap();
    assert_eq!(s.id, "sound");
    assert_eq!(m.id, "music");
    assert!(matches!(
        sound.get_sound(&s).await.unwrap(),
        AudioJobStatus::Running { .. }
    ));
    assert!(matches!(
        music.get_music(&m).await.unwrap(),
        AudioJobStatus::Running { .. }
    ));
}
