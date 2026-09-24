use rath::audio::stt::SttBackend;
use rath::audio::voice_clone::VoiceCloneBackend;
use rath::audio::voice_design::VoiceDesignBackend;
use rath::audio::{stt::*, voice::*, voice_clone::*, voice_design::*};
use rath::images::ImageBackend;
use rath::images::*;
use rath::registry::{ProviderFactory, ProviderRegistry};
use rath::{ModelUrl, RathError};
use serde_json::{Value, json};

struct MediaFixture;
struct Client {
    url: ModelUrl,
    config: Option<Value>,
}

impl ProviderFactory for MediaFixture {
    async fn image(&self, url: &ModelUrl, options: ImageOptions) -> Result<ImageClient, RathError> {
        Ok(ImageClient::from_backend(Client {
            url: url.clone(),
            config: options.provider_config,
        }))
    }
    async fn stt(&self, url: &ModelUrl, options: SttOptions) -> Result<SttClient, RathError> {
        Ok(SttClient::from_backend(Client {
            url: url.clone(),
            config: options.provider_config,
        }))
    }
    async fn voice_design(
        &self,
        url: &ModelUrl,
        options: VoiceDesignOptions,
    ) -> Result<VoiceDesignClient, RathError> {
        Ok(VoiceDesignClient::from_backend(Client {
            url: url.clone(),
            config: options.provider_config,
        }))
    }
    async fn voice_clone(
        &self,
        url: &ModelUrl,
        options: VoiceCloneOptions,
    ) -> Result<VoiceCloneClient, RathError> {
        Ok(VoiceCloneClient::from_backend(Client {
            url: url.clone(),
            config: options.provider_config,
        }))
    }
}

impl ImageBackend for Client {
    async fn generate_image(&self, request: &ImageRequest) -> Result<ImageResponse, RathError> {
        Ok({
            let mut value = ImageResponse::new(vec![ImageData::Url {
                url: "https://example.test/image.png".into(),
            }]);
            value.raw_metadata = Some(
                json!({"model":self.url.model(),"config":self.config,"prompt":request.prompt}),
            );
            value
        })
    }
}

impl SttBackend for Client {
    async fn transcribe_audio(&self, request: &SttRequest) -> Result<SttResponse, RathError> {
        Ok({
            let mut value = SttResponse::new("hello".into());
            value.raw_metadata = Some(json!({"model":self.url.model(),
            "config":self.config,"bytes":request.data.len()}));
            value
        })
    }
}

impl VoiceDesignBackend for Client {
    async fn design_voice(
        &self,
        request: &VoiceDesignRequest,
    ) -> Result<VoiceDesignResponse, RathError> {
        Ok({
            let mut value = VoiceDesignResponse::new(vec![{
                let mut value =
                    VoicePreview::new(sample(&request.text), self.url.provider().as_str().into());
                value.registration_token = None;
                value
            }]);
            value.raw_metadata = Some(json!({"model":self.url.model(),"config":self.config}));
            value
        })
    }
}

impl VoiceCloneBackend for Client {
    async fn clone_voice(&self, request: &VoiceCloneRequest) -> Result<Voice, RathError> {
        Ok({
            Voice::new(
                self.url.provider().as_str().into(),
                VoiceData::ReferenceAudio {
                    samples: request.samples.clone(),
                },
            )
        })
    }
}

fn sample(text: &str) -> VoiceSample {
    VoiceSample {
        mime_type: "audio/wav".into(),
        data: vec![1, 2],
        transcript: Some(text.into()),
    }
}

/// Images and transcription use existing public types and preserve constructor options exactly.
#[tokio::test]
async fn image_and_stt_extensions_preserve_options() {
    let registry = ProviderRegistry::new()
        .register("acme", MediaFixture)
        .unwrap();
    let config = Some(json!({"native":"option"}));
    let image = registry
        .image("acme:///image-model", {
            let mut value = ImageOptions::default();
            value.provider_config = config.clone();
            value
        })
        .await
        .unwrap();
    let stt = registry
        .stt("acme:///stt-model", {
            let mut value = SttOptions::default();
            value.provider_config = config.clone();
            value
        })
        .await
        .unwrap();
    drop(registry);
    let result = image
        .generate_image(&{
            let mut value = ImageRequest::default();
            value.prompt = "hello".into();
            value
        })
        .await
        .unwrap();
    assert_eq!(
        result.raw_metadata.unwrap(),
        json!({"model":"image-model","config":config,"prompt":"hello"})
    );
    assert_eq!(result.images.len(), 1);
    let result = stt
        .transcribe_audio(&{
            let mut value = SttRequest::new("audio/wav".into(), vec![1, 2, 3]);
            value.provider_config = None;
            value
        })
        .await
        .unwrap();
    assert_eq!(result.text, "hello");
    assert_eq!(
        result.raw_metadata.unwrap(),
        json!({"model":"stt-model","config":config,"bytes":3})
    );
}

/// Voice design produces previews only, while explicit cloning returns the existing scoped voice type.
#[tokio::test]
async fn voice_extensions_keep_preview_and_identity_contracts() {
    let registry = ProviderRegistry::new()
        .register("acme", MediaFixture)
        .unwrap();
    let design = registry
        .voice_design("acme:///design", {
            let mut value = VoiceDesignOptions::default();
            value.provider_config = Some(json!({"candidate_count":1}));
            value
        })
        .await
        .unwrap();
    let clone = registry
        .voice_clone("acme:///clone", VoiceCloneOptions::default())
        .await
        .unwrap();
    let result = design
        .design_voice(&{
            let mut value = VoiceDesignRequest::default();
            value.text = "hello".into();
            value
        })
        .await
        .unwrap();
    assert_eq!(
        result.raw_metadata.unwrap()["config"],
        json!({"candidate_count":1})
    );
    assert!(result.previews[0].registration_token.is_none());
    let error = design
        .register_voice(&result.previews[0], "name", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), rath::ErrorKind::UnsupportedCapability);
    let voice = clone
        .clone_voice(&{
            let mut value = VoiceCloneRequest::default();
            value.samples = vec![result.previews[0].sample.clone()];
            value
        })
        .await
        .unwrap();
    voice.validate("acme").unwrap();
    assert!(matches!(voice.data, VoiceData::ReferenceAudio { samples } if samples.len()==1));
}
