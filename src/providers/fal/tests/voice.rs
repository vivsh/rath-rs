use crate::audio::{
    tts::{TtsOptions, TtsRequest},
    voice::{Voice, VoiceData, VoiceSample},
    voice_clone::{VoiceCloneOptions, VoiceCloneRequest},
    voice_design::{VoiceDesignOptions, VoicePreview},
};
use crate::core::{ErrorKind, ModelUrl};
use serde_json::json;

fn url(model: &str) -> ModelUrl {
    let mut url = ModelUrl::parse(&format!("fal:///{model}")).unwrap();
    url.api_key = Some("secret".into());
    url.base_url = Some("http://127.0.0.1:1".into());
    url
}

fn embedding() -> Voice {
    Voice {
        scope: "fal/qwen3-tts-1.7b".into(),
        data: VoiceData::Embedding {
            format: "qwen3-tts-1.7b/safetensors".into(),
            data: vec![1],
            reference_text: Some("Hello".into()),
        },
    }
}

/// Voice scope, representation, language and ignored native instructions fail before network I/O.
#[tokio::test]
async fn unsupported_voice_controls_fail_locally() {
    let client = crate::providers::create_tts_client(
        &url("fal-ai/qwen-3-tts/text-to-speech/1.7b"),
        TtsOptions::default(),
    )
    .unwrap();
    for request in [
        TtsRequest {
            voice: Some(Voice::id("elevenlabs", "speaker")),
            ..Default::default()
        },
        TtsRequest {
            voice: Some(embedding()),
            instructions: Some("Whisper".into()),
            ..Default::default()
        },
        TtsRequest {
            voice: Some(embedding()),
            language: Some("hi".into()),
            ..Default::default()
        },
        TtsRequest {
            voice: Some(Voice {
                scope: "fal/qwen3-tts-1.7b".into(),
                data: VoiceData::ReferenceAudio {
                    samples: vec![VoiceSample {
                        mime_type: "audio/wav".into(),
                        data: vec![1],
                        transcript: None,
                    }],
                },
            }),
            ..Default::default()
        },
    ] {
        let error = client
            .synthesize_speech(&TtsRequest {
                input: "Hello".into(),
                ..request
            })
            .await
            .unwrap_err();
        assert!(matches!(
            error.kind(),
            ErrorKind::Validation | ErrorKind::UnsupportedCapability
        ));
    }
}

/// Separate factories reject design/clone endpoints on TTS and unsupported native registration.
#[tokio::test]
async fn capability_factories_are_independent() {
    for model in [
        "fal-ai/qwen-3-tts/voice-design/1.7b",
        "fal-ai/qwen-3-tts/clone-voice/1.7b",
    ] {
        assert!(crate::providers::create_tts_client(&url(model), TtsOptions::default()).is_err());
    }
    let client = crate::providers::create_voice_design_client(
        &url("fal-ai/qwen-3-tts/voice-design/1.7b"),
        VoiceDesignOptions::default(),
    )
    .unwrap();
    let preview = VoicePreview {
        sample: VoiceSample {
            mime_type: "audio/wav".into(),
            data: vec![1],
            transcript: Some("Hello".into()),
        },
        scope: "fal/qwen3-tts-1.7b".into(),
        registration_token: None,
    };
    assert_eq!(
        client
            .register_voice(&preview, "Nadia", None)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::UnsupportedCapability
    );
}

/// Qwen does not silently drop names, additional samples, absent transcripts or malformed native settings.
#[tokio::test]
async fn cloning_validates_all_typed_fields() {
    let client = crate::providers::create_voice_clone_client(
        &url("fal-ai/qwen-3-tts/clone-voice/1.7b"),
        VoiceCloneOptions::default(),
    )
    .unwrap();
    let sample = VoiceSample {
        mime_type: "audio/wav".into(),
        data: vec![1],
        transcript: Some("Hello".into()),
    };
    for request in [
        VoiceCloneRequest {
            samples: vec![sample.clone()],
            name: Some("Nadia".into()),
            ..Default::default()
        },
        VoiceCloneRequest {
            samples: vec![sample.clone(), sample.clone()],
            ..Default::default()
        },
        VoiceCloneRequest {
            samples: vec![VoiceSample {
                transcript: None,
                ..sample.clone()
            }],
            ..Default::default()
        },
        VoiceCloneRequest {
            samples: vec![sample],
            provider_config: Some(json!([])),
            ..Default::default()
        },
    ] {
        assert_eq!(
            client.clone_voice(&request).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
}
