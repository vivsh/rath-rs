use super::*;

/// All supported representations serialize independently of storage and retain exact material.
#[test]
fn voice_data_round_trips_and_checks_scope() {
    for data in [
        VoiceData::Id("speaker".into()),
        VoiceData::Embedding {
            format: "model/v1".into(),
            data: vec![0, 255, 1],
            reference_text: Some("Hello".into()),
        },
        VoiceData::ReferenceAudio {
            samples: vec![VoiceSample {
                mime_type: "audio/wav".into(),
                data: vec![1],
                transcript: None,
            }],
        },
    ] {
        let voice = Voice {
            scope: "provider/model".into(),
            data,
        };
        voice.validate("provider/model").unwrap();
        assert_eq!(
            voice.validate("other/model").unwrap_err().kind(),
            ErrorKind::Validation
        );
        let restored: Voice = serde_json::from_slice(&serde_json::to_vec(&voice).unwrap()).unwrap();
        assert_eq!(restored, voice);
    }
}

/// Empty audio, invalid MIME parameters and blank transcripts are rejected before submission.
#[test]
fn reference_recordings_require_valid_audio_metadata() {
    assert!(validate_samples(&[]).is_err());
    for (mime, data, transcript) in [
        ("audio/wav", vec![], None),
        ("text/plain", vec![1], None),
        ("audio/wav;code=1", vec![1], None),
        ("audio/wav", vec![1], Some(" ".into())),
    ] {
        assert!(
            validate_samples(&[VoiceSample {
                mime_type: mime.into(),
                data,
                transcript
            }])
            .is_err()
        );
    }
}
