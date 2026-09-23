//! Caller-owned voice identities and reference recordings; Rath never persists these values.
use crate::core::{ErrorKind, RathError};
use serde::{Deserialize, Serialize};
#[cfg(test)]
#[path = "tests/voice.rs"]
mod tests;

/// Encoded reference recording and, when known, its exact spoken transcript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceSample {
    /// Audio MIME type without parameters.
    pub mime_type: String,
    /// Complete encoded audio.
    pub data: Vec<u8>,
    /// Exact reference words; required only by adapters that need them.
    pub transcript: Option<String>,
}

/// Serializable voice identity. Credentials and persistence belong to the caller.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voice {
    /// Adapter-defined provider/model compatibility namespace, not a routing URL.
    pub scope: String,
    /// Native identity or reference material consumed by a compatible adapter.
    pub data: VoiceData,
}

/// Distinguishes provider-owned identities from caller-owned conditioning material.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum VoiceData {
    /// Preset or custom provider identifier; access may be account-specific.
    Id(String),
    /// Opaque speaker data, meaningful only to its compatible model family.
    Embedding {
        /// Adapter-defined format identifier, including a version where necessary.
        format: String,
        /// Complete embedding bytes.
        data: Vec<u8>,
        /// Exact original transcript when needed for conditioning.
        reference_text: Option<String>,
    },
    /// Recordings for models that accept reference audio directly.
    ReferenceAudio {
        /// One or more reference recordings; supported counts are adapter-specific.
        samples: Vec<VoiceSample>,
    },
}

impl Voice {
    /// Constructs an explicitly scoped preset or provider-owned identity.
    pub fn id(scope: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            scope: scope.into(),
            data: VoiceData::Id(id.into()),
        }
    }

    /// Rejects incompatible scopes and malformed voice data without network requests.
    pub fn validate(&self, expected_scope: &str) -> Result<(), RathError> {
        if self.scope.trim().is_empty() || self.scope != expected_scope {
            return Err(invalid(
                "voice is incompatible with the selected provider/model",
            ));
        }
        match &self.data {
            VoiceData::Id(id) if id.trim().is_empty() => Err(invalid("voice ID is empty")),
            VoiceData::Embedding {
                format,
                data,
                reference_text,
            } => {
                if format.trim().is_empty()
                    || data.is_empty()
                    || reference_text
                        .as_ref()
                        .is_some_and(|text| text.trim().is_empty())
                {
                    return Err(invalid("invalid speaker embedding or reference transcript"));
                }
                Ok(())
            }
            VoiceData::ReferenceAudio { samples } => validate_samples(samples),
            _ => Ok(()),
        }
    }
}

/// Validates nonempty encoded recordings; content decoding remains provider-owned.
pub(crate) fn validate_samples(samples: &[VoiceSample]) -> Result<(), RathError> {
    if samples.is_empty() {
        return Err(invalid("at least one voice sample is required"));
    }
    for sample in samples {
        let valid_mime = valid_audio_mime(&sample.mime_type);
        if sample.data.is_empty()
            || !valid_mime
            || sample
                .transcript
                .as_ref()
                .is_some_and(|text| text.trim().is_empty())
        {
            return Err(invalid(
                "voice samples require nonempty audio and a valid audio MIME type",
            ));
        }
    }
    Ok(())
}

/// Accepts a normalized, parameter-free audio MIME token.
pub(crate) fn valid_audio_mime(mime: &str) -> bool {
    mime.strip_prefix("audio/").is_some_and(|subtype| {
        !subtype.is_empty()
            && subtype
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$&^_.+-".contains(&b))
    })
}

pub(crate) fn invalid(message: &str) -> RathError {
    RathError::new(ErrorKind::Validation, message)
}

/// Validates object-shaped provider configuration before any paid operation.
pub(crate) fn validate_config(config: Option<&serde_json::Value>) -> Result<(), RathError> {
    if config.is_some_and(|value| !value.is_object()) {
        return Err(invalid("provider_config must be an object"));
    }
    Ok(())
}
