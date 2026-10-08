use super::*;
use crate::{
    ErrorKind,
    audio::voice::{Voice, VoiceData},
    providers::tests::http::serve,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

/// Constructs the real built-in adapter against a one-request loopback server.
fn client(base: &str, model: &str, options: TtsOptions) -> TtsClient {
    let mut url = ModelUrl::parse(&format!("gemini:///{model}")).unwrap();
    url.api_key = Some("gemini-audio-test-key".into());
    url.base_url = Some(base.into());
    crate::providers::create_tts_client(&url, options).unwrap()
}

fn reply(parts: Value) -> Value {
    json!({"candidates":[{"finishReason":"STOP", "content":{"parts":parts}}]})
}

fn blob(mime: &str, data: &[u8]) -> Value {
    json!({"inlineData":{"mimeType":mime, "data":STANDARD.encode(data)}})
}

/// Actual speech dispatch normalizes HTTP/provider errors and refuses truncated candidate audio.
#[tokio::test]
async fn normalizes_speech_failures_without_retrying() {
    for status in [200, 302, 401, 429, 500] {
        let (base, requests) = serve(
            status,
            json!({"error":{"code":status, "message":"gemini-audio-test-key rejected"}}),
        );
        let error = client(&base, "gemini-3.8-flash-tts", TtsOptions::default())
            .synthesize_speech(&TtsRequest::new("Hello".into()))
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            if status == 200 {
                ErrorKind::Provider
            } else {
                ErrorKind::Http
            }
        );
        assert_eq!(error.http_status(), Some(status));
        assert!(!format!("{error} {error:?}").contains("gemini-audio-test-key"));
        assert!(
            !String::from_utf8_lossy(error.response_body().unwrap().bytes())
                .contains("gemini-audio-test-key")
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
    let mut raw = reply(json!([blob("audio/l16", &[1, 0])]));
    raw["candidates"][0]["finishReason"] = json!("MAX_TOKENS");
    let (base, requests) = serve(200, raw);
    let error = client(&base, "gemini-3.8-flash-tts", TtsOptions::default())
        .synthesize_speech(&TtsRequest::new("Hello".into()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::OutputLimitReached);
    assert_eq!(requests.try_iter().count(), 1);
}

/// Supported TTS models route through generateContent and wrap controlled PCM in a valid WAV.
#[tokio::test]
async fn routes_speech_models_and_wraps_little_endian_pcm() {
    for model in [
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-pro-preview-tts",
        "gemini-3.1-flash-tts-preview",
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
    ] {
        let samples = [0, 128, 255, 127];
        let raw = reply(json!([blob("audio/L16;codec=pcm;rate=24000", &samples)]));
        let (base, requests) = serve(200, raw.clone());
        let response = client(&base, model, TtsOptions::default())
            .synthesize_speech(
                &TtsRequest::new("नमस्ते".into())
                    .with_voice(Voice::id("gemini", "Puck"))
                    .with_language("hi-IN".into())
                    .with_instructions("Speak warmly".into()),
            )
            .await
            .unwrap();
        assert_wav(&response, &samples);
        assert_eq!(response.raw_metadata, Some(raw));
        let wire = requests.try_recv().unwrap();
        assert!(wire.starts_with(&format!("POST /models/{model}:generateContent ")));
        let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_payload(&payload, model);
    }
}

/// Checks every RIFF/PCM format field and preserves sample byte order.
fn assert_wav(response: &TtsResponse, samples: &[u8]) {
    assert_eq!(response.mime_type, "audio/wav");
    assert_eq!(response.data.len(), 48);
    assert_eq!(&response.data[..4], b"RIFF");
    assert_eq!(&response.data[4..8], &40u32.to_le_bytes());
    assert_eq!(&response.data[8..16], b"WAVEfmt ");
    assert_eq!(&response.data[20..24], &[1, 0, 1, 0]);
    assert_eq!(&response.data[24..28], &24000u32.to_le_bytes());
    assert_eq!(&response.data[28..32], &48000u32.to_le_bytes());
    assert_eq!(&response.data[32..36], &[2, 0, 16, 0]);
    assert_eq!(&response.data[36..40], b"data");
    assert_eq!(&response.data[40..44], &4u32.to_le_bytes());
    assert_eq!(&response.data[44..], samples);
}

/// Verifies model-specific directions, voice and output format on the actual wire payload.
fn assert_payload(payload: &Value, model: &str) {
    assert_eq!(
        payload["generationConfig"]["responseModalities"],
        json!(["AUDIO"])
    );
    assert_eq!(
        payload["generationConfig"]["speechConfig"]["languageCode"],
        "hi-IN"
    );
    if model.starts_with("gemini-3.8-") {
        assert_eq!(
            payload["generationConfig"]["speechConfig"]["voiceConfig"]["voice"],
            "Puck"
        );
        assert_eq!(payload["contents"][0]["parts"][0]["text"], "नमस्ते");
        assert_eq!(
            payload["contents"][0]["parts"][0]["speechMetadata"]["style"],
            "Speak warmly"
        );
        assert_eq!(
            payload["generationConfig"]["responseFormat"]["audio"],
            json!({"mimeType":"AUDIO_L16", "sampleRate":24000})
        );
    } else {
        assert_eq!(
            payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Puck"
        );
        assert!(
            payload["contents"][0]["parts"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Speak warmly")
        );
        assert!(payload["generationConfig"].get("responseFormat").is_none());
    }
}

/// Native defaults yield to request controls and typed voice/language while callers remain unchanged.
#[tokio::test]
async fn merges_configuration_with_typed_precedence() {
    let (base, requests) = serve(200, reply(json!([blob("audio/l16", &[1, 0])])));
    let options = TtsOptions::default().with_provider_config(json!({"generationConfig":{"temperature":0.1, "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}}}));
    let request = TtsRequest::new("Hello".into()).with_voice(Voice::id("gemini", "Puck"))
        .with_language("en-IN".into()).with_format("pcm".into())
        .with_provider_config(json!({"generationConfig":{"temperature":0.8, "seed":7, "speechConfig":{"languageCode":"hi-IN"}}}));
    let original = request.clone();
    let response = client(&base, "gemini-2.5-flash-preview-tts", options)
        .synthesize_speech(&request)
        .await
        .unwrap();
    assert_eq!(response.data, [1, 0]);
    assert_eq!(response.mime_type, "audio/L16;codec=pcm;rate=24000");
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    let wire = requests.try_recv().unwrap();
    let payload: Value = serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["generationConfig"]["temperature"], 0.8);
    assert_eq!(payload["generationConfig"]["seed"], 7);
    assert_eq!(
        payload["generationConfig"]["speechConfig"]["languageCode"],
        "en-IN"
    );
    assert_eq!(
        payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
        "Puck"
    );
}

/// Multiple raw PCM parts join in order, and complete WAV data is never wrapped a second time.
#[tokio::test]
async fn combines_pcm_parts_and_preserves_existing_wav() {
    let (base, _requests) = serve(
        200,
        reply(json!([
            blob("audio/l16", &[1, 0]),
            blob("audio/L16;rate=24000", &[2, 0])
        ])),
    );
    let response = client(&base, "gemini-2.5-flash-preview-tts", TtsOptions::default())
        .synthesize_speech(&TtsRequest::new("Hello".into()))
        .await
        .unwrap();
    assert_eq!(&response.data[44..], &[1, 0, 2, 0]);
    let wav = response.data;
    let (base, _requests) = serve(200, reply(json!([blob("audio/wav", &wav)])));
    let response = client(&base, "gemini-3.8-flash-tts", TtsOptions::default())
        .synthesize_speech(&TtsRequest::new("Hello".into()))
        .await
        .unwrap();
    assert_eq!(response.mime_type, "audio/wav");
    assert_eq!(response.data, wav);
}

/// Unsupported model/voice/format/language and unknown or malformed native options fail locally.
#[tokio::test]
async fn validates_supported_controls_without_network() {
    let (base, requests) = serve(200, json!({}));
    let client = client(&base, "gemini-2.5-flash-preview-tts", TtsOptions::default());
    for request in [
        TtsRequest::new(" ".into()),
        TtsRequest::new("Hello".into()).with_format("mp3".into()),
        TtsRequest::new("Hello".into()).with_language("hi".into()),
        TtsRequest::new("Hello".into()).with_instructions(" ".into()),
        TtsRequest::new("Hello".into()).with_voice(Voice::id("openai", "Kore")),
        TtsRequest::new("Hello".into()).with_voice(Voice::id("gemini", "unknown")),
        TtsRequest::new("Hello".into()).with_voice(Voice::new(
            "gemini".into(),
            VoiceData::ReferenceAudio { samples: vec![] },
        )),
        TtsRequest::new("Hello".into())
            .with_provider_config(json!({"generationConfig":{"responseModalities":["TEXT"]}})),
        TtsRequest::new("Hello".into())
            .with_provider_config(json!({"generationConfig":{"temperature":"hot"}})),
        TtsRequest::new("Hello".into()).with_provider_config(
            json!({"generationConfig":{"speechConfig":{"multiSpeakerVoiceConfig":{}}}}),
        ),
        TtsRequest::new("Hello".into())
            .with_provider_config(json!({"headers":{"authorization":"secret"}})),
    ] {
        assert_eq!(
            client.synthesize_speech(&request).await.unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
    assert_eq!(requests.try_iter().count(), 0);
    let mut url = ModelUrl::parse("gemini:///gemini-2.5-flash").unwrap();
    url.api_key = Some("synthetic".into());
    assert_eq!(
        new_tts_client(&url, TtsOptions::default())
            .err()
            .unwrap()
            .kind(),
        ErrorKind::UnsupportedCapability
    );
    url.model = "gemini-2.5-flash-preview-tts".into();
    for config in [
        json!([]),
        json!({"safetySettings":{}}),
        json!({"generationConfig":{"topP":2}}),
    ] {
        assert!(new_tts_client(&url, TtsOptions::default().with_provider_config(config)).is_err());
    }
}

/// Empty/base64/MIME/container failures are rejected with original response evidence intact.
#[tokio::test]
async fn rejects_malformed_speech_output() {
    for parts in [
        json!([]),
        json!([{"text":"not speech"}]),
        json!([{"inlineData":{"data":"AAA="}}]),
        json!([blob("audio/l16", &[])]),
        json!([blob("audio/l16", &[1])]),
        json!([{"inlineData":{"mimeType":"audio/l16", "data":"not base64!"}}]),
        json!([blob("text/plain", &[1, 0])]),
        json!([blob("audio/l16;rate=16000", &[1, 0])]),
        json!([blob("audio/l16;channels=2", &[1, 0])]),
        json!([blob("audio/l16;codec=mp3", &[1, 0])]),
        json!([blob("audio/wav", b"invalid WAV")]),
        json!([blob("audio/l16", &[1, 0]), blob("audio/wav", &[1, 0])]),
    ] {
        let raw = reply(parts);
        let (base, requests) = serve(200, raw.clone());
        let error = client(&base, "gemini-2.5-flash-preview-tts", TtsOptions::default())
            .synthesize_speech(&TtsRequest::new("Hello".into()))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidResponse);
        assert_eq!(error.operation(), Some("text-to-speech"));
        assert_eq!(error.http_status(), Some(200));
        assert_eq!(
            serde_json::from_slice::<Value>(error.response_body().unwrap().bytes()).unwrap(),
            raw
        );
        assert_eq!(requests.try_iter().count(), 1);
    }
}
