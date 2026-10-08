use crate::audio::{
    stt::{SttOptions, SttRequest},
    tts::{TtsOptions, TtsRequest},
};
use crate::images::{ImageOptions, ImageRequest};
use crate::llm::LlmOptions;
use crate::providers::tests::http::serve;
use crate::video::{VideoInput, VideoOptions, VideoRequest};
use crate::{ErrorKind, Provider};
use serde_json::json;
use std::process::Command;

/// Credential resolution tests use child-process environments, never unsafe process-global mutations or locks.
#[test]
fn resolves_shared_gemini_credentials_in_isolated_processes() {
    if let Ok(case) = std::env::var("RATH_GEMINI_MEDIA_CREDENTIAL_CASE") {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(child(&case));
        return;
    }
    for case in ["default", "override", "missing", "missing_override"] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "providers::gemini::tests::media_credentials::resolves_shared_gemini_credentials_in_isolated_processes"])
            .env("RATH_GEMINI_MEDIA_CREDENTIAL_CASE", case)
            .env_remove("GEMINI_API_KEY").env_remove("RATH_GEMINI_MEDIA_TEST_KEY");
        if case != "missing" {
            command.env("GEMINI_API_KEY", "synthetic-default-credential");
        }
        if case == "override" {
            command.env(
                "RATH_GEMINI_MEDIA_TEST_KEY",
                "synthetic-override-credential",
            );
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "credential case {case} failed: {}",
            String::from_utf8_lossy(&output.stdout)
                .replace("synthetic-default-credential", "[REDACTED]")
                .replace("synthetic-override-credential", "[REDACTED]")
        );
    }
}

/// Exercises all public capability constructors and real transports with only the one Gemini environment key.
async fn child(case: &str) {
    if matches!(case, "missing" | "missing_override") {
        missing(case).await;
        return;
    }
    let suffix = if case == "override" {
        "&api_key_env=RATH_GEMINI_MEDIA_TEST_KEY"
    } else {
        ""
    };
    let expected = if case == "override" {
        "synthetic-override-credential"
    } else {
        "synthetic-default-credential"
    };
    // Construction of the existing chat adapter proves it uses the same resolver without provider I/O.
    let chat_suffix = suffix.replace('&', "?");
    assert!(
        LlmOptions::default()
            .create(&format!("gemini:///gemini-3.8-flash{chat_suffix}"))
            .await
            .is_ok()
    );
    transcription(suffix, expected).await;
    speech(suffix, expected).await;
    image(suffix, expected).await;
    video(suffix, expected).await;
}

/// Confirms video submission resolves the same default/overridden key through the public factory.
async fn video(suffix: &str, expected: &str) {
    let (base, requests) = serve(
        200,
        json!({"name":"models/veo-3.1-fast-generate-preview/operations/test-job"}),
    );
    let client = VideoOptions::default()
        .create(&format!(
            "gemini:///veo-3.1-fast-generate-preview?base_url={base}{suffix}"
        ))
        .await
        .unwrap();
    client
        .submit_video(&VideoRequest::new("A bird".into(), VideoInput::TextToVideo))
        .await
        .unwrap();
    assert!(
        requests
            .try_recv()
            .unwrap()
            .contains(&format!("x-goog-api-key: {expected}"))
    );
}

/// Confirms public transcription dispatch uses only the selected shared Gemini key.
async fn transcription(suffix: &str, expected: &str) {
    let transcript =
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":"Hello"}]}}]});
    let (base, requests) = serve(200, transcript);
    let stt = SttOptions::default()
        .create(&format!(
            "gemini:///gemini-3.8-flash?base_url={base}{suffix}"
        ))
        .await
        .unwrap();
    stt.transcribe_audio(&SttRequest::new("audio/wav".into(), vec![1]))
        .await
        .unwrap();
    assert!(
        requests
            .try_recv()
            .unwrap()
            .contains(&format!("x-goog-api-key: {expected}"))
    );
}

/// Confirms public speech dispatch uses only the selected shared Gemini key.
async fn speech(suffix: &str, expected: &str) {
    let speech = json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"inlineData":{"mimeType":"audio/l16", "data":"AQA="}}]}}]});
    let (base, requests) = serve(200, speech);
    let tts = TtsOptions::default()
        .create(&format!(
            "gemini:///gemini-2.5-flash-preview-tts?base_url={base}{suffix}"
        ))
        .await
        .unwrap();
    tts.synthesize_speech(&TtsRequest::new("Hello".into()))
        .await
        .unwrap();
    assert!(
        requests
            .try_recv()
            .unwrap()
            .contains(&format!("x-goog-api-key: {expected}"))
    );
}

/// Confirms image generation uses the same default or explicitly overridden key through public routing.
async fn image(suffix: &str, expected: &str) {
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9N8AAAAASUVORK5CYII=";
    let (base, requests) = serve(
        200,
        json!({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"inlineData":{"mimeType":"image/png", "data":png}}]}}]}),
    );
    let client = ImageOptions::default()
        .create(&format!(
            "gemini:///gemini-3-pro-image?base_url={base}{suffix}"
        ))
        .await
        .unwrap();
    client
        .generate_image(&ImageRequest::new("A bird".into()))
        .await
        .unwrap();
    assert!(
        requests
            .try_recv()
            .unwrap()
            .contains(&format!("x-goog-api-key: {expected}"))
    );
}

/// Missing overrides fail instead of silently substituting an available default Gemini credential.
async fn missing(case: &str) {
    let suffix = if case == "missing_override" {
        "?api_key_env=RATH_GEMINI_MEDIA_TEST_KEY"
    } else {
        ""
    };
    let kind = if case == "missing_override" {
        ErrorKind::InvalidUrl
    } else {
        ErrorKind::Validation
    };
    let errors = [
        SttOptions::default()
            .create(&format!("gemini:///gemini-3.8-flash{suffix}"))
            .await
            .err()
            .unwrap(),
        TtsOptions::default()
            .create(&format!("gemini:///gemini-2.5-flash-preview-tts{suffix}"))
            .await
            .err()
            .unwrap(),
        ImageOptions::default()
            .create(&format!("gemini:///gemini-3-pro-image{suffix}"))
            .await
            .err()
            .unwrap(),
        VideoOptions::default()
            .create(&format!("gemini:///veo-3.1-fast-generate-preview{suffix}"))
            .await
            .err()
            .unwrap(),
    ];
    for error in errors {
        assert_eq!(error.kind(), kind);
        if case == "missing" {
            assert_eq!(error.provider(), Some(&Provider::Gemini));
        }
        assert!(!format!("{error} {error:?}").contains("synthetic-default-credential"));
    }
}
