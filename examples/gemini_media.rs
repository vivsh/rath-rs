//! Shared Gemini credentials for chat, transcription, synthesis and image generation.
//! Running this example only constructs clients; no provider generation is performed.
use rath::RathError;
use rath::audio::{
    stt::{SttOptions, SttRequest},
    tts::{TtsOptions, TtsRequest},
    voice::Voice,
};
use rath::images::{ImageOptions, ImageRequest};
use rath::llm::LlmOptions;

/// Constructs independent model-bound clients using the same GEMINI_API_KEY environment default.
#[tokio::main]
async fn main() -> Result<(), RathError> {
    let _chat = LlmOptions::default()
        .create("gemini:///gemini-3.8-flash")
        .await?;
    let _stt = SttOptions::default()
        .create("gemini:///gemini-3.8-flash")
        .await?;
    let _tts = TtsOptions::default()
        .create("gemini:///gemini-3.8-flash-tts")
        .await?;
    let _images = ImageOptions::default()
        .create("gemini:///gemini-3.1-flash-image")
        .await?;
    println!("Constructed Gemini clients; no generation requested.");
    Ok(())
}

/// Demonstrates paid operations for consumer integration; main deliberately does not invoke this.
/// Requires encoded audio with the supplied MIME type and explicit permission to generate remotely.
pub async fn generate_media(mime: String, audio: Vec<u8>) -> Result<(), RathError> {
    let stt = SttOptions::default()
        .create("gemini:///gemini-3.8-flash")
        .await?;
    let transcript = stt.transcribe_audio(&SttRequest::new(mime, audio)).await?;
    let tts = TtsOptions::default()
        .create("gemini:///gemini-3.8-flash-tts")
        .await?;
    let speech = tts
        .synthesize_speech(
            &TtsRequest::new(transcript.text)
                .with_voice(Voice::id("gemini", "Kore"))
                .with_language("en-IN".into())
                .with_instructions("Speak warmly".into()),
        )
        .await?;
    let images = ImageOptions::default()
        .create("gemini:///gemini-3.1-flash-image")
        .await?;
    let image = images
        .generate_image(
            &ImageRequest::new("A blue bird in a quiet garden".into()).with_size("1K".into()),
        )
        .await?;
    println!(
        "{} bytes of {}, {} image(s)",
        speech.data.len(),
        speech.mime_type,
        image.images.len()
    );
    Ok(())
}
