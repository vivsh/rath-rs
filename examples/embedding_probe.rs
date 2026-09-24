//! Opt-in inference probe. Normal tests never invoke this program or download models.
use rath::embeddings::{EmbedRequest, EmbeddingOptions, FastEmbedOptions};
use std::time::Instant;

/// Takes a model URL and, for FastEmbed, explicit cache and revision arguments.
#[tokio::main]
async fn main() -> Result<(), rath::RathError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let url = args.first().ok_or_else(|| {
        rath::RathError::new(
            rath::ErrorKind::Validation,
            "usage: embedding_probe MODEL_URL [CACHE REVISION [--download]]",
        )
    })?;
    let mut options = EmbeddingOptions::default();
    if url.starts_with("fastembed:") {
        let cache = args.get(1).ok_or_else(|| {
            rath::RathError::new(
                rath::ErrorKind::Validation,
                "FastEmbed requires an explicit cache path",
            )
        })?;
        let revision = args.get(2).ok_or_else(|| {
            rath::RathError::new(
                rath::ErrorKind::Validation,
                "FastEmbed requires an explicit revision",
            )
        })?;
        options = options.with_fastembed(
            FastEmbedOptions::new(cache)
                .with_revision(revision)
                .with_download(args.get(3).is_some_and(|v| v == "--download")),
        );
    }
    let started = Instant::now();
    let client = options.create(url).await?;
    let initialization = started.elapsed();
    let request = EmbedRequest::batch([
        "A cat sits on the mat.".into(),
        "A dog runs in the park.".into(),
    ]);
    let started = Instant::now();
    let result = client.embed(&request).await?;
    println!(
        "model={url} revision={:?} dimensions={:?} batch_size={} initialization_ms={} inference_ms={} usage={:?}",
        args.get(2),
        result.embeddings.first().map(Vec::len),
        request.input.len(),
        initialization.as_millis(),
        started.elapsed().as_millis(),
        result.usage
    );
    Ok(())
}
