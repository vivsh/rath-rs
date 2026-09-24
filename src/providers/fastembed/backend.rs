use super::loading;
use crate::embeddings::{
    EmbedRequest, EmbedResponse, EmbedTaskType, EmbeddingBackend, EmbeddingClient,
    EmbeddingOptions, contract,
};
use crate::{ErrorKind, ModelUrl, Provider, RathError};
use fastembed::{EmbeddingModel, TextEmbedding};
use std::sync::Arc;
use tokio::sync::Mutex;

struct FastEmbedBackend {
    session: Arc<Mutex<TextEmbedding>>,
    model: EmbeddingModel,
    batch_size: usize,
}

/// Prepares all runtime resources on a blocking worker before returning a client.
pub(super) async fn create(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<EmbeddingClient, RathError> {
    runtime()?;
    let setup = options.fastembed.ok_or_else(|| {
        contract::invalid(
            "FastEmbed requires typed FastEmbedOptions with an explicit cache directory",
        )
    })?;
    setup.validate()?;
    contract::config(options.provider_config.as_ref(), None, &[])?;
    let model: EmbeddingModel = url.model.parse().map_err(|error| {
        RathError::new(
            ErrorKind::Validation,
            format!("unknown FastEmbed model: {error}"),
        )
    })?;
    let url = url.clone();
    tokio::task::spawn_blocking(move || {
        let secrets = url.api_key.as_deref().into_iter().collect::<Vec<_>>();
        let session = loading::load(&model, &setup, &url).map_err(|e| e.sanitized(&secrets))?;
        Ok(EmbeddingClient::from_backend(FastEmbedBackend {
            session: Arc::new(Mutex::new(session)),
            model,
            batch_size: setup.batch_size(),
        }))
    })
    .await
    .map_err(|e| normalize(ErrorKind::Other, &e))?
}

impl EmbeddingBackend for FastEmbedBackend {
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        runtime()?;
        let input = prepare(&self.model, request)?;
        let truncate = request.truncate.unwrap_or(false);
        let batch_size = self.batch_size;
        let dimensions = TextEmbedding::get_model_info(&self.model)
            .map_err(|e| normalize(ErrorKind::Validation, &e))?
            .dim;
        let mut session = self.session.clone().lock_owned().await;
        let vectors = tokio::task::spawn_blocking(move || {
            validate_lengths(&session, &input, truncate)?;
            session
                .embed(input, Some(batch_size))
                .map_err(|e| normalize(ErrorKind::Provider, &e))
        })
        .await
        .map_err(|e| normalize(ErrorKind::Other, &e))??;
        if vectors.iter().any(|v| v.len() != dimensions) {
            return Err(RathError::new(
                ErrorKind::InvalidResponse,
                format!("loaded model did not return its native {dimensions} dimensions"),
            ));
        }
        contract::response(
            Provider::FastEmbed,
            serde_json::Value::Null,
            vectors,
            request,
            None,
        )
        .map(|mut response| {
            response.raw_metadata = None;
            response
        })
    }
}

/// Uses upstream model metadata and only the explicitly supported BGE retrieval convention.
fn prepare(model: &EmbeddingModel, request: &EmbedRequest) -> Result<Vec<String>, RathError> {
    contract::validate(request)?;
    contract::config(None, request.provider_config.as_ref(), &[])?;
    let info =
        TextEmbedding::get_model_info(model).map_err(|e| normalize(ErrorKind::Validation, &e))?;
    if request.title.is_some()
        || request
            .output_dimensionality
            .is_some_and(|d| d as usize != info.dim)
    {
        return Err(RathError::unsupported(
            Provider::FastEmbed,
            "titles or non-native embedding dimensions",
        ));
    }
    match &request.task_type {
        None => Ok(request.input.clone()),
        Some(EmbedTaskType::RetrievalDocument) if *model == EmbeddingModel::BGESmallENV15 => {
            Ok(request.input.clone())
        }
        Some(EmbedTaskType::RetrievalQuery) if *model == EmbeddingModel::BGESmallENV15 => {
            let instruction = request
                .instructions
                .as_deref()
                .unwrap_or("Represent this sentence for searching relevant passages:");
            Ok(request
                .input
                .iter()
                .map(|text| format!("{instruction} {text}"))
                .collect())
        }
        _ => Err(RathError::unsupported(
            Provider::FastEmbed,
            "requested retrieval task for this model",
        )),
    }
}

/// Measures the actual tokenizer including special tokens before inference can truncate anything.
fn validate_lengths(
    session: &TextEmbedding,
    input: &[String],
    truncate: bool,
) -> Result<(), RathError> {
    if truncate {
        return Ok(());
    }
    let limit = session
        .tokenizer
        .get_truncation()
        .map(|p| p.max_length)
        .ok_or_else(|| contract::invalid("missing loaded tokenizer length limit"))?;
    let mut tokenizer = session.tokenizer.clone();
    tokenizer
        .with_truncation(None)
        .map_err(|e| normalize(ErrorKind::Validation, e.as_ref()))?;
    tokenizer.with_padding(None);
    for text in input {
        let encoded = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| normalize(ErrorKind::Validation, e.as_ref()))?;
        if encoded.len() > limit {
            return Err(contract::invalid(format!(
                "embedding input has {} tokens, exceeding {limit}; explicitly enable truncate to permit truncation",
                encoded.len()
            )));
        }
    }
    Ok(())
}

/// Returns a typed error instead of allowing Tokio's blocking dispatch to panic.
fn runtime() -> Result<(), RathError> {
    tokio::runtime::Handle::try_current()
        .map(|_| ())
        .map_err(|e| normalize(ErrorKind::Validation, &e))
}

pub(super) fn normalize(kind: ErrorKind, error: &(dyn std::error::Error + 'static)) -> RathError {
    RathError::from_error(kind, error, &[]).with_context(Provider::FastEmbed, "local embeddings")
}

#[cfg(test)]
#[path = "tests/backend.rs"]
mod tests;
