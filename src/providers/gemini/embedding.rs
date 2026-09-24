use crate::core::error::http;
use crate::embeddings::{
    EmbedRequest, EmbedResponse, EmbeddingBackend, EmbeddingClient, EmbeddingOptions, contract,
};
use crate::{ModelUrl, Provider, RathError, TokenUsage};
use serde_json::{Value, json};

struct GeminiEmbeddingBackend {
    http: reqwest::Client,
    url: ModelUrl,
    options: EmbeddingOptions,
}

/// Constructs HTTP embedding support without losing native metadata through SDK decoding.
pub(crate) fn new_embedding_client(
    url: &ModelUrl,
    options: EmbeddingOptions,
) -> Result<EmbeddingClient, RathError> {
    contract::remote_options(&options, &[])?;
    let mut url = url.clone();
    url.api_key = Some(crate::llm::required_api_key(&url, "GEMINI_API_KEY")?);
    Ok(EmbeddingClient::from_backend(GeminiEmbeddingBackend {
        http: reqwest::Client::new(),
        url,
        options,
    }))
}
impl EmbeddingBackend for GeminiEmbeddingBackend {
    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
        validate_request(&self.options, request)?;
        let model = format!("models/{}", self.url.model.trim_start_matches("models/"));
        let endpoint = format!(
            "{}/{model}:batchEmbedContents",
            self.url
                .base_url
                .as_deref()
                .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
                .trim_end_matches('/')
        );
        let config = native_config(request)?;
        let requests: Vec<_> = request.input.iter().map(|text| json!({"model":model,"content":{"parts":[{"text":text}]},"embedContentConfig":config})).collect();
        let key = self.url.api_key.as_deref().unwrap_or_default();
        http::mapped(
            self.http
                .post(endpoint)
                .header("x-goog-api-key", key)
                .json(&json!({"requests":requests})),
            Provider::Gemini,
            "embeddings",
            &[key],
            |raw| decode_embeddings(raw, request),
        )
        .await
    }
}

/// Encodes typed controls using the current embedding configuration envelope.
fn native_config(request: &EmbedRequest) -> Result<Value, RathError> {
    let mut config = json!({});
    if let Some(task) = &request.task_type {
        let value = serde_json::to_value(task)
            .map_err(|e| RathError::from_error(crate::ErrorKind::Serialize, &e, &[]))?;
        config["taskType"] = json!(value.as_str().unwrap_or_default().to_uppercase());
    }
    if let Some(title) = &request.title {
        config["title"] = json!(title);
    }
    if let Some(d) = request.output_dimensionality {
        config["outputDimensionality"] = json!(d);
    }
    if let Some(t) = request.truncate {
        config["autoTruncate"] = json!(t);
    }
    Ok(config)
}

/// Rejects unsupported controls before preparing a batch request.
fn validate_request(options: &EmbeddingOptions, request: &EmbedRequest) -> Result<(), RathError> {
    contract::validate(request)?;
    contract::config(
        options.provider_config.as_ref(),
        request.provider_config.as_ref(),
        &[],
    )?;
    if request.instructions.is_some() {
        return Err(RathError::unsupported(
            Provider::Gemini,
            "embedding instructions",
        ));
    }
    if request
        .output_dimensionality
        .is_some_and(|n| n > i32::MAX as u32)
    {
        return Err(contract::invalid(
            "output_dimensionality exceeds Gemini's range",
        ));
    }
    Ok(())
}

/// Retains aggregate usage and raw provider evidence while checking every vector.
fn decode_embeddings(raw: Value, request: &EmbedRequest) -> Result<EmbedResponse, RathError> {
    let data = raw["embeddings"]
        .as_array()
        .ok_or_else(|| RathError::invalid("missing embeddings", &raw))?;
    let vectors = data
        .iter()
        .map(|v| contract::vector(&v["values"], &raw))
        .collect::<Result<Vec<_>, _>>()?;
    let usage = raw
        .pointer("/usageMetadata/promptTokenCount")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .map(|input| TokenUsage {
            input: Some(input),
            output: None,
        });
    contract::response(Provider::Gemini, raw, vectors, request, usage)
}
