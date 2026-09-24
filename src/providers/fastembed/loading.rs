use super::backend::normalize;
use crate::embeddings::FastEmbedOptions;
use crate::{ErrorKind, ModelUrl, RathError};
use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::{collections::BTreeMap, path::PathBuf};

/// Uses file-based FastEmbed initialization; upstream convenience download methods are never called.
pub(super) fn load(
    model: &EmbeddingModel,
    setup: &FastEmbedOptions,
    url: &ModelUrl,
) -> Result<TextEmbedding, RathError> {
    let info =
        TextEmbedding::get_model_info(model).map_err(|e| normalize(ErrorKind::Validation, &e))?;
    let repo = Repo::with_revision(
        info.model_code.clone(),
        RepoType::Model,
        setup.revision().into(),
    );
    let names: Vec<&str> = [
        info.model_file.as_str(),
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    .into_iter()
    .chain(info.additional_files.iter().map(String::as_str))
    .collect();
    let paths = artifacts(&repo, &names, setup, url)?;
    let definition = definition(model, &paths)?;
    TextEmbedding::try_new_from_user_defined(
        definition,
        InitOptionsUserDefined::new()
            .with_max_length(setup.max_length())
            .with_intra_threads(setup.intra_threads()),
    )
    .map_err(|e| normalize(ErrorKind::Provider, &e))
}

/// Looks up all required files locally first; offline failures never construct a network client.
fn artifacts(
    repo: &Repo,
    names: &[&str],
    setup: &FastEmbedOptions,
    url: &ModelUrl,
) -> Result<BTreeMap<String, PathBuf>, RathError> {
    let cache = Cache::new(setup.cache_dir().into()).repo(repo.clone());
    let mut paths = BTreeMap::new();
    let mut missing = Vec::new();
    for name in names {
        match cache.get(name) {
            Some(path) => {
                paths.insert((*name).into(), path);
            }
            None => missing.push(*name),
        }
    }
    if missing.is_empty() {
        return Ok(paths);
    }
    if !setup.download() {
        return Err(RathError::new(
            ErrorKind::Validation,
            format!(
                "offline FastEmbed cache is missing artifacts for revision {}: {}",
                setup.revision(),
                missing.join(", ")
            ),
        ));
    }
    let api = downloader(setup, url)?;
    let revision = resolve_revision(&api, repo, setup, missing[0])?;
    let pinned = Repo::with_revision(repo.url(), RepoType::Model, revision.clone());
    let api = api.repo(pinned);
    for name in missing {
        let path = api.get(name).map_err(|e| normalize(ErrorKind::Http, &e))?;
        paths.insert(name.into(), path);
    }
    cache
        .create_ref(&revision)
        .map_err(|e| normalize(ErrorKind::Other, &e))?;
    Ok(paths)
}

/// Prevents hf-hub's builder from reading the user's cached token before explicit credentials are set.
fn downloader(
    setup: &FastEmbedOptions,
    url: &ModelUrl,
) -> Result<hf_hub::api::sync::Api, RathError> {
    // hf-hub 0.5 always calls Cache::token in from_cache. A fresh private directory contains none.
    let empty = tempfile::tempdir().map_err(|e| normalize(ErrorKind::Other, &e))?;
    let mut builder = ApiBuilder::from_cache(Cache::new(empty.path().join("hub")))
        .with_cache_dir(setup.cache_dir().into())
        .with_token(url.api_key.clone())
        .with_retries(0)
        .with_progress(false);
    if let Some(endpoint) = &url.base_url {
        builder = builder.with_endpoint(endpoint.clone());
    }
    builder
        .build()
        .map_err(|e| normalize(ErrorKind::Transport, &e))
}

/// Pins partially cached and downloaded artifacts to one commit, never a moving branch per file.
fn resolve_revision(
    api: &hf_hub::api::sync::Api,
    repo: &Repo,
    setup: &FastEmbedOptions,
    file: &str,
) -> Result<String, RathError> {
    let path = setup
        .cache_dir()
        .join(repo.folder_name())
        .join("refs")
        .join(repo.revision());
    let revision = match std::fs::read_to_string(path) {
        Ok(value) => value.trim().to_owned(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => api
            .metadata(&api.repo(repo.clone()).url(file))
            .map_err(|e| normalize(ErrorKind::Http, &e))?
            .commit_hash()
            .to_owned(),
        Err(error) => return Err(normalize(ErrorKind::Other, &error)),
    };
    if revision.is_empty()
        || !revision
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err(RathError::new(
            ErrorKind::InvalidResponse,
            "invalid cached or provider-reported model revision",
        ));
    }
    Ok(revision)
}

#[cfg(test)]
#[path = "tests/loading.rs"]
mod tests;

/// Preserves upstream tokenizer, pooling, quantization and external initializer metadata.
fn definition(
    model: &EmbeddingModel,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<UserDefinedEmbeddingModel, RathError> {
    let info =
        TextEmbedding::get_model_info(model).map_err(|e| normalize(ErrorKind::Validation, &e))?;
    let read = |name: &str| -> Result<Vec<u8>, RathError> {
        let path = paths.get(name).ok_or_else(|| {
            RathError::new(
                ErrorKind::InvalidResponse,
                format!("missing resolved artifact {name}"),
            )
        })?;
        std::fs::read(path).map_err(|e| normalize(ErrorKind::Other, &e))
    };
    let files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    let mut definition = UserDefinedEmbeddingModel::new(read(&info.model_file)?, files)
        .with_quantization(TextEmbedding::get_quantization_mode(model));
    definition.pooling = TextEmbedding::get_default_pooling_method(model);
    definition.output_key = info.output_key.clone();
    let parent = std::path::Path::new(&info.model_file)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""));
    for name in &info.additional_files {
        let relative = std::path::Path::new(name)
            .strip_prefix(parent)
            .map_err(|e| normalize(ErrorKind::Validation, &e))?;
        definition = definition
            .with_external_initializer(relative.to_string_lossy().into_owned(), read(name)?);
    }
    Ok(definition)
}
