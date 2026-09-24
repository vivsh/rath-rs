use crate::{ErrorKind, RathError};
use std::path::{Path, PathBuf};
/// Explicit local model preparation; available even when inference is not compiled in.
#[derive(Debug, Clone)]
pub struct FastEmbedOptions {
    cache_dir: PathBuf,
    download: bool,
    revision: String,
    max_length: usize,
    intra_threads: usize,
    batch_size: usize,
}
impl FastEmbedOptions {
    /// Selects the application-owned artifact directory; downloads default to disabled.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            download: false,
            revision: "main".into(),
            max_length: 512,
            intra_threads: 1,
            batch_size: 32,
        }
    }
    /// Borrows the explicitly selected artifact directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
    /// Sets download; validated during construction.
    pub fn with_download(mut self, value: bool) -> Self {
        self.download = value;
        self
    }
    /// Returns download.
    pub fn download(&self) -> bool {
        self.download
    }
    /// Sets revision; validated during construction.
    pub fn with_revision(mut self, value: impl Into<String>) -> Self {
        self.revision = value.into();
        self
    }
    /// Returns revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }
    /// Sets max length; validated during construction.
    pub fn with_max_length(mut self, value: usize) -> Self {
        self.max_length = value;
        self
    }
    /// Returns max length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }
    /// Sets intra threads; validated during construction.
    pub fn with_intra_threads(mut self, value: usize) -> Self {
        self.intra_threads = value;
        self
    }
    /// Returns intra threads.
    pub fn intra_threads(&self) -> usize {
        self.intra_threads
    }
    /// Sets batch size; validated during construction.
    pub fn with_batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }
    /// Returns batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    /// Rejects malformed setup before filesystem or network operations.
    pub(crate) fn validate(&self) -> Result<(), RathError> {
        if self.cache_dir.as_os_str().is_empty()
            || self.revision.trim().is_empty()
            || std::path::Path::new(&self.revision)
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
            || self.max_length == 0
            || self.intra_threads == 0
            || self.batch_size == 0
        {
            return Err(RathError::new(
                ErrorKind::Validation,
                "FastEmbed requires a cache directory, revision and positive execution limits",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/fastembed_options.rs"]
mod tests;
