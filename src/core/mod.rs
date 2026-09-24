pub(crate) mod error;
mod provider;
pub(crate) use provider::builtin;
/// Model locators and explicit language-model URL controls.
pub mod url;
mod usage;

pub use error::{ErrorBody, ErrorKind, RathError};
pub use provider::Provider;
pub use url::{CacheControl, ModelUrl, ThinkingLevel};
pub use usage::TokenUsage;
