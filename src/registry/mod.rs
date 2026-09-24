//! Application-owned factories for every capability.
mod erased;
mod provider;
mod routing;
mod url;
pub use provider::{BuiltinProviderFactory, ProviderFactory};
pub use routing::ProviderRegistry;

pub(crate) mod prepare;
