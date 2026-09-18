use super::super::{ErrorBody, ErrorKind, Provider, RathError};
use std::mem::size_of;

/// Error values and unit results remain pointer-sized regardless of diagnostic payload size.
#[test]
fn error_storage_is_pointer_sized() {
    assert_eq!(size_of::<RathError>(), size_of::<usize>());
    assert_eq!(size_of::<Option<RathError>>(), size_of::<usize>());
    assert_eq!(size_of::<Result<(), RathError>>(), size_of::<usize>());
}

/// Standard traversal exposes exactly the normalized Rath causes, without a storage wrapper.
#[test]
fn standard_sources_preserve_rath_identity_and_order() {
    let error = RathError::new(ErrorKind::Transport, "connection refused")
        .with_context(Provider::Fal, "download")
        .with_source(RathError::new(ErrorKind::Other, "socket closed"))
        .with_context(Provider::Fal, "audio execution");
    let mut current = &error;
    for message in ["connection refused", "socket closed"] {
        let typed = current.source().unwrap();
        let standard = std::error::Error::source(current)
            .unwrap()
            .downcast_ref::<RathError>()
            .unwrap();
        assert!(std::ptr::eq(typed, standard));
        assert_eq!(standard.message(), message);
        current = standard;
    }
    assert!(std::error::Error::source(current).is_none());
}

/// A cloned snapshot retains its own body and causes after the original is extended and dropped.
#[test]
fn cloned_snapshot_owns_its_diagnostics() {
    let original = RathError::new(ErrorKind::Provider, "failed")
        .with_body(ErrorBody::Complete(b"PRIVATE-PAYLOAD".to_vec()))
        .with_source(RathError::new(ErrorKind::Other, "cause"));
    let cloned = original.clone();
    drop(original.with_source(RathError::new(ErrorKind::Other, "later cause")));
    assert_eq!(cloned.response_body().unwrap().bytes(), b"PRIVATE-PAYLOAD");
    assert!(cloned.response_body().unwrap().is_complete());
    assert_eq!(cloned.source().unwrap().message(), "cause");
    assert!(cloned.source().unwrap().source().is_none());
    assert_eq!(cloned.to_string(), "failed: cause");
    assert!(!format!("{cloned:?}").contains("PRIVATE-PAYLOAD"));
}
