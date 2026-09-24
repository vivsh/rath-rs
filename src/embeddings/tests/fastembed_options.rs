use super::*;
/// Typed setup has one set of explicit defaults and rejects malformed public builder values.
#[test]
fn validates_all_setup_fields() {
    let defaults = FastEmbedOptions::new("models");
    assert!(!defaults.download());
    assert_eq!(defaults.revision(), "main");
    assert_eq!(
        (
            defaults.max_length(),
            defaults.intra_threads(),
            defaults.batch_size()
        ),
        (512, 1, 32)
    );
    for invalid in [
        FastEmbedOptions::new(""),
        defaults.clone().with_revision(""),
        defaults.clone().with_revision("../outside"),
        defaults.clone().with_max_length(0),
        defaults.clone().with_intra_threads(0),
        defaults.clone().with_batch_size(0),
    ] {
        assert_eq!(
            invalid.validate().unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
}
