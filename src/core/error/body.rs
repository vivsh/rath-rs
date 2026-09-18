/// Response bytes retained for explicit inspection, after credential sanitization.
/// Debug describes length/completeness and never displays the bytes.
#[derive(Clone, PartialEq, Eq)]
pub enum ErrorBody {
    /// The complete response or structured partial generation output.
    Complete(Vec<u8>),
    /// Bytes received before the response transfer failed.
    Incomplete(Vec<u8>),
}

impl ErrorBody {
    /// Borrows the retained bytes; they may contain private conversation content.
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Complete(bytes) | Self::Incomplete(bytes) => bytes,
        }
    }
    /// Indicates whether the response transfer completed successfully.
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }
}

impl std::fmt::Debug for ErrorBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorBody")
            .field("complete", &self.is_complete())
            .field("bytes", &self.bytes().len())
            .finish()
    }
}
