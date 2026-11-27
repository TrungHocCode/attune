use std::path::PathBuf;

/// Everything that can go wrong finding the home dir or fetching a file.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("cannot locate the attune home dir: {0}")]
    Home(String),
    #[error("refusing to download from {url}: host not on the allow-list")]
    HostNotAllowed { url: String },
    #[error("{url}: {detail}")]
    Http { url: String, detail: String },
    #[error("checksum mismatch: expected sha256 {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("not enough disk space: need {need} bytes, {free} free")]
    DiskSpace { need: u64, free: u64 },
    #[error("cannot unpack {path}: {detail}")]
    Archive { path: PathBuf, detail: String },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl FetchError {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
