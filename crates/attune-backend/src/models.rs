//! Make sure a profile's model file is on disk (M6): download it from its
//! pinned source, after the caller's consent, when it is missing.

use std::path::{Path, PathBuf};

use attune_fetch::{download, Download, HostPolicy};
use attune_profile::ModelSource;

use crate::BackendError;

/// Return `path` when the file is there with the pinned size; otherwise ask
/// `confirm(size)` and download + verify it. `progress` gets
/// `(bytes_done, bytes_total)`.
pub fn ensure_model(
    path: &Path,
    source: &ModelSource,
    confirm: &mut dyn FnMut(u64) -> bool,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf, BackendError> {
    if std::fs::metadata(path).is_ok_and(|meta| meta.len() == source.size) {
        return Ok(path.to_path_buf());
    }
    if !confirm(source.size) {
        return Err(BackendError::Download {
            detail: format!("download declined; {} is not on disk", path.display()),
        });
    }
    let request = Download {
        url: source.url(),
        dest: path.to_path_buf(),
        size: source.size,
        sha256: source.sha256.clone(),
    };
    download(&request, &HostPolicy::production(), progress).map_err(|err| {
        BackendError::Download {
            detail: err.to_string(),
        }
    })?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(size: u64) -> ModelSource {
        ModelSource {
            repo: "org/repo-GGUF".to_string(),
            file: "m.gguf".to_string(),
            revision: "rev".to_string(),
            sha256: "a".repeat(64),
            size,
        }
    }

    fn temp_file(tag: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-models-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("m.gguf");
        if !bytes.is_empty() {
            std::fs::write(&path, bytes).expect("write");
        }
        path
    }

    #[test]
    fn existing_file_with_right_size_skips_download() {
        let path = temp_file("present", b"12345");
        let got = ensure_model(
            &path,
            &source(5),
            &mut |_| panic!("must not ask"),
            &mut |_, _| {},
        )
        .expect("present");
        assert_eq!(got, path);
    }

    #[test]
    fn declined_confirmation_errors_without_network() {
        let path = temp_file("declined", b"");
        let mut asked = None;
        let err = ensure_model(
            &path,
            &source(2_500_000_000),
            &mut |size| {
                asked = Some(size);
                false
            },
            &mut |_, _| {},
        )
        .expect_err("declined");
        assert_eq!(asked, Some(2_500_000_000));
        assert!(err.to_string().contains("download declined"), "{err}");
        assert!(!path.exists());
    }
}
