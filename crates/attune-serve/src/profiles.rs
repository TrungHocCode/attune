//! Profile lookup by path or by name (`serve --profile`, `attune run`).
//!
//! A bare name such as `qwen3-4b` resolves to `~/.attune/profiles/qwen3-4b.json`;
//! anything that looks like a path is used as-is. The profiles directory is
//! created on first use so saving a profile there never fails for a missing
//! parent.

use std::path::PathBuf;

use crate::ServeError;

/// `~/.attune/profiles`, created if missing.
pub fn profiles_dir() -> Result<PathBuf, ServeError> {
    let dir = attune_fetch::home_dir()?.join("profiles");
    std::fs::create_dir_all(&dir).map_err(|source| ServeError::ProfilesDir {
        path: dir.display().to_string(),
        source,
    })?;
    Ok(dir)
}

/// Resolve a `--profile` argument to a file.
///
/// An existing path always wins. A bare name (no `/` or `\`) is looked up as
/// `profiles/<name>.json`, appending `.json` only when the name lacks it.
/// Anything else is an error that says where we looked and how to save one.
pub fn resolve_profile(arg: &str) -> Result<PathBuf, ServeError> {
    let direct = PathBuf::from(arg);
    if direct.is_file() {
        return Ok(direct);
    }

    let looks_like_path = arg.contains('/') || arg.contains('\\');
    if !looks_like_path {
        let dir = profiles_dir()?;
        let file_name = if arg.to_ascii_lowercase().ends_with(".json") {
            arg.to_string()
        } else {
            format!("{arg}.json")
        };
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        return Err(ServeError::ProfileNotFound {
            arg: arg.to_string(),
            candidates: format!("{} or {}", direct.display(), candidate.display()),
        });
    }

    Err(ServeError::ProfileNotFound {
        arg: arg.to_string(),
        candidates: direct.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::{Mutex, MutexGuard};

    /// `ATTUNE_HOME` is process-wide; one test plus a lock keeps it from
    /// racing tests in this binary.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct HomeGuard {
        previous: Option<std::ffi::OsString>,
        root: PathBuf,
    }

    impl HomeGuard {
        fn set(root: &Path) -> Self {
            let previous = std::env::var_os(attune_fetch::ENV_HOME);
            std::env::set_var(attune_fetch::ENV_HOME, root);
            Self {
                previous,
                root: root.to_path_buf(),
            }
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(attune_fetch::ENV_HOME, value),
                None => std::env::remove_var(attune_fetch::ENV_HOME),
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn profiles_resolve_by_path_and_name() {
        let _lock = lock();
        let root = std::env::temp_dir().join(format!("attune-profiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp home");
        let _guard = HomeGuard::set(&root);

        let dir = profiles_dir().expect("profiles dir");
        assert_eq!(dir, root.join("profiles"));
        assert!(dir.is_dir());

        // An existing path wins, even outside the profiles dir.
        let loose = root.join("elsewhere.json");
        std::fs::write(&loose, b"{}").expect("write loose profile");
        assert_eq!(
            resolve_profile(&loose.display().to_string()).expect("existing path"),
            loose
        );

        // A bare name maps to profiles/<name>.json.
        std::fs::write(dir.join("fast.json"), b"{}").expect("write fast");
        assert_eq!(
            resolve_profile("fast").expect("bare name"),
            dir.join("fast.json")
        );

        // A name that already ends in .json is not doubled.
        std::fs::write(dir.join("slow.json"), b"{}").expect("write slow");
        assert_eq!(
            resolve_profile("slow.json").expect("suffixed name"),
            dir.join("slow.json")
        );

        // A missing name says where it looked and how to save one.
        let error = resolve_profile("missing").expect_err("missing name");
        let text = error.to_string();
        assert!(text.contains("missing.json"), "{text}");
        assert!(text.contains("attune tune --model"), "{text}");

        // A missing explicit path gets the same hint.
        let absent = root.join("nope").join("profile.json");
        let error = resolve_profile(&absent.display().to_string()).expect_err("missing path");
        assert!(error.to_string().contains("attune tune --model"), "{error}");
    }
}
