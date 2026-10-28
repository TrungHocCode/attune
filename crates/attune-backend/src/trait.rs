//! The backend adapter contract (M5 slice 1).
//!
//! The file is named `trait.rs` in the plan §7 layout; the module is exposed as
//! [`crate::backend`] and [`Backend`] is re-exported at the crate root so
//! callers never need the raw identifier.
//!
//! Only [`LlamaServerBackend`] exists today. Shutdown is intentionally not a
//! method: the [`ServerGuard`] owns the child and implements `Drop` with
//! kill + wait + reader-join, so a caller that forgets to shut down explicitly
//! still cannot leak a process.

use std::path::PathBuf;
use std::time::Duration;

use attune_profile::Profile;

use crate::llama_subprocess::{health_wait, server_args, server_args_with_warnings, ServerGuard};
use crate::BackendError;

/// External paths one launch needs.
///
/// `binary` is already resolved (see
/// [`crate::resolve_llama_server`]); `working_dir` is optional and mostly
/// useful for vendor builds that expect relative model paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// The backend executable to spawn.
    pub binary: PathBuf,
    /// Working directory for the child, or `None` to inherit ours.
    pub working_dir: Option<PathBuf>,
}

impl Paths {
    /// Paths for an already-resolved binary, inheriting the current directory.
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            working_dir: None,
        }
    }

    /// Same, with an explicit child working directory.
    pub fn in_dir(binary: impl Into<PathBuf>, working_dir: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            working_dir: Some(working_dir.into()),
        }
    }
}

/// One inference engine adapter.
pub trait Backend {
    /// Human-readable engine name (`llama.cpp`).
    fn name(&self) -> &'static str;

    /// The exact argv that [`Backend::spawn`] would use, without spawning.
    fn args_preview(&self, profile: &Profile, host: &str, port: u16) -> Vec<String>;

    /// Spawn the server for `profile`.
    ///
    /// The returned guard is alive until dropped; dropping it kills, reaps and
    /// joins the child's output readers.
    fn spawn(
        &self,
        profile: &Profile,
        paths: &Paths,
        host: &str,
        port: u16,
    ) -> Result<ServerGuard, BackendError>;

    /// Poll `url` until it reports healthy or `deadline` elapses, returning
    /// how long the wait took.
    fn health_wait(&self, url: &str, deadline: Duration) -> Result<Duration, BackendError>;
}

/// The llama.cpp `llama-server` adapter.
#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaServerBackend;

impl Backend for LlamaServerBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    fn args_preview(&self, profile: &Profile, host: &str, port: u16) -> Vec<String> {
        server_args(profile, host, port)
    }

    fn spawn(
        &self,
        profile: &Profile,
        paths: &Paths,
        host: &str,
        port: u16,
    ) -> Result<ServerGuard, BackendError> {
        let prepared = server_args_with_warnings(profile, host, port);
        ServerGuard::spawn_in(
            &paths.binary,
            &prepared.args,
            paths.working_dir.as_deref(),
            prepared.warnings,
        )
    }

    fn health_wait(&self, url: &str, deadline: Duration) -> Result<Duration, BackendError> {
        health_wait(url, deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attune_profile::{ComputeConfig, LoadConfig, ModelConfig, Profile};

    fn profile() -> Profile {
        Profile {
            model: ModelConfig {
                path: "models/tiny.gguf".to_string(),
                ..ModelConfig::default()
            },
            load_config: LoadConfig {
                n_ctx: 512,
                ..LoadConfig::default()
            },
            compute_config: ComputeConfig {
                n_threads: 2,
                n_threads_batch: 2,
                cpu_affinity: None,
            },
            ..Profile::default()
        }
    }

    #[test]
    fn args_preview_matches_server_args() {
        let backend = LlamaServerBackend;
        let profile = profile();
        assert_eq!(
            backend.args_preview(&profile, "127.0.0.1", 8081),
            server_args(&profile, "127.0.0.1", 8081)
        );
    }

    #[test]
    fn name_is_the_engine_name() {
        assert_eq!(LlamaServerBackend.name(), "llama.cpp");
    }

    #[test]
    fn paths_helpers_set_optional_working_dir() {
        assert_eq!(Paths::new("llama-server").working_dir, None);
        assert_eq!(
            Paths::in_dir("llama-server", "/tmp").working_dir,
            Some(PathBuf::from("/tmp"))
        );
    }
}
