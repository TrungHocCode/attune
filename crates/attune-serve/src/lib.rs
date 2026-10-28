//! attune-serve - run a tuned profile for real (M5 slice 1).
//!
//! Slice 1 covers **launch + health check** only:
//! - read and validate a [`attune_profile::Profile`];
//! - warn on a hardware-fingerprint mismatch and auto-apply the recorded
//!   `fallbacks.if_ram_insufficient.n_ctx` when RAM is short;
//! - resolve and spawn `llama-server` through [`attune_backend`];
//! - poll `/health` with a deadline.
//!
//! The proxy (`/v1`), the metrics endpoints and Ctrl+C handling arrive in later
//! slices. Until then [`launcher::run_forever`] parks the main thread; an
//! interrupt or kill terminates the process without unwinding, so slice 2 will
//! install a Ctrl+C handler that shuts the server down explicitly. The
//! [`attune_backend::ServerGuard`] `Drop` impl still reaps `llama-server` on
//! normal unwinding.

pub mod launcher;

pub use launcher::{
    fingerprint_of, launch, prepare_profile, run_forever, PreparedProfile, ReadyReport,
    ServeOptions, Serving,
};

use thiserror::Error;

/// Errors from preparing or launching a profile.
#[derive(Debug, Error)]
pub enum ServeError {
    /// The profile file could not be read.
    #[error("failed to read profile `{path}`: {source}")]
    ReadProfile {
        /// The path the user passed.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The profile file is not valid profile JSON.
    #[error("invalid profile: {0}")]
    Profile(#[from] attune_profile::ProfileError),
    /// Live hardware inputs needed for validation were unavailable.
    #[error("hardware detection failed: {0}")]
    Hal(#[from] attune_hal::HalError),
    /// The backend could not resolve, spawn or health-check.
    #[error(transparent)]
    Backend(#[from] attune_backend::BackendError),
    /// The caller supplied settings the launcher cannot use.
    #[error("invalid serve configuration: {0}")]
    InvalidConfig(String),
}
