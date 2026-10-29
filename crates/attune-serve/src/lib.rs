//! attune-serve - run a tuned profile for real (M5 slices 1-2).
//!
//! Slice 1 covers **launch + health check**:
//! - read and validate a [`attune_profile::Profile`];
//! - warn on a hardware-fingerprint mismatch and auto-apply the recorded
//!   `fallbacks.if_ram_insufficient.n_ctx` when RAM is short;
//! - resolve and spawn `llama-server` through [`attune_backend`];
//! - poll `/health` with a deadline.
//!
//! Slice 2 adds the **OpenAI-compatible proxy** and **graceful shutdown**:
//! - [`server`] binds a `tiny_http` listener and routes the `/v1` surface plus
//!   `/health`;
//! - [`proxy`] forwards each request to the launched upstream, streaming SSE
//!   responses chunk-by-chunk and returning non-stream bodies verbatim;
//! - [`shutdown`] turns Ctrl+C (`SIGINT`, plus `SIGTERM`/`SIGHUP` on Unix) into
//!   a flag that closes the listener; dropping [`Serving`] then kills and reaps
//!   `llama-server` through [`attune_backend::ServerGuard`].
//!
//! Metrics and `--api-key` enforcement are later slices.

pub mod launcher;
pub mod proxy;
pub mod server;
pub mod shutdown;

pub use launcher::{
    fingerprint_of, launch, prepare_profile, PreparedProfile, ReadyReport, ServeOptions, Serving,
};
pub use server::{Proxy, ProxyConfig};
pub use shutdown::{install_shutdown, ShutdownSignal};

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
    /// The proxy listener could not be bound or served.
    #[error("proxy: {0}")]
    Proxy(String),
    /// The process shutdown handler could not be installed.
    #[error("shutdown handler: {0}")]
    Shutdown(String),
}
