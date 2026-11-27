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
//! Slice 3 adds **proxy-measured metrics** ([`metrics`]): TTFT, TPOT, tok/s and
//! request counters over a bounded window, served by the proxy itself at
//! `GET /v1/metrics` (JSON) and `GET /metrics` (Prometheus text).
//!
//! Slice 4 hardens the proxy: with [`ProxyConfig::api_key`] set, every route
//! except `/health` requires `Authorization: Bearer <key>` (constant-time
//! compare, 401 otherwise) and the client's header is never forwarded to
//! `llama-server`. [`Proxy::bind`] refuses a non-loopback `listen` without a
//! key.

pub mod chat;
pub mod launcher;
pub mod metrics;
pub mod profiles;
pub mod proxy;
pub mod server;
pub mod shutdown;

pub use chat::{parse_sse_line, ChatClient, ChatStats, Message, SseEvent};
pub use launcher::{
    fingerprint_of, launch, prepare_profile, resolve_model_file, DownloadConsent, PreparedProfile,
    ReadyReport, ServeOptions, Serving,
};
pub use profiles::{profiles_dir, resolve_profile};
pub use server::{check_listen, Proxy, ProxyConfig};
pub use shutdown::{install_shutdown, ShutdownSignal};

use thiserror::Error;

/// Errors from preparing or launching a profile.
#[derive(Debug, Error)]
pub enum ServeError {
    /// The profile file could not be read.
    #[error("failed to read profile `{path}`")]
    ReadProfile {
        /// The path the user passed.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The profile file is not valid profile JSON.
    #[error("invalid profile")]
    Profile(#[from] attune_profile::ProfileError),
    /// Live hardware inputs needed for validation were unavailable.
    #[error("hardware detection failed")]
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
    /// The attune home dir could not be resolved.
    #[error("attune home: {0}")]
    Home(#[from] attune_fetch::FetchError),
    /// The `profiles` directory could not be created.
    #[error("cannot create the profiles dir `{path}`: {source}")]
    ProfilesDir {
        /// Directory that could not be created.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `--profile` named a file that does not exist.
    #[error(
        "no profile `{arg}`: looked for {candidates}; save one with \
         `attune tune --model <id> --save <path>`"
    )]
    ProfileNotFound {
        /// The argument the user passed.
        arg: String,
        /// The file paths that were checked.
        candidates: String,
    },
    /// Chat request, streaming or response handling failed.
    #[error("chat: {0}")]
    Chat(String),
    /// No attune server is accepting connections at `url`.
    #[error("No attune server at {url}. Start one with `attune serve --profile <name>`")]
    NoServer {
        /// The URL the client tried.
        url: String,
    },
}

impl ServeError {
    /// Whether this error means no server was listening at the chat URL.
    pub fn is_no_server(&self) -> bool {
        matches!(self, Self::NoServer { .. })
    }
}
