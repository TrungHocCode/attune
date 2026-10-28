//! Errors returned by the backend adapters.
//!
//! Every variant is an expected operating condition (llama.cpp is an optional
//! external tool), so messages carry the action that fixes them. No `.unwrap()`
//! in non-test code (AGENTS.md §3.2).

use thiserror::Error;

/// A backend could not be resolved, spawned, or made healthy.
#[derive(Debug, Error)]
pub enum BackendError {
    /// No usable `llama-server`: nothing matched `--llama-server`,
    /// `ATTUNE_LLAMA_SERVER` or the `PATH` search, or the winner was not a
    /// runnable file.
    #[error("llama-server not found ({detail}) - pass --llama-server or set ATTUNE_LLAMA_SERVER")]
    NotFound {
        /// What was looked for, e.g. "`llama-server` on PATH".
        detail: String,
    },
    /// The child could not be started, or polling it failed.
    #[error("failed to run `{binary}`: {source}")]
    Spawn {
        /// The executable we tried to run.
        binary: String,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },
    /// The health endpoint did not answer healthy before the deadline.
    #[error("`{url}` did not become healthy within {timeout_ms} ms: {detail}")]
    Timeout {
        /// The health URL that was polled.
        url: String,
        /// The deadline that expired, in milliseconds.
        timeout_ms: u64,
        /// Last observed failure (HTTP status or transport error).
        detail: String,
    },
    /// The endpoint answered, but explicitly reported an unhealthy state.
    #[error("health check failed for `{url}`: {detail}")]
    HealthFailed {
        /// The health URL that answered.
        url: String,
        /// The response body (trimmed) that said "not healthy".
        detail: String,
    },
    /// The caller supplied settings the backend cannot use.
    #[error("invalid backend configuration: {0}")]
    InvalidConfig(String),
}
