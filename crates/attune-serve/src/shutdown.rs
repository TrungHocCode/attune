//! Graceful shutdown signalling (M5 slice 2).
//!
//! `ctrlc` owns the OS-specific handler and runs it on its own thread; this
//! module only turns that callback into a pollable flag plus a wake-up hook for
//! the blocking `tiny_http` accept loop. The listener is closed when the loop
//! returns and the [`crate::server::Proxy`] is dropped; the caller then drops
//! [`attune_backend::ServerGuard`], which kills and reaps `llama-server`.
//!
//! # Platform coverage
//!
//! * **Windows**: the handler covers Ctrl+C and console close/logoff/shutdown
//!   events delivered through the console control handler. Windows has no
//!   POSIX `SIGTERM`; `taskkill`/`Stop-Process` terminate the process without
//!   running any handler, so the caller should kill the process tree or accept
//!   that `llama-server` outlives an abrupt kill.
//! * **Linux/macOS**: the `termination` feature of `ctrlc` adds `SIGTERM` and
//!   `SIGHUP` next to `SIGINT`, which is what `systemd stop`, `docker stop` and
//!   `kill` send. A single handler therefore covers interactive Ctrl+C and
//!   service-manager shutdown.
//!
//! Installing a process-wide handler is process-global state, so production
//! code installs it exactly once (from `attune serve`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ServeError;

/// A pollable, cloneable "shutdown requested" flag.
#[derive(Debug, Clone, Default)]
pub struct ShutdownSignal {
    requested: Arc<AtomicBool>,
}

impl ShutdownSignal {
    /// A fresh signal that has not been requested.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark shutdown as requested. Later calls are harmless.
    pub fn request(&self) {
        self.requested.store(true, Ordering::SeqCst);
    }

    /// Whether shutdown has been requested.
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }
}

/// Install the process signal handler.
///
/// `wake` is called from the signal-handler thread after the flag flips; it
/// must unblock the accept loop (`tiny_http::Server::unblock`). Use
/// [`crate::Proxy::unblocker`] so the callback holds only a weak listener
/// reference and cannot keep the socket open after shutdown.
pub fn install_shutdown<F>(wake: F) -> Result<ShutdownSignal, ServeError>
where
    F: Fn() + Send + 'static,
{
    let signal = ShutdownSignal::new();
    let requested = Arc::clone(&signal.requested);
    ctrlc::set_handler(move || {
        // Set the flag *before* waking the loop so an accept call that returns
        // `None` immediately still observes the request.
        requested.store(true, Ordering::SeqCst);
        wake();
    })
    .map_err(|error| ServeError::Shutdown(error.to_string()))?;
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_signal_is_not_requested() {
        let signal = ShutdownSignal::new();
        assert!(!signal.is_requested());
    }

    #[test]
    fn clones_share_the_requested_flag() {
        let signal = ShutdownSignal::new();
        let clone = signal.clone();
        clone.request();
        assert!(signal.is_requested());
        assert!(clone.is_requested());
    }

    #[test]
    fn requesting_twice_is_harmless() {
        let signal = ShutdownSignal::new();
        signal.request();
        signal.request();
        assert!(signal.is_requested());
    }
}
