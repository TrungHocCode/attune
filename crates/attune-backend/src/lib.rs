//! attune-backend - inference backend adapters (M5 slice 1).
//!
//! Public surface:
//! - [`backend`] / [`Backend`] - the adapter trait (`spawn`, `health_wait`,
//!   `args_preview`; shutdown is the [`ServerGuard`] `Drop` impl);
//! - [`llama_subprocess`] - the llama.cpp `llama-server` resolver, argv
//!   mapping and process guard.
//!
//! **Why there is no `attune-bench`/`attune-hal` dependency.** The benchmark
//! crate already knows how to find, spawn, time out and reap a llama.cpp child.
//! Importing it here would drag the whole measurement stack (and `attune-hal`)
//! into the serving path, so the two tiny pieces this crate needs - the
//! precedence resolver and the `Child` guard - are duplicated deliberately.
//! The precedent is `attune_bench::e2e::E2eGuard`, which duplicates
//! `llama::ChildGuard` for the same reason: each call site owns its guard and
//! the dead-code-on-the-other-platform hazard stays local. Keep the copies in
//! sync when the llama.cpp process handling changes.
//!
//! This crate does not proxy HTTP or emit metrics; those are later M5 slices.
//! It only launches a backend, waits for `/health`, and kills the child when
//! the guard is dropped.

pub mod error;
pub mod llama_subprocess;

#[path = "trait.rs"]
pub mod backend;

pub use backend::{Backend, LlamaServerBackend, Paths};
pub use error::BackendError;
pub use llama_subprocess::{
    health_is_ok, health_wait, resolve_llama_server, server_args, server_args_with_warnings,
    LlamaServerSource, ServerArgs, ServerGuard, ENV_LLAMA_SERVER, PATH_CANDIDATES,
};
