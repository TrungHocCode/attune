//! attune-backend - inference backend adapters (M5 slice 1).
//!
//! Public surface:
//! - [`backend`] / [`Backend`] - the adapter trait (`spawn`, `health_wait`,
//!   `args_preview`; shutdown is the [`ServerGuard`] `Drop` impl);
//! - [`llama_subprocess`] - the llama.cpp `llama-server` resolver, argv
//!   mapping and process guard;
//! - [`builds`] - which llama.cpp build variants `attune setup --gpu` installs.
//!
//! **Why the process handling is still duplicated.** `attune-bench` already
//! knows how to find, spawn, time out and reap a llama.cpp child, but importing
//! it just for that would pull the measurement stack into the serving path for
//! no gain. The two tiny pieces this crate needs - the precedence resolver and
//! the `Child` guard - stay local. The precedent is
//! `attune_bench::e2e::E2eGuard`, which duplicates `llama::ChildGuard` for the
//! same reason: each call site owns its guard and the
//! dead-code-on-the-other-platform hazard stays local. Keep the copies in sync
//! when the llama.cpp process handling changes. `builds` is the exception: it
//! reads the catalog pin and the detected GPUs, so it depends on
//! `attune-estimate` (for `LlamaPin`) and `attune-hal` (for `GpuDevice`).
//!
//! This crate does not proxy HTTP or emit metrics; those are later M5 slices.
//! It only launches a backend, waits for `/health`, and kills the child when
//! the guard is dropped.

pub mod builds;
pub mod error;
pub mod llama_subprocess;
pub mod models;

#[path = "trait.rs"]
pub mod backend;

pub use backend::{Backend, LlamaServerBackend, Paths};
pub use builds::{choose_variants, GpuRequest, VariantPlan};
pub use error::BackendError;
pub use llama_subprocess::{
    health_is_ok, health_wait, resolve_llama_server, server_args, server_args_with_warnings,
    LlamaServerSource, ServerArgs, ServerGuard, ENV_LLAMA_SERVER, PATH_CANDIDATES,
};
pub use models::ensure_model;
