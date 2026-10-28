//! attune-profile — versioned inference profiles (M4 slice 3).
//!
//! Public surface:
//! - [`schema`]  — the [`Profile`] JSON contract, versioned by
//!   [`PROFILE_VERSION`];
//! - [`validate`] — tolerant checks against the current machine (never a hard
//!   failure: a mismatch returns a [`ProfileWarning`] so the caller can fall
//!   back);
//! - [`migrate`] — fill gaps in older profile shapes before parsing;
//! - [`args`]    — pure profile → llama.cpp flag mapping.
//!
//! This crate is a leaf by design (AGENTS.md §1): only `serde`, `serde_json`
//! and `thiserror`. It never detects hardware, spawns a process or writes a
//! file — callers supply live inputs and own the I/O.

pub mod args;
pub mod migrate;
pub mod schema;
pub mod validate;

use thiserror::Error;

pub use args::to_llama_args;
pub use migrate::migrate_value;
pub use schema::{
    now_unix, ComputeConfig, DraftConfig, Engine, ExpectedPerformance, Fallbacks, LoadConfig,
    ModelConfig, Profile, RamFallback, SamplingDefaults, DEFAULT_FALLBACK_N_CTX, DEFAULT_N_BATCH,
    DEFAULT_N_CTX, DEFAULT_N_UBATCH,
};
pub use validate::{validate, ProfileWarning, WarningKind, WarningSeverity};

/// Profile schema version written by this crate.
///
/// Bump it only together with a `migrate` step that upgrades the old shape;
/// `serde(default)` on every field keeps older documents parseable in the
/// meantime.
pub const PROFILE_VERSION: u32 = 1;

/// Errors from parsing or serializing a profile.
#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("invalid profile JSON: {0}")]
    Json(#[from] serde_json::Error),
}
