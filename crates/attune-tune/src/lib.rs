//! attune-tune — estimate-only config ranking (M4 slice 1) plus the slice-2
//! full grid, pruning and star ranking.
//!
//! This crate is deliberately pure: it expands a grid of [`TuneConfig`]
//! points, prunes candidates that cannot run, predicts the rest with the
//! `attune-estimate` roofline math, and ranks the results for one
//! [`Objective`]. It never spawns a process, reads a file, detects hardware or
//! writes output — the CLI supplies every live input and owns the rendering.
//!
//! The JSON contract is [`report::TuneReport`]; evolve it with
//! `serde(default)` like `attune-hal` / `attune-bench` / `attune-estimate`,
//! never by silently retyping fields.
//!
//! **Scope:** decode is the weights-only roofline (no `Arch`, so no KV-cache
//! term) and TTFT is the prefill of the full `n_ctx` window. RAM fit and the
//! near-ceiling score penalty therefore count weights only. Slice 3 adds
//! [`export`], which turns the winning config into an `attune-profile`
//! document; later slices add the arch-aware KV term and an e2e verification
//! run.

pub mod config;
pub mod error;
pub mod export;
pub mod objective;
pub mod report;
pub mod sweep;

pub use config::{expand_full_grid, expand_full_grid_with_quants, TuneConfig, FULL_GRID_CAP};
pub use error::TuneError;
pub use export::{from_best, hardware_fingerprint, ProfileInputs};
pub use objective::{ram_penalty, stars, tpot_ms_from_tps, Objective};
pub use report::TuneReport;
pub use sweep::{expand_grid, prune, rank_by_estimate, PruneReason, RankedConfig};

/// Schema stamped on every [`TuneReport`] so consumers can tell this M4
/// document from anything older.
pub const SCHEMA: &str = "attune-tune/1";
