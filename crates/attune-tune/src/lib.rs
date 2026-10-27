//! attune-tune — estimate-only config ranking (M4 slice 1).
//!
//! This slice is deliberately pure: it expands a cartesian grid of
//! [`TuneConfig`] points, predicts each point with the `attune-estimate`
//! roofline math, and ranks the results for one [`Objective`]. It never
//! spawns a process, reads a file, detects hardware or writes output — the
//! CLI supplies every live input and owns the rendering.
//!
//! The JSON contract is [`report::TuneReport`]; evolve it with
//! `serde(default)` like `attune-hal` / `attune-bench` / `attune-estimate`,
//! never by silently retyping fields.
//!
//! **Slice-1 scope:** decode is the weights-only roofline (no `Arch`, so no
//! KV-cache term) and TTFT is the prefill of the full `n_ctx` window. RAM fit
//! therefore counts weights only. Later slices add the arch-aware KV term, an
//! e2e verification run and profile output.

pub mod config;
pub mod error;
pub mod objective;
pub mod report;
pub mod sweep;

pub use config::TuneConfig;
pub use error::TuneError;
pub use objective::Objective;
pub use report::TuneReport;
pub use sweep::{expand_grid, rank_by_estimate, RankedConfig};

/// Schema stamped on every [`TuneReport`] so consumers can tell this M4
/// slice-1 document from anything older.
pub const SCHEMA: &str = "attune-tune/1";
