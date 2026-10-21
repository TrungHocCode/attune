//! attune-bench — micro-benchmarks (M2).
//!
//! Slice 2 ships the **read / write / copy memory-bandwidth kernel** set.
//! FLOPS, power monitoring and llama.cpp e2e land in later slices; do not
//! add them ahead of the phase discipline in AGENTS.md §4.
//!
//! Every public type here is part of the JSON contract (`attune bench --json`),
//! so fields are stable once shipped — evolve with `serde(default)` like
//! `attune-hal` does, never by silently retyping.

pub mod bandwidth;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("hardware detection failed: {0}")]
    Hal(#[from] attune_hal::HalError),
    #[error("failed to allocate a {0} byte benchmark buffer: {1}")]
    Alloc(u64, #[source] std::collections::TryReserveError),
    #[error("benchmark worker thread panicked during a bandwidth pass")]
    WorkerPanicked,
    #[error("invalid bench configuration: {0}")]
    InvalidConfig(String),
}
