//! attune-bench — micro-benchmarks (M2).
//!
//! Slice 2 ships the **read / write / copy memory-bandwidth kernel** set;
//! slice 3 adds the **FP32 FMA throughput** sweep with scalar / AVX2 /
//! AVX-512F path selection. e2e and power monitoring land in later slices;
//! do not add them ahead of the phase discipline in AGENTS.md §4.
//!
//! Every public type here is part of the JSON contract (`attune bench --json`),
//! so fields are stable once shipped — evolve with `serde(default)` like
//! `attune-hal` does, never by silently retyping.

pub mod bandwidth;
pub mod flops;

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

/// One-shot orchestrator that runs the full sweep (bandwidth + FLOPS)
/// end-to-end. Lives at the crate root because it crosses both modules —
/// keeping it out of `bandwidth.rs` keeps that module side-effect-free
/// (no `use crate::flops`) and out of `flops.rs` keeps that module unaware
/// of bandwidth sizing knobs.
///
/// The returned [`bandwidth::FullBandwidthResult`] has `flops` populated.
/// This is the function `attune-cli` calls for full-mode `bench`.
pub fn quick_full_with_flops() -> Result<bandwidth::FullBandwidthResult, BenchError> {
    use attune_hal::HardwareInfo;

    let hw = HardwareInfo::detect()?;
    let bw_cfg = bandwidth::BandwidthConfig::full_from_hardware(&hw);
    let fp_cfg = flops::FlopsConfig::from_hardware(&hw);
    let mut result = bandwidth::full_bandwidth(&bw_cfg)?;
    result.flops = Some(flops::measure_flops(&fp_cfg)?);
    Ok(result)
}
