//! attune-bench — micro-benchmarks (M2).
//!
//! Slice 2 ships the **read / write / copy memory-bandwidth kernel** set;
//! slice 3 adds the **FP32 FMA throughput** sweep with scalar / AVX2 /
//! AVX-512F path selection; slice 4 adds the **resource monitor** that samples
//! RAM / CPU / (Linux) power while a run is in flight; slice 5 adds the
//! **llama.cpp subprocess runner** that finds `llama-cli`, runs it under a
//! timeout and scrapes its version / perf output. A real end-to-end decode
//! sweep against a model lands in a later slice; do not add it ahead of the
//! phase discipline in AGENTS.md §4.
//!
//! Every public type here is part of the JSON contract (`attune bench --json`),
//! so fields are stable once shipped — evolve with `serde(default)` like
//! `attune-hal` does, never by silently retyping.

pub mod bandwidth;
pub mod flops;
pub mod llama;
pub mod monitor;

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
    #[error("failed to spawn the resource monitor thread: {0}")]
    MonitorSpawn(#[source] std::io::Error),
    /// No usable `llama-cli`: nothing matched `--llama-cli`, `ATTUNE_LLAMA_CLI`
    /// or the `PATH` search, or the winner was not a runnable file. The
    /// message always carries the fix — this is an expected condition, not an
    /// internal error.
    #[error("llama-cli not found ({detail}) — pass --llama-cli or set ATTUNE_LLAMA_CLI")]
    LlamaNotFound {
        /// What was looked for, e.g. "`llama-cli` on PATH".
        detail: String,
    },
    /// The child could not be started, or polling it failed.
    #[error("failed to run `{cli}`: {source}")]
    LlamaSpawn {
        /// The executable we tried to run.
        cli: String,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },
    /// The child outlived its deadline and was killed. Never a hang: the
    /// child is reaped before this error is returned.
    #[error("`{cli} {args}` did not finish within {timeout_ms} ms — killed")]
    LlamaTimeout {
        /// The executable that was killed.
        cli: String,
        /// The arguments it was killed with.
        args: String,
        /// The deadline that expired, in milliseconds.
        timeout_ms: u64,
    },
    /// The child ran to completion but reported failure — a non-zero exit or a
    /// missing shared library. `stderr` is the tail of its output.
    #[error("`{cli} {args}` failed{}: {stderr}", exit_code_suffix(*code))]
    LlamaFailed {
        /// The executable that failed.
        cli: String,
        /// The arguments it failed with.
        args: String,
        /// Exit code, or `None` when the process died from a signal.
        code: Option<i32>,
        /// Tail of the child's stderr.
        stderr: String,
    },
    /// `--version` exited cleanly but printed nothing we recognise as a
    /// version, so we refuse to guess a build number.
    #[error("`{cli} --version` printed no recognisable version number")]
    LlamaVersionUnknown {
        /// The executable that gave an unusable answer.
        cli: String,
    },
}

/// Render an optional child exit code for [`BenchError::LlamaFailed`] —
/// ` (exit code 1)`, or ` (killed by a signal)` when the process never exited
/// on its own. Lives next to the variant because the derive expands in here.
fn exit_code_suffix(code: Option<i32>) -> String {
    match code {
        Some(code) => format!(" (exit code {code})"),
        None => " (killed by a signal)".to_string(),
    }
}

/// One-shot orchestrator that runs the full sweep (bandwidth + FLOPS)
/// end-to-end, with the resource monitor sampling this process throughout.
/// Lives at the crate root because it crosses both modules — keeping it out
/// of `bandwidth.rs` keeps that module side-effect-free (no `use crate::flops`)
/// and out of `flops.rs` keeps that module unaware of bandwidth sizing knobs.
///
/// The returned [`bandwidth::FullBandwidthResult`] has `flops` and `resources`
/// populated. This is the function `attune-cli` calls for full-mode `bench`.
pub fn quick_full_with_flops() -> Result<bandwidth::FullBandwidthResult, BenchError> {
    use attune_hal::HardwareInfo;

    let hw = HardwareInfo::detect()?;
    let bw_cfg = bandwidth::BandwidthConfig::full_from_hardware(&hw);
    let fp_cfg = flops::FlopsConfig::from_hardware(&hw);
    // Started before the kernels and stopped after them, so the report covers
    // the whole run. An early `?` drops the handle, which stops and joins the
    // thread (see `Monitor::drop`).
    let monitor = monitor::Monitor::start_sysinfo(monitor::DEFAULT_INTERVAL_MS)?;
    let mut result = bandwidth::full_bandwidth(&bw_cfg)?;
    let flops_result = flops::measure_flops(&fp_cfg)?;
    let resources = monitor.stop();
    result.flops = Some(flops_result);
    result.resources = Some(resources);
    Ok(result)
}
