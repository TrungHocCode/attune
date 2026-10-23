//! attune-bench — micro-benchmarks (M2).
//!
//! Slice 2 ships the **read / write / copy memory-bandwidth kernel** set;
//! slice 3 adds the **FP32 FMA throughput** sweep with scalar / AVX2 /
//! AVX-512F path selection; slice 4 adds the **resource monitor** that samples
//! RAM / CPU / (Linux) power while a run is in flight; slice 5 adds the
//! **llama.cpp subprocess runner** that finds `llama-cli`, runs it under a
//! timeout and scrapes its version / perf output; slice 6 closes M2 with the
//! **end-to-end llama.cpp decode bench** plus the unified [`BenchReport`]
//! that the CLI renders in `--model` mode.
//!
//! Every public type here is part of the JSON contract
//! (`attune bench --json`, `attune bench --model --json`), so fields are
//! stable once shipped — evolve with `serde(default)` like `attune-hal` does,
//! never by silently retyping.

pub mod bandwidth;
pub mod e2e;
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

/// Schema string stamped on every [`BenchReport`] so consumers can tell a
/// slice-6 (M2 closure) document from anything older. Bump this when the
/// shape changes and *also* ship a migration — see the `attune-profile`
/// convention (AGENTS.md §3.7).
pub const SCHEMA: &str = "attune-bench/1";

/// Unified `attune bench` output. The CLI renders this as the human table
/// in full mode and as the JSON contract in either mode. `e2e` is present
/// only when `--model` is passed; everything else is unchanged across
/// modes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// `"quick"` or `"full"`. Discriminates the `bandwidth` shape.
    pub mode: String,
    /// Kernel-level bandwidth numbers, shaped according to `mode`.
    pub bandwidth: BandwidthReport,
    /// FP32 FMA throughput. `None` in quick mode (the sweep is skipped).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flops: Option<flops::FlopsResult>,
    /// Resource-usage summary from the bandwidth + flops phase. `None` in
    /// quick mode (no monitor is running).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<monitor::ResourceStats>,
    /// End-to-end llama.cpp decode measurement. `None` unless `--model`
    /// was passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e: Option<e2e::E2eResult>,
}

/// Bandwidth kernel result. Untagged enum because the two variants have
/// non-overlapping JSON shapes (quick reports one number, full reports
/// three + sizing metadata) and untagged dispatching picks the right one
/// automatically. Test coverage lives next to each variant.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum BandwidthReport {
    /// `--quick`: just the best-of-reps read bandwidth number.
    Quick(bandwidth::ReadResult),
    /// Default (`attune bench` with no `--quick`): read / write / copy
    /// medians + working-set sizing.
    Full(bandwidth::FullBandwidthResult),
}

impl BandwidthReport {
    /// `buffer_mib` regardless of mode. Used by the CLI to render the
    /// "Working set" human-mode line.
    pub fn buffer_mib(&self) -> u64 {
        match self {
            BandwidthReport::Quick(r) => r.buffer_mib,
            BandwidthReport::Full(r) => r.buffer_mib,
        }
    }

    /// Worker thread count. Same purpose as [`Self::buffer_mib`].
    pub fn threads(&self) -> usize {
        match self {
            BandwidthReport::Quick(r) => r.threads,
            BandwidthReport::Full(r) => r.threads,
        }
    }

    /// Timed passes. Same purpose as [`Self::buffer_mib`].
    pub fn reps(&self) -> usize {
        match self {
            BandwidthReport::Quick(r) => r.reps,
            BandwidthReport::Full(r) => r.reps,
        }
    }
}

/// Run just the quick bandwidth probe and wrap it in a [`BenchReport`].
pub fn run_quick() -> Result<BenchReport, BenchError> {
    let result = bandwidth::quick_read()?;
    Ok(BenchReport {
        schema: SCHEMA.to_string(),
        mode: "quick".to_string(),
        bandwidth: BandwidthReport::Quick(result),
        flops: None,
        resources: None,
        e2e: None,
    })
}

/// One-shot orchestrator for the full bench: bandwidth sweep + FP32 FLOPS
/// sweep + resource monitor, plus — when `e2e_cfg` and `llama_cli` are both
/// supplied — the end-to-end llama.cpp decode run.
///
/// Without an e2e config this is equivalent to the previous "quick full
/// with flops" sequence; all earlier behaviour and JSON test invariants
/// stay green.
#[allow(clippy::too_many_arguments)]
pub fn run_full(
    hw: &attune_hal::HardwareInfo,
    e2e_cfg: Option<&e2e::E2eConfig>,
    llama_cli: Option<&std::path::Path>,
) -> Result<BenchReport, BenchError> {
    // Same path the pre-slice-6 `quick_full_with_flops` used. Keeping the
    // monitor pointing at this process (the bench harness) rather than at
    // a kernel worker because the resource sampling covers the *whole*
    // sweep here, including the FLOPS phase that runs alongside it.
    let bw_cfg = bandwidth::BandwidthConfig::full_from_hardware(hw);
    let fp_cfg = flops::FlopsConfig::from_hardware(hw);
    let monitor = monitor::Monitor::start_sysinfo(monitor::DEFAULT_INTERVAL_MS)?;
    let mut bandwidth_result = bandwidth::full_bandwidth(&bw_cfg)?;
    let flops_result = flops::measure_flops(&fp_cfg)?;
    let resources = monitor.stop();
    bandwidth_result.flops = Some(flops_result);
    bandwidth_result.resources = Some(resources);

    // E2e runs only when both an e2e config *and* a llama-cli are present;
    // `--model` without a resolvable backend leaves the bench unchanged.
    let e2e = match (e2e_cfg, llama_cli) {
        (Some(cfg), Some(cli)) => Some(e2e::run_e2e(cli, cfg)?),
        _ => None,
    };

    Ok(BenchReport {
        schema: SCHEMA.to_string(),
        mode: "full".to_string(),
        bandwidth: BandwidthReport::Full(bandwidth_result),
        flops: bandwidth_result.flops,
        resources: bandwidth_result.resources,
        e2e,
    })
}

/// Back-compat shim for callers predating slice 6 that asked for the
/// bandwidth + flops + monitor trio without an [`BenchReport`] wrapper.
/// New code should prefer [`run_full`].
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

// ---- BenchReport tests -----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandwidth::{KernelStats, ReadResult};
    use crate::e2e::E2eResult;
    use crate::flops::{FlopsPath, FlopsResult};
    use crate::monitor::ResourceStats;
    use crate::SCHEMA as BENCH_SCHEMA;

    fn quick_bandwidth() -> BandwidthReport {
        BandwidthReport::Quick(ReadResult {
            read_gbps: 33.2,
            buffer_mib: 256,
            threads: 8,
            reps: 3,
        })
    }

    fn full_bandwidth_value() -> BandwidthReport {
        BandwidthReport::Full(bandwidth::FullBandwidthResult {
            read: KernelStats {
                median_gbps: 22.0,
                best_gbps: 25.0,
            },
            write: KernelStats {
                median_gbps: 14.0,
                best_gbps: 16.0,
            },
            copy: KernelStats {
                median_gbps: 20.0,
                best_gbps: 23.0,
            },
            buffer_mib: 256,
            threads: 8,
            reps: 5,
            flops: None,
            resources: None,
        })
    }

    #[test]
    fn schema_constant_has_the_documented_value() {
        // The schema string is part of the JSON contract; pinning it in a
        // test catches accidental retyping without wiring up a migration.
        assert_eq!(BENCH_SCHEMA, "attune-bench/1");
    }

    #[test]
    fn bandwidth_report_accessors_match_inner_values() {
        assert_eq!(quick_bandwidth().buffer_mib(), 256);
        assert_eq!(quick_bandwidth().threads(), 8);
        assert_eq!(quick_bandwidth().reps(), 3);
        assert_eq!(full_bandwidth_value().buffer_mib(), 256);
        assert_eq!(full_bandwidth_value().threads(), 8);
        assert_eq!(full_bandwidth_value().reps(), 5);
    }

    fn sample_flops() -> FlopsResult {
        FlopsResult {
            fp32_gflops: 50.0,
            best_gflops: 55.0,
            path: FlopsPath::Avx2,
            threads: 8,
            reps: 3,
            theoretical_gflops: 80.0,
            pct_of_theoretical: 62.5,
        }
    }

    fn sample_resources() -> ResourceStats {
        ResourceStats {
            ram_peak_bytes: 1_073_741_824,
            cpu_avg_pct: 42.5,
            cpu_peak_pct: 87.25,
            power_avg_watts: Some(31.75),
            samples: 57,
            interval_ms: 100,
        }
    }

    fn sample_e2e() -> E2eResult {
        E2eResult {
            engine: "5967".into(),
            model: "/models/tiny.gguf".into(),
            prompt_tokens: 512,
            eval_tokens: 128,
            ttft_ms: 120.5,
            tpot_ms: 18.0,
            tok_s: 55.6,
            ram_peak_bytes: 2_147_483_648,
            cpu_avg_pct: 70.0,
            power_avg_watts: None,
            elapsed_ms: 0.0,
        }
    }

    #[test]
    fn quick_report_roundtrip_carries_required_keys() {
        let report = BenchReport {
            schema: SCHEMA.to_string(),
            mode: "quick".to_string(),
            bandwidth: quick_bandwidth(),
            flops: None,
            resources: None,
            e2e: None,
        };
        let json = serde_json::to_string(&report).expect("serialize BenchReport");
        for key in ["schema", "mode", "bandwidth", "read_gbps", "buffer_mib"] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        let back: BenchReport = serde_json::from_str(&json).expect("deserialize BenchReport");
        assert_eq!(back, report);
    }

    #[test]
    fn full_report_roundtrip_carries_required_keys() {
        let report = BenchReport {
            schema: SCHEMA.to_string(),
            mode: "full".to_string(),
            bandwidth: full_bandwidth_value(),
            flops: Some(sample_flops()),
            resources: Some(sample_resources()),
            e2e: None,
        };
        let json = serde_json::to_string(&report).expect("serialize BenchReport");
        for key in ["schema", "mode", "bandwidth", "flops", "resources"] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        let back: BenchReport = serde_json::from_str(&json).expect("deserialize BenchReport");
        assert_eq!(back, report);
    }

    #[test]
    fn full_report_with_e2e_carries_the_e2e_block() {
        let report = BenchReport {
            schema: SCHEMA.to_string(),
            mode: "full".to_string(),
            bandwidth: full_bandwidth_value(),
            flops: Some(sample_flops()),
            resources: Some(sample_resources()),
            e2e: Some(sample_e2e()),
        };
        let json = serde_json::to_string(&report).expect("serialize BenchReport");
        // Required keys per the spec — every required field on e2e lands in
        // the JSON contract too.
        for key in [
            "schema",
            "mode",
            "bandwidth",
            "flops",
            "resources",
            "e2e",
            "ttft_ms",
            "tpot_ms",
            "tok_s",
            "engine",
            "model",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        let back: BenchReport = serde_json::from_str(&json).expect("deserialize BenchReport");
        assert_eq!(back, report);
    }

    #[test]
    fn e2e_none_default_keeps_the_report_compact() {
        // Without `--model`, the e2e block is absent — `skip_serializing_if`
        // keeps it out of the JSON instead of emitting `"e2e": null`.
        let report = BenchReport {
            schema: SCHEMA.to_string(),
            mode: "full".to_string(),
            bandwidth: full_bandwidth_value(),
            flops: None,
            resources: None,
            e2e: None,
        };
        let json = serde_json::to_string(&report).expect("serialize BenchReport");
        assert!(
            !json.contains("\"e2e\""),
            "e2e key must be absent when None, got {json}"
        );
        // But the rest of the document is intact.
        for key in ["schema", "mode", "bandwidth"] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
    }

    #[test]
    fn bandwidth_report_untagged_enum_picks_each_shape() {
        // The untagged enum must dispatch by shape: a `ReadResult` payload
        // does not parse as `FullBandwidthResult`, and vice versa.
        let quick_json = serde_json::to_string(&quick_bandwidth()).expect("serialize");
        let quick_back: BandwidthReport = serde_json::from_str(&quick_json).expect("deserialize");
        assert!(matches!(quick_back, BandwidthReport::Quick(_)));

        let full_json = serde_json::to_string(&full_bandwidth_value()).expect("serialize");
        let full_back: BandwidthReport = serde_json::from_str(&full_json).expect("deserialize");
        assert!(matches!(full_back, BandwidthReport::Full(_)));
    }

    #[test]
    fn report_deserializes_old_slice5_json_without_e2e() {
        // Pre-slice-6 JSON had no `schema` / `mode` / `e2e` keys at the
        // top level; a full-mode `FullBandwidthResult` shape still parses
        // (with the new wrapper-defaults applied).
        let old_json = r#"{
            "schema": "attune-bench/1",
            "mode": "full",
            "bandwidth": {
                "read":   {"median_gbps": 22.0, "best_gbps": 25.0},
                "write":  {"median_gbps": 14.0, "best_gbps": 16.0},
                "copy":   {"median_gbps": 20.0, "best_gbps": 23.0},
                "buffer_mib": 256,
                "threads": 8,
                "reps": 5
            }
        }"#;
        let back: BenchReport = serde_json::from_str(old_json).expect("deserialize");
        assert_eq!(back.schema, "attune-bench/1");
        assert_eq!(back.mode, "full");
        assert!(back.flops.is_none());
        assert!(back.resources.is_none());
        // e2e was absent from the input — back-compat default is None.
        assert!(back.e2e.is_none());
    }
}
