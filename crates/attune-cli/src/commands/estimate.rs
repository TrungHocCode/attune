//! `attune estimate` — roofline prediction (M3 slices 1-3).
//!
//! Thin CLI: parse the model/quant strings, resolve the hardware inputs, call
//! the library, and render either the JSON contract or the human lines. With
//! no overrides the bandwidth comes from the same quick probe as
//! `attune bench --quick` and the FP32 peak comes from `attune-hal`; if the
//! CPU exposes no usable theoretical peak the CLI asks for `--gflops` instead
//! of guessing.
//!
//! Slice 3 adds `--bench-json <PATH>` + `--calibrate`: the saved bench report
//! (either a bare [`BenchReport`] or the `{ "bench": … }` envelope written by
//! `attune bench --model --save`) supplies the measured e2e `tok_s` / TTFT,
//! and the library fits `eta_mem` / `eta_comp` from them. All math lives in
//! `attune-estimate`; this file only wires flags to it.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use attune_bench::{run_quick, BandwidthReport, BenchReport};
use attune_estimate::{
    apply_decode, apply_prefill, classify, decode_ai, parse_params, predict_prefill_ms, ridge_ai,
    CalibratedInputs, CalibrationReport, EstimateError, EstimateInputs, EstimateReport, HwInputs,
    Quant,
};
use attune_hal::HardwareInfo;

/// Resolve the FP32 peak: explicit override first, then the HAL's theoretical
/// estimate, then a clean error that names the fix.
fn resolve_gflops(gflops: Option<f64>) -> Result<f64> {
    match gflops {
        Some(value) => Ok(value),
        None => {
            let hw = HardwareInfo::detect()?;
            let theoretical = hw.cpu.theoretical_fp32_gflops;
            if theoretical.is_finite() && theoretical > 0.0 {
                Ok(theoretical)
            } else {
                Err(anyhow!(
                    "this CPU exposes no theoretical FP32 peak; pass --gflops <GFLOPS>"
                ))
            }
        }
    }
}

/// Resolve bandwidth: explicit override first, then the M2 quick probe.
fn resolve_bandwidth(bandwidth_gbps: Option<f64>) -> Result<f64> {
    match bandwidth_gbps {
        Some(value) => Ok(value),
        None => {
            let report = run_quick().context("failed to measure memory bandwidth")?;
            Ok(match report.bandwidth {
                BandwidthReport::Quick(read) => read.read_gbps,
                BandwidthReport::Full(full) => full.read.median_gbps,
            })
        }
    }
}

/// Parse the text of a saved bench report.
///
/// `attune bench --save` writes a bare [`BenchReport`] when it runs the
/// kernels only, but the `--model` path wraps it in `{ "llama": …, "bench": … }`
/// so the probe block stays alongside the measurements. Accept both shapes:
/// a consumer should not have to know which mode produced the file.
fn parse_bench_report(text: &str) -> Result<BenchReport> {
    let value: serde_json::Value =
        serde_json::from_str(text).context("bench JSON is not valid JSON")?;
    let candidate = value.get("bench").unwrap_or(&value);
    serde_json::from_value(candidate.clone()).context("not an attune bench report")
}

/// Read and parse `--bench-json`.
fn load_bench_report(path: &Path) -> Result<BenchReport> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read --bench-json {}", path.display()))?;
    parse_bench_report(&text)
        .with_context(|| format!("failed to load --bench-json {}", path.display()))
}

/// Fit calibration from a bench report.
///
/// Decode uses `eta_mem = measured / roofline`; prefill latency is inverted
/// (see [`CalibratedInputs::fit`]). A missing or unusable `e2e` block is an
/// expected condition and returns [`EstimateError::CalibrationUnavailable`] —
/// never a panic.
fn calibrate_with_bench(
    bench: &BenchReport,
    roofline_decode_tps: f64,
    roofline_prefill_ms: f64,
    path: &Path,
) -> Result<CalibrationReport, EstimateError> {
    let e2e = bench.e2e.as_ref().ok_or_else(|| {
        EstimateError::CalibrationUnavailable(format!(
            "{} has no e2e block; run `attune bench --model <gguf> --save <PATH>` first",
            path.display()
        ))
    })?;

    if !e2e.tok_s.is_finite() || e2e.tok_s <= 0.0 {
        return Err(EstimateError::CalibrationUnavailable(format!(
            "{} has no usable e2e decode rate (tok_s={}); rerun the bench with --model",
            path.display(),
            e2e.tok_s,
        )));
    }

    let measured_ttft_ms = if e2e.ttft_ms.is_finite() && e2e.ttft_ms > 0.0 {
        Some(e2e.ttft_ms)
    } else {
        None
    };
    let inputs = CalibratedInputs::fit(
        roofline_decode_tps,
        e2e.tok_s,
        Some(roofline_prefill_ms),
        measured_ttft_ms,
    )?;
    let calibrated_decode_tps = apply_decode(roofline_decode_tps, inputs.eta_mem);
    let calibrated_prefill_ms_ttft =
        measured_ttft_ms.map(|_| apply_prefill(roofline_prefill_ms, inputs.eta_comp));

    Ok(CalibrationReport {
        inputs,
        measured_decode_tps: e2e.tok_s,
        calibrated_decode_tps,
        measured_ttft_ms,
        calibrated_prefill_ms_ttft,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    params: &str,
    quant: &str,
    bandwidth_gbps: Option<f64>,
    gflops: Option<f64>,
    prompt_tokens: u64,
    _gen_tokens: u64,
    _ctx: u32,
    bench_json: Option<&Path>,
    calibrate: bool,
    json: bool,
) -> Result<()> {
    let params_count = parse_params(params)?;
    let quant: Quant = quant.parse()?;

    // Resolve the compute peak before the ~1 s bandwidth probe so a missing
    // theoretical peak fails fast.
    let flops_gflops = resolve_gflops(gflops)?;
    let bw_gbps = resolve_bandwidth(bandwidth_gbps)?;

    let hw = HwInputs {
        bw_gbps,
        flops_gflops,
    };
    let decode_tps = hw.predict_decode_tps(params_count, quant.bytes_per_param())?;
    let prefill_ms_ttft = predict_prefill_ms(params_count, prompt_tokens, flops_gflops)?;
    let ai_star = ridge_ai(flops_gflops, bw_gbps)?;
    let workload_ai = decode_ai(quant.bytes_per_param())?;
    let (bottleneck, margin_pct) = classify(workload_ai, ai_star);

    let mut report = EstimateReport::new(params.trim(), quant, decode_tps)
        .with_inputs(EstimateInputs {
            bw_gbps,
            gflops: flops_gflops,
        })
        .with_prefill_ms(prefill_ms_ttft)
        .with_ridge(ai_star, margin_pct)
        .with_bottleneck(bottleneck);

    if calibrate {
        let path = bench_json
            .ok_or_else(|| anyhow!("--calibrate requires --bench-json <PATH> with an e2e block"))?;
        let bench = load_bench_report(path)?;
        let calibration = calibrate_with_bench(&bench, decode_tps, prefill_ms_ttft, path)?;
        report = report.with_calibration(calibration);
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!(
        "Decode:      {:.1} tok/s ({} {})",
        report.decode_tps, report.model, report.quant,
    );
    if let Some(calibration) = report.calibration.as_ref() {
        println!(
            "Calibrated:  {:.1} tok/s (eta={:.3}, calibrated; measured {:.1} tok/s)",
            calibration.calibrated_decode_tps,
            calibration.inputs.eta_mem,
            calibration.measured_decode_tps,
        );
    }
    println!(
        "TTFT:        {:.1} ms (prefill {} tok)",
        prefill_ms_ttft, prompt_tokens
    );
    if let Some(calibration) = report.calibration.as_ref() {
        if let (Some(measured), Some(calibrated)) = (
            calibration.measured_ttft_ms,
            calibration.calibrated_prefill_ms_ttft,
        ) {
            println!(
                "Calibrated TTFT: {:.1} ms (eta={:.3}, calibrated; measured {:.1} ms)",
                calibrated, calibration.inputs.eta_comp, measured,
            );
        }
    }
    println!(
        "Bottleneck:  {} (decode AI {:.2} vs ridge {:.2} FLOP/byte, {:.1}% off)",
        report.bottleneck, workload_ai, ai_star, margin_pct,
    );
    println!(
        "Ridge:       {:.2} FLOP/byte ({:.1} GFLOPS / {:.1} GB/s)",
        ai_star, flops_gflops, bw_gbps,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use attune_bench::bandwidth::{FullBandwidthResult, KernelStats};
    use attune_bench::e2e::E2eResult;

    /// Canned bench report with an e2e block — no real JSON file is committed.
    fn bench_with_e2e(tok_s: f64, ttft_ms: f64) -> BenchReport {
        BenchReport {
            schema: attune_bench::SCHEMA.to_string(),
            mode: "full".to_string(),
            bandwidth: attune_bench::BandwidthReport::Full(FullBandwidthResult {
                read: KernelStats {
                    median_gbps: 20.0,
                    best_gbps: 22.0,
                },
                write: KernelStats {
                    median_gbps: 14.0,
                    best_gbps: 16.0,
                },
                copy: KernelStats {
                    median_gbps: 18.0,
                    best_gbps: 20.0,
                },
                buffer_mib: 256,
                threads: 8,
                reps: 5,
                flops: None,
                resources: None,
            }),
            flops: None,
            resources: None,
            e2e: Some(E2eResult {
                engine: "11804".to_string(),
                model: "/models/tiny.gguf".to_string(),
                prompt_tokens: 128,
                eval_tokens: 32,
                ttft_ms,
                tpot_ms: 1_000.0 / tok_s,
                tok_s,
                ram_peak_bytes: 1_073_741_824,
                cpu_avg_pct: 70.0,
                power_avg_watts: None,
            }),
        }
    }

    #[test]
    fn parses_a_bare_bench_report() {
        let report = bench_with_e2e(55.6, 120.0);
        let text = serde_json::to_string(&report).expect("serialize BenchReport");
        let parsed = parse_bench_report(&text).expect("parse bare report");
        assert_eq!(parsed, report);
    }

    #[test]
    fn parses_the_cli_envelope_written_by_save() {
        let report = bench_with_e2e(55.6, 120.0);
        let text = serde_json::json!({
            "llama": { "version": "11804", "model": "/models/tiny.gguf" },
            "bench": report,
        })
        .to_string();
        let parsed = parse_bench_report(&text).expect("parse envelope");
        assert_eq!(parsed.e2e.expect("e2e block").tok_s, 55.6);
    }

    #[test]
    fn calibration_fixture_matches_measured_by_construction() {
        let bench = bench_with_e2e(55.6, 120.0);
        let calibration = calibrate_with_bench(&bench, 246.9, 2764.8, Path::new("bench.json"))
            .expect("calibration should succeed");

        // 55.6 / 246.9 = 0.22519…
        assert!(
            (calibration.inputs.eta_mem - 0.225).abs() < 0.001,
            "got {}",
            calibration.inputs.eta_mem
        );
        // calibrated = raw × eta = measured, so the error is zero by
        // construction (up to float noise).
        assert!(
            (calibration.calibrated_decode_tps - 55.6).abs() < 1e-9,
            "got {}",
            calibration.calibrated_decode_tps
        );
        // Prefill is inverted and clamped: 2764.8 / 120 is far above MAX_ETA.
        assert_eq!(calibration.inputs.eta_comp, attune_estimate::MAX_ETA);
        assert_eq!(calibration.measured_ttft_ms, Some(120.0));
        assert!(calibration.calibrated_prefill_ms_ttft.is_some());
    }

    #[test]
    fn missing_e2e_block_is_a_clean_error() {
        let mut bench = bench_with_e2e(55.6, 120.0);
        bench.e2e = None;
        let err = calibrate_with_bench(&bench, 246.9, 2764.8, Path::new("bench.json"))
            .expect_err("missing e2e must be an error");
        assert!(
            matches!(err, EstimateError::CalibrationUnavailable(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_decode_rate_is_a_clean_error() {
        let bench = bench_with_e2e(0.0, 120.0);
        let err = calibrate_with_bench(&bench, 246.9, 2764.8, Path::new("bench.json"))
            .expect_err("zero tok/s must be an error");
        assert!(
            matches!(err, EstimateError::CalibrationUnavailable(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_ttft_only_skips_prefill_calibration() {
        let bench = bench_with_e2e(55.6, 0.0);
        let calibration = calibrate_with_bench(&bench, 246.9, 2764.8, Path::new("bench.json"))
            .expect("decode-only calibration should succeed");
        assert_eq!(calibration.inputs.eta_comp, 1.0);
        assert_eq!(calibration.measured_ttft_ms, None);
        assert_eq!(calibration.calibrated_prefill_ms_ttft, None);
    }
}
