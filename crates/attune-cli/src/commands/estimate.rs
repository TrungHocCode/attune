//! `attune estimate` — roofline prediction (M3 slices 1-4).
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
//! and the library fits `eta_mem` / `eta_comp` from them.
//!
//! Slice 4 adds confidence intervals and multi-config compares. A comma
//! separated `--quant` produces one report per format (a bare report when only
//! one format is named, so the single-quant contract is unchanged). Intervals
//! come from the full bench read-kernel spread `(best - median) / median` when
//! a full `--bench-json` is supplied, or from an explicit `--spread-pct`; the
//! quick probe cannot supply a spread, so those bounds stay `None` with the
//! reason recorded in `confidence.method`. All math lives in
//! `attune-estimate`; this file only wires flags to it.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use attune_bench::{run_quick, BandwidthReport, BenchReport};
use attune_estimate::{
    apply_decode, apply_prefill, classify, decode_ai, parse_params, parse_quants,
    predict_prefill_ms, ridge_ai, spread_pct_from_stats, CalibratedInputs, CalibrationReport,
    Confidence, EstimateError, EstimateInputs, EstimateReport, HwInputs, MultiEstimateReport,
    Quant,
};
use attune_hal::HardwareInfo;

/// Resolved FP32 peak plus the detected max clock it came from.
struct ComputeResolution {
    gflops: f64,
    /// Max CPU clock in GHz, or 0.0 when the user supplied `--gflops` (the
    /// override skips detection) or no clock could be read.
    clock_ghz: f64,
}

/// Resolve the FP32 peak: explicit override first, then the HAL estimate
/// (cores × SIMD path × detected max clock), then a clean error that names
/// the fix.
fn resolve_compute(gflops: Option<f64>) -> Result<ComputeResolution> {
    match gflops {
        Some(value) => Ok(ComputeResolution {
            gflops: value,
            clock_ghz: 0.0,
        }),
        None => {
            let hw = HardwareInfo::detect()?;
            let theoretical = hw.cpu.theoretical_fp32_gflops;
            if theoretical.is_finite() && theoretical > 0.0 {
                Ok(ComputeResolution {
                    gflops: theoretical,
                    clock_ghz: hw.cpu.clock_ghz,
                })
            } else {
                Err(anyhow!(
                    "cannot derive an FP32 peak from this CPU (unknown clock or no FMA SIMD path); pass --gflops <GFLOPS>"
                ))
            }
        }
    }
}

/// Resolved bandwidth plus the spread used for confidence intervals.
struct BandwidthResolution {
    gbps: f64,
    /// Measured (or explicit) half-spread in percent, or `None` when no
    /// KernelStats are available.
    spread_pct: Option<f64>,
    /// Human-readable provenance for `spread_pct`, including why it is
    /// unavailable.
    method: String,
}

/// Read the confidence spread from a saved full bench report.
///
/// Only a `Full` bandwidth report carries a median *and* a best sample; the
/// quick report is best-of-reps only, so it cannot produce a spread.
fn spread_from_bench(bench: &BenchReport) -> Option<f64> {
    match &bench.bandwidth {
        BandwidthReport::Full(full) => spread_pct_from_stats(&full.read),
        BandwidthReport::Quick(_) => None,
    }
}

/// Resolve bandwidth: explicit override first, then the M2 quick probe.
///
/// The spread source is independent of the bandwidth source: an explicit
/// `--spread-pct` wins, then a full `--bench-json` read-kernel spread. A
/// quick probe or a bare `--bandwidth-gbps` override has no KernelStats, so
/// the interval stays unavailable and `method` says why.
fn resolve_bandwidth(
    bandwidth_gbps: Option<f64>,
    spread_pct: Option<f64>,
    bench: Option<&BenchReport>,
) -> Result<BandwidthResolution> {
    let gbps = match bandwidth_gbps {
        Some(value) => value,
        None => {
            let report = run_quick().context("failed to measure memory bandwidth")?;
            match report.bandwidth {
                BandwidthReport::Quick(read) => read.read_gbps,
                BandwidthReport::Full(full) => full.read.median_gbps,
            }
        }
    };

    let (spread_pct, method) = if let Some(pct) = spread_pct {
        if !pct.is_finite() || pct < 0.0 || pct >= 100.0 {
            return Err(anyhow!(
                "--spread-pct must be a finite number >= 0 and < 100"
            ));
        }
        (Some(pct), "explicit --spread-pct".to_string())
    } else if let Some(bench) = bench {
        match spread_from_bench(bench) {
            Some(pct) => (
                Some(pct),
                "the --bench-json read kernel (best-median)/median".to_string(),
            ),
            None => (
                None,
                "unavailable: --bench-json has no median/best read spread; pass --spread-pct"
                    .to_string(),
            ),
        }
    } else if bandwidth_gbps.is_some() {
        (
            None,
            "unavailable: --bandwidth-gbps has no measured spread; pass --spread-pct or a full --bench-json"
                .to_string(),
        )
    } else {
        (
            None,
            "unavailable: the quick bandwidth probe reports no spread; pass --spread-pct or a full --bench-json"
                .to_string(),
        )
    };

    Ok(BandwidthResolution {
        gbps,
        spread_pct,
        method,
    })
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

/// Everything shared by every config in one `estimate` invocation.
struct ReportContext<'a> {
    label: &'a str,
    params: u64,
    prompt_tokens: u64,
    hw: HwInputs,
    /// Max CPU clock behind `hw.flops_gflops`; 0.0 for an explicit override.
    clock_ghz: f64,
    spread_pct: Option<f64>,
    spread_method: &'a str,
}

/// Build one config's report by calling the pure `attune-estimate` math.
fn build_report(ctx: &ReportContext<'_>, quant: Quant) -> Result<EstimateReport, EstimateError> {
    let decode_tps = ctx
        .hw
        .predict_decode_tps(ctx.params, quant.bytes_per_param())?;
    let prefill_ms_ttft = predict_prefill_ms(ctx.params, ctx.prompt_tokens, ctx.hw.flops_gflops)?;
    let ai_star = ridge_ai(ctx.hw.flops_gflops, ctx.hw.bw_gbps)?;
    let workload_ai = decode_ai(quant.bytes_per_param())?;
    let (bottleneck, margin_pct) = classify(workload_ai, ai_star);

    let report = EstimateReport::new(ctx.label, quant, decode_tps)
        .with_inputs(EstimateInputs {
            bw_gbps: ctx.hw.bw_gbps,
            gflops: ctx.hw.flops_gflops,
            clock_ghz: ctx.clock_ghz,
        })
        .with_prefill_ms(prefill_ms_ttft)
        .with_ridge(ai_star, margin_pct)
        .with_bottleneck(bottleneck);

    Ok(match ctx.spread_pct {
        Some(spread) => report
            .with_intervals(spread)
            .with_confidence(Confidence::measured(spread, ctx.spread_method)),
        None => report.with_confidence(Confidence::unavailable(ctx.spread_method)),
    })
}

/// Sort configs fastest-first; ties keep the order the user typed.
fn sort_by_decode(reports: &mut [EstimateReport]) {
    reports.sort_by(|a, b| {
        b.decode_tps
            .partial_cmp(&a.decode_tps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Render the JSON contract: a bare report for one config, or the
/// `{ schema, model, configs: [...] }` document for several.
fn render_json(model: &str, reports: &[EstimateReport]) -> Result<String> {
    match reports {
        [single] => Ok(serde_json::to_string_pretty(single)?),
        _ => Ok(serde_json::to_string_pretty(&MultiEstimateReport::new(
            model,
            reports.to_vec(),
        ))?),
    }
}

/// `point [lo, hi]`, or `point [-]` when no interval is available.
fn format_interval(value: f64, lo: Option<f64>, hi: Option<f64>) -> String {
    match (lo, hi) {
        (Some(lo), Some(hi)) => format!("{value:.1} [{lo:.1}, {hi:.1}]"),
        _ => format!("{value:.1} [-]"),
    }
}

/// Human-readable peak line. A non-zero clock means the HAL derived the
/// GFLOPS from the detected max clock; 0.0 means the user passed `--gflops`.
fn format_peak(gflops: f64, clock_ghz: f64) -> String {
    if clock_ghz.is_finite() && clock_ghz > 0.0 {
        format!("{gflops:.1} GFLOPS @ {clock_ghz:.2} GHz max")
    } else {
        format!("{gflops:.1} GFLOPS (explicit --gflops)")
    }
}

/// Existing single-config human output. Interval lines are only added when a
/// measured spread produced bounds, so the no-spread output is unchanged.
fn print_single(report: &EstimateReport, prompt_tokens: u64) {
    let bw_gbps = report.inputs.map_or(0.0, |inputs| inputs.bw_gbps);
    let flops_gflops = report.inputs.map_or(0.0, |inputs| inputs.gflops);
    let clock_ghz = report.inputs.map_or(0.0, |inputs| inputs.clock_ghz);
    let ai_star = report.ridge_ai.unwrap_or(0.0);
    let margin_pct = report.margin_pct.unwrap_or(0.0);
    let workload_ai = decode_ai(report.quant.bytes_per_param()).unwrap_or(0.0);
    let prefill_ms_ttft = report.prefill_ms_ttft.unwrap_or(0.0);

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
    if let (Some(lo), Some(hi)) = (report.decode_tps_lo, report.decode_tps_hi) {
        let level = report
            .confidence
            .as_ref()
            .map_or("unknown", |confidence| confidence.level.as_str());
        println!("Decode range: {lo:.1} - {hi:.1} tok/s ({level} confidence)");
    }
    println!("TTFT:        {prefill_ms_ttft:.1} ms (prefill {prompt_tokens} tok)");
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
    if let (Some(lo), Some(hi)) = (report.prefill_ms_lo, report.prefill_ms_hi) {
        println!("TTFT range:  {lo:.1} - {hi:.1} ms");
    }
    println!(
        "Bottleneck:  {} (decode AI {:.2} vs ridge {:.2} FLOP/byte, {:.1}% off)",
        report.bottleneck, workload_ai, ai_star, margin_pct,
    );
    println!("Peak:        {}", format_peak(flops_gflops, clock_ghz));
    println!(
        "Ridge:       {:.2} FLOP/byte ({:.1} GFLOPS / {:.1} GB/s)",
        ai_star, flops_gflops, bw_gbps,
    );
}

/// Multi-config human output, already sorted fastest-first by the caller.
fn print_table(reports: &[EstimateReport], prompt_tokens: u64) {
    let model = reports.first().map_or("", |report| report.model.as_str());
    println!("Model:  {model} (prompt {prompt_tokens} tok)");
    if let Some(inputs) = reports.first().and_then(|report| report.inputs) {
        println!(
            "Inputs: {:.1} GB/s, {}",
            inputs.bw_gbps,
            format_peak(inputs.gflops, inputs.clock_ghz)
        );
    }
    println!();
    println!(
        "{:<8} {:<26} {:<26} {:<17} Confidence",
        "Quant", "Decode tok/s [lo, hi]", "TTFT ms [lo, hi]", "Bottleneck"
    );
    for report in reports {
        let decode = format_interval(
            report.decode_tps,
            report.decode_tps_lo,
            report.decode_tps_hi,
        );
        let prefill = report.prefill_ms_ttft.map_or_else(
            || "-".to_string(),
            |value| format_interval(value, report.prefill_ms_lo, report.prefill_ms_hi),
        );
        let level = report
            .confidence
            .as_ref()
            .map_or("unavailable", |confidence| confidence.level.as_str());
        // `Quant`/`Bottleneck` have custom `Display` impls that bypass
        // formatter padding, so pad their string forms explicitly.
        println!(
            "{:<8} {:<26} {:<26} {:<17} {}",
            report.quant.as_str(),
            decode,
            prefill,
            report.bottleneck.to_string(),
            level,
        );
    }

    if let Some(confidence) = reports
        .first()
        .and_then(|report| report.confidence.as_ref())
    {
        println!("\nConfidence: {} ({})", confidence.level, confidence.method);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    params: &str,
    quant: &str,
    bandwidth_gbps: Option<f64>,
    spread_pct: Option<f64>,
    gflops: Option<f64>,
    prompt_tokens: u64,
    _gen_tokens: u64,
    _ctx: u32,
    bench_json: Option<&Path>,
    calibrate: bool,
    json: bool,
) -> Result<()> {
    let params_count = parse_params(params)?;
    let quants = parse_quants(quant)?;

    // Resolve the compute peak before the ~1 s bandwidth probe so a missing
    // theoretical peak fails fast.
    let compute = resolve_compute(gflops)?;

    // A supplied bench report is also the confidence-spread source, so load it
    // before resolving bandwidth.
    let bench = match bench_json {
        Some(path) => Some(load_bench_report(path)?),
        None => None,
    };
    let bandwidth = resolve_bandwidth(bandwidth_gbps, spread_pct, bench.as_ref())?;

    let hw = HwInputs {
        bw_gbps: bandwidth.gbps,
        flops_gflops: compute.gflops,
    };
    let ctx = ReportContext {
        label: params.trim(),
        params: params_count,
        prompt_tokens,
        hw,
        clock_ghz: compute.clock_ghz,
        spread_pct: bandwidth.spread_pct,
        spread_method: &bandwidth.method,
    };

    let mut reports = Vec::with_capacity(quants.len());
    for quant in quants {
        let mut report = build_report(&ctx, quant)?;
        if calibrate {
            let bench = bench.as_ref().ok_or_else(|| {
                anyhow!("--calibrate requires --bench-json <PATH> with an e2e block")
            })?;
            let path = bench_json.ok_or_else(|| {
                anyhow!("--calibrate requires --bench-json <PATH> with an e2e block")
            })?;
            let calibration = calibrate_with_bench(
                bench,
                report.decode_tps,
                report.prefill_ms_ttft.unwrap_or(0.0),
                path,
            )?;
            report = report.with_calibration(calibration);
        }
        reports.push(report);
    }

    sort_by_decode(&mut reports);

    if json {
        println!("{}", render_json(params.trim(), &reports)?);
        return Ok(());
    }

    if let [report] = reports.as_slice() {
        print_single(report, prompt_tokens);
    } else {
        print_table(&reports, prompt_tokens);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use attune_bench::bandwidth::{FullBandwidthResult, KernelStats, ReadResult};
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
                elapsed_ms: 0.0,
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

    fn context(spread_pct: Option<f64>, spread_method: &'static str) -> ReportContext<'static> {
        ReportContext {
            label: "135M",
            params: 135_000_000,
            prompt_tokens: 512,
            hw: HwInputs {
                bw_gbps: 20.0,
                flops_gflops: 50.0,
            },
            clock_ghz: 4.0,
            spread_pct,
            spread_method,
        }
    }

    #[test]
    fn format_peak_names_the_clock_or_the_override() {
        assert_eq!(format_peak(1024.0, 4.0), "1024.0 GFLOPS @ 4.00 GHz max");
        assert_eq!(format_peak(50.0, 0.0), "50.0 GFLOPS (explicit --gflops)");
        assert_eq!(
            format_peak(50.0, f64::NAN),
            "50.0 GFLOPS (explicit --gflops)"
        );
    }

    #[test]
    fn build_report_attaches_intervals_and_confidence() {
        let ctx = context(Some(10.0), "test spread");
        let report = build_report(&ctx, Quant::Q4_K_M).expect("valid report");

        let expected = ctx
            .hw
            .predict_decode_tps(ctx.params, Quant::Q4_K_M.bytes_per_param())
            .expect("expected decode rate");
        assert!((report.decode_tps - expected).abs() < 1e-9);
        let (lo, hi) = (
            report.decode_tps_lo.expect("decode lo"),
            report.decode_tps_hi.expect("decode hi"),
        );
        assert!((lo - report.decode_tps * 0.9).abs() < 1e-9, "lo {lo}");
        assert!((hi - report.decode_tps * 1.1).abs() < 1e-9, "hi {hi}");
        assert!(report.prefill_ms_lo.is_some());
        assert!(report.prefill_ms_hi.is_some());

        let confidence = report.confidence.expect("confidence");
        assert_eq!(confidence.level, "medium");
        assert!(
            confidence.method.contains("10.0%"),
            "method: {}",
            confidence.method
        );
    }

    #[test]
    fn build_report_without_spread_leaves_bounds_none() {
        let ctx = context(None, "unavailable: quick probe");
        let report = build_report(&ctx, Quant::Q4_K_M).expect("valid report");

        assert_eq!(report.decode_tps_lo, None);
        assert_eq!(report.decode_tps_hi, None);
        assert_eq!(report.prefill_ms_lo, None);
        assert_eq!(report.prefill_ms_hi, None);

        let confidence = report.confidence.expect("confidence");
        assert_eq!(confidence.level, "unavailable");
        assert!(
            confidence.method.contains("unavailable"),
            "method: {}",
            confidence.method
        );
    }

    #[test]
    fn spread_from_bench_uses_the_full_read_kernel() {
        // read: median 20, best 22 => (22 - 20) / 20 = 10%.
        let bench = bench_with_e2e(55.6, 120.0);
        let spread = spread_from_bench(&bench).expect("full report spread");
        assert!((spread - 10.0).abs() < 1e-12, "got {spread}");
    }

    #[test]
    fn spread_from_quick_bench_is_unavailable() {
        let mut bench = bench_with_e2e(55.6, 120.0);
        bench.bandwidth = BandwidthReport::Quick(ReadResult {
            read_gbps: 20.0,
            buffer_mib: 256,
            threads: 8,
            reps: 3,
        });
        assert_eq!(spread_from_bench(&bench), None);
    }

    #[test]
    fn resolve_bandwidth_honours_explicit_spread_without_probing() {
        let resolved = resolve_bandwidth(Some(20.0), Some(13.6), None).expect("explicit override");
        assert_eq!(resolved.gbps, 20.0);
        assert_eq!(resolved.spread_pct, Some(13.6));
        assert!(
            resolved.method.contains("explicit --spread-pct"),
            "method: {}",
            resolved.method
        );
    }

    #[test]
    fn resolve_bandwidth_uses_full_bench_stats() {
        let bench = bench_with_e2e(55.6, 120.0);
        let resolved =
            resolve_bandwidth(Some(20.0), None, Some(&bench)).expect("bench spread should resolve");
        assert_eq!(resolved.gbps, 20.0);
        assert!((resolved.spread_pct.expect("spread") - 10.0).abs() < 1e-12);
        assert!(
            resolved.method.contains("--bench-json"),
            "method: {}",
            resolved.method
        );
    }

    #[test]
    fn resolve_bandwidth_without_stats_notes_the_override() {
        let resolved = resolve_bandwidth(Some(20.0), None, None).expect("override");
        assert_eq!(resolved.spread_pct, None);
        assert!(
            resolved.method.contains("--bandwidth-gbps"),
            "method: {}",
            resolved.method
        );
    }

    #[test]
    fn resolve_bandwidth_rejects_bad_explicit_spread() {
        for pct in [-1.0, 100.0, f64::NAN, f64::INFINITY] {
            assert!(
                resolve_bandwidth(Some(20.0), Some(pct), None).is_err(),
                "spread {pct} should be rejected"
            );
        }
    }

    #[test]
    fn render_json_keeps_the_single_quant_shape() {
        let report = build_report(&context(None, "unavailable: quick probe"), Quant::Q4_K_M)
            .expect("report");
        let json = render_json("135M", &[report]).expect("render");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

        assert!(value.get("configs").is_none(), "single shape: {json}");
        assert!(value.get("decode_tps").is_some(), "single shape: {json}");
        let back: EstimateReport = serde_json::from_str(&json).expect("deserialize single");
        assert_eq!(back.model, "135M");
    }

    #[test]
    fn render_json_wraps_multiple_configs() {
        let reports = vec![
            build_report(&context(None, "unavailable: quick probe"), Quant::Q4_K_M)
                .expect("report"),
            build_report(&context(None, "unavailable: quick probe"), Quant::Fp16).expect("report"),
        ];
        let json = render_json("135M", &reports).expect("render");
        let back: MultiEstimateReport =
            serde_json::from_str(&json).expect("deserialize multi-config");
        assert_eq!(back.configs.len(), 2);
        assert_eq!(back.model, "135M");
    }

    #[test]
    fn sort_by_decode_orders_fastest_first() {
        let mut reports = vec![
            EstimateReport::new("135M", Quant::Fp16, 50.0),
            EstimateReport::new("135M", Quant::Q4_K_M, 250.0),
            EstimateReport::new("135M", Quant::Q8_0, 100.0),
        ];
        sort_by_decode(&mut reports);
        assert_eq!(
            reports.iter().map(|r| r.quant).collect::<Vec<_>>(),
            vec![Quant::Q4_K_M, Quant::Q8_0, Quant::Fp16]
        );
    }
}
