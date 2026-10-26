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
    apply_decode, apply_prefill, clamped, classify, decode_ai, kv_bytes_per_token, parse_arch,
    parse_model, parse_params, parse_quants, predict_prefill_ms_quant, probe_gguf_file, ridge_ai,
    spread_pct_from_stats, CalibratedInputs, CalibrationReport, Confidence, EstimateError,
    EstimateInputs, EstimateReport, GgufProbe, HwInputs, MultiEstimateReport, Quant, RamFit,
};
use attune_hal::cpu::{theoretical_fp32_gflops, BASELINE_CLOCK_GHZ};
use attune_hal::HardwareInfo;

/// Resolved FP32 peak plus the detected max clock it came from.
struct ComputeResolution {
    gflops: f64,
    /// Max CPU clock in GHz, or 0.0 when the user supplied `--gflops` (the
    /// override skips detection) or no clock could be read.
    clock_ghz: f64,
    /// True when the peak fell back to [`BASELINE_CLOCK_GHZ`] because the
    /// clock is unknown. The warning in [`CLOCK_FALLBACK_WARNING`] must be
    /// surfaced then — a silent baseline once skewed TTFT 2.5x.
    clock_fallback: bool,
}

/// Warning attached to every report (pretty + JSON) built on the 1 GHz
/// baseline fallback.
pub const CLOCK_FALLBACK_WARNING: &str =
    "clock unknown, assuming 1 GHz baseline — pass --gflops for real numbers";

/// Notice attached when no `--arch` was supplied: the decode prediction then
/// omits the KV-cache read, which is optimistic at large contexts.
pub const KV_EXCLUDED_WARNING: &str = "KV excluded, unknown arch (pass --arch)";

/// Notice attached when a probed GGUF has no usable attention head count, so
/// the KV-cache shape cannot be derived from the header.
pub const GGUF_KV_EXCLUDED_WARNING: &str = "KV excluded, GGUF has no usable attention head count";

/// Peak from the 1 GHz baseline for CPUs whose clock is unknown but whose
/// SIMD path is known. `None` when even the baseline yields nothing (no FMA
/// path), in which case the caller keeps the loud `--gflops` error.
fn baseline_peak(physical_cores: u32, simd_flags: &[String]) -> Option<f64> {
    let peak = theoretical_fp32_gflops(physical_cores, simd_flags, BASELINE_CLOCK_GHZ);
    (peak.is_finite() && peak > 0.0).then_some(peak)
}

/// Resolve the FP32 peak: explicit override first, then the HAL estimate
/// (cores × SIMD path × detected max clock), then the 1 GHz baseline with a
/// loud warning when only the clock is missing, then a clean error naming
/// the fix when not even the baseline applies (no FMA SIMD path).
fn resolve_compute(gflops: Option<f64>) -> Result<ComputeResolution> {
    match gflops {
        Some(value) => Ok(ComputeResolution {
            gflops: value,
            clock_ghz: 0.0,
            clock_fallback: false,
        }),
        None => {
            let hw = HardwareInfo::detect()?;
            let theoretical = hw.cpu.theoretical_fp32_gflops;
            if theoretical.is_finite() && theoretical > 0.0 {
                Ok(ComputeResolution {
                    gflops: theoretical,
                    clock_ghz: hw.cpu.clock_ghz,
                    clock_fallback: false,
                })
            } else if let Some(peak) = baseline_peak(hw.cpu.physical_cores, &hw.cpu.simd_flags) {
                Ok(ComputeResolution {
                    gflops: peak,
                    clock_ghz: BASELINE_CLOCK_GHZ,
                    clock_fallback: true,
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

/// KV-cache traffic resolved from an optional `--arch` preset.
struct KvResolution {
    /// Bytes of KV cache read per generated token; 0 when unknown.
    per_token: f64,
    /// Notice explaining the omission, when no preset was supplied.
    warning: Option<&'static str>,
}

/// Resolve `--arch` into a per-token KV size.
///
/// Omitting the flag is allowed but never silent: the decode prediction then
/// ignores KV traffic and carries [`KV_EXCLUDED_WARNING`].
fn resolve_kv(arch: Option<&str>) -> Result<KvResolution, EstimateError> {
    match arch {
        Some(name) => Ok(KvResolution {
            per_token: kv_bytes_per_token(parse_arch(name)?),
            warning: None,
        }),
        None => Ok(KvResolution {
            per_token: 0.0,
            warning: Some(KV_EXCLUDED_WARNING),
        }),
    }
}

/// Model inputs resolved from either `--model` or `--params` + `--arch`.
struct ModelResolution<'a> {
    /// Report label: catalog name, GGUF path or the literal `--params` text.
    label: &'a str,
    /// Parameter count used for the weight footprint.
    params: u64,
    /// KV-cache traffic for the resolved architecture.
    kv: KvResolution,
    /// Probe result when `--model` was a `.gguf` file path. Drives the
    /// human-readable probe line and the catalog mismatch warning.
    gguf: Option<GgufProbe>,
}

/// True when `--model` points at an existing `.gguf` file (case-insensitive
/// extension), which is probed instead of being looked up as a catalog name.
fn as_gguf_path(value: &str) -> Option<&Path> {
    let path = Path::new(value.trim());
    let is_gguf = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"));
    (is_gguf && path.is_file()).then_some(path)
}

/// Resolve `--model` first, then fall back to `--params` + `--arch`.
///
/// An existing `.gguf` file is probed from its header: `general.parameter_count`
/// supplies the size when present, otherwise a catalog entry inferred from the
/// file name does. The file's KV shape wins over a catalog disagreement, which
/// is recorded as a warning. Plain names keep the catalog path; the two flags
/// stay the escape hatch for models outside both. Clap makes `--model` mutually
/// exclusive with `--params` / `--arch`, so the paths cannot silently disagree.
/// The catalog's `ctx_default` is deliberately not applied here yet: `--ctx`
/// remains the context source until the wiring slice consumes it.
fn resolve_model<'a>(
    model: Option<&'a str>,
    params: Option<&'a str>,
    arch: Option<&str>,
) -> Result<ModelResolution<'a>> {
    if let Some(name) = model {
        if let Some(path) = as_gguf_path(name) {
            let probe = probe_gguf_file(path)
                .map_err(|err| anyhow!("failed to probe {}: {err}", path.display()))?;
            let params = probe.params().ok_or_else(|| {
                anyhow!(
                    "{} has no general.parameter_count and no catalog match; use --params <SIZE> --arch <NAME> instead of --model",
                    path.display()
                )
            })?;
            let kv = match probe.arch {
                Some(arch) => KvResolution {
                    per_token: kv_bytes_per_token(&arch),
                    warning: None,
                },
                None => KvResolution {
                    per_token: 0.0,
                    warning: Some(GGUF_KV_EXCLUDED_WARNING),
                },
            };
            return Ok(ModelResolution {
                label: name,
                params,
                kv,
                gguf: Some(probe),
            });
        }

        let entry = parse_model(name).map_err(|err| {
            anyhow!("{err}; pass --params <SIZE> --arch <NAME> for a model outside the catalog")
        })?;
        return Ok(ModelResolution {
            label: entry.name,
            params: entry.params,
            kv: KvResolution {
                per_token: kv_bytes_per_token(&entry.arch),
                warning: None,
            },
            gguf: None,
        });
    }

    let params = params.ok_or_else(|| anyhow!("pass --model <NAME> or --params <SIZE>"))?;
    Ok(ModelResolution {
        label: params.trim(),
        params: parse_params(params)?,
        kv: resolve_kv(arch)?,
        gguf: None,
    })
}

/// Best-effort available RAM for the OOM check. `None` when the HAL cannot
/// read it (0), in which case the report simply omits `ram_fit` rather than
/// claiming an OOM from unknown numbers.
fn detect_available_ram_bytes() -> Option<f64> {
    attune_hal::memory::detect()
        .ok()
        .map(|memory| memory.available_bytes as f64)
        .filter(|bytes| bytes.is_finite() && *bytes > 0.0)
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
///
/// Cross-workload application is allowed but loud: `bench_quant` (the format
/// the bench measured, when the user passes `--bench-quant`) and the bench's
/// prompt length are compared against this estimate, and any mismatch — or
/// an eta sitting exactly on a clamp bound — becomes a `warnings` entry
/// instead of a silent mis-calibration.
#[allow(clippy::too_many_arguments)]
fn calibrate_with_bench(
    bench: &BenchReport,
    roofline_decode_tps: f64,
    roofline_prefill_ms: f64,
    path: &Path,
    bench_quant: Option<Quant>,
    estimate_quant: Quant,
    estimate_prompt_tokens: u64,
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

    let mut warnings = Vec::new();
    if let Some(bench_quant) = bench_quant {
        if bench_quant != estimate_quant {
            warnings.push(format!(
                "eta from {bench_quant} applied to {estimate_quant}"
            ));
        }
    }
    if e2e.prompt_tokens != estimate_prompt_tokens {
        warnings.push(format!(
            "prompt mismatch {} vs {}",
            e2e.prompt_tokens, estimate_prompt_tokens
        ));
    }
    if clamped(inputs.eta_mem) || (measured_ttft_ms.is_some() && clamped(inputs.eta_comp)) {
        warnings.push("eta clamped, unreliable".to_string());
    }

    Ok(CalibrationReport {
        inputs,
        measured_decode_tps: e2e.tok_s,
        calibrated_decode_tps,
        measured_ttft_ms,
        calibrated_prefill_ms_ttft,
        warnings,
    })
}

/// Everything shared by every config in one `estimate` invocation.
struct ReportContext<'a> {
    label: &'a str,
    params: u64,
    prompt_tokens: u64,
    /// Context length used for KV sizing.
    ctx: u32,
    /// Bytes of KV cache read per generated token; 0 when `--arch` is absent.
    kv_per_token: f64,
    hw: HwInputs,
    /// Max CPU clock behind `hw.flops_gflops`; 0.0 for an explicit override.
    clock_ghz: f64,
    /// Available RAM for the OOM check, when detection succeeded.
    ram_avail_bytes: Option<f64>,
    spread_pct: Option<f64>,
    spread_method: &'a str,
}

/// Build one config's report by calling the pure `attune-estimate` math.
fn build_report(ctx: &ReportContext<'_>, quant: Quant) -> Result<EstimateReport, EstimateError> {
    let bytes_per_param = quant.bytes_per_param();
    let decode_tps = ctx.hw.predict_decode_tps_with_kv(
        ctx.params,
        bytes_per_param,
        ctx.kv_per_token,
        ctx.ctx,
    )?;
    let prefill_ms_ttft =
        predict_prefill_ms_quant(ctx.params, ctx.prompt_tokens, ctx.hw.flops_gflops, quant)?;
    let ai_star = ridge_ai(ctx.hw.flops_gflops, ctx.hw.bw_gbps)?;
    let workload_ai = decode_ai(bytes_per_param)?;
    let (bottleneck, margin_pct) = classify(workload_ai, ai_star);

    let mut report = EstimateReport::new(ctx.label, quant, decode_tps)
        .with_inputs(EstimateInputs {
            bw_gbps: ctx.hw.bw_gbps,
            gflops: ctx.hw.flops_gflops,
            clock_ghz: ctx.clock_ghz,
        })
        .with_prefill_ms(prefill_ms_ttft)
        .with_ridge(ai_star, margin_pct)
        .with_bottleneck(bottleneck);

    // Weights + KV must fit in RAM before tok/s means anything. Sizing lives
    // in `attune-estimate`; the CLI only supplies the detected available RAM.
    if let Some(avail_bytes) = ctx.ram_avail_bytes {
        if let Some(ram_fit) =
            RamFit::for_model(ctx.params, quant, ctx.kv_per_token, ctx.ctx, avail_bytes)
        {
            report = report.with_ram_fit(ram_fit);
        }
    }

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
/// A fallback peak is labelled as a baseline so it is never mistaken for a
/// measured max clock.
fn format_peak(gflops: f64, clock_ghz: f64, clock_fallback: bool) -> String {
    if clock_fallback {
        format!("{gflops:.1} GFLOPS @ {clock_ghz:.2} GHz baseline (clock unknown)")
    } else if clock_ghz.is_finite() && clock_ghz > 0.0 {
        format!("{gflops:.1} GFLOPS @ {clock_ghz:.2} GHz max")
    } else {
        format!("{gflops:.1} GFLOPS (explicit --gflops)")
    }
}

/// Compact parameter label for the probe line (`135M`, `8.03B`, …).
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1_000_000.0)
    } else {
        params.to_string()
    }
}

/// Human-readable summary of a probed GGUF header.
///
/// Shows the facts the file won with; `(catalog)` marks a parameter count that
/// came from the catalog because the file did not store
/// `general.parameter_count`.
fn print_probe(probe: &GgufProbe, params: u64) {
    let facts = &probe.facts;
    let source = if facts.param_estimate == 0 {
        " (catalog)"
    } else {
        ""
    };
    println!(
        "Probed:      {} | {} blocks | {} KV heads | {} params{}",
        facts.architecture,
        facts.block_count,
        facts.n_kv_heads,
        format_params(params),
        source,
    );
}

/// Existing single-config human output. Interval lines are only added when a
/// measured spread produced bounds, so the no-spread output is unchanged.
/// Warnings print last so a degraded input is impossible to miss.
fn print_single(report: &EstimateReport, prompt_tokens: u64, clock_fallback: bool) {
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
    println!(
        "Peak:        {}",
        format_peak(flops_gflops, clock_ghz, clock_fallback)
    );
    println!(
        "Ridge:       {:.2} FLOP/byte ({:.1} GFLOPS / {:.1} GB/s)",
        ai_star, flops_gflops, bw_gbps,
    );
    if let Some(fit) = report.ram_fit {
        println!(
            "RAM:         {:.1} GB needed (weights {:.1} + KV {:.1}), {:.1} GB available ({})",
            fit.peak_bytes / 1e9,
            fit.weight_bytes / 1e9,
            fit.kv_bytes / 1e9,
            fit.avail_bytes / 1e9,
            if fit.fits { "fits" } else { "OOM" },
        );
    }
    print_warnings(report);
}

/// Print the report's non-fatal notices, if any — report-level first, then
/// the calibration block's (same strings as the JSON `warnings` fields).
fn print_warnings(report: &EstimateReport) {
    for warning in &report.warnings {
        println!("Warning:     {warning}");
    }
    if let Some(calibration) = report.calibration.as_ref() {
        for warning in &calibration.warnings {
            println!("Warning:     {warning}");
        }
    }
}

/// Multi-config human output, already sorted fastest-first by the caller.
fn print_table(reports: &[EstimateReport], prompt_tokens: u64, clock_fallback: bool) {
    let model = reports.first().map_or("", |report| report.model.as_str());
    println!("Model:  {model} (prompt {prompt_tokens} tok)");
    if let Some(inputs) = reports.first().and_then(|report| report.inputs) {
        println!(
            "Inputs: {:.1} GB/s, {}",
            inputs.bw_gbps,
            format_peak(inputs.gflops, inputs.clock_ghz, clock_fallback)
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
    // Warnings are input-level (identical across configs), so the first
    // report represents all of them.
    if let Some(first) = reports.first() {
        for warning in &first.warnings {
            println!("Warning:     {warning}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Option<&str>,
    params: Option<&str>,
    quant: &str,
    bandwidth_gbps: Option<f64>,
    spread_pct: Option<f64>,
    gflops: Option<f64>,
    arch: Option<&str>,
    prompt_tokens: u64,
    _gen_tokens: u64,
    ctx: u32,
    bench_json: Option<&Path>,
    bench_quant: Option<&str>,
    calibrate: bool,
    json: bool,
) -> Result<()> {
    let ModelResolution {
        label,
        params: params_count,
        kv,
        gguf,
    } = resolve_model(model, params, arch)?;
    let quants = parse_quants(quant)?;
    // The bench's own quant is only known when the user states it; without
    // the flag no quant-mismatch warning can be produced (and none is).
    let bench_quant = bench_quant
        .map(str::parse::<Quant>)
        .transpose()
        .map_err(|e: EstimateError| anyhow!("{e}"))?;

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
    let context = ReportContext {
        label,
        params: params_count,
        prompt_tokens,
        ctx,
        kv_per_token: kv.per_token,
        hw,
        clock_ghz: compute.clock_ghz,
        ram_avail_bytes: detect_available_ram_bytes(),
        spread_pct: bandwidth.spread_pct,
        spread_method: &bandwidth.method,
    };

    let mut reports = Vec::with_capacity(quants.len());
    for quant in quants {
        let mut report = build_report(&context, quant)?;
        if let Some(probe) = gguf.as_ref() {
            if let Some(warning) = probe.warning.as_deref() {
                report = report.with_warning(warning);
            }
        }
        if let Some(warning) = kv.warning {
            report = report.with_warning(warning);
        }
        if compute.clock_fallback {
            report = report.with_warning(CLOCK_FALLBACK_WARNING);
        }
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
                bench_quant,
                quant,
                prompt_tokens,
            )?;
            report = report.with_calibration(calibration);
        }
        reports.push(report);
    }

    sort_by_decode(&mut reports);

    if json {
        println!("{}", render_json(label, &reports)?);
        return Ok(());
    }

    if let Some(probe) = gguf.as_ref() {
        print_probe(probe, params_count);
    }
    if let [report] = reports.as_slice() {
        print_single(report, prompt_tokens, compute.clock_fallback);
    } else {
        print_table(&reports, prompt_tokens, compute.clock_fallback);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use attune_bench::bandwidth::{FullBandwidthResult, KernelStats, ReadResult};
    use attune_bench::e2e::E2eResult;

    const TYPE_STRING: u32 = 8;
    const TYPE_UINT32: u32 = 4;
    const TYPE_UINT64: u32 = 10;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_key(bytes: &mut Vec<u8>, key: &str, value_type: u32) {
        push_string(bytes, key);
        push_u32(bytes, value_type);
    }

    /// Build a fake GGUF header (no tensor payload) for the CLI tests.
    fn fake_gguf(
        architecture: &str,
        block_count: u32,
        embedding_length: u32,
        head_count: u32,
        kv_heads: Option<u32>,
        params: Option<u64>,
    ) -> Vec<u8> {
        let kv_count =
            4 + if kv_heads.is_some() { 1 } else { 0 } + if params.is_some() { 1 } else { 0 };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0); // tensor_count
        push_u64(&mut bytes, kv_count);

        push_key(&mut bytes, "general.architecture", TYPE_STRING);
        push_string(&mut bytes, architecture);
        push_key(
            &mut bytes,
            &format!("{architecture}.block_count"),
            TYPE_UINT32,
        );
        push_u32(&mut bytes, block_count);
        push_key(
            &mut bytes,
            &format!("{architecture}.embedding_length"),
            TYPE_UINT32,
        );
        push_u32(&mut bytes, embedding_length);
        push_key(
            &mut bytes,
            &format!("{architecture}.attention.head_count"),
            TYPE_UINT32,
        );
        push_u32(&mut bytes, head_count);
        if let Some(kv_heads) = kv_heads {
            push_key(
                &mut bytes,
                &format!("{architecture}.attention.head_count_kv"),
                TYPE_UINT32,
            );
            push_u32(&mut bytes, kv_heads);
        }
        if let Some(params) = params {
            push_key(&mut bytes, "general.parameter_count", TYPE_UINT64);
            push_u64(&mut bytes, params);
        }
        bytes
    }

    fn write_temp_gguf(tag: &str, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-cli-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write fake GGUF");
        path
    }

    fn cleanup_temp_gguf(path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

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
        let calibration = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            128,
        )
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
        // The clamped prefill eta must warn (bench prompt 128 matches the
        // estimate prompt here, so clamping is the only warning).
        assert_eq!(
            calibration.warnings,
            vec!["eta clamped, unreliable".to_string()]
        );
    }

    #[test]
    fn matching_workload_calibrates_without_warnings() {
        // eta_mem = 55.6/246.9 ≈ 0.225 (unclamped); eta_comp neutral via
        // missing TTFT, so nothing is clamped and prompts match (128).
        let bench = bench_with_e2e(55.6, 0.0);
        let calibration = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            128,
        )
        .expect("calibration should succeed");
        assert!(
            calibration.warnings.is_empty(),
            "got {:?}",
            calibration.warnings
        );
    }

    #[test]
    fn quant_mismatch_warns_but_still_calibrates() {
        let bench = bench_with_e2e(55.6, 120.0);
        let calibration = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            Some(Quant::Q4_K_M),
            Quant::Fp16,
            128,
        )
        .expect("mismatch must warn, not fail");
        assert!(
            calibration
                .warnings
                .iter()
                .any(|w| w == "eta from q4_k_m applied to fp16"),
            "got {:?}",
            calibration.warnings
        );
        // Same quant on both sides: no quant warning (clamp warning may
        // still apply — here eta_comp is clamped, so exactly one remains).
        let same = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            Some(Quant::Q4_K_M),
            Quant::Q4_K_M,
            128,
        )
        .expect("same quant should succeed");
        assert!(
            !same.warnings.iter().any(|w| w.contains("applied to")),
            "got {:?}",
            same.warnings
        );
    }

    #[test]
    fn prompt_mismatch_warns_with_both_lengths() {
        let bench = bench_with_e2e(55.6, 120.0);
        let calibration = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            512,
        )
        .expect("mismatch must warn, not fail");
        assert!(
            calibration
                .warnings
                .iter()
                .any(|w| w == "prompt mismatch 128 vs 512"),
            "got {:?}",
            calibration.warnings
        );
    }

    #[test]
    fn missing_e2e_block_is_a_clean_error() {
        let mut bench = bench_with_e2e(55.6, 120.0);
        bench.e2e = None;
        let err = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            128,
        )
        .expect_err("missing e2e must be an error");
        assert!(
            matches!(err, EstimateError::CalibrationUnavailable(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_decode_rate_is_a_clean_error() {
        let bench = bench_with_e2e(0.0, 120.0);
        let err = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            128,
        )
        .expect_err("zero tok/s must be an error");
        assert!(
            matches!(err, EstimateError::CalibrationUnavailable(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_ttft_only_skips_prefill_calibration() {
        let bench = bench_with_e2e(55.6, 0.0);
        let calibration = calibrate_with_bench(
            &bench,
            246.9,
            2764.8,
            Path::new("bench.json"),
            None,
            Quant::Q4_K_M,
            128,
        )
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
            ctx: 2048,
            kv_per_token: 0.0,
            hw: HwInputs {
                bw_gbps: 20.0,
                flops_gflops: 50.0,
            },
            clock_ghz: 4.0,
            ram_avail_bytes: None,
            spread_pct,
            spread_method,
        }
    }

    #[test]
    fn format_peak_names_the_clock_or_the_override() {
        assert_eq!(
            format_peak(1024.0, 4.0, false),
            "1024.0 GFLOPS @ 4.00 GHz max"
        );
        assert_eq!(
            format_peak(50.0, 0.0, false),
            "50.0 GFLOPS (explicit --gflops)"
        );
        assert_eq!(
            format_peak(50.0, f64::NAN, false),
            "50.0 GFLOPS (explicit --gflops)"
        );
        assert_eq!(
            format_peak(256.0, 1.0, true),
            "256.0 GFLOPS @ 1.00 GHz baseline (clock unknown)"
        );
    }

    #[test]
    fn baseline_peak_matches_the_1ghz_convention() {
        let avx2 = vec!["AVX2".to_string()];
        assert_eq!(baseline_peak(8, &avx2), Some(256.0));
        // No FMA path: even the baseline yields nothing, so the caller keeps
        // the loud --gflops error.
        assert_eq!(baseline_peak(8, &[]), None);
        assert_eq!(baseline_peak(0, &avx2), None);
    }

    #[test]
    fn resolve_kv_uses_the_preset_or_warns_loudly() {
        let known = resolve_kv(Some("llama-3.1-8b")).expect("known preset");
        assert_eq!(known.per_token, 131_072.0);
        assert_eq!(known.warning, None);

        let unknown = resolve_kv(None).expect("omitting --arch is allowed");
        assert_eq!(unknown.per_token, 0.0);
        assert_eq!(unknown.warning, Some(KV_EXCLUDED_WARNING));

        assert!(resolve_kv(Some("not-a-model")).is_err());
    }

    #[test]
    fn resolve_model_prefers_the_catalog_entry() {
        let resolved = resolve_model(Some("Llama-3.1-8B"), None, None).expect("catalog model");
        assert_eq!(resolved.label, "llama-3.1-8b");
        assert_eq!(resolved.params, 8_030_000_000);
        assert_eq!(resolved.kv.per_token, 131_072.0);
        assert_eq!(resolved.kv.warning, None);
    }

    #[test]
    fn resolve_model_falls_back_to_params_and_arch() {
        let resolved =
            resolve_model(None, Some("8.03B"), Some("llama-3.1-8b")).expect("explicit flags");
        assert_eq!(resolved.label, "8.03B");
        assert_eq!(resolved.params, 8_030_000_000);
        assert_eq!(resolved.kv.per_token, 131_072.0);
        assert_eq!(resolved.kv.warning, None);

        let weights_only = resolve_model(None, Some("135M"), None).expect("no arch");
        assert_eq!(weights_only.params, 135_000_000);
        assert_eq!(weights_only.kv.per_token, 0.0);
        assert_eq!(weights_only.kv.warning, Some(KV_EXCLUDED_WARNING));
    }

    #[test]
    fn resolve_model_paths_resolve_to_the_same_inputs() {
        let via_model = resolve_model(Some("llama-3.1-8b"), None, None).expect("catalog");
        let via_flags =
            resolve_model(None, Some("8.03B"), Some("llama-3.1-8b")).expect("explicit flags");
        assert_eq!(via_model.params, via_flags.params);
        assert_eq!(via_model.kv.per_token, via_flags.kv.per_token);
    }

    #[test]
    fn resolve_model_unknown_name_suggests_params() {
        let err = resolve_model(Some("not-a-model"), None, None)
            .err()
            .expect("unknown model must fail");
        let message = err.to_string();
        assert!(message.contains("not-a-model"), "message: {message}");
        assert!(message.contains("--params"), "message: {message}");
    }

    #[test]
    fn resolve_model_probes_a_gguf_file_without_params() {
        let bytes = fake_gguf("llama", 36, 4096, 32, Some(8), Some(8_200_000_000));
        let path = write_temp_gguf("probe", "local-model.gguf", &bytes);
        let name = path.to_str().expect("temp path is UTF-8");

        let resolved = resolve_model(Some(name), None, None).expect("GGUF file should resolve");
        assert_eq!(resolved.label, name);
        assert_eq!(resolved.params, 8_200_000_000);
        // 2 (K+V) × 36 layers × 8 KV heads × (4096/32) × 2 bytes.
        assert_eq!(resolved.kv.per_token, 2.0 * 36.0 * 8.0 * 128.0 * 2.0);
        assert_eq!(resolved.kv.warning, None);

        let probe = resolved.gguf.as_ref().expect("probe result");
        assert_eq!(probe.facts.architecture, "llama");
        assert_eq!(probe.facts.block_count, 36);
        assert_eq!(probe.facts.n_kv_heads, 8);
        assert_eq!(probe.warning, None);
        cleanup_temp_gguf(&path);
    }

    #[test]
    fn resolve_model_file_mismatch_warns_and_file_wins() {
        // No `general.parameter_count`: the catalog supplies 8.03B, while the
        // file says 36 layers against the catalog's 32.
        let bytes = fake_gguf("llama", 36, 4096, 32, Some(8), None);
        let path = write_temp_gguf("mismatch", "llama-3.1-8b.gguf", &bytes);
        let name = path.to_str().expect("temp path is UTF-8");

        let resolved = resolve_model(Some(name), None, None).expect("GGUF file should resolve");
        assert_eq!(resolved.params, 8_030_000_000);
        assert_eq!(
            resolved
                .gguf
                .as_ref()
                .and_then(|probe| probe.warning.as_deref()),
            Some("catalog says 32 layers, file has 36")
        );
        // The file's 36 layers drive the KV size, not the catalog's 32.
        assert_eq!(resolved.kv.per_token, 2.0 * 36.0 * 8.0 * 128.0 * 2.0);
        cleanup_temp_gguf(&path);
    }

    #[test]
    fn resolve_model_corrupt_gguf_is_a_clean_error() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NOPE");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        let path = write_temp_gguf("corrupt", "bad.gguf", &bytes);
        let name = path.to_str().expect("temp path is UTF-8");

        let err = resolve_model(Some(name), None, None)
            .err()
            .expect("corrupt header must fail");
        let message = err.to_string();
        assert!(message.contains("failed to probe"), "message: {message}");
        assert!(message.contains("magic"), "message: {message}");
        cleanup_temp_gguf(&path);
    }

    #[test]
    fn build_report_includes_kv_and_lowers_raw_decode() {
        let mut ctx = context(None, "unavailable: test");
        ctx.kv_per_token = 38_400.0;

        let report = build_report(&ctx, Quant::Q4_K_M).expect("report");
        let bytes_per_param = Quant::Q4_K_M.bytes_per_param();
        let weights_only = ctx
            .hw
            .predict_decode_tps(ctx.params, bytes_per_param)
            .expect("weights-only prediction");
        let expected = ctx
            .hw
            .predict_decode_tps_with_kv(ctx.params, bytes_per_param, ctx.kv_per_token, ctx.ctx)
            .expect("kv prediction");

        assert_eq!(report.decode_tps, expected);
        assert!(
            report.decode_tps < weights_only,
            "KV must lower decode: {} vs {}",
            report.decode_tps,
            weights_only
        );
    }

    #[test]
    fn build_report_flags_oom_when_the_peak_exceeds_ram() {
        let mut ctx = context(None, "unavailable: test");
        ctx.params = 70_000_000_000;
        ctx.kv_per_token = kv_bytes_per_token(parse_arch("llama-3.1-70b").expect("preset"));
        ctx.ram_avail_bytes = Some(16e9);

        let report = build_report(&ctx, Quant::Q4_K_M).expect("report");
        let fit = report.ram_fit.expect("ram fit should be attached");
        assert!(!fit.fits, "70B Q4 must not fit in 16 GB");
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.starts_with("OOM:")),
            "warnings: {:?}",
            report.warnings
        );
    }

    #[test]
    fn build_report_marks_a_small_model_as_fitting() {
        let mut ctx = context(None, "unavailable: test");
        ctx.ram_avail_bytes = Some(16e9);

        let report = build_report(&ctx, Quant::Q4_K_M).expect("report");
        let fit = report.ram_fit.expect("ram fit should be attached");
        assert!(fit.fits, "135M Q4 should fit in 16 GB");
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
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
