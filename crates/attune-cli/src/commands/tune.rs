//! `attune tune` — estimate-only config ranking (M4 slice 1).
//!
//! Thin CLI: resolve the model, expand a small grid, call the pure
//! `attune-tune` ranking and render either the JSON contract or a human table.
//! No llama.cpp process is spawned and no file is written in this slice.
//!
//! Model resolution mirrors `attune estimate` (GGUF file -> `org/model`
//! fetch/cache -> catalog name -> manual `--params`/`--arch`). The resolver is
//! deliberately duplicated from `commands/estimate.rs` for this slice; see the
//! TODO on [`resolve_model`] — promoting it to one shared helper is a
//! follow-up, not part of M4 slice 1.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use attune_bench::{run_quick, BandwidthReport};
use attune_estimate::{
    is_hf_id, parse_arch, parse_model, parse_params, parse_quants, probe_gguf_file,
    resolve_hf_model, Arch, HwInputs, Quant,
};
use attune_hal::cpu::{theoretical_fp32_gflops, BASELINE_CLOCK_GHZ};
use attune_hal::HardwareInfo;
use attune_tune::{expand_grid, rank_by_estimate, Objective, TuneError, TuneReport};

/// Context windows the slice-1 sweep tries. Kept small so the estimate-only
/// ranking finishes instantly; later slices add explicit `--ctx` flags.
const CTX_GRID: &[u32] = &[2_048, 4_096];
/// Prompt batch sizes the slice-1 sweep tries. `n_batch` is carried into the
/// report but does not change the roofline score yet.
const BATCH_GRID: &[u32] = &[512];
/// KV-cache formats the slice-1 sweep tries. Carried into the report; the
/// weights-only prediction does not read it yet.
const KV_GRID: &[Quant] = &[Quant::Fp16];

/// Warning attached to every report (pretty + JSON) built on the 1 GHz
/// baseline fallback. Same text as `attune estimate` so users see one story.
const CLOCK_FALLBACK_WARNING: &str =
    "clock unknown, assuming 1 GHz baseline — pass --gflops for real numbers";

/// Model inputs resolved from GGUF, fetch/cache, catalog or manual flags.
struct ModelResolution {
    /// Report label: catalog name, GGUF path or the literal `--params` text.
    label: String,
    /// Human-readable source label for the pretty `Model:` line.
    display: String,
    /// Parameter count used for the weight footprint.
    params: u64,
    /// Resolved KV shape, when the source had one (display only in slice 1).
    arch: Option<Arch>,
    /// Every non-fatal notice from resolution.
    warnings: Vec<String>,
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

/// Resolve model facts with the same priority order as `attune estimate`:
/// existing `.gguf` file, `org/model` fetch/cache, catalog name, then manual
/// `--params` + optional `--arch`.
// TODO(m4): promote `commands/estimate.rs::resolve_model` to a shared helper
// and delete this duplicate. It is copied here on purpose so slice 1 does not
// refactor the estimate command.
fn resolve_model(
    model: Option<&str>,
    params: Option<&str>,
    arch: Option<&str>,
) -> Result<ModelResolution> {
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
            let mut warnings = Vec::new();
            if let Some(warning) = probe.warning.clone() {
                warnings.push(warning);
            }
            return Ok(ModelResolution {
                label: name.to_string(),
                display: name.to_string(),
                params,
                arch: probe.arch,
                warnings,
            });
        }

        if is_hf_id(name) {
            let resolved = resolve_hf_model(name, false).map_err(|err| anyhow!("{err}"))?;
            return Ok(ModelResolution {
                label: resolved.label.clone(),
                display: resolved.label,
                params: resolved.params,
                arch: Some(resolved.arch),
                warnings: resolved.warnings,
            });
        }

        let entry = parse_model(name).map_err(|catalog_err| {
            anyhow!(
                "{catalog_err}; pass --params <SIZE> --arch <NAME> for a model outside the catalog"
            )
        })?;
        return Ok(ModelResolution {
            label: entry.name.to_string(),
            display: format!("{}/{}", entry.provider, entry.name),
            params: entry.params,
            arch: Some(entry.arch),
            warnings: Vec::new(),
        });
    }

    let params = params.ok_or_else(|| anyhow!("pass --model <NAME_OR_PATH> or --params <SIZE>"))?;
    let arch = match arch {
        Some(name) => Some(*parse_arch(name)?),
        None => None,
    };
    Ok(ModelResolution {
        label: params.trim().to_string(),
        display: params.trim().to_string(),
        params: parse_params(params)?,
        arch,
        warnings: Vec::new(),
    })
}

/// Resolve bandwidth: explicit override first, then the M2 quick probe.
fn resolve_bandwidth(bandwidth_gbps: Option<f64>) -> Result<f64> {
    match bandwidth_gbps {
        Some(value) => {
            if !value.is_finite() || value <= 0.0 {
                return Err(anyhow!("--bandwidth-gbps must be a finite number > 0"));
            }
            Ok(value)
        }
        None => {
            let report = run_quick().context("failed to measure memory bandwidth")?;
            match report.bandwidth {
                BandwidthReport::Quick(read) => Ok(read.read_gbps),
                BandwidthReport::Full(full) => Ok(full.read.median_gbps),
            }
        }
    }
}

/// Resolved hardware inputs for one tune run.
struct HardwareResolution {
    gflops: f64,
    clock_fallback: bool,
    avail_bytes: Option<f64>,
}

/// Resolve the FP32 peak and available RAM in one detection pass.
///
/// An explicit `--gflops` skips the CPU probe but still reads available RAM.
/// Otherwise [`HardwareInfo::detect`] supplies both the peak and the memory
/// figure, so the CLI never detects the same hardware twice.
fn resolve_hardware(gflops: Option<f64>) -> Result<HardwareResolution> {
    if let Some(value) = gflops {
        if !value.is_finite() || value <= 0.0 {
            return Err(anyhow!("--gflops must be a finite number > 0"));
        }
        return Ok(HardwareResolution {
            gflops: value,
            clock_fallback: false,
            avail_bytes: detect_available_ram_bytes(),
        });
    }

    let hw = HardwareInfo::detect()?;
    let available = hw.memory.available_bytes as f64;
    let avail_bytes = (available.is_finite() && available > 0.0).then_some(available);

    let theoretical = hw.cpu.theoretical_fp32_gflops;
    if theoretical.is_finite() && theoretical > 0.0 {
        return Ok(HardwareResolution {
            gflops: theoretical,
            clock_fallback: false,
            avail_bytes,
        });
    }

    let baseline = theoretical_fp32_gflops(
        hw.cpu.physical_cores,
        &hw.cpu.simd_flags,
        BASELINE_CLOCK_GHZ,
    );
    if baseline.is_finite() && baseline > 0.0 {
        return Ok(HardwareResolution {
            gflops: baseline,
            clock_fallback: true,
            avail_bytes,
        });
    }

    Err(anyhow!(
        "cannot derive an FP32 peak from this CPU (unknown clock or no FMA SIMD path); pass --gflops <GFLOPS>"
    ))
}

/// Best-effort available RAM for the OOM check. `None` when the HAL cannot
/// read it (0), in which case the ranking skips the fit check rather than
/// claiming an OOM from unknown numbers.
fn detect_available_ram_bytes() -> Option<f64> {
    attune_hal::memory::detect()
        .ok()
        .map(|memory| memory.available_bytes as f64)
        .filter(|bytes| bytes.is_finite() && *bytes > 0.0)
}

/// Default thread count for the grid. Slice-1 roofline math has no thread
/// model, so one value is enough; using the machine's parallelism keeps the
/// reported config realistic.
fn default_threads() -> u32 {
    std::thread::available_parallelism()
        .ok()
        .and_then(|parallelism| u32::try_from(parallelism.get()).ok())
        .unwrap_or(8)
}

/// Compact parameter label (`135M`, `8.00B`, ...).
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1_000_000.0)
    } else {
        params.to_string()
    }
}

/// Human-readable rank table. The score column prints `-inf` for OOM configs
/// (the JSON contract writes those as `null`).
fn print_table(report: &TuneReport, resolution: &ModelResolution) {
    let arch = resolution.arch.map_or("unknown", |arch| arch.name);
    println!(
        "Model:  {} ({}) | Arch: {arch}",
        resolution.display,
        format_params(report.params)
    );
    println!(
        "Inputs: {:.1} GB/s, {:.1} GFLOPS | Objective: {}",
        report.bw_gbps, report.gflops, report.objective
    );
    match report.avail_bytes {
        Some(avail) => println!("RAM:    {:.1} GB available", avail / 1e9),
        None => println!("RAM:    unknown (fit check skipped)"),
    }
    println!();
    println!(
        "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14} {:<14} {:<5} Score",
        "Rank", "Quant", "Ctx", "Threads", "Batch", "KV", "Decode tok/s", "TTFT ms", "Fit"
    );
    for (index, ranked) in report.configs.iter().enumerate() {
        let score = if ranked.score.is_finite() {
            format!("{:.4}", ranked.score)
        } else {
            "-inf".to_string()
        };
        println!(
            "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14.1} {:<14.1} {:<5} {}",
            index + 1,
            ranked.config.quant.as_str(),
            ranked.config.n_ctx,
            ranked.config.n_threads,
            ranked.config.n_batch,
            ranked.config.cache_kv.as_str(),
            ranked.decode_tps,
            ranked.ttft_ms,
            if ranked.fits { "yes" } else { "OOM" },
            score,
        );
    }
    println!();
    println!(
        "Note: estimate-only ranking (weights-only decode, full-context prefill TTFT); no llama.cpp run, no file written."
    );
    for warning in &report.warnings {
        println!("Warning:     {warning}");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Option<&str>,
    params: Option<&str>,
    arch: Option<&str>,
    quant: &str,
    objective: &str,
    bandwidth_gbps: Option<f64>,
    gflops: Option<f64>,
    json: bool,
) -> Result<()> {
    let resolution = resolve_model(model, params, arch)?;
    let quants = parse_quants(quant)?;
    let objective: Objective = objective
        .parse()
        .map_err(|err: TuneError| anyhow!("{err}"))?;

    // Fail fast on unusable overrides before starting either probe.
    for (name, value) in [("--bandwidth-gbps", bandwidth_gbps), ("--gflops", gflops)] {
        if let Some(value) = value {
            if !value.is_finite() || value <= 0.0 {
                return Err(anyhow!("{name} must be a finite number > 0"));
            }
        }
    }

    // The CPU/RAM probe and the bandwidth kernel are independent, so run
    // them concurrently: the live path then costs one probe, not their sum,
    // and still spawns no subprocess. `resolve_hardware` reuses a single HAL
    // detection for both the FP32 peak and available RAM.
    let (hardware, bw_gbps) = std::thread::scope(|scope| -> Result<(HardwareResolution, f64)> {
        let hardware = scope.spawn(move || resolve_hardware(gflops));
        let bandwidth = resolve_bandwidth(bandwidth_gbps);
        let hardware = hardware
            .join()
            .map_err(|_| anyhow!("hardware detection worker panicked"))??;
        Ok((hardware, bandwidth?))
    })?;
    let HardwareResolution {
        gflops,
        clock_fallback,
        avail_bytes,
    } = hardware;
    let hw = HwInputs {
        bw_gbps,
        flops_gflops: gflops,
    };

    let grid = expand_grid(&quants, CTX_GRID, &[default_threads()], BATCH_GRID, KV_GRID);
    let ranked = rank_by_estimate(
        &grid,
        resolution.params,
        hw,
        avail_bytes.unwrap_or(0.0),
        objective,
    )?;

    let mut warnings = resolution.warnings.clone();
    if clock_fallback {
        warnings.push(CLOCK_FALLBACK_WARNING.to_string());
    }
    let report = TuneReport::new(
        resolution.label.clone(),
        resolution.params,
        objective,
        hw,
        avail_bytes,
        ranked,
    )
    .with_warnings(warnings);

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    print_table(&report, &resolution);
    Ok(())
}
