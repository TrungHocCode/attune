//! `attune tune` — config ranking (M4 slices 1–5).
//!
//! Thin CLI: resolve the model, expand the full thread/context/batch/KV grid,
//! prune candidates that cannot run, call the pure `attune-tune` ranking and
//! render either the JSON contract or a human table. Slice 3 adds `--save`,
//! which exports the winner as an `attune-profile` document. Slice 4 adds the
//! opt-in `--verify`, which measures the best `--top-k` configs with a real
//! llama.cpp e2e run and re-scores them. Without `--verify` nothing is
//! spawned and the output is unchanged. Slice 5 adds
//! `--bench-json`/`--bench-quant`/`--calibrate`: one eta pair is fitted from
//! the saved bench and applied to every ranked config.
//!
//! Model resolution mirrors `attune estimate` (GGUF file -> `org/model`
//! fetch/cache -> catalog name -> manual `--params`/`--arch`). The resolver is
//! deliberately duplicated from `commands/estimate.rs` for this slice; see the
//! TODO on [`resolve_model`] — promoting it to one shared helper is a
//! follow-up, not part of M4 slice 1.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use attune_bench::e2e::{DEFAULT_GEN_TOKENS, DEFAULT_PROMPT_TOKENS};
use attune_bench::{run_quick, BandwidthReport, BenchReport};
use attune_estimate::model::registry::{load_catalog, CatalogModel};
use attune_estimate::{
    is_gguf_file, is_hf_id, kv_bytes_per_token, parse_arch, parse_bench_report, parse_model,
    parse_params, parse_quants, probe_gguf_file, resolve_hf_model, weight_bytes_for, Arch,
    BenchCalibration, HwInputs, Quant,
};
use attune_hal::cpu::{theoretical_fp32_gflops, BASELINE_CLOCK_GHZ};
use attune_hal::HardwareInfo;
use attune_profile::{ModelSource, Profile};
use attune_tune::ProfileInputs;
use attune_tune::{
    calibrate_ranked, expand_full_grid_with_quants, from_best, hardware_fingerprint, prune,
    rank_by_estimate, run_verify_topk, stars, Objective, PruneReason, RankedConfig, TuneConfig,
    TuneError, TuneModel, TuneReport,
};

/// `engine.version` when no llama.cpp build was observed (no `--verify`, or
/// every verification run failed), so the field says so instead of guessing.
const ENGINE_VERSION_ESTIMATE_ONLY: &str = "unknown (estimate-only)";

/// Confidence recorded for an uncalibrated slice-3 prediction.
const ESTIMATE_ONLY_CONFIDENCE: f64 = 0.5;

/// Prompt length assumed by the prune check. The slice-1 sweep ranks the full
/// context as TTFT, so this is only used to drop a context window that cannot
/// hold a realistic prompt + generation.
const PROMPT_TOKENS: u64 = 512;
/// Decode length assumed by the prune check.
const GEN_TOKENS: u64 = 128;

/// Warning attached to every report (pretty + JSON) built on the 1 GHz
/// baseline fallback. Same text as `attune estimate` so users see one story.
const CLOCK_FALLBACK_WARNING: &str =
    "clock unknown, assuming 1 GHz baseline — pass --gflops for real numbers";

/// Warning attached when transient available RAM would prune every candidate.
const TOTAL_RAM_FALLBACK_WARNING: &str =
    "available RAM can hold no candidate; planning against total RAM — runtime may still OOM";

/// Verify-mode knobs from the CLI. Grouped so [`run`] does not keep growing
/// one positional argument per flag.
pub struct VerifyArgs<'a> {
    /// `--verify`: measure the top-K estimate configs with llama.cpp.
    pub enabled: bool,
    /// `--top-k`: how many of the best estimate configs to measure.
    pub top_k: usize,
    /// `--llama-cli`: explicit binary path; `None` uses env/PATH.
    pub llama_cli: Option<&'a Path>,
    /// `--prompt-tokens`: prefill size for the verification run.
    pub prompt_tokens: Option<usize>,
    /// `--gen-tokens`: decode length for the verification run.
    pub gen_tokens: Option<u32>,
    /// `--ctx`: context window override for every verification run.
    pub ctx: Option<u32>,
}

/// Recommend-mode knobs (`tune` without `--model`/`--params`).
pub struct RecommendArgs<'a> {
    /// `--max-ram`: budget to plan against instead of available RAM.
    pub max_ram: Option<&'a str>,
    /// `--all`: every model and quant, not just the picks.
    pub all: bool,
    /// `--refresh-catalog`: fetch the catalog even when the cache is fresh.
    pub refresh_catalog: bool,
}

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
    /// GGUF file size; the real weight footprint when the source is a file.
    weight_bytes: Option<u64>,
    /// Every non-fatal notice from resolution.
    warnings: Vec<String>,
    /// `model.path` for a saved profile when it differs from `label` (a
    /// catalog model's portable `~/.attune/models/...` path).
    model_path: Option<String>,
    /// Download source for a catalog model (profile v2).
    source: Option<ModelSource>,
}

/// The path when `--model` is an existing GGUF file (`.gguf` extension or
/// `GGUF` magic), which is probed instead of being looked up as a catalog name.
fn as_gguf_path(value: &str) -> Option<&Path> {
    let path = Path::new(value.trim());
    is_gguf_file(path).then_some(path)
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
    quant: Quant,
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
            let weight_bytes = fs::metadata(path)
                .map_err(|err| anyhow!("failed to stat {}: {err}", path.display()))?
                .len();
            let mut warnings = Vec::new();
            if let Some(warning) = probe.warning.clone() {
                warnings.push(warning);
            }
            return Ok(ModelResolution {
                label: name.to_string(),
                display: name.to_string(),
                params,
                arch: probe.arch,
                weight_bytes: Some(weight_bytes),
                warnings,
                model_path: None,
                source: None,
            });
        }

        if is_hf_id(name) {
            let resolved = resolve_hf_model(name, false).map_err(|err| anyhow!("{err}"))?;
            return Ok(ModelResolution {
                label: resolved.label.clone(),
                display: resolved.label,
                params: resolved.params,
                arch: Some(resolved.arch),
                weight_bytes: None,
                warnings: resolved.warnings,
                model_path: None,
                source: None,
            });
        }

        // Downloadable catalog first (it pins files), then the estimate-only
        // names such as `llama-3.1-8b`.
        let loaded = load_catalog(false);
        let catalog_err = match loaded.catalog.find(name) {
            Ok(model) => {
                let mut resolution = resolve_catalog_model(model, quant)?;
                resolution.warnings.extend(loaded.warnings);
                return Ok(resolution);
            }
            Err(err) => err,
        };
        let entry = parse_model(name).map_err(|_| {
            anyhow!(
                "{catalog_err}; run `attune tune` without --model to list models, or pass --params <SIZE> --arch <NAME>"
            )
        })?;
        return Ok(ModelResolution {
            label: entry.name.to_string(),
            display: format!("{}/{}", entry.provider, entry.name),
            params: entry.params,
            arch: Some(entry.arch),
            weight_bytes: None,
            warnings: Vec::new(),
            model_path: None,
            source: None,
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
        weight_bytes: None,
        warnings: Vec::new(),
        model_path: None,
        source: None,
    })
}

/// Facts and download source for one catalog model at `quant`. The file
/// does not need to exist: the catalog pins its exact size and shape.
fn resolve_catalog_model(model: &CatalogModel, quant: Quant) -> Result<ModelResolution> {
    let file = model.file_for(quant).ok_or_else(|| {
        let have: Vec<&str> = model.files.iter().map(|f| f.quant.as_str()).collect();
        anyhow!(
            "{} has no {} file in the catalog; available: {}",
            model.id,
            quant.as_str(),
            have.join(", ")
        )
    })?;
    let source = ModelSource {
        repo: file.repo.clone(),
        file: file.file.clone(),
        revision: file.revision.clone(),
        sha256: file.sha256.clone(),
        size: file.size,
    };
    Ok(ModelResolution {
        label: model.id.clone(),
        display: format!("{}/{}", model.publisher, model.display),
        params: model.params,
        arch: Some(model.arch.to_arch()),
        weight_bytes: Some(file.size),
        warnings: Vec::new(),
        model_path: Some(source.home_path()),
        source: Some(source),
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

/// Read and parse `--bench-json`.
///
/// Parsing (bare report or `{ "bench": … }` envelope) and the calibration
/// math live in `attune-estimate`; this wrapper only adds file/path context.
fn load_bench_report(path: &Path) -> Result<BenchReport> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read --bench-json {}", path.display()))?;
    parse_bench_report(&text)
        .with_context(|| format!("failed to load --bench-json {}", path.display()))
}

/// Resolved hardware inputs for one tune run.
struct HardwareResolution {
    gflops: f64,
    clock_fallback: bool,
    avail_bytes: Option<f64>,
    total_bytes: Option<f64>,
    physical_cores: u32,
}

/// Resolve the FP32 peak, memory limits and physical core count in one pass.
///
/// An explicit `--gflops` skips the CPU peak probe but still reads memory and
/// the physical core count (needed to build and prune the grid). Otherwise
/// [`HardwareInfo::detect`] supplies all of them, so the CLI never detects the
/// same hardware twice.
fn resolve_hardware(gflops: Option<f64>) -> Result<HardwareResolution> {
    if let Some(value) = gflops {
        if !value.is_finite() || value <= 0.0 {
            return Err(anyhow!("--gflops must be a finite number > 0"));
        }
        let (avail_bytes, total_bytes) = detect_memory_limits();
        return Ok(HardwareResolution {
            gflops: value,
            clock_fallback: false,
            avail_bytes,
            total_bytes,
            physical_cores: detect_physical_cores(),
        });
    }

    let hw = HardwareInfo::detect()?;
    let available = hw.memory.available_bytes as f64;
    let avail_bytes = (available.is_finite() && available > 0.0).then_some(available);
    let total = hw.memory.total_bytes as f64;
    let total_bytes = (total.is_finite() && total > 0.0).then_some(total);
    let physical_cores = hw.cpu.physical_cores.max(1);

    let theoretical = hw.cpu.theoretical_fp32_gflops;
    if theoretical.is_finite() && theoretical > 0.0 {
        return Ok(HardwareResolution {
            gflops: theoretical,
            clock_fallback: false,
            avail_bytes,
            total_bytes,
            physical_cores,
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
            total_bytes,
            physical_cores,
        });
    }

    Err(anyhow!(
        "cannot derive an FP32 peak from this CPU (unknown clock or no FMA SIMD path); pass --gflops <GFLOPS>"
    ))
}

/// Best-effort `(available, total)` RAM for the OOM check.
///
/// Each value is `None` when the HAL reports 0/unusable, in which case the
/// ranking skips the fit check rather than claiming an OOM from unknown
/// numbers. The total is only used by the CLI's all-pruned fallback.
fn detect_memory_limits() -> (Option<f64>, Option<f64>) {
    match attune_hal::memory::detect() {
        Ok(memory) => {
            let available = memory.available_bytes as f64;
            let total = memory.total_bytes as f64;
            (
                (available.is_finite() && available > 0.0).then_some(available),
                (total.is_finite() && total > 0.0).then_some(total),
            )
        }
        Err(_) => (None, None),
    }
}

/// Best-effort physical core count for the thread axis and prune check.
///
/// Falls back to the logical parallelism (then 1) when the HAL cannot report
/// physical cores, so the grid always has at least one valid thread value.
fn detect_physical_cores() -> u32 {
    attune_hal::cpu::detect()
        .ok()
        .map(|cpu| cpu.physical_cores)
        .filter(|cores| *cores > 0)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .and_then(|parallelism| u32::try_from(parallelism.get()).ok())
        })
        .unwrap_or(1)
}

/// Readable fingerprint for the machine this profile is being saved on.
///
/// Only called when `--save` is used, so the normal estimate path never pays
/// for a second HAL detection. Detection failures degrade to the same
/// `unknown|0|unknown|0` marker rather than blocking the save.
fn current_hardware_fingerprint() -> String {
    let cpu = attune_hal::cpu::detect();
    let memory = attune_hal::memory::detect();
    let brand = cpu
        .as_ref()
        .map(|cpu| cpu.brand.as_str())
        .unwrap_or("unknown");
    let cores = cpu.as_ref().map_or(0, |cpu| cpu.physical_cores);
    let simd = cpu
        .as_ref()
        .map(|cpu| cpu.simd.as_str())
        .unwrap_or("unknown");
    let total_ram = memory.as_ref().map_or(0, |memory| memory.total_bytes);
    hardware_fingerprint(brand, cores, simd, total_ram)
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
///
/// `estimates` is the pre-verification ranking, kept only so `--verify` can
/// show the estimate next to the measured numbers (the report itself carries
/// the measured values in `decode_tps` / `ttft_ms`).
fn print_table(
    report: &TuneReport,
    resolution: &ModelResolution,
    total_ram_fallback: bool,
    estimates: Option<&[RankedConfig]>,
) {
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
    match (report.avail_bytes, total_ram_fallback) {
        (Some(ram), true) => println!(
            "RAM:    {:.1} GB total (fallback; available RAM was too low)",
            ram / 1e9
        ),
        (Some(ram), false) => println!("RAM:    {:.1} GB available", ram / 1e9),
        (None, _) => println!("RAM:    unknown (fit check skipped)"),
    }
    // One eta pair was fitted once and applied to every config (slice 5), so
    // the first calibrated entry describes the whole sweep.
    if let Some(calibration) = report
        .configs
        .iter()
        .find_map(|ranked| ranked.calibration.as_ref())
    {
        let ttft = calibration
            .measured_ttft_ms
            .map_or_else(|| "no TTFT".to_string(), |ms| format!("{ms:.1} ms TTFT"));
        println!(
            "Calib:  eta_mem={:.3} (decode), eta_comp={:.3} (prefill) | bench measured {:.1} tok/s, {ttft}",
            calibration.inputs.eta_mem,
            calibration.inputs.eta_comp,
            calibration.measured_decode_tps,
        );
    }
    // Only widen the table when at least one config was measured, so the
    // estimate-only output every earlier slice produced stays byte-identical.
    let show_verify = report.configs.iter().any(|ranked| ranked.verified);
    println!();
    if show_verify {
        println!(
            "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14} {:<14} {:<14} {:<14} {:<8} {:<5} {:<6} Score",
            "Rank",
            "Quant",
            "Ctx",
            "Threads",
            "Batch",
            "KV",
            "Est. tok/s",
            "Est. TTFT ms",
            "E2E tok/s",
            "E2E TTFT ms",
            "Verified",
            "Fit",
            "Stars"
        );
    } else {
        println!(
            "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14} {:<14} {:<5} {:<6} Score",
            "Rank",
            "Quant",
            "Ctx",
            "Threads",
            "Batch",
            "KV",
            "Decode tok/s",
            "TTFT ms",
            "Fit",
            "Stars"
        );
    }
    let total = report.configs.len();
    for (index, ranked) in report.configs.iter().enumerate() {
        let score = if ranked.score.is_finite() {
            format!("{:.4}", ranked.score)
        } else {
            "-inf".to_string()
        };
        let star_marks = "*".repeat(usize::from(stars(index, total)));
        if show_verify {
            let estimate = estimates
                .and_then(|list| list.iter().find(|item| item.config == ranked.config))
                .unwrap_or(ranked);
            let (e2e_tps, e2e_ttft, verified) = match ranked.measured.as_ref() {
                Some(measured) if ranked.verified => (
                    format!("{:.1}", measured.tok_s),
                    format!("{:.1}", measured.ttft_ms),
                    "yes".to_string(),
                ),
                _ => ("-".to_string(), "-".to_string(), "-".to_string()),
            };
            println!(
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14.1} {:<14.1} {:<14} {:<14} {:<8} {:<5} {:<6} {}",
                index + 1,
                ranked.config.quant.as_str(),
                ranked.config.n_ctx,
                ranked.config.n_threads,
                ranked.config.n_batch,
                ranked.config.cache_kv.as_str(),
                estimate.decode_tps,
                estimate.ttft_ms,
                e2e_tps,
                e2e_ttft,
                verified,
                if ranked.fits { "yes" } else { "OOM" },
                star_marks,
                score,
            );
        } else {
            println!(
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<14.1} {:<14.1} {:<5} {:<6} {}",
                index + 1,
                ranked.config.quant.as_str(),
                ranked.config.n_ctx,
                ranked.config.n_threads,
                ranked.config.n_batch,
                ranked.config.cache_kv.as_str(),
                ranked.decode_tps,
                ranked.ttft_ms,
                if ranked.fits { "yes" } else { "OOM" },
                star_marks,
                score,
            );
        }
    }
    println!();
    if show_verify {
        let measured = report
            .configs
            .iter()
            .filter(|ranked| ranked.verified)
            .count();
        println!(
            "Note: {measured} config(s) re-scored from a measured llama.cpp e2e run; the rest are estimate-only."
        );
    } else {
        println!(
            "Note: estimate-only ranking (decode = weights + filled KV, prompt-length prefill TTFT); no llama.cpp run."
        );
    }
    for warning in &report.warnings {
        println!("Warning:     {warning}");
    }
}

/// Print the one-line prune summary shown before the table.
///
/// `cap_from_expand` counts points already dropped by the 64-point full-grid
/// guard; the rest come from [`prune`] and are counted by reason.
fn print_pruned(pruned: &[(TuneConfig, PruneReason)], cap_from_expand: usize) {
    let mut oom = 0;
    let mut ctx = 0;
    let mut threads = 0;
    let mut cap = cap_from_expand;
    for (_, reason) in pruned {
        match reason {
            PruneReason::Oom => oom += 1,
            PruneReason::CtxOverflow => ctx += 1,
            PruneReason::TooManyThreads => threads += 1,
            PruneReason::OverCap => cap += 1,
        }
    }
    println!(
        "pruned {} ({} OOM, {} ctx-overflow, {} threads, {} cap)",
        oom + ctx + threads + cap,
        oom,
        ctx,
        threads,
        cap
    );
}

/// Default profile name: `<model>-<objective>`, where a GGUF source contributes
/// its file stem (not the full path) and any path separator becomes `-`, so
/// the name is portable and leaks no directory layout.
fn default_profile_name(label: &str, objective: Objective) -> String {
    let model = match as_gguf_path(label) {
        Some(path) => path.file_stem().map_or_else(
            || label.to_string(),
            |stem| stem.to_string_lossy().into_owned(),
        ),
        None => label.to_string(),
    };
    format!("{model}-{objective}").replace(['/', '\\'], "-")
}

/// Build the versioned profile for the highest-ranked config.
fn build_profile(
    report: &TuneReport,
    resolution: &ModelResolution,
    name: &str,
    engine_version: Option<&str>,
) -> Result<Profile> {
    let best = report
        .configs
        .first()
        .ok_or_else(|| anyhow!("cannot save a profile without a ranked config"))?;
    let quant = best.config.quant;
    let size_bytes = weight_bytes_for(report.params, quant, resolution.weight_bytes).round() as u64;
    let fingerprint = current_hardware_fingerprint();
    let inputs = ProfileInputs {
        name,
        fingerprint: &fingerprint,
        engine_version: engine_version.unwrap_or(ENGINE_VERSION_ESTIMATE_ONLY),
        model_path: resolution
            .model_path
            .as_deref()
            .unwrap_or(&resolution.label),
        quant,
        size_bytes,
        ram_peak_gb: size_bytes as f64 / 1e9,
        confidence: ESTIMATE_ONLY_CONFIDENCE,
        source: resolution.source.clone(),
    };
    Ok(from_best(
        &best.config,
        best.decode_tps,
        best.ttft_ms,
        &inputs,
    ))
}

/// One-line summary of the winner, printed before the save path.
fn print_best(report: &TuneReport) {
    if let Some(best) = report.configs.first() {
        println!(
            "Best: {} ctx={} threads={} batch={} kv={} | {:.1} tok/s decode, {:.1} ms TTFT",
            best.config.quant.as_str(),
            best.config.n_ctx,
            best.config.n_threads,
            best.config.n_batch,
            best.config.cache_kv.as_str(),
            best.decode_tps,
            best.ttft_ms,
        );
    }
}

/// Convert the `--llama-cli` path to the `&str` the resolver takes. A
/// non-UTF-8 path is named rather than silently ignored (same policy as
/// `attune bench`).
fn llama_explicit(path: Option<&Path>) -> Result<Option<&str>> {
    match path {
        None => Ok(None),
        Some(path) => path
            .to_str()
            .ok_or_else(|| {
                anyhow!(
                    "--llama-cli path is not valid UTF-8: {}",
                    path.to_string_lossy()
                )
            })
            .map(Some),
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
    max_configs: usize,
    save: Option<&Path>,
    name: Option<&str>,
    verify: VerifyArgs<'_>,
    bench_json: Option<&Path>,
    bench_quant: Option<&str>,
    calibrate: bool,
    recommend_args: RecommendArgs<'_>,
    json: bool,
) -> Result<()> {
    let recommend_mode = model.is_none() && params.is_none();
    let max_ram = recommend_args
        .max_ram
        .map(super::recommend::parse_size)
        .transpose()?;
    if max_configs == 0 {
        return Err(anyhow!("--max-configs must be > 0"));
    }
    if verify.enabled && verify.top_k == 0 {
        return Err(anyhow!("--top-k must be > 0"));
    }
    if calibrate && bench_json.is_none() {
        return Err(anyhow!(
            "--calibrate requires --bench-json <PATH> with an e2e block"
        ));
    }

    // The bench report is only read once, but parse it before the probes so a
    // typo in the path fails fast. `--calibrate` is the only consumer.
    let bench = match bench_json {
        Some(path) => Some(load_bench_report(path)?),
        None => None,
    };
    let bench_quant = bench_quant
        .map(str::parse::<Quant>)
        .transpose()
        .map_err(|err: attune_estimate::EstimateError| anyhow!("{err}"))?;

    // `--verify` can only measure a real file: a catalog name or `--params`
    // has no GGUF for llama.cpp to load. Fail before the hardware probes.
    // A catalog id is downloaded (after consent) once it is resolved below.
    let mut verify_model = if verify.enabled {
        let Some(requested) = model else {
            return Err(anyhow!(
                "--verify requires --model: a local .gguf file or a catalog model"
            ));
        };
        as_gguf_path(requested).map(Path::to_path_buf)
    } else {
        None
    };

    let quants = parse_quants(quant)?;
    let resolution = if recommend_mode {
        None
    } else {
        Some(resolve_model(model, params, arch, quants[0])?)
    };
    if verify.enabled && verify_model.is_none() {
        let Some(source) = resolution.as_ref().and_then(|r| r.source.as_ref()) else {
            return Err(anyhow!(
                "--verify needs a model file: pass a local .gguf path or a catalog model id"
            ));
        };
        let home = attune_fetch::home_dir().map_err(|err| anyhow!("{err}"))?;
        let path = attune_profile::expand_home(&source.home_path(), &home);
        let mut profile = attune_profile::Profile::default();
        profile.model.path = source.home_path();
        profile.model.source = Some(source.clone());
        attune_serve::resolve_model_file(
            &mut profile,
            &home,
            super::download::consent_for_stdin(false),
            &mut super::download::progress,
        )?;
        verify_model = Some(path);
    }
    if resolution.as_ref().is_some_and(|r| r.source.is_some()) && quants.len() > 1 {
        return Err(anyhow!(
            "a catalog model is tuned at one --quant at a time (each quant is a different file)"
        ));
    }
    let objective: Objective = objective
        .parse()
        .map_err(|err: TuneError| anyhow!("{err}"))?;

    let explicit_gflops = gflops.is_some();
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
    // detection for the FP32 peak, available RAM and physical core count.
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
        total_bytes,
        physical_cores,
    } = hardware;
    // Sequential with the bandwidth probe above so the two never compete.
    let int8_gops = if explicit_gflops {
        None
    } else {
        super::estimate::measured_int8_gops(None)
    };
    let hw = HwInputs {
        bw_gbps,
        flops_gflops: gflops,
        int8_gops,
    };
    let Some(resolution) = resolution else {
        return super::recommend::run(
            hw,
            avail_bytes.or(total_bytes).unwrap_or(0.0),
            max_ram,
            recommend_args.all,
            recommend_args.refresh_catalog,
            json,
        );
    };

    // Full slice-2 grid (threads/ctx/batch/KV) restricted to the requested
    // formats, then prune by OOM, context fit, thread count and the cap.
    let model = TuneModel {
        params: resolution.params,
        weight_bytes: resolution.weight_bytes,
        kv_f16_per_token: resolution.arch.as_ref().map_or(0.0, kv_bytes_per_token),
    };
    let (grid, cap_from_expand) = expand_full_grid_with_quants(physical_cores, &quants);
    let (mut kept, mut pruned) = prune(
        &grid,
        &model,
        avail_bytes.unwrap_or(0.0),
        PROMPT_TOKENS,
        GEN_TOKENS,
        physical_cores,
        max_configs,
    );
    let mut effective_avail = avail_bytes;
    let mut total_ram_fallback = false;
    // A busy machine can report almost no *available* RAM even though the
    // model would fit once the user frees memory. If the OOM check would drop
    // every candidate, plan against total RAM instead and say so loudly rather
    // than returning an empty table.
    if kept.is_empty() {
        if let Some(total) =
            total_bytes.filter(|total| avail_bytes.is_none_or(|avail| *total > avail))
        {
            let (fallback_kept, fallback_pruned) = prune(
                &grid,
                &model,
                total,
                PROMPT_TOKENS,
                GEN_TOKENS,
                physical_cores,
                max_configs,
            );
            if !fallback_kept.is_empty() {
                kept = fallback_kept;
                pruned = fallback_pruned;
                effective_avail = Some(total);
                total_ram_fallback = true;
            }
        }
    }
    if kept.is_empty() {
        return Err(anyhow!(
            "every candidate config was pruned ({} before the cap); try a smaller --quant, a lower context, or a larger --max-configs",
            grid.len().saturating_add(cap_from_expand)
        ));
    }

    let mut ranked = rank_by_estimate(
        &kept,
        &model,
        hw,
        effective_avail.unwrap_or(0.0),
        objective,
        PROMPT_TOKENS,
        GEN_TOKENS,
    )?;

    // Opt-in slice 5: fit one eta pair against the best estimate config and
    // apply it to every ranked point. Fitting per config would give each grid
    // point a different correction; the grid must stay internally consistent.
    let mut calibration_warnings: Vec<String> = Vec::new();
    if calibrate {
        let path = bench_json
            .ok_or_else(|| anyhow!("--calibrate requires --bench-json <PATH> with an e2e block"))?;
        let bench = bench
            .as_ref()
            .ok_or_else(|| anyhow!("--calibrate requires --bench-json <PATH> with an e2e block"))?;
        let reference = ranked
            .first()
            .ok_or_else(|| anyhow!("cannot calibrate an empty ranking"))?;
        let fit = BenchCalibration::fit(bench, reference.decode_tps, Some(reference.ttft_ms), path)
            .map_err(|err| anyhow!("{err}"))?;
        ranked = calibrate_ranked(&ranked, &fit, bench_quant, objective, PROMPT_TOKENS);
        // Each config carries its own notices; the human table shows the
        // deduplicated union once instead of one line per row.
        for entry in &ranked {
            if let Some(report) = &entry.calibration {
                for warning in &report.warnings {
                    if !calibration_warnings.contains(warning) {
                        calibration_warnings.push(warning.clone());
                    }
                }
            }
        }
    }

    // Keep the estimate numbers for the human `--verify` table; the report
    // itself carries the measured values after re-scoring.
    let estimate_snapshot = verify.enabled.then(|| ranked.clone());

    let mut warnings = resolution.warnings.clone();
    if clock_fallback {
        warnings.push(CLOCK_FALLBACK_WARNING.to_string());
    }
    if total_ram_fallback {
        warnings.push(TOTAL_RAM_FALLBACK_WARNING.to_string());
    }
    warnings.extend(calibration_warnings);
    // Opt-in slice 4: measure the top-K estimate configs and re-score them.
    // Setup errors fail here; a single bad run becomes a warning and keeps
    // that config's estimate score.
    let mut observed_engine_version = None;
    if let Some(model_path) = verify_model.as_deref() {
        let outcome = run_verify_topk(
            &ranked,
            objective,
            verify.top_k,
            model_path,
            llama_explicit(verify.llama_cli)?,
            verify.prompt_tokens.unwrap_or(DEFAULT_PROMPT_TOKENS),
            verify.gen_tokens.unwrap_or(DEFAULT_GEN_TOKENS),
            verify.ctx,
        )?;
        ranked = outcome.configs;
        warnings.extend(outcome.warnings);
        observed_engine_version = outcome.engine_version;
    }
    let report = TuneReport::new(
        resolution.label.clone(),
        resolution.params,
        objective,
        hw,
        effective_avail,
        ranked,
    )
    .with_warnings(warnings);

    let profile = match save {
        Some(path) => {
            let default_name = default_profile_name(&resolution.label, objective);
            let profile_name = name.unwrap_or(default_name.as_str());
            let profile = build_profile(
                &report,
                &resolution,
                profile_name,
                observed_engine_version.as_deref(),
            )?;
            let document = profile.to_json_string()?;
            std::fs::write(path, document)
                .with_context(|| format!("failed to write profile to {}", path.display()))?;
            Some(profile)
        }
        None => None,
    };

    if json {
        let mut value = serde_json::to_value(&report)?;
        if let Some(profile) = &profile {
            if let Some(object) = value.as_object_mut() {
                object.insert("profile".to_string(), serde_json::to_value(profile)?);
                if let Some(path) = save {
                    object.insert(
                        "profile_path".to_string(),
                        serde_json::Value::String(path.display().to_string()),
                    );
                }
            }
        }
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    print_pruned(&pruned, cap_from_expand);
    print_table(
        &report,
        &resolution,
        total_ram_fallback,
        estimate_snapshot.as_deref(),
    );
    if let (Some(path), Some(_)) = (save, profile.as_ref()) {
        print_best(&report);
        println!("Profile saved: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_for(label: &str) -> TuneReport {
        let config = TuneConfig::new(Quant::Q4_K_M, 2048, 4, 512, Quant::Fp16);
        let ranked = RankedConfig {
            config,
            decode_tps: 100.0,
            ttft_ms: 50.0,
            fits: true,
            score: 1.0,
            verified: false,
            measured: None,
            calibration: None,
            grid_index: 0,
        };
        let hw = HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 100.0,
            int8_gops: None,
        };
        TuneReport::new(
            label,
            135_000_000,
            Objective::Throughput,
            hw,
            None,
            vec![ranked],
        )
    }

    fn resolution(label: &str, weight_bytes: Option<u64>) -> ModelResolution {
        ModelResolution {
            label: label.to_string(),
            display: label.to_string(),
            params: 135_000_000,
            arch: None,
            weight_bytes,
            warnings: Vec::new(),
            model_path: None,
            source: None,
        }
    }

    #[test]
    fn gguf_default_name_is_the_file_stem_plus_objective() {
        let dir = std::env::temp_dir().join("attune-tune-name-test");
        fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("smollm2-135m-instruct-q4_k_m.gguf");
        fs::write(&file, b"GGUF").expect("write");
        let label = file.display().to_string();
        let name = default_profile_name(&label, Objective::Throughput);
        assert_eq!(name, "smollm2-135m-instruct-q4_k_m-throughput");
        assert!(!name.contains('/') && !name.contains('\\'));
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn non_file_labels_have_separators_replaced() {
        let name = default_profile_name("org/model", Objective::Throughput);
        assert_eq!(name, "org-model-throughput");
        let name = default_profile_name("C:\\x\\missing.gguf", Objective::Throughput);
        assert!(!name.contains('/') && !name.contains('\\'));
    }

    #[test]
    fn profile_uses_observed_engine_version_else_estimate_only() {
        let report = report_for("m");
        let res = resolution("m", None);
        let observed = build_profile(&report, &res, "n", Some("0.7.0-dev")).expect("profile");
        assert_eq!(observed.engine.version, "0.7.0-dev");
        let estimate = build_profile(&report, &res, "n", None).expect("profile");
        assert_eq!(estimate.engine.version, ENGINE_VERSION_ESTIMATE_ONLY);
    }

    #[test]
    fn profile_size_bytes_is_the_gguf_file_size() {
        let report = report_for("m");
        let res = resolution("m", Some(105_454_432));
        let profile = build_profile(&report, &res, "n", None).expect("profile");
        assert_eq!(profile.model.size_bytes, 105_454_432);
        assert_eq!(profile.model.path, "m");
    }

    #[test]
    fn catalog_id_tune_needs_no_file_and_saves_source() {
        use attune_estimate::model::registry::{parse_catalog, SNAPSHOT};
        let catalog = parse_catalog(SNAPSHOT).expect("snapshot");
        let model = catalog.find("qwen3-4b").expect("prefix");
        let resolution = resolve_catalog_model(model, Quant::Q4_K_M).expect("resolve");
        let file = model.file_for(Quant::Q4_K_M).expect("file");
        assert_eq!(resolution.weight_bytes, Some(file.size));
        assert_eq!(resolution.params, model.params);
        assert_eq!(
            resolution.model_path.as_deref(),
            Some("~/.attune/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        );
        let source = resolution.source.expect("source");
        assert_eq!(source.sha256, file.sha256);
        assert_eq!(source.revision, file.revision);
    }

    #[test]
    fn catalog_model_without_the_quant_lists_what_exists() {
        use attune_estimate::model::registry::{parse_catalog, SNAPSHOT};
        let catalog = parse_catalog(SNAPSHOT).expect("snapshot");
        let model = catalog.find("qwen3-0.6b").expect("id");
        let err = resolve_catalog_model(model, Quant::Q4_K_M)
            .err()
            .expect("no q4_k_m file")
            .to_string();
        assert!(err.contains("q8_0"), "{err}");
    }
}
