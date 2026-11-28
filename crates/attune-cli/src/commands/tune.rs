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
//! Model resolution is shared with `attune estimate` (see
//! [`super::model::resolve_model`]): GGUF file -> `org/model` fetch/cache ->
//! live catalog id -> built-in registry -> manual `--params`/`--arch`.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use attune_bench::e2e::{DEFAULT_GEN_TOKENS, DEFAULT_PROMPT_TOKENS};
use attune_bench::gpu::GpuE2eResult;
use attune_bench::{run_quick, BandwidthReport, BenchReport};
use attune_estimate::{
    gpu_bandwidth_gbps, kv_bytes_per_token, parse_bench_report, parse_quants, weight_bytes_for,
    BenchCalibration, HwInputs, Quant, GPU_DECODE_EFFICIENCY,
};
use attune_hal::cpu::{theoretical_fp32_gflops, BASELINE_CLOCK_GHZ};
use attune_hal::{GpuDevice, HardwareInfo};
use attune_profile::Profile;
use attune_tune::export::hardware_fingerprint_with_gpu;
use attune_tune::ProfileInputs;
use attune_tune::{
    calibrate_ranked, expand_full_grid_with_quants, from_best, prune, rank_by_estimate,
    run_verify_topk, select_configs, stars, Device, GpuTarget, Objective, PruneReason,
    RankedConfig, TuneConfig, TuneError, TuneModel, TuneReport,
};

use super::model::{as_gguf_path, resolve_model, ModelResolution};

/// Which device `attune tune` should plan for.
///
/// A CLI-side mirror of [`Device`] because clap's `ValueEnum` derive must not
/// leak into `attune-tune` (the library owns the sweep, not argument parsing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum TuneDevice {
    /// CPU configs plus GPU configs when a GPU is usable.
    Auto,
    /// CPU-only configs.
    Cpu,
    /// GPU configs only.
    Gpu,
}

impl From<TuneDevice> for Device {
    fn from(device: TuneDevice) -> Self {
        match device {
            TuneDevice::Auto => Device::Auto,
            TuneDevice::Cpu => Device::Cpu,
            TuneDevice::Gpu => Device::Gpu,
        }
    }
}

/// Effective prefill throughput assumed for a GPU target (decimal TFLOP/s).
/// Measured at ~15 TFLOP/s effective on an RTX 4050 Laptop Vulkan build; low
/// confidence on other GPUs, so treat this as a rough constant until a GPU
/// prefill calibration exists.
const GPU_PREFILL_TFLOPS: f64 = 15.0;

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

/// Rows shown by the human table when `--verbose` is absent.
const TOP_CONFIGS: usize = 5;

/// Defaults used by `attune recommend`, which only runs the recommend half.
const RECOMMEND_QUANT: &str = "q4_k_m";
const RECOMMEND_OBJECTIVE: &str = "balanced";
const RECOMMEND_MAX_CONFIGS: usize = 64;

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

// Model resolution (`--model` / `--params`) is shared with `attune estimate`;
// see `super::model` for the priority order and the one place that decides
// where facts come from.

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
    /// Detected adapters, empty when the caller only plans for the CPU.
    gpu_devices: Vec<GpuDevice>,
}

/// Resolve the FP32 peak, memory limits and physical core count in one pass.
///
/// An explicit `--gflops` skips the CPU peak probe but still reads memory and
/// the physical core count (needed to build and prune the grid). Otherwise
/// [`HardwareInfo::detect`] supplies all of them, so the CLI never detects the
/// same hardware twice. `detect_gpu` is false for `--device cpu`, which skips
/// the (possibly slow) GPU probe entirely.
fn resolve_hardware(gflops: Option<f64>, detect_gpu: bool) -> Result<HardwareResolution> {
    if let Some(value) = gflops {
        if !value.is_finite() || value <= 0.0 {
            return Err(anyhow!("--gflops must be a finite number > 0"));
        }
        let (avail_bytes, total_bytes) = detect_memory_limits();
        let gpu_devices = if detect_gpu {
            attune_hal::gpu::detect().devices
        } else {
            Vec::new()
        };
        return Ok(HardwareResolution {
            gflops: value,
            clock_fallback: false,
            avail_bytes,
            total_bytes,
            physical_cores: detect_physical_cores(),
            gpu_devices,
        });
    }

    let hw = HardwareInfo::detect()?;
    let available = hw.memory.available_bytes as f64;
    let avail_bytes = (available.is_finite() && available > 0.0).then_some(available);
    let total = hw.memory.total_bytes as f64;
    let total_bytes = (total.is_finite() && total > 0.0).then_some(total);
    let physical_cores = hw.cpu.physical_cores.max(1);
    let gpu_devices = if detect_gpu {
        hw.gpu.devices.clone()
    } else {
        Vec::new()
    };

    let theoretical = hw.cpu.theoretical_fp32_gflops;
    if theoretical.is_finite() && theoretical > 0.0 {
        return Ok(HardwareResolution {
            gflops: theoretical,
            clock_fallback: false,
            avail_bytes,
            total_bytes,
            physical_cores,
            gpu_devices,
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
            gpu_devices,
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

/// A detected GPU that the sweep can plan a layer split against.
#[derive(Debug)]
struct GpuPlan {
    target: GpuTarget,
    /// Physical VRAM, recorded in the profile fingerprint.
    vram_total_bytes: u64,
}

/// Outcome of looking for a usable GPU for this run.
struct GpuSelection {
    plan: Option<GpuPlan>,
    /// Set when a discrete GPU exists but its bandwidth is neither in the
    /// table nor in `--bench-json`, so no target can be planned.
    unknown_bandwidth: Option<String>,
}

impl GpuSelection {
    fn none() -> Self {
        Self {
            plan: None,
            unknown_bandwidth: None,
        }
    }
}

/// Pick the first NVIDIA/AMD adapter with known free VRAM and derive its
/// effective bandwidth.
///
/// A `--bench-json` GPU block is a direct measurement of the effective VRAM
/// bandwidth and wins over the name table. The table value is multiplied by
/// [`GPU_DECODE_EFFICIENCY`] because the roofline wants effective GB/s, not
/// the marketing number. An unknown name with no measurement leaves the plan
/// empty and names the GPU in `unknown_bandwidth`.
fn select_gpu(devices: &[GpuDevice], bench: Option<&GpuE2eResult>) -> GpuSelection {
    let Some(device) = devices.iter().find(|device| {
        matches!(device.vendor.as_str(), "nvidia" | "amd") && device.vram_free_bytes.is_some()
    }) else {
        return GpuSelection::none();
    };
    let Some(vram_free_bytes) = device.vram_free_bytes else {
        return GpuSelection::none();
    };
    let vram_gbps = bench
        .map(|gpu| gpu.effective_vram_gbps)
        .filter(|gbps| gbps.is_finite() && *gbps > 0.0)
        .or_else(|| {
            gpu_bandwidth_gbps(&device.name).map(|theoretical| theoretical * GPU_DECODE_EFFICIENCY)
        });
    let Some(vram_gbps) = vram_gbps else {
        return GpuSelection {
            plan: None,
            unknown_bandwidth: Some(device.name.clone()),
        };
    };
    GpuSelection {
        plan: Some(GpuPlan {
            target: GpuTarget {
                name: device.name.clone(),
                vram_free_bytes,
                vram_gbps,
                prefill_tflops: GPU_PREFILL_TFLOPS,
            },
            vram_total_bytes: device.vram_total_bytes.unwrap_or(vram_free_bytes),
        }),
        unknown_bandwidth: None,
    }
}

/// Actionable hint appended to every `--device gpu` planning failure.
const GPU_FIX_HINT: &str = "fix: `attune setup --gpu auto`, or use --device cpu";

/// Decide the GPU for recommend mode from a [`GpuSelection`].
///
/// `cpu` never plans one, `gpu` fails with [`GPU_FIX_HINT`] when none is
/// usable, and `auto` quietly falls back to the CPU (the caller renders CPU
/// rows and never names a GPU in the header).
fn recommend_gpu_plan(device: Device, selection: GpuSelection) -> Result<Option<GpuPlan>> {
    match device {
        Device::Cpu => Ok(None),
        Device::Gpu => selection.plan.map(Some).ok_or_else(|| {
            let reason = selection.unknown_bandwidth.map_or_else(
                || "no GPU detected".to_string(),
                |name| format!("GPU {name}: bandwidth unknown"),
            );
            anyhow!("{reason}\n{GPU_FIX_HINT}")
        }),
        Device::Auto => Ok(selection.plan),
    }
}

/// The GPU llama.cpp build a saved GPU profile should be served with.
///
/// Vulkan wins when installed, then CUDA 12; a machine with neither still
/// records `vulkan` (the portable default) rather than inventing a build.
fn gpu_engine_variant() -> &'static str {
    let Ok(home) = attune_fetch::home_dir() else {
        return "vulkan";
    };
    if attune_fetch::find_installed_variant(&home, "llama-server", "vulkan").is_some() {
        "vulkan"
    } else if attune_fetch::find_installed_variant(&home, "llama-server", "cuda12").is_some() {
        "cuda12"
    } else {
        "vulkan"
    }
}

/// Readable fingerprint for the machine this profile is being saved on.
///
/// Only called when `--save` (or `attune run`) needs it, so the normal estimate
/// path never pays for a second HAL detection. Detection failures degrade to
/// the same `unknown|0|unknown|0` marker rather than blocking the save.
pub fn current_hardware_fingerprint() -> String {
    current_hardware_fingerprint_with_gpu(None)
}

/// Same as [`current_hardware_fingerprint`], but appends the GPU name and
/// rounded VRAM when a GPU profile is being saved or compared.
pub fn current_hardware_fingerprint_with_gpu(gpu: Option<(&str, u64)>) -> String {
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
    hardware_fingerprint_with_gpu(brand, cores, simd, total_ram, gpu)
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

/// GPU-column value: `<n_gpu_layers>/<n_layers>` for GPU rows, `-` for CPU.
fn gpu_layers_cell(ranked: &RankedConfig, n_layers: Option<u32>) -> String {
    if ranked.device == Device::Gpu {
        let layers = n_layers.map_or_else(|| "?".to_string(), |layers| layers.to_string());
        format!("{}/{}", ranked.config.n_gpu_layers, layers)
    } else {
        "-".to_string()
    }
}

/// Where the winning config runs, for the quiet `attune run` line.
fn placement_label(ranked: &RankedConfig, n_layers: Option<u32>) -> String {
    if ranked.device == Device::Gpu {
        let layers = n_layers.map_or_else(|| "?".to_string(), |layers| layers.to_string());
        format!("on GPU {}/{}", ranked.config.n_gpu_layers, layers)
    } else {
        "on CPU".to_string()
    }
}

/// Human-readable rank table. The score column prints `-inf` for OOM configs
/// (the JSON contract writes those as `null`).
///
/// `estimates` is the pre-verification ranking, kept only so `--verify` can
/// show the estimate next to the measured numbers (the report itself carries
/// the measured values in `decode_tps` / `ttft_ms`).
///
/// Without `verbose` only the top [`TOP_CONFIGS`] rows are shown and the score
/// column is omitted, so a newcomer sees the useful part first.
fn render_table(
    report: &TuneReport,
    resolution: &ModelResolution,
    total_ram_fallback: bool,
    gpu: Option<&GpuTarget>,
    n_layers: Option<u32>,
    estimates: Option<&[RankedConfig]>,
    verbose: bool,
) -> String {
    let mut out = String::new();
    let arch = resolution.arch.map_or("unknown", |arch| arch.name);
    let _ = writeln!(
        out,
        "Model:  {} ({}) | Arch: {arch}",
        resolution.display,
        format_params(report.params)
    );
    let _ = writeln!(
        out,
        "Inputs: {:.1} GB/s, {:.1} GFLOPS | Objective: {}",
        report.bw_gbps, report.gflops, report.objective
    );
    match (report.avail_bytes, total_ram_fallback) {
        (Some(ram), true) => {
            let _ = writeln!(
                out,
                "RAM:    {:.1} GB total (fallback; available RAM was too low)",
                ram / 1e9
            );
        }
        (Some(ram), false) => {
            let _ = writeln!(out, "RAM:    {:.1} GB available", ram / 1e9);
        }
        (None, _) => {
            let _ = writeln!(out, "RAM:    unknown (fit check skipped)");
        }
    }
    if let Some(gpu) = gpu.filter(|_| report.configs.iter().any(|r| r.device == Device::Gpu)) {
        let _ = writeln!(
            out,
            "GPU:    {} ({:.1} GB free, {:.0} GB/s effective)",
            gpu.name,
            gpu.vram_free_bytes as f64 / 1e9,
            gpu.vram_gbps,
        );
    }
    // One eta pair was fitted once and applied to every config, so the first
    // calibrated entry describes the whole sweep.
    if let Some(calibration) = report
        .configs
        .iter()
        .find_map(|ranked| ranked.calibration.as_ref())
    {
        let ttft = calibration
            .measured_ttft_ms
            .map_or_else(|| "no TTFT".to_string(), |ms| format!("{ms:.1} ms TTFT"));
        let _ = writeln!(
            out,
            "Calib:  eta_mem={:.3} (decode), eta_comp={:.3} (prefill) | bench measured {:.1} tok/s, {ttft}",
            calibration.inputs.eta_mem,
            calibration.inputs.eta_comp,
            calibration.measured_decode_tps,
        );
    }
    // Only widen the table when at least one config was measured.
    let show_verify = report.configs.iter().any(|ranked| ranked.verified);
    let total = report.configs.len();
    let shown = if verbose {
        total
    } else {
        total.min(TOP_CONFIGS)
    };
    let _ = writeln!(out);
    match (show_verify, verbose) {
        (true, true) => {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14} {:<14} {:<14} {:<14} {:<8} {:<5} {:<6} Score",
                "Rank",
                "Quant",
                "Ctx",
                "Threads",
                "Batch",
                "KV",
                "GPU",
                "Est. tok/s",
                "Est. TTFT ms",
                "E2E tok/s",
                "E2E TTFT ms",
                "Verified",
                "Fit",
                "Stars"
            );
        }
        (true, false) => {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14} {:<14} {:<14} {:<14} {:<8} {:<5} {:<6}",
                "Rank",
                "Quant",
                "Ctx",
                "Threads",
                "Batch",
                "KV",
                "GPU",
                "Est. tok/s",
                "Est. TTFT ms",
                "E2E tok/s",
                "E2E TTFT ms",
                "Verified",
                "Fit",
                "Stars"
            );
        }
        (false, true) => {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14} {:<14} {:<5} {:<6} Score",
                "Rank",
                "Quant",
                "Ctx",
                "Threads",
                "Batch",
                "KV",
                "GPU",
                "Decode tok/s",
                "TTFT ms",
                "Fit",
                "Stars"
            );
        }
        (false, false) => {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14} {:<14} {:<5} {:<6}",
                "Rank",
                "Quant",
                "Ctx",
                "Threads",
                "Batch",
                "KV",
                "GPU",
                "Decode tok/s",
                "TTFT ms",
                "Fit",
                "Stars"
            );
        }
    }
    for (index, ranked) in report.configs.iter().take(shown).enumerate() {
        let score = if ranked.score.is_finite() {
            format!("{:.4}", ranked.score)
        } else {
            "-inf".to_string()
        };
        let star_marks = "*".repeat(usize::from(stars(index, total)));
        let gpu_cell = gpu_layers_cell(ranked, n_layers);
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
            if verbose {
                let _ = writeln!(
                    out,
                    "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14.1} {:<14.1} {:<14} {:<14} {:<8} {:<5} {:<6} {}",
                    index + 1,
                    ranked.config.quant.as_str(),
                    ranked.config.n_ctx,
                    ranked.config.n_threads,
                    ranked.config.n_batch,
                    ranked.config.cache_kv.as_str(),
                    gpu_cell,
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
                let _ = writeln!(
                    out,
                    "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14.1} {:<14.1} {:<14} {:<14} {:<8} {:<5} {:<6}",
                    index + 1,
                    ranked.config.quant.as_str(),
                    ranked.config.n_ctx,
                    ranked.config.n_threads,
                    ranked.config.n_batch,
                    ranked.config.cache_kv.as_str(),
                    gpu_cell,
                    estimate.decode_tps,
                    estimate.ttft_ms,
                    e2e_tps,
                    e2e_ttft,
                    verified,
                    if ranked.fits { "yes" } else { "OOM" },
                    star_marks,
                );
            }
        } else if verbose {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14.1} {:<14.1} {:<5} {:<6} {}",
                index + 1,
                ranked.config.quant.as_str(),
                ranked.config.n_ctx,
                ranked.config.n_threads,
                ranked.config.n_batch,
                ranked.config.cache_kv.as_str(),
                gpu_cell,
                ranked.decode_tps,
                ranked.ttft_ms,
                if ranked.fits { "yes" } else { "OOM" },
                star_marks,
                score,
            );
        } else {
            let _ = writeln!(
                out,
                "{:<5} {:<8} {:<6} {:<8} {:<7} {:<6} {:<8} {:<14.1} {:<14.1} {:<5} {:<6}",
                index + 1,
                ranked.config.quant.as_str(),
                ranked.config.n_ctx,
                ranked.config.n_threads,
                ranked.config.n_batch,
                ranked.config.cache_kv.as_str(),
                gpu_cell,
                ranked.decode_tps,
                ranked.ttft_ms,
                if ranked.fits { "yes" } else { "OOM" },
                star_marks,
            );
        }
    }
    if shown < total {
        let _ = writeln!(
            out,
            "Showing the top {shown} of {total} configs; pass --verbose to see all ranked configs and scores."
        );
    }
    let _ = writeln!(out);
    if show_verify {
        let measured = report
            .configs
            .iter()
            .filter(|ranked| ranked.verified)
            .count();
        let _ = writeln!(
            out,
            "Note: {measured} config(s) re-scored from a measured llama.cpp e2e run; the rest are estimate-only."
        );
    } else {
        let _ = writeln!(
            out,
            "Note: estimate-only ranking (decode = weights + filled KV, prompt-length prefill TTFT); no llama.cpp run."
        );
    }
    for warning in &report.warnings {
        let _ = writeln!(out, "Warning:     {warning}");
    }
    out
}

/// The one-line prune summary shown before the table (verbose only).
///
/// `cap_from_expand` counts points already dropped by the 64-point full-grid
/// guard; the rest come from [`prune`] and are counted by reason.
fn render_pruned_line(pruned: &[(TuneConfig, PruneReason)], cap_from_expand: usize) -> String {
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
    format!(
        "pruned {} ({} OOM, {} ctx-overflow, {} threads, {} cap)\n",
        oom + ctx + threads + cap,
        oom,
        ctx,
        threads,
        cap
    )
}

/// Everything the human `tune` output prints, built in one place so tests can
/// assert what a newcomer sees. The prune line is verbose-only.
#[allow(clippy::too_many_arguments)]
fn render_human(
    report: &TuneReport,
    resolution: &ModelResolution,
    total_ram_fallback: bool,
    gpu: Option<&GpuTarget>,
    n_layers: Option<u32>,
    estimates: Option<&[RankedConfig]>,
    pruned: &[(TuneConfig, PruneReason)],
    cap_from_expand: usize,
    verbose: bool,
) -> String {
    let mut out = String::new();
    if verbose {
        out.push_str(&render_pruned_line(pruned, cap_from_expand));
    }
    out.push_str(&render_table(
        report,
        resolution,
        total_ram_fallback,
        gpu,
        n_layers,
        estimates,
        verbose,
    ));
    out
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

/// GPU identity a saved profile records when the winning config is a GPU one.
struct ProfileGpu<'a> {
    /// Adapter name, e.g. `NVIDIA GeForce RTX 4050 Laptop GPU`.
    name: &'a str,
    /// Physical VRAM, appended to the hardware fingerprint.
    vram_total_bytes: u64,
    /// Installed llama.cpp build (`vulkan`/`cuda12`).
    engine_variant: &'a str,
}

/// Build the versioned profile for the highest-ranked config.
///
/// `gpu` is `Some` only when the winning config is a GPU one; CPU profiles pass
/// `None` and keep exactly the pre-M7 fields (variant `cpu`, no `gpu_name`,
/// four-part fingerprint).
fn build_profile(
    report: &TuneReport,
    resolution: &ModelResolution,
    name: &str,
    engine_version: Option<&str>,
    gpu: Option<ProfileGpu<'_>>,
) -> Result<Profile> {
    let best = report
        .configs
        .first()
        .ok_or_else(|| anyhow!("cannot save a profile without a ranked config"))?;
    let quant = best.config.quant;
    let size_bytes = weight_bytes_for(report.params, quant, resolution.weight_bytes).round() as u64;
    let fingerprint = match gpu.as_ref() {
        Some(gpu) => current_hardware_fingerprint_with_gpu(Some((gpu.name, gpu.vram_total_bytes))),
        None => current_hardware_fingerprint(),
    };
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
    let mut profile = from_best(&best.config, best.decode_tps, best.ttft_ms, &inputs);
    if let Some(gpu) = gpu {
        profile.engine.variant = gpu.engine_variant.to_string();
        profile.load_config.gpu_name = Some(gpu.name.to_string());
    }
    Ok(profile)
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

/// `attune recommend`: the recommend half of [`run`] with no model source.
///
/// The defaults below only satisfy argument parsing; recommend mode ignores
/// the tuning knobs before they are used.
pub fn run_recommend(
    max_ram: Option<&str>,
    all: bool,
    refresh_catalog: bool,
    device: TuneDevice,
    json: bool,
) -> Result<()> {
    run(
        None,
        None,
        None,
        RECOMMEND_QUANT,
        RECOMMEND_OBJECTIVE,
        None,
        None,
        device,
        RECOMMEND_MAX_CONFIGS,
        None,
        None,
        VerifyArgs {
            enabled: false,
            top_k: 2,
            llama_cli: None,
            prompt_tokens: None,
            gen_tokens: None,
            ctx: None,
        },
        None,
        None,
        false,
        RecommendArgs {
            max_ram,
            all,
            refresh_catalog,
        },
        json,
        false,
        false,
    )
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
    device: TuneDevice,
    max_configs: usize,
    save: Option<&Path>,
    name: Option<&str>,
    verify: VerifyArgs<'_>,
    bench_json: Option<&Path>,
    bench_quant: Option<&str>,
    calibrate: bool,
    recommend_args: RecommendArgs<'_>,
    json: bool,
    verbose: bool,
    quiet: bool,
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
        Some(resolve_model(model, params, arch, false, Some(quants[0]))?)
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

    let device: Device = device.into();
    // Sequential on purpose: run concurrently, HAL detection competed with
    // the bandwidth kernel and the probe read 13-36 GB/s on one machine
    // (34.5 standalone). `resolve_hardware` reuses a single HAL detection for
    // the FP32 peak, available RAM and physical core count.
    let hardware = resolve_hardware(gflops, device != Device::Cpu)?;
    let bw_gbps = resolve_bandwidth(bandwidth_gbps)?;
    let HardwareResolution {
        gflops,
        clock_fallback,
        avail_bytes,
        total_bytes,
        physical_cores,
        gpu_devices,
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
        let selection = select_gpu(
            &gpu_devices,
            bench.as_ref().and_then(|bench| bench.gpu.as_ref()),
        );
        let gpu_plan = recommend_gpu_plan(device, selection)?;
        let gpu_target = gpu_plan.as_ref().map(|plan| &plan.target);
        return super::recommend::run(
            hw,
            avail_bytes.or(total_bytes).unwrap_or(0.0),
            max_ram,
            recommend_args.all,
            recommend_args.refresh_catalog,
            json,
            device,
            gpu_target,
        );
    };

    // Resolve the GPU target (auto/gpu only) and build the candidate set for
    // the requested device. A GPU with no known bandwidth is left unusable:
    // auto stays CPU-only with a note, `--device gpu` fails with a fix hint.
    let GpuSelection {
        plan,
        unknown_bandwidth,
    } = select_gpu(
        &gpu_devices,
        bench.as_ref().and_then(|bench| bench.gpu.as_ref()),
    );
    let mut gpu_note: Option<String> = None;
    let gpu_plan = match device {
        Device::Cpu => None,
        Device::Gpu => match plan {
            Some(plan) => Some(plan),
            None => {
                let reason = unknown_bandwidth.map_or_else(
                    || "no GPU detected".to_string(),
                    |name| format!("GPU {name}: bandwidth unknown"),
                );
                return Err(anyhow!("{reason}\n{GPU_FIX_HINT}"));
            }
        },
        Device::Auto => {
            if plan.is_none() {
                if let Some(name) = unknown_bandwidth.as_deref() {
                    gpu_note = Some(format!(
                        "GPU {name}: bandwidth unknown, CPU only (run `attune bench --model <file> --device gpu` to calibrate)"
                    ));
                }
            }
            plan
        }
    };
    let gpu_target = gpu_plan.as_ref().map(|plan| &plan.target);

    // Full slice-2 grid (threads/ctx/batch/KV) restricted to the requested
    // formats, then prune by OOM, context fit, thread count and the cap.
    let model = TuneModel {
        params: resolution.params,
        weight_bytes: resolution.weight_bytes,
        kv_f16_per_token: resolution.arch.as_ref().map_or(0.0, kv_bytes_per_token),
        n_layers: resolution.arch.as_ref().map(|arch| arch.n_layer),
    };
    let (cpu_grid, cap_from_expand) = expand_full_grid_with_quants(physical_cores, &quants);
    let grid =
        select_configs(device, cpu_grid, &model, gpu_target, physical_cores).map_err(|err| {
            match err {
                TuneError::NoGpu(reason) => {
                    anyhow!("GPU tuning unavailable: {reason}\n{GPU_FIX_HINT}")
                }
                other => anyhow!("{other}"),
            }
        })?;
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
        gpu_target,
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
    if let Some(note) = gpu_note {
        warnings.push(note);
    }
    warnings.extend(calibration_warnings);
    // Opt-in slice 4: measure the top-K estimate configs and re-score them.
    // Setup errors fail here; a single bad run becomes a warning and keeps
    // that config's estimate score.
    let mut observed_engine_version = None;
    if let Some(model_path) = verify_model.as_deref() {
        let llama_cli = super::setup::llama_cli_or_installed(verify.llama_cli);
        let outcome = run_verify_topk(
            &ranked,
            objective,
            verify.top_k,
            model_path,
            llama_explicit(llama_cli.as_deref())?,
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

    let best_is_gpu = report
        .configs
        .first()
        .is_some_and(|best| best.device == Device::Gpu);
    let profile_gpu = if best_is_gpu {
        gpu_plan.as_ref().map(|plan| ProfileGpu {
            name: &plan.target.name,
            vram_total_bytes: plan.vram_total_bytes,
            engine_variant: gpu_engine_variant(),
        })
    } else {
        None
    };
    let profile = match save {
        Some(path) => {
            let default_name = default_profile_name(&resolution.label, objective);
            let profile_name = name.unwrap_or(default_name.as_str());
            let profile = build_profile(
                &report,
                &resolution,
                profile_name,
                observed_engine_version.as_deref(),
                profile_gpu,
            )?;
            let document = profile.to_json_string()?;
            std::fs::write(path, document)
                .with_context(|| format!("failed to write profile to {}", path.display()))?;
            Some(profile)
        }
        None => None,
    };

    if quiet {
        if let Some(best) = report.configs.first() {
            println!(
                "Tuned: {} ctx={} threads={} batch={} kv={} {} - about {:.1} tok/s, first reply ~{:.1} s",
                best.config.quant.as_str(),
                best.config.n_ctx,
                best.config.n_threads,
                best.config.n_batch,
                best.config.cache_kv.as_str(),
                placement_label(best, model.n_layers),
                best.decode_tps,
                best.ttft_ms / 1000.0,
            );
        }
        if let (Some(_), Some(profile)) = (save, profile.as_ref()) {
            println!("Saved profile {}", profile.name);
        }
        return Ok(());
    }

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

    print!(
        "{}",
        render_human(
            &report,
            &resolution,
            total_ram_fallback,
            gpu_target,
            model.n_layers,
            estimate_snapshot.as_deref(),
            &pruned,
            cap_from_expand,
            verbose,
        )
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
    use crate::commands::model::resolve_catalog_model;
    use attune_tune::Device;

    fn report_for(label: &str) -> TuneReport {
        let config = TuneConfig::new(Quant::Q4_K_M, 2048, 4, 512, Quant::Fp16);
        let ranked = RankedConfig {
            config,
            device: Device::Cpu,
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
            kv_per_token: 0.0,
            ctx_default: None,
            provenance: crate::commands::model::ModelProvenance::Manual,
            gguf: None,
            weight_bytes,
            warnings: Vec::new(),
            model_path: None,
            source: None,
        }
    }

    fn gpu_device(
        vendor: &str,
        name: &str,
        vram_free_bytes: Option<u64>,
        vram_total_bytes: Option<u64>,
    ) -> GpuDevice {
        GpuDevice {
            vendor: vendor.to_string(),
            name: name.to_string(),
            vram_total_bytes,
            vram_free_bytes,
            driver: None,
            source: "test".to_string(),
        }
    }

    fn gpu_bench(effective_vram_gbps: f64) -> GpuE2eResult {
        GpuE2eResult {
            device: "Vulkan1".to_string(),
            device_name: Some("NVIDIA GeForce RTX 4050 Laptop GPU".to_string()),
            build: "vulkan".to_string(),
            decode_tps: 34.6,
            prefill_tps: 500.0,
            weight_bytes: 5_027_783_488,
            effective_vram_gbps,
            theoretical_vram_gbps: Some(192.0),
            eta_gpu: Some(effective_vram_gbps / 192.0),
        }
    }

    fn gpu_report_for(label: &str) -> TuneReport {
        let config = TuneConfig {
            n_gpu_layers: 24,
            ..TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16)
        };
        let ranked = RankedConfig {
            config,
            device: Device::Gpu,
            decode_tps: 300.0,
            ttft_ms: 40.0,
            fits: true,
            score: 3.0,
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

    #[test]
    fn placement_label_names_the_cpu_or_gpu_split() {
        let cpu = report_for("m");
        assert_eq!(placement_label(&cpu.configs[0], Some(36)), "on CPU");
        let gpu = gpu_report_for("m");
        assert_eq!(placement_label(&gpu.configs[0], Some(36)), "on GPU 24/36");
        assert_eq!(placement_label(&gpu.configs[0], None), "on GPU 24/?");
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
        let observed = build_profile(&report, &res, "n", Some("0.7.0-dev"), None).expect("profile");
        assert_eq!(observed.engine.version, "0.7.0-dev");
        let estimate = build_profile(&report, &res, "n", None, None).expect("profile");
        assert_eq!(estimate.engine.version, ENGINE_VERSION_ESTIMATE_ONLY);
    }

    #[test]
    fn profile_size_bytes_is_the_gguf_file_size() {
        let report = report_for("m");
        let res = resolution("m", Some(105_454_432));
        let profile = build_profile(&report, &res, "n", None, None).expect("profile");
        assert_eq!(profile.model.size_bytes, 105_454_432);
        assert_eq!(profile.model.path, "m");
    }

    #[test]
    fn gpu_best_profile_sets_variant_gpu_name_and_fingerprint() {
        let report = gpu_report_for("m");
        let res = resolution("m", None);
        let gpu = ProfileGpu {
            name: "NVIDIA GeForce RTX 4050 Laptop GPU",
            vram_total_bytes: 6_442_450_944,
            engine_variant: "vulkan",
        };
        let profile = build_profile(&report, &res, "n", None, Some(gpu)).expect("profile");
        assert_eq!(profile.engine.variant, "vulkan");
        assert_eq!(
            profile.load_config.gpu_name.as_deref(),
            Some("NVIDIA GeForce RTX 4050 Laptop GPU")
        );
        assert_eq!(profile.device(), "gpu");
        assert!(
            profile
                .hardware_fingerprint
                .ends_with("|NVIDIA GeForce RTX 4050 Laptop GPU|6"),
            "{}",
            profile.hardware_fingerprint
        );
    }

    #[test]
    fn cpu_best_profile_keeps_cpu_defaults() {
        let report = report_for("m");
        let res = resolution("m", None);
        let profile = build_profile(&report, &res, "n", None, None).expect("profile");
        assert_eq!(profile.engine.variant, "cpu");
        assert!(profile.load_config.gpu_name.is_none());
        assert_eq!(profile.device(), "cpu");
        assert_eq!(profile.hardware_fingerprint.split('|').count(), 4);
    }

    #[test]
    fn gpu_table_shows_layers_over_total_and_the_gpu_header() {
        let report = gpu_report_for("m");
        let res = resolution("m", None);
        let gpu = GpuTarget {
            name: "NVIDIA GeForce RTX 4050 Laptop GPU".to_string(),
            vram_free_bytes: 6_200_000_000,
            vram_gbps: 163.2,
            prefill_tflops: GPU_PREFILL_TFLOPS,
        };
        let text = render_human(
            &report,
            &res,
            false,
            Some(&gpu),
            Some(36),
            None,
            &[],
            0,
            false,
        );
        assert!(
            text.contains(
                "GPU:    NVIDIA GeForce RTX 4050 Laptop GPU (6.2 GB free, 163 GB/s effective)"
            ),
            "{text}"
        );
        assert!(text.contains("24/36"), "{text}");
    }

    #[test]
    fn select_gpu_builds_a_target_from_an_nvidia_device() {
        let devices = [gpu_device(
            "nvidia",
            "NVIDIA GeForce RTX 4050 Laptop GPU",
            Some(6_200_000_000),
            Some(6_442_450_944),
        )];
        let selection = select_gpu(&devices, None);
        let plan = selection.plan.expect("nvidia target");
        assert_eq!(plan.target.name, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(plan.target.vram_free_bytes, 6_200_000_000);
        assert!(
            (plan.target.vram_gbps - 192.0 * GPU_DECODE_EFFICIENCY).abs() < 1e-9,
            "{}",
            plan.target.vram_gbps
        );
        assert_eq!(plan.target.prefill_tflops, GPU_PREFILL_TFLOPS);
        assert_eq!(plan.vram_total_bytes, 6_442_450_944);
        assert!(selection.unknown_bandwidth.is_none());
    }

    #[test]
    fn select_gpu_ignores_an_intel_only_machine() {
        let devices = [gpu_device("intel", "Intel UHD Graphics", Some(1), Some(1))];
        let selection = select_gpu(&devices, None);
        assert!(selection.plan.is_none());
        assert!(selection.unknown_bandwidth.is_none());
    }

    #[test]
    fn select_gpu_marks_an_unknown_name_without_a_bench_as_unavailable() {
        let devices = [gpu_device(
            "nvidia",
            "Mystery GPU 9000",
            Some(8_000_000_000),
            Some(9_000_000_000),
        )];
        let selection = select_gpu(&devices, None);
        assert!(selection.plan.is_none());
        assert_eq!(
            selection.unknown_bandwidth.as_deref(),
            Some("Mystery GPU 9000")
        );
    }

    #[test]
    fn select_gpu_prefers_the_bench_gpu_block_over_the_name_table() {
        let devices = [gpu_device(
            "nvidia",
            "NVIDIA GeForce RTX 4050 Laptop GPU",
            Some(6_200_000_000),
            Some(6_442_450_944),
        )];
        let bench = gpu_bench(123.4);
        let selection = select_gpu(&devices, Some(&bench));
        let plan = selection.plan.expect("target from bench");
        assert!((plan.target.vram_gbps - 123.4).abs() < 1e-9);
    }

    #[test]
    fn recommend_device_gpu_errors_without_one_but_auto_falls_back() {
        let devices = [gpu_device(
            "nvidia",
            "NVIDIA GeForce RTX 4050 Laptop GPU",
            Some(6_200_000_000),
            Some(6_442_450_944),
        )];
        assert!(recommend_gpu_plan(Device::Gpu, select_gpu(&devices, None))
            .expect("gpu is usable")
            .is_some());

        let err = recommend_gpu_plan(Device::Gpu, select_gpu(&[], None))
            .expect_err("gpu without a GPU must fail");
        let text = err.to_string();
        assert!(text.contains("no GPU detected"), "{text}");
        assert!(text.contains("attune setup --gpu auto"), "{text}");
        assert!(text.contains("--device cpu"), "{text}");

        assert!(recommend_gpu_plan(Device::Auto, select_gpu(&[], None))
            .expect("auto falls back")
            .is_none());
        assert!(recommend_gpu_plan(Device::Cpu, select_gpu(&devices, None))
            .expect("cpu never needs a GPU")
            .is_none());

        let unknown = [gpu_device(
            "nvidia",
            "Mystery GPU 9000",
            Some(8_000_000_000),
            Some(9_000_000_000),
        )];
        let err = recommend_gpu_plan(Device::Gpu, select_gpu(&unknown, None))
            .expect_err("unknown bandwidth is unusable");
        assert!(err.to_string().contains("Mystery GPU 9000"), "{err}");
    }

    #[test]
    fn catalog_id_tune_needs_no_file_and_saves_source() {
        use attune_estimate::model::registry::{parse_catalog, SNAPSHOT};
        let catalog = parse_catalog(SNAPSHOT).expect("snapshot");
        let model = catalog.find("qwen3-4b-instruct-2507").expect("id");
        let resolution = resolve_catalog_model(model, Some(Quant::Q4_K_M)).expect("resolve");
        let file = model.file_for(Quant::Q4_K_M).expect("file");
        assert_eq!(resolution.weight_bytes, Some(file.size));
        assert_eq!(resolution.params, model.params);
        let expected_path = format!("~/.attune/models/{}/{}", file.repo, file.file);
        assert_eq!(
            resolution.model_path.as_deref(),
            Some(expected_path.as_str())
        );
        let source = resolution.source.expect("source");
        assert_eq!(source.sha256, file.sha256);
        assert_eq!(source.revision, file.revision);
    }

    #[test]
    fn catalog_model_without_the_quant_lists_what_exists() {
        use attune_estimate::model::registry::{parse_catalog, SNAPSHOT};
        let catalog = parse_catalog(SNAPSHOT).expect("snapshot");
        let mut model = catalog.find("qwen3-0.6b").expect("id").clone();
        model.files.retain(|file| file.quant != Quant::Q4_K_M);
        assert!(
            model.files.iter().any(|file| file.quant == Quant::Q8_0),
            "the snapshot must keep a q8_0 file for this test"
        );
        let err = resolve_catalog_model(&model, Some(Quant::Q4_K_M))
            .expect_err("no q4_k_m file")
            .to_string();
        assert!(err.contains("q8_0"), "{err}");
    }

    fn report_with(n: usize) -> TuneReport {
        let hw = HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 100.0,
            int8_gops: None,
        };
        let configs = (0..n)
            .map(|index| RankedConfig {
                config: TuneConfig::new(Quant::Q4_K_M, 2048 + index as u32, 4, 512, Quant::Fp16),
                device: Device::Cpu,
                decode_tps: 100.0 - index as f64,
                ttft_ms: 50.0 + index as f64,
                fits: true,
                score: 1.0 - index as f64 * 0.01,
                verified: false,
                measured: None,
                calibration: None,
                grid_index: index,
            })
            .collect();
        TuneReport::new("m", 135_000_000, Objective::Throughput, hw, None, configs)
    }

    fn row_count(text: &str) -> usize {
        text.lines()
            .filter(|line| {
                line.split_whitespace()
                    .next()
                    .is_some_and(|token| token.parse::<usize>().is_ok())
            })
            .count()
    }

    #[test]
    fn default_table_hides_score_and_prune_line_and_caps_rows() {
        let report = report_with(7);
        let res = resolution("m", None);
        let text = render_human(&report, &res, false, None, None, None, &[], 0, false);
        assert!(!text.contains(" Score"), "{text}");
        assert!(!text.contains("pruned"), "{text}");
        assert_eq!(row_count(&text), TOP_CONFIGS, "{text}");
        assert!(text.contains("pass --verbose"), "{text}");
    }

    #[test]
    fn verbose_table_shows_scores_all_rows_and_prune_summary() {
        let report = report_with(7);
        let res = resolution("m", None);
        let pruned = vec![(
            TuneConfig::new(Quant::Q4_K_M, 2048, 4, 512, Quant::Fp16),
            PruneReason::Oom,
        )];
        let text = render_human(&report, &res, false, None, None, None, &pruned, 2, true);
        assert!(text.contains(" Score"), "{text}");
        assert!(text.contains("pruned 3 (1 OOM"), "{text}");
        assert_eq!(row_count(&text), 7, "{text}");
        assert!(!text.contains("pass --verbose"), "{text}");
    }
}
