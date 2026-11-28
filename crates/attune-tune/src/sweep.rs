//! Cartesian grid expansion, pruning and estimate-only ranking.

use attune_bench::e2e::E2eResult;
use attune_estimate::{
    default_compute_buffer_bytes, kv_elem_bytes, kv_filled_tokens, max_gpu_layers,
    predict_decode_tps_from_bytes, predict_split_decode_tps, weight_bytes_for, BenchCalibration,
    CalibrationReport, HwInputs, Quant, RamFit, SplitInputs,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::str::FromStr;

use crate::{Objective, TuneConfig, TuneError};

/// Context tokens reserved for chat templates / BOS overhead when checking
/// whether a config can hold `prompt_tokens + gen_tokens`.
pub const CTX_SAFETY_MARGIN: u64 = 32;

/// What the sweep needs to know about the model's bytes.
///
/// `weight_bytes` is the GGUF file size when the source is a file (it replaces
/// the nominal bytes-per-param table for every config); `kv_f16_per_token` is
/// the f16 KV size per token, `0.0` when the architecture is unknown. Each
/// config scales it by its own KV cache type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneModel {
    pub params: u64,
    pub weight_bytes: Option<u64>,
    pub kv_f16_per_token: f64,
    /// Number of transformer layers when the architecture is known; `None`
    /// disables GPU layer planning and split predictions.
    pub n_layers: Option<u32>,
}

impl TuneModel {
    /// Weights only, nominal bytes per parameter (no file, no KV shape).
    pub fn nominal(params: u64) -> Self {
        Self {
            params,
            weight_bytes: None,
            kv_f16_per_token: 0.0,
            n_layers: None,
        }
    }

    /// Weight bytes for `quant`: the file size when known, else nominal.
    pub fn weight_bytes_for(&self, quant: Quant) -> f64 {
        weight_bytes_for(self.params, quant, self.weight_bytes)
    }

    /// KV bytes per token for this config's cache type.
    pub fn kv_per_token(&self, cache_kv: Quant) -> f64 {
        self.kv_f16_per_token * kv_elem_bytes(cache_kv) / 2.0
    }
}

/// A GPU the sweep can plan layer offload against.
///
/// `vram_gbps` and `prefill_tflops` are already *effective* (the caller has
/// multiplied by the measured efficiency or applied a bench calibration), so
/// the pure ranking can use them without detecting anything.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuTarget {
    /// Device name, used in warnings and the report.
    pub name: String,
    /// Free VRAM in bytes, before the safety margin.
    pub vram_free_bytes: u64,
    /// Effective VRAM bandwidth in decimal GB/s.
    pub vram_gbps: f64,
    /// Effective prefill throughput in decimal TFLOP/s.
    pub prefill_tflops: f64,
}

/// Which device the sweep should plan for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    /// CPU configs plus GPU configs when a GPU is available.
    #[default]
    Auto,
    /// CPU-only configs.
    Cpu,
    /// GPU configs only; an error when none can be planned.
    Gpu,
}

impl Device {
    /// Canonical lowercase name (also the JSON value).
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }

    /// The device a ranked config actually runs on, derived from its
    /// `n_gpu_layers`.
    pub fn of_config(config: &TuneConfig) -> Self {
        if config.n_gpu_layers > 0 {
            Device::Gpu
        } else {
            Device::Cpu
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Device {
    type Err = TuneError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(TuneError::InvalidConfig(format!(
                "invalid device: {other} (expected auto, cpu or gpu)"
            ))),
        }
    }
}

/// Why [`prune`] dropped a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PruneReason {
    /// Weights do not fit in the supplied available RAM.
    Oom,
    /// `prompt_tokens + gen_tokens + margin` exceeds the context window.
    CtxOverflow,
    /// The config uses more threads than the machine has physical cores.
    TooManyThreads,
    /// The grid exceeded `--max-configs` after the checks above.
    OverCap,
}

impl PruneReason {
    /// Canonical lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            PruneReason::Oom => "oom",
            PruneReason::CtxOverflow => "ctx_overflow",
            PruneReason::TooManyThreads => "too_many_threads",
            PruneReason::OverCap => "over_cap",
        }
    }

    /// Actionable hint shown next to an OOM or overflow notice.
    ///
    /// The OOM text intentionally names the two knobs that reduce the
    /// footprint: a smaller context window and a smaller quantization.
    pub fn suggestion(self) -> &'static str {
        match self {
            PruneReason::Oom => "try a lower n_ctx or a smaller quant",
            PruneReason::CtxOverflow => "raise n_ctx or lower prompt/gen tokens",
            PruneReason::TooManyThreads => "use at most the physical core count",
            PruneReason::OverCap => "raise --max-configs to keep more candidates",
        }
    }
}

impl std::fmt::Display for PruneReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One ranked grid point plus the prediction it was scored from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedConfig {
    /// The swept config.
    pub config: TuneConfig,
    /// Device this config runs on, derived from `config.n_gpu_layers`.
    #[serde(default)]
    pub device: Device,
    /// Predicted decode throughput in tokens/second (weights-only in slice 1).
    #[serde(default)]
    pub decode_tps: f64,
    /// Predicted prefill latency / TTFT in milliseconds.
    #[serde(default)]
    pub ttft_ms: f64,
    /// Whether the weights fit in the supplied available RAM.
    #[serde(default)]
    pub fits: bool,
    /// Objective score; higher is better. OOM configs carry `-inf`, which JSON
    /// cannot represent and therefore writes as `null` (see
    /// [`serialize_score`]).
    #[serde(
        default,
        serialize_with = "serialize_score",
        deserialize_with = "deserialize_score"
    )]
    pub score: f64,
    /// Whether a real llama.cpp run measured this config (M4 slice 4).
    #[serde(default)]
    pub verified: bool,
    /// The end-to-end measurement behind [`Self::verified`]. `None` when this
    /// config was only scored by the roofline estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<E2eResult>,
    /// The bench calibration applied to this config's estimate (M4 slice 5,
    /// `attune tune --bench-json --calibrate`). `None` when the sweep ran
    /// uncalibrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationReport>,
    /// Position in the swept grid, so near-ties can go to the earliest
    /// (llama.cpp-default) config. Not part of the JSON contract.
    #[serde(skip)]
    pub grid_index: usize,
}

/// Predicted gains within this fraction of the top config are treated as ties.
/// It is the estimate's own error band (M3 uncalibrated decode error is about
/// +-6%), so smaller differences are not real and the defaults win.
const NEAR_TIE: f64 = 0.05;

/// Move the earliest-in-grid fitting config that is within [`NEAR_TIE`] of the
/// top (same quant, so a genuinely faster format still wins) to rank 1.
fn promote_near_tie(ranked: &mut Vec<RankedConfig>) {
    let Some(top) = ranked.first() else { return };
    if !top.fits {
        return;
    }
    let pick = ranked
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            r.fits
                && r.config.quant == top.config.quant
                && r.decode_tps >= (1.0 - NEAR_TIE) * top.decode_tps
                && r.ttft_ms <= (1.0 + NEAR_TIE) * top.ttft_ms
        })
        .min_by_key(|(_, r)| r.grid_index)
        .map(|(i, _)| i);
    if let Some(i) = pick.filter(|&i| i > 0) {
        let chosen = ranked.remove(i);
        ranked.insert(0, chosen);
    }
}

/// Expand a cartesian product into configs.
///
/// Loop order is fixed — `quants`, then `ctxs`, then `threads`, then
/// `batches`, then `kv_formats` (innermost) — so the same inputs always yield
/// the same vector. Empty axes simply produce an empty vector; the caller can
/// decide whether that is an error ([`rank_by_estimate`] rejects it).
pub fn expand_grid(
    quants: &[Quant],
    ctxs: &[u32],
    threads: &[u32],
    batches: &[u32],
    kv_formats: &[Quant],
) -> Vec<TuneConfig> {
    let mut configs = Vec::new();
    for &quant in quants {
        for &n_ctx in ctxs {
            for &n_threads in threads {
                for &n_batch in batches {
                    for &cache_kv in kv_formats {
                        configs.push(TuneConfig::new(quant, n_ctx, n_threads, n_batch, cache_kv));
                    }
                }
            }
        }
    }
    configs
}

/// Turn CPU grid points into GPU layer-offload candidates.
///
/// Thread count does not change decode at full offload, so every distinct
/// non-thread combination of `cpu_grid` is collapsed to a single config with
/// `n_threads = physical_cores`. For each combination the largest number of
/// layers that fit VRAM is computed with [`max_gpu_layers`]; candidates use
/// `max_fit`, `max_fit - 2` and `max_fit - 4`, keeping only values `>= 1` and
/// dropping duplicates. Returns empty when the model has no known layer count
/// or nothing fits.
pub fn gpu_configs(
    cpu_grid: &[TuneConfig],
    model: &TuneModel,
    gpu: &GpuTarget,
    physical_cores: u32,
) -> Vec<TuneConfig> {
    let Some(n_layers) = model.n_layers.filter(|&layers| layers > 0) else {
        return Vec::new();
    };
    let threads = physical_cores.max(1);
    let mut seen: Vec<TuneConfig> = Vec::new();
    let mut configs = Vec::new();

    for base in cpu_grid {
        let mut normalized = base.clone();
        normalized.n_threads = threads;
        normalized.n_gpu_layers = 0;
        if seen.contains(&normalized) {
            continue;
        }
        seen.push(normalized.clone());

        let max_fit = max_gpu_layers(
            model.weight_bytes_for(base.quant),
            n_layers,
            model.kv_per_token(base.cache_kv) * f64::from(base.n_ctx),
            default_compute_buffer_bytes(base.n_ctx),
            gpu.vram_free_bytes,
        );

        let mut emitted: Vec<u32> = Vec::new();
        for step in [0, 2, 4] {
            let Some(layers) = max_fit.checked_sub(step).filter(|&layers| layers >= 1) else {
                continue;
            };
            if emitted.contains(&layers) {
                continue;
            }
            emitted.push(layers);
            let mut config = normalized.clone();
            config.n_gpu_layers = layers;
            configs.push(config);
        }
    }

    configs
}

/// Pick the candidate set for the requested [`Device`].
///
/// `Cpu` returns the CPU grid unchanged; `Gpu` returns only fitted GPU configs
/// and fails with [`TuneError::NoGpu`] and an actionable reason when it cannot;
/// `Auto` concatenates the two, falling back to CPU-only when no GPU was
/// supplied or the model cannot be planned on one.
pub fn select_configs(
    device: Device,
    cpu_grid: Vec<TuneConfig>,
    model: &TuneModel,
    gpu: Option<&GpuTarget>,
    physical_cores: u32,
) -> Result<Vec<TuneConfig>, TuneError> {
    match device {
        Device::Cpu => Ok(cpu_grid),
        Device::Gpu => {
            let gpu = gpu.ok_or_else(|| TuneError::NoGpu("no GPU detected".to_string()))?;
            if gpu.vram_free_bytes == 0 {
                return Err(TuneError::NoGpu("GPU VRAM unknown".to_string()));
            }
            if model.n_layers.is_none_or(|layers| layers == 0) {
                return Err(TuneError::NoGpu("model layer count unknown".to_string()));
            }
            let configs = gpu_configs(&cpu_grid, model, gpu, physical_cores);
            if configs.is_empty() {
                return Err(TuneError::NoGpu(
                    "model does not fit in VRAM even with 1 layer".to_string(),
                ));
            }
            Ok(configs)
        }
        Device::Auto => {
            let mut configs = cpu_grid;
            let gpu_list = gpu
                .map(|gpu| gpu_configs(&configs, model, gpu, physical_cores))
                .unwrap_or_default();
            configs.extend(gpu_list);
            Ok(configs)
        }
    }
}

/// Split the full-window footprint into the part that stays in system RAM.
///
/// A CPU config keeps everything; a GPU config only keeps the layers it did not
/// offload, so RAM fit must count `weights × (1 - f)` and `kv × (1 - f)`.
fn cpu_resident_bytes(model: &TuneModel, config: &TuneConfig, kv_bytes: f64) -> (f64, f64) {
    let weight_bytes = model.weight_bytes_for(config.quant);
    if config.n_gpu_layers == 0 {
        return (weight_bytes, kv_bytes);
    }
    let Some(n_layers) = model.n_layers.filter(|&layers| layers > 0) else {
        return (weight_bytes, kv_bytes);
    };
    let frac = f64::from(config.n_gpu_layers.min(n_layers)) / f64::from(n_layers);
    (weight_bytes * (1.0 - frac), kv_bytes * (1.0 - frac))
}

/// Drop configs that cannot run, then enforce the candidate cap.
///
/// Checks run in a fixed order so the same grid always reports the same reason:
/// OOM, context overflow, too many threads, then `max_configs`. A non-positive
/// or non-finite `avail_bytes` (or `physical_cores == 0`) means "unknown", so
/// that check is skipped rather than pruning everything. The kept order is the
/// input order, which keeps cap truncation deterministic.
pub fn prune(
    configs: &[TuneConfig],
    model: &TuneModel,
    avail_bytes: f64,
    prompt_tokens: u64,
    gen_tokens: u64,
    physical_cores: u32,
    max_configs: usize,
) -> (Vec<TuneConfig>, Vec<(TuneConfig, PruneReason)>) {
    let required_ctx = prompt_tokens
        .saturating_add(gen_tokens)
        .saturating_add(CTX_SAFETY_MARGIN);
    let mut kept = Vec::new();
    let mut pruned = Vec::new();

    for config in configs {
        // Capacity: weights plus the KV cache for the full window. GPU configs
        // only keep the CPU-resident share in RAM; the rest lives in VRAM.
        let kv_bytes = model.kv_per_token(config.cache_kv) * f64::from(config.n_ctx);
        let (weight_bytes, kv_bytes) = cpu_resident_bytes(model, config, kv_bytes);
        let fits = RamFit::new(weight_bytes, kv_bytes, avail_bytes).is_none_or(|fit| fit.fits);
        if !fits {
            pruned.push((config.clone(), PruneReason::Oom));
            continue;
        }
        if u64::from(config.n_ctx) < required_ctx {
            pruned.push((config.clone(), PruneReason::CtxOverflow));
            continue;
        }
        if physical_cores > 0 && config.n_threads > physical_cores {
            pruned.push((config.clone(), PruneReason::TooManyThreads));
            continue;
        }
        if kept.len() >= max_configs {
            pruned.push((config.clone(), PruneReason::OverCap));
            continue;
        }
        kept.push(config.clone());
    }

    (kept, pruned)
}

/// Predict and rank every config, best score first.
///
/// All math is the pure `attune-estimate` roofline; nothing is detected or
/// spawned here. `avail_bytes` is supplied by the caller (the CLI passes the
/// HAL's available RAM) so tests are hermetic. A non-positive or non-finite
/// `avail_bytes` means "unknown": the fit check is skipped and `fits` stays
/// `true` rather than claiming an OOM from missing data.
///
/// **Prediction model:**
/// - decode is [`predict_decode_tps_from_bytes`]: [`TuneModel`] weight bytes
///   (file size for a GGUF source) plus the KV cache read at the expected
///   filled length `prompt_tokens + gen_tokens / 2` (capped at `n_ctx`), using
///   each config's own KV cache element size; no `Arch` means no KV term;
/// - TTFT is [`predict_prefill_ms_quant`] for the prompt (`prompt_tokens`,
///   clamped to `n_ctx`), the same model `attune estimate` uses;
/// - RAM fit and the near-ceiling score penalty count weights plus the KV
///   cache for the full `n_ctx` window.
///
/// Ties keep the input (grid) order because [`slice::sort_by`] is stable, and a
/// near-tie with the top goes to the earliest grid point ([`NEAR_TIE`]).
/// The `Result` exists because the roofline math rejects invalid inputs
/// (zero params, bad bandwidth/FLOPS); swallowing those would silently rank
/// nonsense.
#[allow(clippy::too_many_arguments)]
pub fn rank_by_estimate(
    configs: &[TuneConfig],
    model: &TuneModel,
    gpu: Option<&GpuTarget>,
    hw: HwInputs,
    avail_bytes: f64,
    objective: Objective,
    prompt_tokens: u64,
    gen_tokens: u64,
) -> Result<Vec<RankedConfig>, TuneError> {
    if configs.is_empty() {
        return Err(TuneError::EmptyGrid);
    }

    let mut ranked = Vec::with_capacity(configs.len());
    for (grid_index, config) in configs.iter().enumerate() {
        config.validate()?;

        let weight_bytes = model.weight_bytes_for(config.quant);
        let kv_per_token = model.kv_per_token(config.cache_kv);
        let (decode_tps, ttft_ms) = if config.n_gpu_layers > 0 {
            let Some(gpu) = gpu else {
                return Err(TuneError::NoGpu(
                    "GPU config ranking requires a GPU target".to_string(),
                ));
            };
            let Some(n_layers) = model.n_layers.filter(|&layers| layers > 0) else {
                return Err(TuneError::NoGpu("model layer count unknown".to_string()));
            };
            if !gpu.prefill_tflops.is_finite() || gpu.prefill_tflops <= 0.0 {
                return Err(TuneError::NoGpu(format!(
                    "GPU `{}` has no usable prefill throughput",
                    gpu.name
                )));
            }

            let frac = f64::from(config.n_gpu_layers.min(n_layers)) / f64::from(n_layers);
            // Decode reads the same filled-KV length as the CPU path, split
            // between VRAM and RAM by the offloaded layer fraction.
            let decode_tps = predict_split_decode_tps(&SplitInputs {
                weight_bytes,
                n_layers,
                gpu_layers: config.n_gpu_layers,
                kv_bytes: kv_per_token
                    * f64::from(kv_filled_tokens(prompt_tokens, gen_tokens, config.n_ctx)),
                vram_gbps: gpu.vram_gbps,
                ram_gbps: hw.bw_gbps,
            })?;

            // Prefill: GPU layers compute at `prefill_tflops`; the CPU-resident
            // weights stream over PCIe at ~12 GB/s once per prompt batch
            // (measured assumption from the M7 spike).
            let cpu_bytes = weight_bytes * (1.0 - frac);
            let prefill_secs = 2.0 * model.params as f64 * prompt_tokens as f64
                / (gpu.prefill_tflops * 1e12)
                + cpu_bytes / 12e9 * (prompt_tokens as f64 / f64::from(config.n_batch)).ceil();
            (decode_tps, prefill_secs * 1000.0)
        } else {
            // Decode reads the expected filled KV; RAM fit below uses the full window.
            let decode_tps = predict_decode_tps_from_bytes(
                weight_bytes,
                hw.bw_gbps,
                kv_per_token,
                kv_filled_tokens(prompt_tokens, gen_tokens, config.n_ctx),
            )?;
            let ttft_ms = hw.predict_prefill_ms(
                model.params,
                prompt_tokens.min(u64::from(config.n_ctx)),
                config.quant,
            )?;
            (decode_tps, ttft_ms)
        };

        // RAM fit counts only the CPU-resident share of weights + KV.
        let kv_full = kv_per_token * f64::from(config.n_ctx);
        let (cpu_weight_bytes, cpu_kv_bytes) = cpu_resident_bytes(model, config, kv_full);
        let fit = RamFit::new(cpu_weight_bytes, cpu_kv_bytes, avail_bytes);
        let fits = fit.is_none_or(|fit| fit.fits);
        let ram_ratio = fit.map_or(0.0, |fit| fit.peak_bytes / fit.avail_bytes);
        let score = objective.score(decode_tps, ttft_ms, fits, ram_ratio);

        ranked.push(RankedConfig {
            config: config.clone(),
            device: Device::of_config(config),
            decode_tps,
            ttft_ms,
            fits,
            score,
            verified: false,
            measured: None,
            calibration: None,
            grid_index,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    promote_near_tie(&mut ranked);
    Ok(ranked)
}

/// Apply one fitted bench calibration to every ranked config (M4 slice 5).
///
/// The caller fits [`BenchCalibration`] **once** against a reference roofline
/// (the CLI uses the best estimate config) and this function applies the same
/// `eta_mem` / `eta_comp` to every point: decode is scaled, prefill latency is
/// divided. Re-fitting per config would give each point a different factor and
/// make the grid internally inconsistent.
///
/// Each entry stores its own [`CalibrationReport`] — that config's calibrated
/// numbers plus its cross-workload warnings (quant mismatch, prompt mismatch,
/// clamped eta) — and is re-scored with the same [`Objective`]. The estimate's
/// near-ceiling RAM penalty is recovered and preserved through
/// [`crate::verify::rescore`], then the list is re-sorted best-score-first.
pub fn calibrate_ranked(
    ranked: &[RankedConfig],
    calibration: &BenchCalibration,
    bench_quant: Option<Quant>,
    objective: Objective,
    prompt_tokens: u64,
) -> Vec<RankedConfig> {
    let mut calibrated: Vec<RankedConfig> = ranked
        .iter()
        .map(|entry| {
            // TTFT is predicted for the prompt clamped to `n_ctx` (see
            // `rank_by_estimate`); report that as the modelled prompt length.
            let report = calibration.report_for(
                entry.decode_tps,
                Some(entry.ttft_ms),
                bench_quant,
                entry.config.quant,
                prompt_tokens.min(u64::from(entry.config.n_ctx)),
            );
            let decode_tps = report.calibrated_decode_tps;
            let ttft_ms = report.calibrated_prefill_ms_ttft.unwrap_or(entry.ttft_ms);

            let mut updated = entry.clone();
            updated.decode_tps = decode_tps;
            updated.ttft_ms = ttft_ms;
            updated.score = crate::verify::rescore(entry, decode_tps, ttft_ms, objective);
            updated.calibration = Some(report);
            updated
        })
        .collect();

    calibrated.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    promote_near_tie(&mut calibrated);
    calibrated
}

/// Write a finite score as a number and an OOM score as `null`.
///
/// JSON has no `-inf`; a `null` score is the documented OOM marker and
/// [`deserialize_score`] maps it back to [`f64::NEG_INFINITY`].
fn serialize_score<S>(score: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if score.is_finite() {
        serializer.serialize_some(score)
    } else {
        serializer.serialize_none()
    }
}

/// Read a score, treating `null` (or a missing field default) as `-inf`.
fn deserialize_score<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    Ok(value.unwrap_or(f64::NEG_INFINITY))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Weights-only nominal model with the default 512/128 workload.
    fn rank_nominal(
        configs: &[TuneConfig],
        params: u64,
        hw: HwInputs,
        avail: f64,
        objective: Objective,
    ) -> Result<Vec<RankedConfig>, TuneError> {
        rank_by_estimate(
            configs,
            &TuneModel::nominal(params),
            None,
            hw,
            avail,
            objective,
            512,
            128,
        )
    }

    fn hw() -> HwInputs {
        HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 50.0,
            int8_gops: None,
        }
    }

    #[test]
    fn expand_grid_is_cartesian_and_deterministic() {
        let quants = [Quant::Q4_K_M, Quant::Q8_0];
        let ctxs = [2_048, 4_096];
        let threads = [4];
        let batches = [512];
        let kv = [Quant::Fp16];

        let first = expand_grid(&quants, &ctxs, &threads, &batches, &kv);
        let second = expand_grid(&quants, &ctxs, &threads, &batches, &kv);

        assert_eq!(first.len(), 4);
        assert_eq!(first, second, "same inputs must produce the same order");
        assert_eq!(
            first[0],
            TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16)
        );
        assert_eq!(
            first[1],
            TuneConfig::new(Quant::Q4_K_M, 4_096, 4, 512, Quant::Fp16)
        );
        assert_eq!(
            first[2],
            TuneConfig::new(Quant::Q8_0, 2_048, 4, 512, Quant::Fp16)
        );
        assert_eq!(
            first[3],
            TuneConfig::new(Quant::Q8_0, 4_096, 4, 512, Quant::Fp16)
        );
    }

    #[test]
    fn expand_grid_with_an_empty_axis_is_empty() {
        assert!(expand_grid(&[], &[2_048], &[4], &[512], &[Quant::Fp16]).is_empty());
        assert!(expand_grid(&[Quant::Q4_K_M], &[], &[4], &[512], &[Quant::Fp16]).is_empty());
    }

    #[test]
    fn rank_is_best_first_for_throughput() {
        let configs = expand_grid(
            &[Quant::Q8_0, Quant::Q4_K_M],
            &[2_048],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        let ranked =
            rank_nominal(&configs, 135_000_000, hw(), 16e9, Objective::Throughput).expect("rank");

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].config.quant, Quant::Q4_K_M);
        assert_eq!(ranked[1].config.quant, Quant::Q8_0);
        assert!(ranked[0].score > ranked[1].score);
        assert!(ranked[0].fits && ranked[1].fits);
    }

    #[test]
    fn ttft_follows_the_prompt_length_and_clamps_to_n_ctx() {
        let params = 135_000_000;
        let configs = expand_grid(
            &[Quant::Q4_K_M],
            &[4_096, 2_048],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        let model = TuneModel::nominal(params);

        // Prompt fits in every window: TTFT is the prompt prefill, ctx-independent.
        let ranked = rank_nominal(&configs, params, hw(), 16e9, Objective::Latency).expect("rank");
        let want = hw()
            .predict_prefill_ms(params, 512, Quant::Q4_K_M)
            .expect("prefill");
        for entry in &ranked {
            assert!((entry.ttft_ms - want).abs() < 1e-9, "{}", entry.ttft_ms);
        }

        // Prompt larger than the smaller window: clamps to n_ctx there only.
        let ranked = rank_by_estimate(
            &configs,
            &model,
            None,
            hw(),
            16e9,
            Objective::Latency,
            3_000,
            128,
        )
        .expect("rank");
        for (n_ctx, tokens) in [(4_096_u32, 3_000_u64), (2_048, 2_048)] {
            let want = hw()
                .predict_prefill_ms(params, tokens, Quant::Q4_K_M)
                .expect("prefill");
            let entry = ranked
                .iter()
                .find(|r| r.config.n_ctx == n_ctx)
                .expect("config");
            assert!((entry.ttft_ms - want).abs() < 1e-9, "{}", entry.ttft_ms);
        }
        assert_eq!(ranked[0].config.n_ctx, 2_048);
    }

    #[test]
    fn ttft_matches_estimate_for_same_model_quant_prompt() {
        let configs = [
            TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16),
            TuneConfig::new(Quant::Q4_K_M, 4_096, 4, 512, Quant::Fp16),
        ];
        let prompt_tokens = 512;
        let ranked = rank_by_estimate(
            &configs,
            &TuneModel::nominal(135_000_000),
            None,
            hw(),
            16e9,
            Objective::Latency,
            prompt_tokens,
            128,
        )
        .expect("rank");

        let expected_ttft = hw()
            .predict_prefill_ms(135_000_000, prompt_tokens, Quant::Q4_K_M)
            .expect("estimate prefill");

        // tune and estimate agree on TTFT for the same model/quant/prompt (<= 1% tolerance)
        assert!((ranked[0].ttft_ms - expected_ttft).abs() / expected_ttft < 0.01);
        assert!((ranked[1].ttft_ms - expected_ttft).abs() / expected_ttft < 0.01);
    }

    #[test]
    fn oom_configs_score_negative_infinity() {
        let configs = expand_grid(
            &[Quant::Fp16, Quant::Q4_K_M],
            &[2_048],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        // 70B FP16 (140 GB) and 70B Q4_K_M (~43 GB) both exceed 16 GB.
        let ranked = rank_nominal(&configs, 70_000_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");

        assert!(ranked.iter().all(|config| !config.fits));
        assert!(ranked
            .iter()
            .all(|config| config.score == f64::NEG_INFINITY));
    }

    #[test]
    fn equal_scores_keep_the_input_order() {
        let configs = expand_grid(&[Quant::Q4_K_M], &[2_048], &[4, 8], &[512], &[Quant::Fp16]);
        let ranked =
            rank_nominal(&configs, 135_000_000, hw(), 16e9, Objective::Throughput).expect("rank");

        assert_eq!(ranked[0].config.n_threads, 4);
        assert_eq!(ranked[1].config.n_threads, 8);
        assert!(
            (ranked[0].score - ranked[1].score).abs() < 1e-12,
            "scores must tie: {} vs {}",
            ranked[0].score,
            ranked[1].score
        );
    }

    #[test]
    fn unknown_ram_does_not_claim_oom() {
        let configs = expand_grid(&[Quant::Q4_K_M], &[2_048], &[4], &[512], &[Quant::Fp16]);
        let ranked =
            rank_nominal(&configs, 135_000_000, hw(), 0.0, Objective::Throughput).expect("rank");
        assert!(ranked[0].fits);
    }

    #[test]
    fn empty_grid_is_an_error() {
        let err = rank_nominal(&[], 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect_err("empty grid must fail");
        assert!(matches!(err, TuneError::EmptyGrid), "got {err:?}");
    }

    #[test]
    fn invalid_config_is_an_error() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 0, 4, 512, Quant::Fp16)];
        let err = rank_nominal(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect_err("invalid config must fail");
        assert!(matches!(err, TuneError::InvalidConfig(_)), "got {err:?}");
    }

    #[test]
    fn decode_matches_the_estimate_weights_only_formula() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16)];
        let ranked =
            rank_nominal(&configs, 135_000_000, hw(), 16e9, Objective::Throughput).expect("rank");
        let expected =
            attune_estimate::predict_decode_tps(135_000_000, Quant::Q4_K_M.bytes_per_param(), 20.0)
                .expect("estimate");
        assert!(
            (ranked[0].decode_tps - expected).abs() < 1e-9,
            "got {} want {expected}",
            ranked[0].decode_tps
        );
    }

    #[test]
    fn rank_feeds_the_near_ceiling_ram_ratio_into_the_score() {
        let config = TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16);
        let params = 1_000_000_000_u64;
        let avail = 700_000_000.0;
        let ranked =
            rank_nominal(&[config], params, hw(), avail, Objective::Throughput).expect("rank");

        let weight = params as f64 * Quant::Q4_K_M.bytes_per_param();
        let ratio = weight / avail;
        let expected = ranked[0].decode_tps - (ratio - crate::objective::RAM_WARN_RATIO) * 10.0;
        assert!(ranked[0].fits);
        assert!(ratio > crate::objective::RAM_WARN_RATIO, "ratio {ratio}");
        assert!(
            (ranked[0].score - expected).abs() < 1e-9,
            "got {} want {expected}",
            ranked[0].score
        );
    }

    /// Canned bench report with the measured e2e block slice 5 calibrates from.
    fn canned_bench(prompt_tokens: u64, tok_s: f64, ttft_ms: f64) -> attune_bench::BenchReport {
        use attune_bench::bandwidth::{FullBandwidthResult, KernelStats};

        attune_bench::BenchReport {
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
            int8: None,
            resources: None,
            e2e: Some(E2eResult {
                engine: "test".to_string(),
                model: "/models/tiny.gguf".to_string(),
                prompt_tokens,
                eval_tokens: 32,
                ttft_ms,
                tpot_ms: 1_000.0 / tok_s,
                tok_s,
                ram_peak_bytes: 1_073_741_824,
                cpu_avg_pct: 70.0,
                power_avg_watts: None,
                elapsed_ms: 0.0,
            }),
            gpu: None,
        }
    }

    #[test]
    fn one_calibration_scales_the_whole_grid_and_propagates_warnings() {
        use std::path::Path;

        let configs = expand_grid(
            &[Quant::Q4_K_M],
            &[2_048, 4_096],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        let raw =
            rank_nominal(&configs, 135_000_000, hw(), 16e9, Objective::Throughput).expect("rank");

        // Fit against the best estimate: half the predicted decode and a
        // prefill 1.25x slower than predicted (both inside the clamp bounds).
        let reference = raw.first().expect("ranked config");
        let bench = canned_bench(128, reference.decode_tps * 0.5, reference.ttft_ms / 1.25);
        let fit = BenchCalibration::fit(
            &bench,
            reference.decode_tps,
            Some(reference.ttft_ms),
            Path::new("bench.json"),
        )
        .expect("canned calibration should fit");
        assert!(
            (fit.inputs.eta_mem - 0.5).abs() < 1e-9,
            "eta_mem {}",
            fit.inputs.eta_mem
        );
        assert!(
            (fit.inputs.eta_comp - 1.25).abs() < 1e-9,
            "eta_comp {}",
            fit.inputs.eta_comp
        );

        let calibrated =
            calibrate_ranked(&raw, &fit, Some(Quant::Fp16), Objective::Throughput, 512);
        assert_eq!(calibrated.len(), raw.len());

        for entry in &calibrated {
            let original = raw
                .iter()
                .find(|original| original.config == entry.config)
                .expect("raw entry for every config");
            let report = entry.calibration.as_ref().expect("calibration block");

            // The same fitted eta scales every grid point.
            assert!((report.inputs.eta_mem - fit.inputs.eta_mem).abs() < 1e-12);
            assert!((report.inputs.eta_comp - fit.inputs.eta_comp).abs() < 1e-12);
            assert!(
                (entry.decode_tps - original.decode_tps * 0.5).abs() < 1e-9,
                "decode {} vs {}",
                entry.decode_tps,
                original.decode_tps * 0.5
            );
            assert!(
                (entry.ttft_ms - original.ttft_ms / 1.25).abs() < 1e-9,
                "ttft {} vs {}",
                entry.ttft_ms,
                original.ttft_ms / 1.25
            );
            assert!((report.calibrated_decode_tps - entry.decode_tps).abs() < 1e-12);

            // Cross-workload warnings survive per config.
            assert!(
                report
                    .warnings
                    .iter()
                    .any(|warning| warning == "eta from fp16 applied to q4_k_m"),
                "{:?}",
                report.warnings
            );
            assert!(
                report
                    .warnings
                    .iter()
                    .any(|warning| warning == "prompt mismatch 128 vs 512"),
                "{:?}",
                report.warnings
            );
            assert!(
                !report
                    .warnings
                    .iter()
                    .any(|warning| warning == "eta clamped, unreliable"),
                "{:?}",
                report.warnings
            );
        }
    }

    #[test]
    fn prune_drops_oom_with_a_lower_n_ctx_or_quant_suggestion() {
        let configs = expand_grid(
            &[Quant::Fp16, Quant::Q4_K_M],
            &[2_048],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        let (kept, pruned) = prune(
            &configs,
            &TuneModel::nominal(70_000_000_000),
            16e9,
            512,
            128,
            8,
            64,
        );

        assert!(kept.is_empty());
        assert_eq!(pruned.len(), 2);
        assert!(pruned.iter().all(|(_, reason)| *reason == PruneReason::Oom));
        let suggestion = PruneReason::Oom.suggestion();
        assert!(
            suggestion.contains("n_ctx") && suggestion.contains("quant"),
            "suggestion must name both knobs: {suggestion}"
        );
    }

    #[test]
    fn prune_drops_context_overflow() {
        let configs = expand_grid(&[Quant::Q4_K_M], &[2_048], &[4], &[512], &[Quant::Fp16]);
        // 2048 + 128 + 32 = 2208 > 2048.
        let (kept, pruned) = prune(
            &configs,
            &TuneModel::nominal(135_000_000),
            16e9,
            2048,
            128,
            8,
            64,
        );
        assert!(kept.is_empty());
        assert_eq!(pruned[0].1, PruneReason::CtxOverflow);
    }

    #[test]
    fn prune_drops_too_many_threads() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 2_048, 16, 512, Quant::Fp16)];
        let (kept, pruned) = prune(
            &configs,
            &TuneModel::nominal(135_000_000),
            16e9,
            512,
            128,
            8,
            64,
        );
        assert!(kept.is_empty());
        assert_eq!(pruned[0].1, PruneReason::TooManyThreads);
    }

    #[test]
    fn prune_cap_is_deterministic_and_reports_the_count() {
        let (configs, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q4_K_M]);
        let (first, first_pruned) = prune(
            &configs,
            &TuneModel::nominal(135_000_000),
            16e9,
            512,
            128,
            8,
            5,
        );
        let (second, second_pruned) = prune(
            &configs,
            &TuneModel::nominal(135_000_000),
            16e9,
            512,
            128,
            8,
            5,
        );

        assert_eq!(first, second);
        assert_eq!(first.len(), 5);
        assert_eq!(first, configs[..5].to_vec());
        assert_eq!(
            first_pruned
                .iter()
                .filter(|(_, reason)| *reason == PruneReason::OverCap)
                .count(),
            11
        );
        assert_eq!(first_pruned, second_pruned);
    }

    #[test]
    fn prune_skips_checks_when_ram_or_cores_are_unknown() {
        let configs = expand_grid(&[Quant::Q4_K_M], &[2_048], &[4], &[512], &[Quant::Fp16]);
        let (kept, pruned) = prune(&configs, &TuneModel::nominal(135_000_000), 0.0, 0, 0, 0, 64);
        assert_eq!(kept.len(), 1);
        assert!(pruned.is_empty());
    }

    fn kv_model() -> TuneModel {
        TuneModel {
            params: 135_000_000,
            weight_bytes: Some(103_700_000),
            kv_f16_per_token: 100_000.0,
            n_layers: None,
        }
    }

    #[test]
    fn decode_uses_file_bytes_and_ctx_independent_kv_with_per_config_elem_size() {
        let configs = [
            TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16),
            TuneConfig::new(Quant::Q4_K_M, 8_192, 4, 512, Quant::Fp16),
            TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Q8_0),
        ];
        let ranked = rank_by_estimate(
            &configs,
            &kv_model(),
            None,
            hw(),
            16e9,
            Objective::Throughput,
            512,
            128,
        )
        .expect("rank");
        let tps = |ctx: u32, kv: Quant| {
            ranked
                .iter()
                .find(|r| r.config.n_ctx == ctx && r.config.cache_kv == kv)
                .expect("config")
                .decode_tps
        };
        // 20 GB/s over file bytes + 576 tokens of f16 KV, whatever the ctx.
        let expected = 20e9 / (103_700_000.0 + 576.0 * 100_000.0);
        assert!((tps(2_048, Quant::Fp16) - expected).abs() < 1e-6 * expected);
        assert!((tps(8_192, Quant::Fp16) - expected).abs() < 1e-6 * expected);
        // q8_0 KV is charged 34/32 B per element, not 2 B.
        let expected_q8 = 20e9 / (103_700_000.0 + 576.0 * 100_000.0 * 34.0 / 64.0);
        assert!((tps(2_048, Quant::Q8_0) - expected_q8).abs() < 1e-6 * expected_q8);
    }

    #[test]
    fn prune_counts_full_ctx_kv_for_capacity() {
        let configs = [
            TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16),
            TuneConfig::new(Quant::Q4_K_M, 65_536, 4, 512, Quant::Fp16),
        ];
        // 65536 x 100 kB = 6.5 GB of KV does not fit in 2 GB; 2048 x 100 kB does.
        let (kept, pruned) = prune(&configs, &kv_model(), 2e9, 512, 128, 8, 64);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].n_ctx, 2_048);
        assert_eq!(pruned[0].1, PruneReason::Oom);
    }

    #[test]
    fn near_tie_goes_to_llama_cpp_defaults() {
        let (grid, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q4_K_M]);
        let ranked = rank_by_estimate(
            &grid,
            &TuneModel::nominal(4_000_000_000),
            None,
            hw(),
            16e9,
            Objective::Throughput,
            512,
            128,
        )
        .expect("rank");
        let top = &ranked[0].config;
        assert_eq!((top.n_threads, top.n_batch, top.n_ubatch), (8, 512, 512));
        assert_eq!(top.cache_kv, Quant::Fp16);
    }

    #[test]
    fn real_gains_still_beat_the_defaults() {
        // A smaller weight format is far faster than 5%.
        let (grid, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q8_0, Quant::Q4_K_M]);
        let ranked = rank_by_estimate(
            &grid,
            &TuneModel::nominal(4_000_000_000),
            None,
            hw(),
            16e9,
            Objective::Throughput,
            512,
            128,
        )
        .expect("rank");
        assert_eq!(ranked[0].config.quant, Quant::Q4_K_M);

        // f16 KV does not fit in 250 MB, q8_0 KV does.
        let (grid, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q4_K_M]);
        let ranked = rank_by_estimate(
            &grid,
            &kv_model(),
            None,
            hw(),
            250e6,
            Objective::Throughput,
            512,
            128,
        )
        .expect("rank");
        assert!(ranked[0].fits);
        assert_eq!(ranked[0].config.cache_kv, Quant::Q8_0);
        assert_eq!(ranked[0].config.n_batch, 512);
    }

    fn rtx_4050() -> GpuTarget {
        GpuTarget {
            name: "RTX 4050 Laptop".to_string(),
            vram_free_bytes: 5_921 * 1024 * 1024,
            vram_gbps: 192.0 * attune_estimate::GPU_DECODE_EFFICIENCY,
            prefill_tflops: 15.0,
        }
    }

    fn hw33() -> HwInputs {
        HwInputs {
            bw_gbps: 33.0,
            flops_gflops: 50.0,
            int8_gops: None,
        }
    }

    fn qwen3_8b() -> TuneModel {
        TuneModel {
            params: 8_000_000_000,
            weight_bytes: Some(5_027_783_488),
            // 36 layers × 8 KV heads × 128 dim × 2 (K+V) × 2 bytes.
            kv_f16_per_token: 147_456.0,
            n_layers: Some(36),
        }
    }

    #[test]
    fn auto_with_gpu_ranks_a_split_config_above_cpu_by_3x() {
        let model = qwen3_8b();
        let gpu = rtx_4050();
        let cpu_grid = expand_grid(
            &[Quant::Q4_K_M],
            &[4_096],
            &[4, 8],
            &[512, 128],
            &[Quant::Fp16],
        );
        let configs =
            select_configs(Device::Auto, cpu_grid, &model, Some(&gpu), 8).expect("select configs");
        assert!(
            configs.iter().any(|config| config.n_gpu_layers > 0),
            "auto must add GPU configs"
        );

        let ranked = rank_by_estimate(
            &configs,
            &model,
            Some(&gpu),
            hw33(),
            16e9,
            Objective::Throughput,
            512,
            128,
        )
        .expect("rank");

        let top = &ranked[0];
        assert_eq!(top.device, Device::Gpu);
        let expected = max_gpu_layers(
            model.weight_bytes_for(top.config.quant),
            36,
            model.kv_per_token(top.config.cache_kv) * f64::from(top.config.n_ctx),
            default_compute_buffer_bytes(top.config.n_ctx),
            gpu.vram_free_bytes,
        );
        assert_eq!(top.config.n_gpu_layers, expected);
        assert!(top.config.n_gpu_layers >= 1);
        assert_eq!(top.config.n_threads, 8, "GPU configs keep one thread value");

        let best_cpu = ranked
            .iter()
            .filter(|entry| entry.device == Device::Cpu)
            .map(|entry| entry.decode_tps)
            .fold(f64::MIN, f64::max);
        assert!(
            top.decode_tps > 3.0 * best_cpu,
            "GPU {} tok/s must beat CPU {best_cpu} tok/s by 3x",
            top.decode_tps
        );
    }

    #[test]
    fn gpu_configs_respect_the_vram_fit_for_a_14b() {
        let model = TuneModel {
            params: 14_000_000_000,
            weight_bytes: Some(8_988_110_976),
            // 48 layers × 8 KV heads × 128 dim × 2 (K+V) × 2 bytes.
            kv_f16_per_token: 196_608.0,
            n_layers: Some(48),
        };
        let gpu = rtx_4050();
        let (cpu_grid, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q4_K_M]);
        let configs = gpu_configs(&cpu_grid, &model, &gpu, 8);
        assert!(!configs.is_empty(), "14B must partially fit the GPU");

        for config in &configs {
            let max = max_gpu_layers(
                model.weight_bytes_for(config.quant),
                48,
                model.kv_per_token(config.cache_kv) * f64::from(config.n_ctx),
                default_compute_buffer_bytes(config.n_ctx),
                gpu.vram_free_bytes,
            );
            assert!(
                config.n_gpu_layers <= max,
                "{} layers exceed the {max} that fit at ctx {}",
                config.n_gpu_layers,
                config.n_ctx
            );
            assert!(config.n_gpu_layers >= 1);
        }
    }

    #[test]
    fn select_cpu_returns_the_grid_and_gpu_requires_a_usable_target() {
        let model = qwen3_8b();
        let gpu = rtx_4050();
        let grid = vec![TuneConfig::new(Quant::Q4_K_M, 4_096, 4, 512, Quant::Fp16)];

        assert_eq!(
            select_configs(Device::Cpu, grid.clone(), &model, Some(&gpu), 8).expect("cpu"),
            grid
        );

        let err = select_configs(Device::Gpu, grid.clone(), &model, None, 8)
            .expect_err("gpu without a target must fail");
        assert!(matches!(err, TuneError::NoGpu(_)), "got {err:?}");
        assert!(err.to_string().contains("no GPU detected"), "{err}");

        let err = select_configs(
            Device::Gpu,
            grid.clone(),
            &TuneModel::nominal(8_000_000_000),
            Some(&gpu),
            8,
        )
        .expect_err("unknown layer count must fail");
        assert!(matches!(err, TuneError::NoGpu(_)), "got {err:?}");
        assert!(err.to_string().contains("layer count unknown"), "{err}");

        let mut blind = gpu.clone();
        blind.vram_free_bytes = 0;
        let err = select_configs(Device::Gpu, grid.clone(), &model, Some(&blind), 8)
            .expect_err("unknown VRAM must fail");
        assert!(err.to_string().contains("VRAM unknown"), "{err}");

        let huge = TuneModel {
            weight_bytes: Some(200_000_000_000),
            ..model
        };
        let err = select_configs(Device::Gpu, grid, &huge, Some(&gpu), 8)
            .expect_err("a model larger than VRAM must fail");
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn device_parses_displays_and_serializes() {
        for (input, expected) in [
            ("auto", Device::Auto),
            ("CPU", Device::Cpu),
            (" gpu ", Device::Gpu),
        ] {
            assert_eq!(input.parse::<Device>().expect("device"), expected);
        }
        assert_eq!(Device::Gpu.to_string(), "gpu");
        assert_eq!(
            serde_json::to_string(&Device::Cpu).expect("serialize"),
            "\"cpu\""
        );
        assert!("tpu".parse::<Device>().is_err());
    }

    #[test]
    fn ranking_a_gpu_config_without_a_target_is_an_error() {
        let config = TuneConfig {
            n_gpu_layers: 4,
            ..TuneConfig::new(Quant::Q4_K_M, 4_096, 4, 512, Quant::Fp16)
        };
        let err = rank_by_estimate(
            &[config],
            &qwen3_8b(),
            None,
            hw33(),
            16e9,
            Objective::Throughput,
            512,
            128,
        )
        .expect_err("GPU config requires a GPU target");
        assert!(matches!(err, TuneError::NoGpu(_)), "got {err:?}");
    }

    #[test]
    fn prune_counts_only_cpu_resident_bytes_for_gpu_configs() {
        let model = qwen3_8b();
        let cpu = TuneConfig::new(Quant::Q4_K_M, 4_096, 8, 512, Quant::Fp16);
        let gpu = TuneConfig {
            n_gpu_layers: 18,
            ..cpu.clone()
        };
        // Full footprint is ~5.6 GB; half offloaded is ~2.8 GB.
        let (kept, pruned) = prune(&[cpu, gpu.clone()], &model, 3.5e9, 512, 128, 8, 64);
        assert_eq!(kept, vec![gpu]);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].1, PruneReason::Oom);
    }
}
