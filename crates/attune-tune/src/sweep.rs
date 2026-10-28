//! Cartesian grid expansion, pruning and estimate-only ranking.

use attune_bench::e2e::E2eResult;
use attune_estimate::{
    predict_decode_tps, predict_prefill_ms_quant, BenchCalibration, CalibrationReport, HwInputs,
    Quant, RamFit,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Objective, TuneConfig, TuneError};

/// Context tokens reserved for chat templates / BOS overhead when checking
/// whether a config can hold `prompt_tokens + gen_tokens`.
pub const CTX_SAFETY_MARGIN: u64 = 32;

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

/// Drop configs that cannot run, then enforce the candidate cap.
///
/// Checks run in a fixed order so the same grid always reports the same reason:
/// OOM, context overflow, too many threads, then `max_configs`. A non-positive
/// or non-finite `avail_bytes` (or `physical_cores == 0`) means "unknown", so
/// that check is skipped rather than pruning everything. The kept order is the
/// input order, which keeps cap truncation deterministic.
pub fn prune(
    configs: &[TuneConfig],
    params: u64,
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
        let weight_bytes = params as f64 * config.quant.bytes_per_param();
        let fits = RamFit::new(weight_bytes, 0.0, avail_bytes).is_none_or(|fit| fit.fits);
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
/// **Slice-1 prediction model:**
/// - decode is [`predict_decode_tps`] (weights only; no `Arch`, so the KV
///   cache read is omitted and the number is optimistic at large contexts);
/// - TTFT is [`predict_prefill_ms_quant`] for the full `n_ctx` window, the
///   worst-case prefill available without a separate prompt-length input;
/// - RAM fit and the near-ceiling score penalty count weights only, for the
///   same no-`Arch` reason.
///
/// Ties keep the input (grid) order because [`slice::sort_by`] is stable.
/// The `Result` exists because the roofline math rejects invalid inputs
/// (zero params, bad bandwidth/FLOPS); swallowing those would silently rank
/// nonsense.
pub fn rank_by_estimate(
    configs: &[TuneConfig],
    params: u64,
    hw: HwInputs,
    avail_bytes: f64,
    objective: Objective,
) -> Result<Vec<RankedConfig>, TuneError> {
    if configs.is_empty() {
        return Err(TuneError::EmptyGrid);
    }

    let mut ranked = Vec::with_capacity(configs.len());
    for config in configs {
        config.validate()?;

        let decode_tps = predict_decode_tps(params, config.quant.bytes_per_param(), hw.bw_gbps)?;
        let ttft_ms = predict_prefill_ms_quant(
            params,
            u64::from(config.n_ctx),
            hw.flops_gflops,
            config.quant,
        )?;
        let fit = RamFit::new(
            params as f64 * config.quant.bytes_per_param(),
            0.0,
            avail_bytes,
        );
        let fits = fit.is_none_or(|fit| fit.fits);
        let ram_ratio = fit.map_or(0.0, |fit| fit.peak_bytes / fit.avail_bytes);
        let score = objective.score(decode_tps, ttft_ms, fits, ram_ratio);

        ranked.push(RankedConfig {
            config: config.clone(),
            decode_tps,
            ttft_ms,
            fits,
            score,
            verified: false,
            measured: None,
            calibration: None,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
) -> Vec<RankedConfig> {
    let mut calibrated: Vec<RankedConfig> = ranked
        .iter()
        .map(|entry| {
            // `attune tune` predicts TTFT as the full-context prefill, so the
            // config's `n_ctx` is the prompt length this prediction models.
            let report = calibration.report_for(
                entry.decode_tps,
                Some(entry.ttft_ms),
                bench_quant,
                entry.config.quant,
                u64::from(entry.config.n_ctx),
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

    fn hw() -> HwInputs {
        HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 50.0,
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
        let ranked = rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].config.quant, Quant::Q4_K_M);
        assert_eq!(ranked[1].config.quant, Quant::Q8_0);
        assert!(ranked[0].score > ranked[1].score);
        assert!(ranked[0].fits && ranked[1].fits);
    }

    #[test]
    fn latency_rank_prefers_the_smaller_context() {
        let configs = expand_grid(
            &[Quant::Q4_K_M],
            &[4_096, 2_048],
            &[4],
            &[512],
            &[Quant::Fp16],
        );
        let ranked =
            rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Latency).expect("rank");

        assert_eq!(ranked[0].config.n_ctx, 2_048);
        assert!(ranked[0].score > ranked[1].score);
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
        let ranked = rank_by_estimate(&configs, 70_000_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");

        assert!(ranked.iter().all(|config| !config.fits));
        assert!(ranked
            .iter()
            .all(|config| config.score == f64::NEG_INFINITY));
    }

    #[test]
    fn equal_scores_keep_the_input_order() {
        let configs = expand_grid(&[Quant::Q4_K_M], &[2_048], &[4, 8], &[512], &[Quant::Fp16]);
        let ranked = rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");

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
        let ranked = rank_by_estimate(&configs, 135_000_000, hw(), 0.0, Objective::Throughput)
            .expect("rank");
        assert!(ranked[0].fits);
    }

    #[test]
    fn empty_grid_is_an_error() {
        let err = rank_by_estimate(&[], 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect_err("empty grid must fail");
        assert!(matches!(err, TuneError::EmptyGrid), "got {err:?}");
    }

    #[test]
    fn invalid_config_is_an_error() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 0, 4, 512, Quant::Fp16)];
        let err = rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect_err("invalid config must fail");
        assert!(matches!(err, TuneError::InvalidConfig(_)), "got {err:?}");
    }

    #[test]
    fn decode_matches_the_estimate_weights_only_formula() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16)];
        let ranked = rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");
        let expected = predict_decode_tps(135_000_000, Quant::Q4_K_M.bytes_per_param(), 20.0)
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
            rank_by_estimate(&[config], params, hw(), avail, Objective::Throughput).expect("rank");

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
        let raw = rank_by_estimate(&configs, 135_000_000, hw(), 16e9, Objective::Throughput)
            .expect("rank");

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

        let calibrated = calibrate_ranked(&raw, &fit, Some(Quant::Fp16), Objective::Throughput);
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
                    .any(|warning| warning
                        == &format!("prompt mismatch 128 vs {}", entry.config.n_ctx)),
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
        let (kept, pruned) = prune(&configs, 70_000_000_000, 16e9, 512, 128, 8, 64);

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
        let (kept, pruned) = prune(&configs, 135_000_000, 16e9, 2048, 128, 8, 64);
        assert!(kept.is_empty());
        assert_eq!(pruned[0].1, PruneReason::CtxOverflow);
    }

    #[test]
    fn prune_drops_too_many_threads() {
        let configs = [TuneConfig::new(Quant::Q4_K_M, 2_048, 16, 512, Quant::Fp16)];
        let (kept, pruned) = prune(&configs, 135_000_000, 16e9, 512, 128, 8, 64);
        assert!(kept.is_empty());
        assert_eq!(pruned[0].1, PruneReason::TooManyThreads);
    }

    #[test]
    fn prune_cap_is_deterministic_and_reports_the_count() {
        let (configs, _) = crate::expand_full_grid_with_quants(8, &[Quant::Q4_K_M]);
        let (first, first_pruned) = prune(&configs, 135_000_000, 16e9, 512, 128, 8, 5);
        let (second, second_pruned) = prune(&configs, 135_000_000, 16e9, 512, 128, 8, 5);

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
        let (kept, pruned) = prune(&configs, 135_000_000, 0.0, 0, 0, 0, 64);
        assert_eq!(kept.len(), 1);
        assert!(pruned.is_empty());
    }
}
