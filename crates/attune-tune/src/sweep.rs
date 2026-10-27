//! Cartesian grid expansion and estimate-only ranking.

use attune_estimate::{predict_decode_tps, predict_prefill_ms_quant, HwInputs, Quant, RamFit};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Objective, TuneConfig, TuneError};

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
/// - RAM fit counts weights only, for the same no-`Arch` reason.
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
        let fits = RamFit::new(
            params as f64 * config.quant.bytes_per_param(),
            0.0,
            avail_bytes,
        )
        .is_none_or(|fit| fit.fits);
        let score = objective.score(decode_tps, ttft_ms, fits);

        ranked.push(RankedConfig {
            config: *config,
            decode_tps,
            ttft_ms,
            fits,
            score,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(ranked)
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
}
