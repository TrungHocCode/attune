//! Ranking objectives for an estimate-only sweep.
//!
//! Every objective returns a score where **higher is better**, so the ranker
//! always sorts descending. The formulas are intentionally small and exact:
//!
//! | objective    | score (before `ram_penalty`)                       |
//! |--------------|----------------------------------------------------|
//! | `throughput` | `decode_tps`                                       |
//! | `latency`    | `-(ttft_ms + tpot_ms)`                             |
//! | `balanced`   | `0.5 × tps/1000 − 0.5 × (ttft+tpot)/1000`          |
//!
//! `tpot_ms` is `1000 / decode_tps` ([`tpot_ms_from_tps`]). `ram_penalty` is
//! added to every objective as a subtraction:
//!
//! - `+INFINITY` when the config does not fit, so an OOM config can never
//!   outrank one that fits (`score = raw − inf = −inf`);
//! - a finite warning-level penalty `(ratio − 0.8) × 10` when the predicted
//!   peak exceeds [`RAM_WARN_RATIO`] of available RAM but still fits. This
//!   nudges a tight-but-valid config below a roomier one instead of banning
//!   it outright.
//!
//! [`stars`] converts a zero-based rank into 0–5 stars by percentile for the
//! human table; it plays no role in ranking.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::TuneError;

/// Fraction of available RAM above which a fitting config earns a finite
/// warning-level score penalty.
pub const RAM_WARN_RATIO: f64 = 0.8;

/// Score points subtracted per unit of peak/available RAM above
/// [`RAM_WARN_RATIO`]. At the 0.9 fit ceiling this is 1.0 point; the value is
/// intentionally small relative to decode tok/s but large enough to break a
/// tie between an otherwise-equal roomy and tight config.
pub const RAM_PENALTY_SCALE: f64 = 10.0;

/// Decode-throughput reference for the balanced blend, in tokens/second.
pub const BALANCED_TPS_REF: f64 = 1_000.0;

/// End-to-end token-latency reference for the balanced blend, in milliseconds.
pub const BALANCED_LATENCY_REF_MS: f64 = 1_000.0;

/// Weight of the throughput half of the balanced blend. The latency half gets
/// `1 − BALANCED_TPS_WEIGHT`.
pub const BALANCED_TPS_WEIGHT: f64 = 0.5;

/// How to order candidate configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Maximise predicted decode tokens/second.
    Throughput,
    /// Minimise predicted time-to-first-token plus per-token latency.
    Latency,
    /// Blend throughput and latency on reference-normalised scales.
    #[default]
    Balanced,
}

/// Time per output token in milliseconds, or `+inf` for an unusable rate.
///
/// Pure and total so callers never divide by zero: a non-finite or
/// non-positive tokens/second figure makes the latency term infinite, which
/// turns the objective score into `-inf`.
pub fn tpot_ms_from_tps(tps: f64) -> f64 {
    if tps.is_finite() && tps > 0.0 {
        1_000.0 / tps
    } else {
        f64::INFINITY
    }
}

/// Score penalty for running close to the memory ceiling.
///
/// `fits == false` returns [`f64::INFINITY`] (the caller subtracts it, which
/// yields `-inf`). A fitting config pays nothing up to [`RAM_WARN_RATIO`] and
/// a finite `(ratio − 0.8) × 10` above it. Unknown ratios (zero, negative or
/// non-finite) are treated as roomy rather than inventing a penalty.
pub fn ram_penalty(fits: bool, ram_ratio: f64) -> f64 {
    if !fits {
        return f64::INFINITY;
    }
    if ram_ratio.is_finite() && ram_ratio > RAM_WARN_RATIO {
        (ram_ratio - RAM_WARN_RATIO) * RAM_PENALTY_SCALE
    } else {
        0.0
    }
}

/// Convert a zero-based rank (`0` = best) into 0–5 stars by percentile.
///
/// `total` is the number of ranked configs. The best config is always 5 stars;
/// the percentile `1 − rank/total` is rounded to the nearest star, so the last
/// of 64 entries gets 0 stars. Out-of-range ranks and an empty ranking return
/// 0 instead of panicking.
pub fn stars(rank_idx: usize, total: usize) -> u8 {
    if total == 0 || rank_idx >= total {
        return 0;
    }
    let percentile = 1.0 - rank_idx as f64 / total as f64;
    (percentile * 5.0).round().clamp(0.0, 5.0) as u8
}

impl Objective {
    /// Canonical lowercase name (also the JSON value).
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::Throughput => "throughput",
            Objective::Latency => "latency",
            Objective::Balanced => "balanced",
        }
    }

    /// Score one config. Higher is always better.
    ///
    /// `ram_ratio` is `peak_bytes / avail_bytes` (pass `0.0` when RAM is
    /// unknown). See the module docs for the exact per-objective formula and
    /// the RAM penalty. Non-finite/non-positive inputs score `-inf` instead of
    /// leaking `NaN` into the sort.
    pub fn score(self, decode_tps: f64, ttft_ms: f64, fits: bool, ram_ratio: f64) -> f64 {
        let penalty = ram_penalty(fits, ram_ratio);
        let tpot_ms = tpot_ms_from_tps(decode_tps);
        let tps_ok = decode_tps.is_finite() && decode_tps >= 0.0;
        let ttft_ok = ttft_ms.is_finite() && ttft_ms > 0.0;

        match self {
            Objective::Throughput => {
                if tps_ok {
                    decode_tps - penalty
                } else {
                    f64::NEG_INFINITY
                }
            }
            Objective::Latency => {
                if tps_ok && ttft_ok && tpot_ms.is_finite() {
                    -(ttft_ms + tpot_ms) - penalty
                } else {
                    f64::NEG_INFINITY
                }
            }
            Objective::Balanced => {
                if tps_ok && ttft_ok && tpot_ms.is_finite() {
                    let latency_ms = ttft_ms + tpot_ms;
                    BALANCED_TPS_WEIGHT * (decode_tps / BALANCED_TPS_REF)
                        - (1.0 - BALANCED_TPS_WEIGHT) * (latency_ms / BALANCED_LATENCY_REF_MS)
                        - penalty
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }
}

impl std::fmt::Display for Objective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Objective {
    type Err = TuneError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "throughput" => Ok(Self::Throughput),
            "latency" => Ok(Self::Latency),
            "balanced" => Ok(Self::Balanced),
            other => Err(TuneError::InvalidObjective(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_case_insensitively_with_whitespace() {
        assert_eq!(
            "throughput".parse::<Objective>().expect("throughput"),
            Objective::Throughput
        );
        assert_eq!(
            " LATENCY ".parse::<Objective>().expect("latency"),
            Objective::Latency
        );
        assert_eq!(
            "Balanced".parse::<Objective>().expect("balanced"),
            Objective::Balanced
        );
    }

    #[test]
    fn rejects_unknown_objectives() {
        let err = "fastest".parse::<Objective>().expect_err("unknown");
        assert!(matches!(err, TuneError::InvalidObjective(_)), "got {err:?}");
        assert!(err.to_string().contains("throughput"), "message: {err}");
    }

    #[test]
    fn throughput_prefers_higher_tok_s() {
        let fast = Objective::Throughput.score(100.0, 500.0, true, 0.0);
        let slow = Objective::Throughput.score(50.0, 500.0, true, 0.0);
        assert!(fast > slow, "{fast} should beat {slow}");
        assert_eq!(fast, 100.0);
    }

    #[test]
    fn latency_prefers_lower_ttft_and_tpot() {
        let quick = Objective::Latency.score(50.0, 100.0, true, 0.0);
        let slow = Objective::Latency.score(100.0, 500.0, true, 0.0);
        assert!(quick > slow, "{quick} should beat {slow}");
        // -(100 + 1000/50) = -120.
        assert!((quick + 120.0).abs() < 1e-9, "got {quick}");
    }

    #[test]
    fn tpot_is_the_inverse_rate() {
        assert!((tpot_ms_from_tps(100.0) - 10.0).abs() < 1e-12);
        assert!(tpot_ms_from_tps(0.0).is_infinite());
        assert!(tpot_ms_from_tps(f64::NAN).is_infinite());
        assert!(tpot_ms_from_tps(-1.0).is_infinite());
    }

    #[test]
    fn balanced_rewards_both_directions() {
        let base = Objective::Balanced.score(100.0, 100.0, true, 0.0);
        assert!(Objective::Balanced.score(200.0, 100.0, true, 0.0) > base);
        assert!(Objective::Balanced.score(100.0, 50.0, true, 0.0) > base);
        // At both reference points the normalized blend is the weights' sum
        // (the tiny gap is the 1000 tok/s tpot of 1 ms).
        let reference = Objective::Balanced.score(1_000.0, 1_000.0, true, 0.0);
        assert!(reference.abs() < 1e-3, "got {reference}");
    }

    #[test]
    fn three_objectives_choose_different_winners_on_a_crafted_grid() {
        // Hand-picked predictions on one grid: the fast config has the worst
        // latency, the small-context config is the quickest, and the middle
        // config is the balanced optimum.
        let grid = [
            ("fast", 200.0_f64, 200.0_f64),
            ("middle", 140.0, 100.0),
            ("quick", 50.0, 10.0),
        ];

        let winner = |objective: Objective| {
            grid.iter()
                .enumerate()
                .map(|(idx, (_, tps, ttft))| (idx, objective.score(*tps, *ttft, true, 0.0)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(usize::MAX)
        };

        assert_eq!(winner(Objective::Throughput), 0);
        assert_eq!(winner(Objective::Latency), 2);
        assert_eq!(winner(Objective::Balanced), 1);
    }

    #[test]
    fn ram_penalty_is_infinite_for_oom_and_finite_near_the_ceiling() {
        assert!(ram_penalty(false, 0.0).is_infinite());
        assert_eq!(ram_penalty(true, 0.5), 0.0);
        assert_eq!(ram_penalty(true, RAM_WARN_RATIO), 0.0);
        let tight = ram_penalty(true, 0.85);
        assert!(tight.is_finite() && tight > 0.0, "got {tight}");
        assert!((tight - 0.5).abs() < 1e-9, "got {tight}");
        // Unknown ratios are roomy, not penalised.
        assert_eq!(ram_penalty(true, f64::NAN), 0.0);
        assert_eq!(ram_penalty(true, -1.0), 0.0);
    }

    #[test]
    fn ram_penalty_lowers_a_fitting_score_without_making_it_infinite() {
        let roomy = Objective::Throughput.score(100.0, 10.0, true, 0.5);
        let tight = Objective::Throughput.score(100.0, 10.0, true, 0.85);
        assert!(roomy > tight, "{roomy} should beat {tight}");
        assert!(tight.is_finite());
    }

    #[test]
    fn oom_scores_negative_infinity_for_every_objective() {
        for objective in [
            Objective::Throughput,
            Objective::Latency,
            Objective::Balanced,
        ] {
            let score = objective.score(100.0, 100.0, false, 0.0);
            assert_eq!(score, f64::NEG_INFINITY);
        }
    }

    #[test]
    fn non_finite_inputs_score_negative_infinity() {
        assert_eq!(
            Objective::Throughput.score(f64::NAN, 100.0, true, 0.0),
            f64::NEG_INFINITY
        );
        assert_eq!(
            Objective::Latency.score(100.0, f64::NAN, true, 0.0),
            f64::NEG_INFINITY
        );
        assert_eq!(
            Objective::Balanced.score(100.0, 0.0, true, 0.0),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn stars_follow_the_percentile_bands() {
        assert_eq!(stars(0, 0), 0);
        assert_eq!(stars(0, 1), 5);
        assert_eq!(
            (0..5).map(|rank| stars(rank, 5)).collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1]
        );
        assert_eq!(stars(63, 64), 0);
        assert_eq!(stars(64, 64), 0);
        assert_eq!(stars(usize::MAX, 64), 0);
        assert!(stars(0, 100) <= 5);
    }

    #[test]
    fn display_and_json_names_are_pinned() {
        assert_eq!(Objective::Throughput.to_string(), "throughput");
        assert_eq!(Objective::Latency.to_string(), "latency");
        assert_eq!(Objective::Balanced.to_string(), "balanced");
        assert_eq!(
            serde_json::to_string(&Objective::Balanced).expect("serialize objective"),
            "\"balanced\""
        );
    }
}
