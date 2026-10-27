//! Ranking objectives for an estimate-only sweep.
//!
//! Every objective returns a score where **higher is better**, so the ranker
//! always sorts descending. The formulas are intentionally small and exact:
//!
//! | objective    | score when `fits`          | score when `!fits` |
//! |--------------|----------------------------|--------------------|
//! | `throughput` | `decode_tps`               | `-inf`             |
//! | `latency`    | `-ttft_ms`                 | `-inf`             |
//! | `balanced`   | `decode_tps / ttft_ms`     | `-inf`             |
//!
//! `-inf` guarantees an OOM config can never outrank a config that fits,
//! whatever the raw numbers look like. Non-finite or non-positive inputs also
//! score `-inf` instead of leaking `NaN` into the sort.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::TuneError;

/// How to order candidate configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Maximise predicted decode tokens/second.
    Throughput,
    /// Minimise predicted time-to-first-token (prefill latency).
    Latency,
    /// Blend both: `decode_tps / ttft_ms` (higher is better).
    #[default]
    Balanced,
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
    /// `fits == false` short-circuits to [`f64::NEG_INFINITY`]. See the module
    /// docs for the exact per-objective formula; `balanced` is
    /// `decode_tps / ttft_ms`, which rises when throughput grows *or* TTFT
    /// falls, so neither axis can be ignored.
    pub fn score(self, decode_tps: f64, ttft_ms: f64, fits: bool) -> f64 {
        if !fits {
            return f64::NEG_INFINITY;
        }

        match self {
            Objective::Throughput => {
                if decode_tps.is_finite() && decode_tps >= 0.0 {
                    decode_tps
                } else {
                    f64::NEG_INFINITY
                }
            }
            Objective::Latency => {
                if ttft_ms.is_finite() && ttft_ms > 0.0 {
                    -ttft_ms
                } else {
                    f64::NEG_INFINITY
                }
            }
            Objective::Balanced => {
                if decode_tps.is_finite()
                    && decode_tps >= 0.0
                    && ttft_ms.is_finite()
                    && ttft_ms > 0.0
                {
                    decode_tps / ttft_ms
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
        let fast = Objective::Throughput.score(100.0, 500.0, true);
        let slow = Objective::Throughput.score(50.0, 500.0, true);
        assert!(fast > slow, "{fast} should beat {slow}");
        assert_eq!(fast, 100.0);
    }

    #[test]
    fn latency_prefers_lower_ttft() {
        let quick = Objective::Latency.score(50.0, 100.0, true);
        let slow = Objective::Latency.score(100.0, 500.0, true);
        assert!(quick > slow, "{quick} should beat {slow}");
        assert_eq!(quick, -100.0);
    }

    #[test]
    fn balanced_rewards_both_directions() {
        let base = Objective::Balanced.score(100.0, 100.0, true);
        assert!(Objective::Balanced.score(200.0, 100.0, true) > base);
        assert!(Objective::Balanced.score(100.0, 50.0, true) > base);
    }

    #[test]
    fn oom_scores_negative_infinity_for_every_objective() {
        for objective in [
            Objective::Throughput,
            Objective::Latency,
            Objective::Balanced,
        ] {
            assert_eq!(objective.score(100.0, 100.0, false), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn non_finite_inputs_score_negative_infinity() {
        assert_eq!(
            Objective::Throughput.score(f64::NAN, 100.0, true),
            f64::NEG_INFINITY
        );
        assert_eq!(
            Objective::Latency.score(100.0, f64::NAN, true),
            f64::NEG_INFINITY
        );
        assert_eq!(
            Objective::Balanced.score(100.0, 0.0, true),
            f64::NEG_INFINITY
        );
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
