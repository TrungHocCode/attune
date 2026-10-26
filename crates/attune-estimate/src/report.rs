//! The `attune estimate` JSON contract.

use attune_bench::bandwidth::KernelStats;
use serde::{Deserialize, Serialize};

use crate::calibrate::CalibratedInputs;
use crate::model::Quant;
use crate::SCHEMA;

/// What is limiting throughput. Slice 1 only ever emits
/// [`Bottleneck::MemoryBandwidth`]; slice 2 computes the roofline ridge
/// (`AI* = FLOPS_peak / BW`) and returns [`Bottleneck::Compute`] when the
/// workload's arithmetic intensity is at or above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    #[default]
    MemoryBandwidth,
    Compute,
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bottleneck::MemoryBandwidth => f.write_str("memory bandwidth"),
            Bottleneck::Compute => f.write_str("compute"),
        }
    }
}

/// Calibration learned from a saved `attune bench --model` report.
///
/// Present only when the user passes both `--bench-json` and `--calibrate`.
/// The nested object keeps every new field grouped (and optional at the
/// `EstimateReport` level) so older slice-1/2 documents still parse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// The fitted factors. `eta_comp` is the neutral `1.0` when the bench had
    /// no measurable TTFT.
    #[serde(flatten)]
    pub inputs: CalibratedInputs,
    /// Decode throughput the bench measured, in tokens/second.
    #[serde(default)]
    pub measured_decode_tps: f64,
    /// Roofline decode throughput scaled by `eta_mem`.
    #[serde(default)]
    pub calibrated_decode_tps: f64,
    /// TTFT the bench measured, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_ttft_ms: Option<f64>,
    /// Roofline prefill latency divided by `eta_comp`, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_prefill_ms_ttft: Option<f64>,
}

/// How much trust to put in the prediction interval (M3 slice 4).
///
/// `level` is a coarse bucket derived from the measured bandwidth spread:
/// `high` at ≤5%, `medium` at ≤15%, `low` above that, and `unavailable` when
/// no spread could be measured. `method` is the human-readable provenance —
/// it always says what the interval was computed from, including why it is
/// missing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Confidence {
    /// `high`, `medium`, `low` or `unavailable`.
    pub level: String,
    /// Human-readable derivation (or the reason the interval is unavailable).
    pub method: String,
}

impl Confidence {
    /// Spreads at or below this percentage are `high` confidence.
    pub const HIGH_MAX_SPREAD_PCT: f64 = 5.0;
    /// Spreads at or below this percentage are `medium` confidence; anything
    /// larger is `low`.
    pub const MEDIUM_MAX_SPREAD_PCT: f64 = 15.0;

    /// Build a confidence block from a measured spread percentage.
    ///
    /// A non-finite spread, a negative spread, or one of 100% or more falls
    /// back to [`Confidence::unavailable`] so a corrupt measurement cannot
    /// produce a nonsensical level.
    pub fn measured(spread_pct: f64, source: impl Into<String>) -> Self {
        if !spread_pct.is_finite() || spread_pct < 0.0 || spread_pct >= 100.0 {
            return Self::unavailable(format!("unavailable: invalid spread {spread_pct}"));
        }

        let level = if spread_pct <= Self::HIGH_MAX_SPREAD_PCT {
            "high"
        } else if spread_pct <= Self::MEDIUM_MAX_SPREAD_PCT {
            "medium"
        } else {
            "low"
        };

        let source: String = source.into();
        Self {
            level: level.to_string(),
            method: format!("half-spread {spread_pct:.1}% from {source}"),
        }
    }

    /// Build a confidence block when no spread is available, carrying the
    /// reason in `method`.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            level: "unavailable".to_string(),
            method: reason.into(),
        }
    }
}

/// Symmetric prediction interval `pred × (1 ± spread_pct / 100)`.
///
/// `spread_pct` is a percentage half-width (`13.6` means ±13.6%). Returns
/// `None` for a non-finite or negative prediction, a non-finite/negative
/// spread, or a spread of 100% or more (which would make the lower bound
/// non-positive). The pure formula is unit-tested directly; callers normally
/// obtain `spread_pct` from [`spread_pct_from_stats`].
pub fn interval(pred: f64, spread_pct: f64) -> Option<(f64, f64)> {
    if !pred.is_finite()
        || pred < 0.0
        || !spread_pct.is_finite()
        || spread_pct < 0.0
        || spread_pct >= 100.0
    {
        return None;
    }

    let half = spread_pct / 100.0;
    let lo = pred * (1.0 - half);
    let hi = pred * (1.0 + half);
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// Half-spread of a measured kernel, as a percentage:
/// `(best - median) / median × 100`.
///
/// Returns `None` when the median is not finite and positive, when `best` is
/// not finite or below the median, or when the result would not be finite.
/// A zero spread is valid and yields `Some(0.0)`.
pub fn spread_pct_from_stats(stats: &KernelStats) -> Option<f64> {
    let median = stats.median_gbps;
    let best = stats.best_gbps;
    if !median.is_finite() || median <= 0.0 || !best.is_finite() || best < median {
        return None;
    }

    let spread = (best - median) / median * 100.0;
    spread.is_finite().then_some(spread)
}

/// Hardware inputs a slice-2 report was computed from, echoed so a consumer
/// can reproduce the prediction without re-running detection.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EstimateInputs {
    /// Memory bandwidth in decimal GB/s.
    #[serde(default)]
    pub bw_gbps: f64,
    /// FP32 peak in decimal GFLOPS.
    #[serde(default)]
    pub gflops: f64,
    /// Max CPU clock in GHz used to derive `gflops`. 0.0 when the caller
    /// supplied an explicit peak or the clock could not be detected.
    /// `serde(default)` keeps older report JSON without this field parseable.
    #[serde(default)]
    pub clock_ghz: f64,
}

/// One `attune estimate` result.
///
/// `serde(default)` is on every field except the schema stamp so a future
/// slice can add/omit fields without breaking older consumers — the same
/// convention `attune-hal` and `attune-bench` use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstimateReport {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// Model-size label as supplied by the user (e.g. `"135M"`).
    #[serde(default)]
    pub model: String,
    /// Weight format.
    #[serde(default)]
    pub quant: Quant,
    /// Predicted decode throughput in tokens/second.
    #[serde(default)]
    pub decode_tps: f64,
    /// Lower bound of the decode-throughput interval, when a measured
    /// bandwidth spread was available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tps_lo: Option<f64>,
    /// Upper bound of the decode-throughput interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tps_hi: Option<f64>,
    /// Dominant limiter for the current (decode) prediction.
    #[serde(default)]
    pub bottleneck: Bottleneck,
    /// Predicted prefill latency / time-to-first-token in milliseconds, when
    /// slice-2 inputs were available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms_ttft: Option<f64>,
    /// Lower bound of the prefill/TTFT interval, when a spread was available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms_lo: Option<f64>,
    /// Upper bound of the prefill/TTFT interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms_hi: Option<f64>,
    /// Roofline ridge point `AI* = FLOPS / BW`, in FLOP per byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ridge_ai: Option<f64>,
    /// Distance of the workload's arithmetic intensity from the ridge, in
    /// percent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin_pct: Option<f64>,
    /// Hardware inputs the prediction was built from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<EstimateInputs>,
    /// Calibration against a saved bench report, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationReport>,
    /// Interval provenance / confidence level (M3 slice 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
}

impl EstimateReport {
    /// Build a report for the slice-1 decode prediction. The bottleneck is
    /// memory bandwidth by construction; slice-2 callers replace it with the
    /// real ridge comparison via [`EstimateReport::with_bottleneck`].
    pub fn new(model: impl Into<String>, quant: Quant, decode_tps: f64) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            model: model.into(),
            quant,
            decode_tps,
            decode_tps_lo: None,
            decode_tps_hi: None,
            bottleneck: Bottleneck::MemoryBandwidth,
            prefill_ms_ttft: None,
            prefill_ms_lo: None,
            prefill_ms_hi: None,
            ridge_ai: None,
            margin_pct: None,
            inputs: None,
            calibration: None,
            confidence: None,
        }
    }

    /// Attach the hardware inputs the prediction used.
    pub fn with_inputs(mut self, inputs: EstimateInputs) -> Self {
        self.inputs = Some(inputs);
        self
    }

    /// Attach the prefill/TTFT latency in milliseconds.
    pub fn with_prefill_ms(mut self, prefill_ms_ttft: f64) -> Self {
        self.prefill_ms_ttft = Some(prefill_ms_ttft);
        self
    }

    /// Attach the ridge point and the workload's margin from it.
    pub fn with_ridge(mut self, ridge_ai: f64, margin_pct: f64) -> Self {
        self.ridge_ai = Some(ridge_ai);
        self.margin_pct = Some(margin_pct);
        self
    }

    /// Override the bottleneck classification.
    pub fn with_bottleneck(mut self, bottleneck: Bottleneck) -> Self {
        self.bottleneck = bottleneck;
        self
    }

    /// Attach the calibration block produced by `--calibrate`.
    pub fn with_calibration(mut self, calibration: CalibrationReport) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// Attach confidence-interval bounds around the decode and prefill point
    /// estimates. Both use the same half-spread; `prefill_ms_lo/hi` stay
    /// `None` when no prefill prediction is present.
    pub fn with_intervals(mut self, spread_pct: f64) -> Self {
        if let Some((lo, hi)) = interval(self.decode_tps, spread_pct) {
            self.decode_tps_lo = Some(lo);
            self.decode_tps_hi = Some(hi);
        }
        if let Some(prefill) = self.prefill_ms_ttft {
            if let Some((lo, hi)) = interval(prefill, spread_pct) {
                self.prefill_ms_lo = Some(lo);
                self.prefill_ms_hi = Some(hi);
            }
        }
        self
    }

    /// Attach the interval provenance / confidence level.
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// Multi-config `attune estimate` output (M3 slice 4).
///
/// Emitted instead of a bare [`EstimateReport`] when `--quant` names more than
/// one format. The `schema`/`model` fields mirror the single-report contract
/// while each entry in `configs` is a full [`EstimateReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiEstimateReport {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// Model-size label as supplied by the user (e.g. `"135M"`).
    #[serde(default)]
    pub model: String,
    /// One report per requested weight format.
    #[serde(default)]
    pub configs: Vec<EstimateReport>,
}

impl MultiEstimateReport {
    /// Wrap a list of per-config reports in the multi-config contract.
    pub fn new(model: impl Into<String>, configs: Vec<EstimateReport>) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            model: model.into(),
            configs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> EstimateReport {
        EstimateReport::new("135M", Quant::Q4_K_M, 246.913_58)
    }

    #[test]
    fn new_stamps_the_schema_and_memory_bottleneck() {
        let report = sample();
        assert_eq!(report.schema, SCHEMA);
        assert_eq!(report.bottleneck, Bottleneck::MemoryBandwidth);
        assert_eq!(report.decode_tps_lo, None);
        assert_eq!(report.decode_tps_hi, None);
        assert_eq!(report.prefill_ms_ttft, None);
        assert_eq!(report.prefill_ms_lo, None);
        assert_eq!(report.prefill_ms_hi, None);
        assert_eq!(report.ridge_ai, None);
        assert_eq!(report.margin_pct, None);
        assert_eq!(report.inputs, None);
        assert_eq!(report.calibration, None);
        assert_eq!(report.confidence, None);
    }

    #[test]
    fn builders_attach_the_slice_two_fields() {
        let report = sample()
            .with_inputs(EstimateInputs {
                bw_gbps: 20.0,
                gflops: 50.0,
                clock_ghz: 4.0,
            })
            .with_prefill_ms(2764.8)
            .with_ridge(2.5, 33.333_333)
            .with_bottleneck(Bottleneck::Compute);

        assert_eq!(report.bottleneck, Bottleneck::Compute);
        assert_eq!(report.prefill_ms_ttft, Some(2764.8));
        assert_eq!(report.ridge_ai, Some(2.5));
        assert_eq!(report.margin_pct, Some(33.333_333));
        let inputs = report.inputs.expect("inputs");
        assert_eq!(inputs.bw_gbps, 20.0);
        assert_eq!(inputs.clock_ghz, 4.0);
    }

    #[test]
    fn json_roundtrip_carries_the_documented_keys() {
        let report = sample();
        let json = serde_json::to_string(&report).expect("serialize EstimateReport");
        for key in ["schema", "model", "quant", "decode_tps", "bottleneck"] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        assert!(
            json.contains("\"schema\":\"attune-estimate/1\""),
            "schema value pinned: {json}"
        );
        assert!(
            json.contains("\"quant\":\"q4_k_m\""),
            "quant value pinned: {json}"
        );
        assert!(
            json.contains("\"bottleneck\":\"memory_bandwidth\""),
            "bottleneck value pinned: {json}"
        );

        let back: EstimateReport = serde_json::from_str(&json).expect("deserialize EstimateReport");
        assert_eq!(back, report);
        assert_eq!(back.schema, SCHEMA);
    }

    #[test]
    fn json_roundtrip_carries_the_slice_two_fields() {
        let report = sample()
            .with_inputs(EstimateInputs {
                bw_gbps: 20.0,
                gflops: 50.0,
                clock_ghz: 4.0,
            })
            .with_prefill_ms(2764.8)
            .with_ridge(2.5, 33.333_333);
        let json = serde_json::to_string(&report).expect("serialize EstimateReport");
        for key in [
            "prefill_ms_ttft",
            "ridge_ai",
            "margin_pct",
            "inputs",
            "bw_gbps",
            "gflops",
            "clock_ghz",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }

        let back: EstimateReport = serde_json::from_str(&json).expect("deserialize EstimateReport");
        assert_eq!(back, report);
    }

    #[test]
    fn slice_one_json_still_parses_with_defaults() {
        // Exactly the shape slice 1 emitted: no slice-2 keys at all.
        let old_json = r#"{
            "schema": "attune-estimate/1",
            "model": "135M",
            "quant": "q4_k_m",
            "decode_tps": 246.91358,
            "bottleneck": "memory_bandwidth"
        }"#;
        let back: EstimateReport =
            serde_json::from_str(old_json).expect("deserialize slice-1 JSON");
        assert_eq!(back.decode_tps, 246.913_58);
        assert_eq!(back.bottleneck, Bottleneck::MemoryBandwidth);
        assert_eq!(back.decode_tps_lo, None);
        assert_eq!(back.decode_tps_hi, None);
        assert_eq!(back.prefill_ms_ttft, None);
        assert_eq!(back.prefill_ms_lo, None);
        assert_eq!(back.prefill_ms_hi, None);
        assert_eq!(back.ridge_ai, None);
        assert_eq!(back.margin_pct, None);
        assert_eq!(back.inputs, None);
        assert_eq!(back.calibration, None);
        assert_eq!(back.confidence, None);
    }

    #[test]
    fn slice_two_inputs_without_clock_still_parse() {
        // Slice-2/3 reports had `inputs` without `clock_ghz`; the new field
        // must default to 0.0 so those documents keep loading.
        let old_json = r#"{
            "schema": "attune-estimate/1",
            "model": "135M",
            "quant": "q4_k_m",
            "decode_tps": 246.91358,
            "bottleneck": "memory_bandwidth",
            "inputs": { "bw_gbps": 20.0, "gflops": 50.0 }
        }"#;
        let back: EstimateReport =
            serde_json::from_str(old_json).expect("deserialize pre-clock JSON");
        let inputs = back.inputs.expect("inputs");
        assert_eq!(inputs.bw_gbps, 20.0);
        assert_eq!(inputs.gflops, 50.0);
        assert_eq!(inputs.clock_ghz, 0.0);
    }

    #[test]
    fn partial_json_uses_the_future_facing_defaults() {
        let old_json = r#"{"schema": "attune-estimate/1"}"#;
        let back: EstimateReport = serde_json::from_str(old_json).expect("deserialize partial");
        assert_eq!(back.schema, SCHEMA);
        assert!(back.model.is_empty());
        assert_eq!(back.quant, Quant::Fp16);
        assert_eq!(back.decode_tps, 0.0);
        assert_eq!(back.bottleneck, Bottleneck::MemoryBandwidth);
        assert_eq!(back.decode_tps_lo, None);
        assert_eq!(back.decode_tps_hi, None);
        assert_eq!(back.prefill_ms_ttft, None);
        assert_eq!(back.prefill_ms_lo, None);
        assert_eq!(back.prefill_ms_hi, None);
        assert_eq!(back.ridge_ai, None);
        assert_eq!(back.margin_pct, None);
        assert_eq!(back.inputs, None);
        assert_eq!(back.calibration, None);
        assert_eq!(back.confidence, None);
    }

    #[test]
    fn json_roundtrip_carries_the_calibration_block() {
        let report = sample().with_calibration(CalibrationReport {
            inputs: CalibratedInputs {
                eta_mem: 0.225,
                eta_comp: 0.5,
            },
            measured_decode_tps: 55.6,
            calibrated_decode_tps: 55.5525,
            measured_ttft_ms: Some(120.0),
            calibrated_prefill_ms_ttft: Some(5529.6),
        });
        let json = serde_json::to_string(&report).expect("serialize EstimateReport");
        for key in [
            "calibration",
            "eta_mem",
            "eta_comp",
            "measured_decode_tps",
            "calibrated_decode_tps",
            "measured_ttft_ms",
            "calibrated_prefill_ms_ttft",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }

        let back: EstimateReport = serde_json::from_str(&json).expect("deserialize EstimateReport");
        assert_eq!(back, report);
    }

    #[test]
    fn interval_applies_the_symmetric_half_spread() {
        // Spread (25 - 22) / 22 = 13.636…%; around 246.91358 tok/s the
        // formula predicts roughly [213.2, 280.6].
        let spread_pct = (25.0 - 22.0) / 22.0 * 100.0;
        let pred = 246.913_58;
        let (lo, hi) = interval(pred, spread_pct).expect("valid interval");
        let half = spread_pct / 100.0;

        assert!((lo - pred * (1.0 - half)).abs() < 1e-9, "lo {lo}");
        assert!((hi - pred * (1.0 + half)).abs() < 1e-9, "hi {hi}");
        assert!(lo > 213.0 && lo < 214.0, "lo {lo}");
        assert!(hi > 280.0 && hi < 281.0, "hi {hi}");
    }

    #[test]
    fn interval_rejects_invalid_inputs() {
        for (pred, spread) in [
            (-1.0, 10.0),
            (f64::NAN, 10.0),
            (f64::INFINITY, 10.0),
            (100.0, -1.0),
            (100.0, f64::NAN),
            (100.0, 100.0),
            (100.0, 150.0),
        ] {
            assert!(
                interval(pred, spread).is_none(),
                "({pred}, {spread}) should be rejected"
            );
        }
    }

    #[test]
    fn kernel_stats_spread_is_best_minus_median_over_median() {
        let stats = KernelStats {
            median_gbps: 22.0,
            best_gbps: 25.0,
        };
        let spread = spread_pct_from_stats(&stats).expect("spread");
        let expected = (25.0 - 22.0) / 22.0 * 100.0;
        assert!((spread - expected).abs() < 1e-12, "got {spread}");
        assert!(spread > 13.6 && spread < 13.7, "got {spread}");
    }

    #[test]
    fn kernel_stats_spread_is_none_when_unavailable() {
        let cases = [
            KernelStats {
                median_gbps: 0.0,
                best_gbps: 10.0,
            },
            KernelStats {
                median_gbps: -1.0,
                best_gbps: 10.0,
            },
            KernelStats {
                median_gbps: f64::NAN,
                best_gbps: 10.0,
            },
            KernelStats {
                median_gbps: 10.0,
                best_gbps: 5.0,
            },
            KernelStats {
                median_gbps: 10.0,
                best_gbps: f64::INFINITY,
            },
        ];
        for stats in cases {
            assert!(
                spread_pct_from_stats(&stats).is_none(),
                "spread for {stats:?} should be unavailable"
            );
        }

        let flat = KernelStats {
            median_gbps: 10.0,
            best_gbps: 10.0,
        };
        assert_eq!(spread_pct_from_stats(&flat), Some(0.0));
    }

    #[test]
    fn confidence_levels_follow_the_spread_thresholds() {
        assert_eq!(Confidence::measured(0.0, "test").level, "high");
        assert_eq!(Confidence::measured(5.0, "test").level, "high");
        assert_eq!(Confidence::measured(5.01, "test").level, "medium");
        assert_eq!(Confidence::measured(15.0, "test").level, "medium");
        assert_eq!(Confidence::measured(15.01, "test").level, "low");
        assert_eq!(Confidence::measured(f64::NAN, "test").level, "unavailable");
        assert_eq!(Confidence::measured(100.0, "test").level, "unavailable");
        assert_eq!(Confidence::unavailable("no spread").level, "unavailable");
    }

    #[test]
    fn with_intervals_fills_decode_and_prefill_bounds() {
        let report = sample().with_prefill_ms(100.0).with_intervals(10.0);
        let (lo, hi) = (
            report.decode_tps_lo.expect("decode lo"),
            report.decode_tps_hi.expect("decode hi"),
        );
        assert!((lo - report.decode_tps * 0.9).abs() < 1e-9, "lo {lo}");
        assert!((hi - report.decode_tps * 1.1).abs() < 1e-9, "hi {hi}");
        let (prefill_lo, prefill_hi) = (
            report.prefill_ms_lo.expect("prefill lo"),
            report.prefill_ms_hi.expect("prefill hi"),
        );
        assert!((prefill_lo - 90.0).abs() < 1e-9, "prefill lo {prefill_lo}");
        assert!((prefill_hi - 110.0).abs() < 1e-9, "prefill hi {prefill_hi}");
    }

    #[test]
    fn with_intervals_without_prefill_only_bounds_decode() {
        let report = sample().with_intervals(10.0);
        assert!(report.decode_tps_lo.is_some());
        assert!(report.decode_tps_hi.is_some());
        assert_eq!(report.prefill_ms_lo, None);
        assert_eq!(report.prefill_ms_hi, None);
    }

    #[test]
    fn json_roundtrip_carries_the_interval_fields() {
        let report = sample()
            .with_prefill_ms(100.0)
            .with_intervals(13.636_36)
            .with_confidence(Confidence::measured(13.636_36, "test"));
        let json = serde_json::to_string(&report).expect("serialize EstimateReport");
        for key in [
            "decode_tps_lo",
            "decode_tps_hi",
            "prefill_ms_lo",
            "prefill_ms_hi",
            "confidence",
            "level",
            "method",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }

        let back: EstimateReport = serde_json::from_str(&json).expect("deserialize EstimateReport");
        assert_eq!(back.quant, report.quant);
        assert_eq!(back.decode_tps, report.decode_tps);
        assert!(back.decode_tps_lo.is_some());
        assert!(back.prefill_ms_lo.is_some());
        assert_eq!(back.confidence.expect("confidence").level, "medium");
    }

    #[test]
    fn multi_report_carries_the_configs_array() {
        let multi = MultiEstimateReport::new(
            "135M",
            vec![
                sample().with_intervals(5.0),
                EstimateReport::new("135M", Quant::Q8_0, 114.1).with_intervals(5.0),
            ],
        );
        let json = serde_json::to_string(&multi).expect("serialize MultiEstimateReport");
        assert!(json.contains("\"configs\""), "missing configs: {json}");
        assert!(
            json.contains("\"schema\":\"attune-estimate/1\""),
            "schema value pinned: {json}"
        );

        let back: MultiEstimateReport =
            serde_json::from_str(&json).expect("deserialize MultiEstimateReport");
        assert_eq!(back.schema, SCHEMA);
        assert_eq!(back.model, multi.model);
        assert_eq!(back.configs.len(), multi.configs.len());
        // Float intervals can differ in the last bit after a JSON roundtrip,
        // so compare the structural fields rather than the whole document.
        for (back, original) in back.configs.iter().zip(multi.configs.iter()) {
            assert_eq!(back.quant, original.quant);
            assert_eq!(back.decode_tps, original.decode_tps);
            assert!(back.decode_tps_lo.is_some());
            assert!(back.decode_tps_hi.is_some());
        }
    }

    #[test]
    fn bottleneck_display_is_human_readable() {
        assert_eq!(Bottleneck::MemoryBandwidth.to_string(), "memory bandwidth");
        assert_eq!(Bottleneck::Compute.to_string(), "compute");
    }

    #[test]
    fn bottleneck_json_names_are_pinned() {
        assert_eq!(
            serde_json::to_string(&Bottleneck::MemoryBandwidth).expect("serialize"),
            "\"memory_bandwidth\""
        );
        assert_eq!(
            serde_json::to_string(&Bottleneck::Compute).expect("serialize"),
            "\"compute\""
        );
    }
}
