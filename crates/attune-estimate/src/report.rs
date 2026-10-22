//! The `attune estimate` JSON contract.

use serde::{Deserialize, Serialize};

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
    /// Dominant limiter for the current (decode) prediction.
    #[serde(default)]
    pub bottleneck: Bottleneck,
    /// Predicted prefill latency / time-to-first-token in milliseconds, when
    /// slice-2 inputs were available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms_ttft: Option<f64>,
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
            bottleneck: Bottleneck::MemoryBandwidth,
            prefill_ms_ttft: None,
            ridge_ai: None,
            margin_pct: None,
            inputs: None,
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
        assert_eq!(report.prefill_ms_ttft, None);
        assert_eq!(report.ridge_ai, None);
        assert_eq!(report.margin_pct, None);
        assert_eq!(report.inputs, None);
    }

    #[test]
    fn builders_attach_the_slice_two_fields() {
        let report = sample()
            .with_inputs(EstimateInputs {
                bw_gbps: 20.0,
                gflops: 50.0,
            })
            .with_prefill_ms(2764.8)
            .with_ridge(2.5, 33.333_333)
            .with_bottleneck(Bottleneck::Compute);

        assert_eq!(report.bottleneck, Bottleneck::Compute);
        assert_eq!(report.prefill_ms_ttft, Some(2764.8));
        assert_eq!(report.ridge_ai, Some(2.5));
        assert_eq!(report.margin_pct, Some(33.333_333));
        assert_eq!(report.inputs.expect("inputs").bw_gbps, 20.0);
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
        assert_eq!(back.prefill_ms_ttft, None);
        assert_eq!(back.ridge_ai, None);
        assert_eq!(back.margin_pct, None);
        assert_eq!(back.inputs, None);
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
        assert_eq!(back.prefill_ms_ttft, None);
        assert_eq!(back.ridge_ai, None);
        assert_eq!(back.margin_pct, None);
        assert_eq!(back.inputs, None);
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
