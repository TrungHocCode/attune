//! The `attune estimate` JSON contract.

use serde::{Deserialize, Serialize};

use crate::model::Quant;
use crate::SCHEMA;

/// What is limiting throughput. Slice 1 only ever emits
/// [`Bottleneck::MemoryBandwidth`]; slice 2 computes the roofline ridge
/// (`AI* = FLOPS_peak / BW`) and can return [`Bottleneck::Compute`] once
/// prefill is modelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    #[default]
    MemoryBandwidth,
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bottleneck::MemoryBandwidth => f.write_str("memory bandwidth"),
        }
    }
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
    /// Dominant limiter for the current prediction.
    #[serde(default)]
    pub bottleneck: Bottleneck,
}

impl EstimateReport {
    /// Build a report for the slice-1 decode prediction. The bottleneck is
    /// memory bandwidth by construction — slice 2 replaces this with the
    /// real ridge comparison.
    pub fn new(model: impl Into<String>, quant: Quant, decode_tps: f64) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            model: model.into(),
            quant,
            decode_tps,
            bottleneck: Bottleneck::MemoryBandwidth,
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
        assert_eq!(report.schema, "attune-estimate/1");
        assert_eq!(report.bottleneck, Bottleneck::MemoryBandwidth);
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
    fn partial_json_uses_the_future_facing_defaults() {
        let old_json = r#"{"schema": "attune-estimate/1"}"#;
        let back: EstimateReport = serde_json::from_str(old_json).expect("deserialize partial");
        assert_eq!(back.schema, SCHEMA);
        assert!(back.model.is_empty());
        assert_eq!(back.quant, Quant::Fp16);
        assert_eq!(back.decode_tps, 0.0);
        assert_eq!(back.bottleneck, Bottleneck::MemoryBandwidth);
    }

    #[test]
    fn bottleneck_display_is_human_readable() {
        assert_eq!(Bottleneck::MemoryBandwidth.to_string(), "memory bandwidth");
    }
}
