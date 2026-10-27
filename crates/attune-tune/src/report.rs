//! The `attune tune` JSON contract.

use attune_estimate::HwInputs;
use serde::{Deserialize, Serialize};

use crate::sweep::RankedConfig;
use crate::{Objective, SCHEMA};

/// One `attune tune` result: the ranked configs plus the inputs they were
/// scored from, echoed so a consumer can reproduce the run without
/// re-detecting hardware.
///
/// `serde(default)` is on every field except the schema stamp so later slices
/// can add/omit fields without breaking older consumers — the same convention
/// `attune-hal`, `attune-bench` and `attune-estimate` use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneReport {
    /// Always [`SCHEMA`].
    pub schema: String,
    /// Model label as resolved by the CLI (catalog name, path or `--params`).
    #[serde(default)]
    pub model: String,
    /// Parameter count used for the weight footprint.
    #[serde(default)]
    pub params: u64,
    /// Objective the configs were ranked for.
    #[serde(default)]
    pub objective: Objective,
    /// Memory bandwidth in decimal GB/s.
    #[serde(default)]
    pub bw_gbps: f64,
    /// FP32 peak in decimal GFLOPS.
    #[serde(default)]
    pub gflops: f64,
    /// Available RAM in bytes, when the HAL could read it. Absent means the
    /// fit check was skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avail_bytes: Option<f64>,
    /// Ranked configs, best score first.
    #[serde(default)]
    pub configs: Vec<RankedConfig>,
    /// Non-fatal notices from model/hardware resolution. Empty in the common
    /// case and omitted from JSON then.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl TuneReport {
    /// Build a report around an already-ranked config list.
    pub fn new(
        model: impl Into<String>,
        params: u64,
        objective: Objective,
        hw: HwInputs,
        avail_bytes: Option<f64>,
        configs: Vec<RankedConfig>,
    ) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            model: model.into(),
            params,
            objective,
            bw_gbps: hw.bw_gbps,
            gflops: hw.flops_gflops,
            avail_bytes,
            configs,
            warnings: Vec::new(),
        }
    }

    /// Attach non-fatal notices (model resolution, clock fallback).
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TuneConfig;
    use attune_estimate::Quant;

    fn ranked(score: f64) -> RankedConfig {
        RankedConfig {
            config: TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16),
            decode_tps: 246.913_58,
            ttft_ms: 2_764.8,
            fits: true,
            score,
        }
    }

    fn sample() -> TuneReport {
        TuneReport::new(
            "135M",
            135_000_000,
            Objective::Throughput,
            HwInputs {
                bw_gbps: 20.0,
                flops_gflops: 50.0,
            },
            Some(16e9),
            vec![ranked(246.913_58), ranked(114.1)],
        )
    }

    #[test]
    fn new_stamps_the_schema_and_echoes_inputs() {
        let report = sample();
        assert_eq!(report.schema, SCHEMA);
        assert_eq!(report.params, 135_000_000);
        assert_eq!(report.objective, Objective::Throughput);
        assert_eq!(report.bw_gbps, 20.0);
        assert_eq!(report.gflops, 50.0);
        assert_eq!(report.avail_bytes, Some(16e9));
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn json_roundtrip_preserves_the_document() {
        let report = sample();
        let json = serde_json::to_string(&report).expect("serialize TuneReport");
        for key in [
            "schema",
            "model",
            "params",
            "objective",
            "bw_gbps",
            "gflops",
            "avail_bytes",
            "configs",
            "score",
            "decode_tps",
            "ttft_ms",
            "fits",
            "n_ubatch",
            "n_gpu_layers",
            "mmap",
            "flash_attn",
            "numa",
            "cache_type_k",
            "cache_type_v",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        assert!(
            json.contains("\"schema\":\"attune-tune/1\""),
            "schema value pinned: {json}"
        );
        assert!(
            json.contains("\"objective\":\"throughput\""),
            "objective value pinned: {json}"
        );

        let back: TuneReport = serde_json::from_str(&json).expect("deserialize TuneReport");
        assert_eq!(back.schema, report.schema);
        assert_eq!(back.model, report.model);
        assert_eq!(back.params, report.params);
        assert_eq!(back.objective, report.objective);
        assert_eq!(back.configs.len(), report.configs.len());
        for (back, original) in back.configs.iter().zip(report.configs.iter()) {
            assert_eq!(back.config, original.config);
            assert_eq!(back.fits, original.fits);
            // Float JSON roundtrips are compared with tolerance (AGENTS §6).
            assert!((back.decode_tps - original.decode_tps).abs() < 1e-9);
            assert!((back.ttft_ms - original.ttft_ms).abs() < 1e-9);
            assert!((back.score - original.score).abs() < 1e-9);
        }
    }

    #[test]
    fn oom_score_serializes_as_null_and_reads_back_as_negative_infinity() {
        let report = TuneReport::new(
            "70B",
            70_000_000_000,
            Objective::Throughput,
            HwInputs {
                bw_gbps: 20.0,
                flops_gflops: 50.0,
            },
            Some(16e9),
            vec![ranked(f64::NEG_INFINITY)],
        );
        let json = serde_json::to_string(&report).expect("serialize non-finite score");
        assert!(
            json.contains("\"score\":null"),
            "non-finite score must be null: {json}"
        );

        let back: TuneReport = serde_json::from_str(&json).expect("deserialize");
        assert!(back.configs[0].score.is_infinite());
        assert!(back.configs[0].score < 0.0);
    }

    #[test]
    fn empty_warnings_are_omitted() {
        let json = serde_json::to_string(&sample()).expect("serialize");
        assert!(
            !json.contains("\"warnings\""),
            "empty warnings must be omitted: {json}"
        );
    }
}
