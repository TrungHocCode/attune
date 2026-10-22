//! Roofline ridge and bottleneck classification (M3 slice 2).
//!
//! The ridge point is the arithmetic intensity where a workload stops being
//! memory-bound and starts being compute-bound:
//!
//! ```text
//! AI* = FLOPS_peak / Bandwidth   (FLOP per byte)
//! ```
//!
//! A workload's own arithmetic intensity is compared against that ridge; the
//! side it falls on determines the [`Bottleneck`].

use crate::{Bottleneck, EstimateError};

/// Compute the roofline ridge point `AI* = FLOPS / BW` in FLOP per byte.
///
/// `flops_gflops` and `bw_gbps` are both decimal (GFLOPS, GB/s), so the
/// `1e9` factors cancel: `GFLOPS / GB/s` is already FLOP/byte.
pub fn ridge_ai(flops_gflops: f64, bw_gbps: f64) -> Result<f64, EstimateError> {
    if !flops_gflops.is_finite() || flops_gflops <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "FP32 peak must be finite and > 0 GFLOPS, got {flops_gflops}"
        )));
    }
    if !bw_gbps.is_finite() || bw_gbps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "bandwidth must be finite and > 0 GB/s, got {bw_gbps}"
        )));
    }

    let ai = flops_gflops / bw_gbps;
    if !ai.is_finite() || ai <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "ridge point is not a finite positive number, got {ai}"
        )));
    }
    Ok(ai)
}

/// Arithmetic intensity of decode: `1 / bytes_per_param` FLOP per byte.
///
/// Decode streams the weights once per token. This is the slice-2
/// approximation; slice 3 calibrates it against measured end-to-end runs.
pub fn decode_ai(bytes_per_param: f64) -> Result<f64, EstimateError> {
    if !bytes_per_param.is_finite() || bytes_per_param <= 0.0 {
        return Err(EstimateError::InvalidQuant(format!(
            "bytes per parameter must be finite and > 0, got {bytes_per_param}"
        )));
    }
    let ai = 1.0 / bytes_per_param;
    if !ai.is_finite() || ai <= 0.0 {
        return Err(EstimateError::InvalidQuant(format!(
            "decode arithmetic intensity is not finite, got {ai}"
        )));
    }
    Ok(ai)
}

/// Arithmetic intensity of prefill: `prompt_tokens` FLOP per byte.
///
/// This is the documented slice-2 approximation (every prompt token processes
/// the whole weight matrix); slice 3 refines it with calibration.
pub fn prefill_ai(prompt_tokens: u64) -> Result<f64, EstimateError> {
    if prompt_tokens == 0 {
        return Err(EstimateError::InvalidModel(
            "prefill token count must be > 0".to_string(),
        ));
    }
    Ok(prompt_tokens as f64)
}

/// Classify a workload against the ridge point.
///
/// Returns the limiting resource and how far the workload's arithmetic
/// intensity is from the ridge, in percent:
/// `margin_pct = |AI_workload − AI*| / AI* × 100`.
///
/// At the ridge (`AI_workload == AI*`) the workload is compute-bound by
/// convention: compute and memory are equally limiting, and the roofline
/// ceiling is reached either way. Callers are expected to pass an `ai_star`
/// produced by [`ridge_ai`]; a non-positive value is reported as zero margin
/// rather than leaking `NaN`.
pub fn classify(ai_workload: f64, ai_star: f64) -> (Bottleneck, f64) {
    let bottleneck = if ai_workload >= ai_star {
        Bottleneck::Compute
    } else {
        Bottleneck::MemoryBandwidth
    };

    let margin_pct = if ai_workload.is_finite() && ai_star.is_finite() && ai_star > 0.0 {
        (ai_workload - ai_star).abs() / ai_star * 100.0
    } else {
        0.0
    };

    (bottleneck, margin_pct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_is_flops_over_bandwidth() {
        let ai = ridge_ai(50.0, 20.0).expect("valid inputs");
        assert!((ai - 2.5).abs() < 1e-12, "got {ai}");
    }

    #[test]
    fn ridge_rejects_bad_hardware() {
        for (flops, bw) in [
            (0.0, 20.0),
            (-50.0, 20.0),
            (f64::NAN, 20.0),
            (f64::INFINITY, 20.0),
            (50.0, 0.0),
            (50.0, -20.0),
            (50.0, f64::NAN),
            (50.0, f64::NEG_INFINITY),
        ] {
            assert!(
                ridge_ai(flops, bw).is_err(),
                "{flops}/{bw} should be rejected"
            );
        }
    }

    #[test]
    fn decode_ai_is_inverse_bytes_per_param() {
        let ai = decode_ai(0.6).expect("valid");
        assert!((ai - 1.0 / 0.6).abs() < 1e-12, "got {ai}");
        assert!(decode_ai(0.0).is_err());
        assert!(decode_ai(-0.5).is_err());
        assert!(decode_ai(f64::NAN).is_err());
        assert!(decode_ai(f64::INFINITY).is_err());
    }

    #[test]
    fn prefill_ai_is_prompt_tokens() {
        assert_eq!(prefill_ai(512).expect("valid"), 512.0);
        assert!(prefill_ai(0).is_err());
    }

    #[test]
    fn classify_below_ridge_is_memory_bound() {
        // AI 1.667 vs ridge 2.5 => 33.33% below.
        let (bottleneck, margin) = classify(1.0 / 0.6, 2.5);
        assert_eq!(bottleneck, Bottleneck::MemoryBandwidth);
        assert!((margin - 33.333_333).abs() < 0.001, "got {margin}");
    }

    #[test]
    fn classify_above_ridge_is_compute_bound() {
        // AI 512 vs ridge 2.5 => 20380% above.
        let (bottleneck, margin) = classify(512.0, 2.5);
        assert_eq!(bottleneck, Bottleneck::Compute);
        assert!((margin - 20_380.0).abs() < 0.01, "got {margin}");
    }

    #[test]
    fn classify_at_the_ridge_is_compute_with_zero_margin() {
        let (bottleneck, margin) = classify(2.5, 2.5);
        assert_eq!(bottleneck, Bottleneck::Compute);
        assert_eq!(margin, 0.0);
    }

    #[test]
    fn classify_never_leaks_nan_for_invalid_ridge() {
        for ai_star in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (_, margin) = classify(1.0, ai_star);
            assert!(margin.is_finite(), "margin for ai_star {ai_star}: {margin}");
        }
    }
}
