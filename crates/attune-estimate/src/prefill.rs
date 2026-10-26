//! Prefill / time-to-first-token math (M3 slice 2).
//!
//! Prefill is compute-bound: it runs roughly two FLOPs per parameter per
//! prompt token, so
//!
//! ```text
//! time_ms = 2 × params × n_tokens / FLOPS_peak × 1000
//! ```
//!
//! `FLOPS_peak` is supplied in decimal GFLOPS. Inputs are validated so a bad
//! value returns [`EstimateError::BadHw`] / [`EstimateError::InvalidModel`]
//! instead of leaking `NaN` or `inf` into a report.

use crate::{EstimateError, Quant};

/// Predict prefill latency (TTFT) in milliseconds:
///
/// ```text
/// ms = 2 × params × n_tokens / (flops_gflops × 1e9) × 1000
/// ```
///
/// `params` and `n_tokens` must be non-zero and `flops_gflops` must be a
/// finite positive number; the computed latency is checked the same way.
pub fn predict_prefill_ms(
    params: u64,
    n_tokens: u64,
    flops_gflops: f64,
) -> Result<f64, EstimateError> {
    if params == 0 {
        return Err(EstimateError::InvalidModel(
            "parameter count must be > 0".to_string(),
        ));
    }
    if n_tokens == 0 {
        return Err(EstimateError::InvalidModel(
            "prefill token count must be > 0".to_string(),
        ));
    }
    if !flops_gflops.is_finite() || flops_gflops <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "FP32 peak must be finite and > 0 GFLOPS, got {flops_gflops}"
        )));
    }

    let flops = 2.0 * params as f64 * n_tokens as f64;
    if !flops.is_finite() || flops <= 0.0 {
        return Err(EstimateError::InvalidModel(
            "prefill FLOP count overflowed".to_string(),
        ));
    }

    let ms = flops / (flops_gflops * 1e9) * 1000.0;
    if !ms.is_finite() || ms <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "prefill prediction is not a finite positive number, got {ms}"
        )));
    }
    Ok(ms)
}

/// Predict prefill latency including the per-format dequant cost.
///
/// Same FLOP math as [`predict_prefill_ms`], scaled by
/// [`Quant::dequant_penalty`]: quantized weights must be unpacked during
/// the compute-bound prefill, so lower precision is slower here even
/// though it is faster at decode time.
pub fn predict_prefill_ms_quant(
    params: u64,
    n_tokens: u64,
    flops_gflops: f64,
    quant: Quant,
) -> Result<f64, EstimateError> {
    let ms = predict_prefill_ms(params, n_tokens, flops_gflops)?;
    let penalized = ms * quant.dequant_penalty();
    if !penalized.is_finite() || penalized <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "penalized prefill prediction is not finite positive, got {penalized}"
        )));
    }
    Ok(penalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canned_135m_512_tokens_on_50_gflops() {
        // 2 × 135e6 × 512 / 50e9 × 1000 = 2764.8 ms.
        let ms = predict_prefill_ms(135_000_000, 512, 50.0).expect("valid inputs");
        assert!((ms - 2764.8).abs() < 0.1, "got {ms}");
    }

    #[test]
    fn scales_linearly_with_tokens_and_inversely_with_flops() {
        let one = predict_prefill_ms(7_000_000_000, 1, 100.0).expect("valid");
        let many = predict_prefill_ms(7_000_000_000, 100, 100.0).expect("valid");
        let faster = predict_prefill_ms(7_000_000_000, 1, 200.0).expect("valid");

        assert!((many - one * 100.0).abs() < 1e-6, "got {many} vs {one}");
        assert!((faster - one / 2.0).abs() < 1e-9, "got {faster} vs {one}");
    }

    #[test]
    fn rejects_zero_parameters() {
        assert!(matches!(
            predict_prefill_ms(0, 512, 50.0),
            Err(EstimateError::InvalidModel(_))
        ));
    }

    #[test]
    fn rejects_zero_tokens() {
        assert!(matches!(
            predict_prefill_ms(135_000_000, 0, 50.0),
            Err(EstimateError::InvalidModel(_))
        ));
    }

    #[test]
    fn rejects_bad_flops() {
        for flops in [0.0, -50.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                predict_prefill_ms(135_000_000, 512, flops).is_err(),
                "flops {flops} should be rejected"
            );
        }
    }

    #[test]
    fn never_returns_non_finite_or_non_positive() {
        let cases = [
            (u64::MAX, u64::MAX, 1e-300),
            (u64::MAX, 1, 1.0),
            (1, u64::MAX, 1.0),
        ];
        for (params, tokens, flops) in cases {
            if let Ok(ms) = predict_prefill_ms(params, tokens, flops) {
                assert!(ms.is_finite() && ms > 0.0, "got {ms}");
            }
        }
    }

    #[test]
    fn quant_penalty_orders_fp16_above_q8_above_q4() {
        let fp16 = predict_prefill_ms_quant(135_000_000, 512, 50.0, Quant::Fp16).expect("valid");
        let q8 = predict_prefill_ms_quant(135_000_000, 512, 50.0, Quant::Q8_0).expect("valid");
        let q4 = predict_prefill_ms_quant(135_000_000, 512, 50.0, Quant::Q4_K_M).expect("valid");
        // 2764.8 ms base × penalty.
        assert!((fp16 - 2764.8).abs() < 0.1, "got {fp16}");
        assert!((q8 - 2764.8 * 1.15).abs() < 0.5, "got {q8}");
        assert!((q4 - 2764.8 * 1.3).abs() < 0.5, "got {q4}");
        assert!(
            fp16 < q8 && q8 < q4,
            "ranking must hold: {fp16} < {q8} < {q4}"
        );
        assert_eq!(Quant::Q4_0.dequant_penalty(), 1.3);
    }
}
