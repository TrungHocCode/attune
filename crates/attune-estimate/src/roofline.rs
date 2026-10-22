//! Roofline math (M3 slice 1: decode only).
//!
//! Decode is memory-bound: every generated token streams the weight matrix
//! once, so `tok/s ≈ bandwidth / bytes-per-weight`. The compute side
//! (prefill) and the ridge comparison live in [`crate::prefill`] and
//! [`crate::bottleneck`].

use crate::EstimateError;

/// Hardware measurements a roofline needs. `flops_gflops` is unused by the
/// decode prediction in slice 1 but carried here so slice 2's ridge
/// (`AI* = FLOPS / BW`) has one stable input type.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HwInputs {
    /// Measured (or overridden) memory bandwidth in decimal GB/s.
    pub bw_gbps: f64,
    /// Theoretical (or overridden) FP32 peak in decimal GFLOPS.
    pub flops_gflops: f64,
}

impl HwInputs {
    /// Convenience wrapper around [`predict_decode_tps`] that uses
    /// `self.bw_gbps`.
    pub fn predict_decode_tps(
        &self,
        params: u64,
        bytes_per_param: f64,
    ) -> Result<f64, EstimateError> {
        predict_decode_tps(params, bytes_per_param, self.bw_gbps)
    }
}

/// Predict decode throughput in tokens/second:
///
/// ```text
/// tok/s = bandwidth_bytes_per_second / (params × bytes_per_param)
/// ```
///
/// All inputs are validated: zero, negative, NaN or infinite values return
/// an error instead of leaking `NaN` / `inf` into a report. The result is
/// likewise checked before it is returned.
pub fn predict_decode_tps(
    params: u64,
    bytes_per_param: f64,
    bw_gbps: f64,
) -> Result<f64, EstimateError> {
    if params == 0 {
        return Err(EstimateError::InvalidModel(
            "parameter count must be > 0".to_string(),
        ));
    }
    if !bytes_per_param.is_finite() || bytes_per_param <= 0.0 {
        return Err(EstimateError::InvalidQuant(format!(
            "bytes per parameter must be finite and > 0, got {bytes_per_param}"
        )));
    }
    if !bw_gbps.is_finite() || bw_gbps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "bandwidth must be finite and > 0 GB/s, got {bw_gbps}"
        )));
    }

    let weight_bytes = params as f64 * bytes_per_param;
    if !weight_bytes.is_finite() || weight_bytes <= 0.0 {
        return Err(EstimateError::InvalidModel(
            "model weight footprint overflowed".to_string(),
        ));
    }

    let tps = bw_gbps * 1e9 / weight_bytes;
    if !tps.is_finite() || tps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "prediction is not a finite positive number, got {tps}"
        )));
    }
    Ok(tps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canned_q4_k_m_135m_on_20_gbps() {
        // 20e9 / (135e6 × 0.6) = 246.91358 tok/s.
        let tps = predict_decode_tps(135_000_000, 0.6, 20.0).expect("valid inputs");
        assert!((tps - 246.913_58).abs() < 0.1, "got {tps}");
    }

    #[test]
    fn canned_fp16_7b_on_20_gbps() {
        // 20e9 / (7e9 × 2) = 1.42857 tok/s.
        let tps = predict_decode_tps(7_000_000_000, 2.0, 20.0).expect("valid inputs");
        assert!((tps - 1.428_571).abs() < 0.01, "got {tps}");
    }

    #[test]
    fn hw_inputs_method_uses_bandwidth() {
        let hw = HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 50.0,
        };
        let via_method = hw.predict_decode_tps(135_000_000, 0.6).expect("valid");
        let via_fn = predict_decode_tps(135_000_000, 0.6, 20.0).expect("valid");
        assert_eq!(via_method, via_fn);
    }

    #[test]
    fn rejects_zero_parameters() {
        assert!(matches!(
            predict_decode_tps(0, 0.6, 20.0),
            Err(EstimateError::InvalidModel(_))
        ));
    }

    #[test]
    fn rejects_bad_bytes_per_param() {
        for bpp in [0.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                predict_decode_tps(135_000_000, bpp, 20.0).is_err(),
                "bpp {bpp} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_bad_bandwidth() {
        for bw in [0.0, -20.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                predict_decode_tps(135_000_000, 0.6, bw).is_err(),
                "bandwidth {bw} should be rejected"
            );
        }
    }

    #[test]
    fn never_returns_non_finite_or_non_positive() {
        let cases = [
            (1_u64, 0.0, 20.0),
            (1, 1e-300, 20.0),
            (u64::MAX, 2.0, 1e-300),
            (u64::MAX, 2.0, f64::MAX),
        ];
        for (params, bpp, bw) in cases {
            if let Ok(tps) = predict_decode_tps(params, bpp, bw) {
                assert!(tps.is_finite() && tps > 0.0, "got {tps}");
            }
        }
    }
}
