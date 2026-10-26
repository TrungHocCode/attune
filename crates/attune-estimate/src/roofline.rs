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
    ///
    /// Weights-only: prefer [`HwInputs::predict_decode_tps_with_kv`] when the
    /// architecture and context size are known.
    pub fn predict_decode_tps(
        &self,
        params: u64,
        bytes_per_param: f64,
    ) -> Result<f64, EstimateError> {
        predict_decode_tps(params, bytes_per_param, self.bw_gbps)
    }

    /// Convenience wrapper around [`predict_decode_tps_with_kv`] that uses
    /// `self.bw_gbps`.
    pub fn predict_decode_tps_with_kv(
        &self,
        params: u64,
        bytes_per_param: f64,
        kv_per_token: f64,
        ctx: u32,
    ) -> Result<f64, EstimateError> {
        predict_decode_tps_with_kv(params, bytes_per_param, self.bw_gbps, kv_per_token, ctx)
    }
}

/// Predict decode throughput in tokens/second, weights only:
///
/// ```text
/// tok/s = bandwidth_bytes_per_second / (params × bytes_per_param)
/// ```
///
/// This is the pre-KV-cache approximation. It over-predicts as the context
/// grows because every generated token also streams the KV cache; use
/// [`predict_decode_tps_with_kv`] when the architecture is known. All inputs
/// are validated: zero, negative, NaN or infinite values return an error
/// instead of leaking `NaN` / `inf` into a report.
pub fn predict_decode_tps(
    params: u64,
    bytes_per_param: f64,
    bw_gbps: f64,
) -> Result<f64, EstimateError> {
    predict_decode_tps_with_kv(params, bytes_per_param, bw_gbps, 0.0, 0)
}

/// Predict decode throughput in tokens/second including the KV-cache read:
///
/// ```text
/// tok/s = bandwidth_bytes_per_second / (weights + kv_per_token × ctx)
/// ```
///
/// `kv_per_token` is the per-token KV traffic in bytes (see
/// [`crate::kv_bytes_per_token`]); pass `0.0` to reproduce the weights-only
/// prediction. A zero `ctx` therefore also drops the KV term. All inputs are
/// validated: zero parameters, a non-positive weight size, a negative or
/// non-finite `kv_per_token`, or a bad bandwidth return an error instead of
/// leaking `NaN` / `inf` into a report.
pub fn predict_decode_tps_with_kv(
    params: u64,
    bytes_per_param: f64,
    bw_gbps: f64,
    kv_per_token: f64,
    ctx: u32,
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
    if !kv_per_token.is_finite() || kv_per_token < 0.0 {
        return Err(EstimateError::BadHw(format!(
            "KV bytes per token must be finite and >= 0, got {kv_per_token}"
        )));
    }

    let weight_bytes = params as f64 * bytes_per_param;
    if !weight_bytes.is_finite() || weight_bytes <= 0.0 {
        return Err(EstimateError::InvalidModel(
            "model weight footprint overflowed".to_string(),
        ));
    }

    let kv_bytes = kv_per_token * ctx as f64;
    let total_bytes = weight_bytes + kv_bytes;
    if !total_bytes.is_finite() || total_bytes <= 0.0 {
        return Err(EstimateError::InvalidModel(
            "decode footprint overflowed".to_string(),
        ));
    }

    let tps = bw_gbps * 1e9 / total_bytes;
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
    use crate::model::{kv_bytes_per_token, Arch, Quant};

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

        let with_kv = hw
            .predict_decode_tps_with_kv(135_000_000, 0.6, 1_000.0, 2_048)
            .expect("valid");
        let with_kv_fn =
            predict_decode_tps_with_kv(135_000_000, 0.6, 20.0, 1_000.0, 2_048).expect("valid");
        assert_eq!(with_kv, with_kv_fn);
    }

    #[test]
    fn zero_kv_matches_the_weights_only_prediction() {
        for (params, bpp, bw) in [(135_000_000, 0.6118, 29.1), (7_000_000_000, 2.0, 50.0)] {
            let old = predict_decode_tps(params, bpp, bw).expect("weights only");
            let new = predict_decode_tps_with_kv(params, bpp, bw, 0.0, 2048).expect("zero kv");
            assert_eq!(old, new);
        }
    }

    #[test]
    fn kv_term_follows_the_documented_formula_exactly() {
        // Synthetic 30-layer / 5-KV-head / head_dim-64 shape: 38_400 B/token.
        let arch = Arch {
            name: "test-135m-kv",
            n_layer: 30,
            n_kv_heads: 5,
            head_dim: 64,
        };
        let kv_per_token = kv_bytes_per_token(&arch);
        assert_eq!(kv_per_token, 38_400.0);

        let params = 135_000_000_u64;
        let bpp = Quant::Q4_K_M.bytes_per_param();
        let bw = 29.1;
        let ctx = 2048_u32;

        let got =
            predict_decode_tps_with_kv(params, bpp, bw, kv_per_token, ctx).expect("valid inputs");
        let expected = bw * 1e9 / (params as f64 * bpp + kv_per_token * ctx as f64);
        assert!((got - expected).abs() < 1e-9, "got {got}, want {expected}");
        // Sanity range for the canned acceptance fixture (~180.5 tok/s).
        assert!((got - 180.48).abs() < 0.5, "got {got}");
    }

    #[test]
    fn kv_term_lowers_the_prediction_and_scales_with_context() {
        let params = 135_000_000_u64;
        let bpp = 0.6118;
        let bw = 29.1;
        let kv = 38_400.0;

        let none = predict_decode_tps_with_kv(params, bpp, bw, kv, 0).expect("ctx 0");
        let small = predict_decode_tps_with_kv(params, bpp, bw, kv, 2_048).expect("ctx 2048");
        let large = predict_decode_tps_with_kv(params, bpp, bw, kv, 8_192).expect("ctx 8192");

        assert!(small < none, "{small} should be below {none}");
        assert!(large < small, "{large} should be below {small}");
    }

    #[test]
    fn rejects_bad_kv_per_token() {
        for kv in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                predict_decode_tps_with_kv(135_000_000, 0.6, 20.0, kv, 2048).is_err(),
                "kv {kv} should be rejected"
            );
        }
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
