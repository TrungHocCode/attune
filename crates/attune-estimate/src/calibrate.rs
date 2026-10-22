//! Calibration against a real end-to-end bench (M3 slice 3).
//!
//! The roofline models ideal streaming and ideal compute. Real inference also
//! touches the KV cache, runs dequantisation kernels, pays attention
//! overhead, and rarely reaches the quoted FP32 peak. Rather than model every
//! term, slice 3 learns two scalar correction factors from one measured run:
//!
//! * `eta_mem = measured_tok_s / roofline_tok_s` — decode throughput is scaled
//!   by this factor.
//! * `eta_comp = roofline_prefill_ms / measured_ttft_ms` — prefill *latency*
//!   scales inversely, so the calibrated time is `raw_ms / eta_comp`.
//!
//! Both factors are clamped to [`MIN_ETA`]..=[`MAX_ETA`], so a bogus bench (or
//! a wildly different workload) cannot turn a report into nonsense. The
//! roofline's decode bytes are deliberately weights-only: the KV cache grows
//! with context and is not modelled yet; `eta_mem` absorbs it (plus every
//! other per-token overhead) for this slice.
//!
//! Everything here is pure: no I/O, no clock, no allocation beyond the error
//! strings. The CLI reads the bench JSON and feeds the numbers in.

use crate::EstimateError;

/// Lower clamp for any fitted eta. A measured/predicted ratio below this is
/// more likely a broken measurement than a 20× error in the model.
pub const MIN_ETA: f64 = 0.05;
/// Upper clamp for any fitted eta. Symmetric guard for the other direction.
pub const MAX_ETA: f64 = 1.5;

/// Correction factors learned from a measured end-to-end run.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibratedInputs {
    /// Decode throughput correction: `measured_tok_s / roofline_tok_s`.
    pub eta_mem: f64,
    /// Prefill latency correction: `roofline_ms / measured_ttft_ms`. Applied
    /// as a divisor, because lower latency means a larger factor.
    pub eta_comp: f64,
}

impl CalibratedInputs {
    /// Neutral factors: predictions pass through unchanged.
    pub const NEUTRAL: Self = Self {
        eta_mem: 1.0,
        eta_comp: 1.0,
    };

    /// Fit both factors from the roofline and the measured values.
    ///
    /// Decode uses the throughput ratio directly. Prefill latency is inverted,
    /// so its factor is `predicted_ms / measured_ms`. Passing `None` for
    /// either TTFT side leaves `eta_comp` at the neutral `1.0` — a report can
    /// still be calibrated on decode when the bench did not time prefill.
    pub fn fit(
        roofline_decode_tps: f64,
        measured_decode_tps: f64,
        roofline_prefill_ms: Option<f64>,
        measured_ttft_ms: Option<f64>,
    ) -> Result<Self, EstimateError> {
        let eta_mem = fit_eta(roofline_decode_tps, measured_decode_tps)?;

        let eta_comp = match (roofline_prefill_ms, measured_ttft_ms) {
            (Some(predicted), Some(measured))
                if predicted.is_finite()
                    && predicted > 0.0
                    && measured.is_finite()
                    && measured > 0.0 =>
            {
                // Time is inverted: eta = predicted_ms / measured_ms, so
                // feeding `fit_eta` the measured value as "predicted" yields
                // the divisor `apply_prefill` expects.
                fit_eta(measured, predicted)?
            }
            _ => 1.0,
        };

        Ok(Self { eta_mem, eta_comp })
    }
}

/// Fit a throughput-style ratio `measured / predicted`, clamped to
/// [`MIN_ETA`]..=[`MAX_ETA`].
///
/// Returns [`EstimateError::InvalidCalibration`] when `predicted` is not a
/// finite positive number or `measured` is not finite. A finite non-positive
/// measurement is clamped to [`MIN_ETA`] rather than rejected, so a stalled
/// bench still produces a bounded (useless but safe) factor.
pub fn fit_eta(predicted: f64, measured: f64) -> Result<f64, EstimateError> {
    if !predicted.is_finite() || predicted <= 0.0 {
        return Err(EstimateError::InvalidCalibration(format!(
            "predicted value must be finite and > 0, got {predicted}"
        )));
    }
    if !measured.is_finite() {
        return Err(EstimateError::InvalidCalibration(format!(
            "measured value must be finite, got {measured}"
        )));
    }

    Ok((measured / predicted).clamp(MIN_ETA, MAX_ETA))
}

/// Scale a decode throughput prediction by `eta_mem`.
///
/// Returns `0.0` for non-finite inputs instead of leaking `NaN` into a report.
pub fn apply_decode(raw_tps: f64, eta_mem: f64) -> f64 {
    if raw_tps.is_finite() && eta_mem.is_finite() {
        raw_tps * eta_mem
    } else {
        0.0
    }
}

/// Scale a prefill latency prediction by `eta_comp`.
///
/// Prefill *time* scales inversely with the correction factor:
/// `calibrated = raw / eta`. Returns `0.0` for non-finite inputs or a
/// non-positive factor.
pub fn apply_prefill(raw_ms: f64, eta_comp: f64) -> f64 {
    if raw_ms.is_finite() && eta_comp.is_finite() && eta_comp > 0.0 {
        raw_ms / eta_comp
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative error in percent, the shape the M3 acceptance gate uses.
    fn rel_err_pct(predicted: f64, measured: f64) -> f64 {
        (predicted - measured).abs() / measured * 100.0
    }

    #[test]
    fn canned_fixture_fits_the_documented_eta() {
        // The slice-3 fixture: 20 GB/s × 135M at the old 0.6 B/param
        // approximation predicts 246.9 tok/s; the measured e2e run gives
        // 55.6 tok/s, so eta = 55.6 / 246.9 ≈ 0.2252.
        let eta = fit_eta(246.9, 55.6).expect("valid fixture");
        assert!((eta - 0.225).abs() < 0.001, "got {eta}");
    }

    #[test]
    fn calibrated_prediction_is_exact_on_the_fixture() {
        let predicted = 246.9;
        let measured = 55.6;
        let eta = fit_eta(predicted, measured).expect("valid fixture");
        let calibrated = apply_decode(predicted, eta);
        // Calibrated = predicted × (measured / predicted) = measured, so the
        // residual is pure floating-point noise.
        assert!(rel_err_pct(calibrated, measured) < 1e-9, "got {calibrated}");
    }

    #[test]
    fn uncalibrated_vs_measured_uses_the_acceptance_bound_pattern() {
        // A roofline that lands within the M3 uncalibrated acceptance bound
        // (<25%) needs no rescue...
        let close = rel_err_pct(60.0, 55.6);
        assert!(close < 25.0, "close prediction error {close}%");

        // ...while the canned fixture is intentionally outside it, which is
        // exactly why slice 3 adds calibration.
        let stale = rel_err_pct(246.9, 55.6);
        assert!(stale > 25.0, "fixture should need calibration: {stale}%");
    }

    #[test]
    fn fit_eta_clamps_both_ends() {
        // Measured far below the model → floor.
        assert_eq!(fit_eta(1000.0, 1.0).expect("valid"), MIN_ETA);
        // Measured far above the model → ceiling.
        assert_eq!(fit_eta(1.0, 1000.0).expect("valid"), MAX_ETA);
        // A finite zero measurement clamps rather than errors.
        assert_eq!(fit_eta(100.0, 0.0).expect("valid"), MIN_ETA);
    }

    #[test]
    fn fit_eta_rejects_bad_predictions() {
        for predicted in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    fit_eta(predicted, 55.6),
                    Err(EstimateError::InvalidCalibration(_))
                ),
                "predicted {predicted} should be rejected"
            );
        }
    }

    #[test]
    fn fit_eta_rejects_non_finite_measurements() {
        for measured in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                fit_eta(246.9, measured).is_err(),
                "measured {measured} should be rejected"
            );
        }
    }

    #[test]
    fn fit_combines_decode_and_inverted_prefill() {
        // Decode: 50 / 100 = 0.5.
        // Prefill: predicted 200 ms / measured 400 ms = 0.5.
        let inputs = CalibratedInputs::fit(100.0, 50.0, Some(200.0), Some(400.0))
            .expect("valid calibration");
        assert!((inputs.eta_mem - 0.5).abs() < 1e-12);
        assert!((inputs.eta_comp - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fit_uses_neutral_eta_comp_when_ttft_is_missing() {
        let inputs =
            CalibratedInputs::fit(100.0, 50.0, Some(200.0), None).expect("valid calibration");
        assert_eq!(inputs.eta_mem, 0.5);
        assert_eq!(inputs.eta_comp, CalibratedInputs::NEUTRAL.eta_comp);

        let zero_ttft =
            CalibratedInputs::fit(100.0, 50.0, Some(200.0), Some(0.0)).expect("valid calibration");
        assert_eq!(zero_ttft.eta_comp, 1.0);
    }

    #[test]
    fn apply_decode_is_a_plain_scale() {
        assert!((apply_decode(246.9, 0.225) - 55.5525).abs() < 1e-9);
        assert_eq!(apply_decode(f64::NAN, 1.0), 0.0);
        assert_eq!(apply_decode(1.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn apply_prefill_divides_by_eta() {
        // Half the latency means the factor is 2 and the prediction shrinks.
        assert!((apply_prefill(200.0, 2.0) - 100.0).abs() < 1e-12);
        assert!((apply_prefill(100.0, 0.5) - 200.0).abs() < 1e-12);
        assert_eq!(apply_prefill(100.0, 0.0), 0.0);
        assert_eq!(apply_prefill(f64::NAN, 1.0), 0.0);
    }
}
