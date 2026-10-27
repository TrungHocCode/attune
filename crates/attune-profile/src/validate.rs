//! Tolerant profile validation (M4 slice 3).
//!
//! [`validate`] never returns `Err`: a profile is still useful on a related
//! machine or after the user frees RAM, so mismatches come back as
//! [`ProfileWarning`]s. The caller decides whether to warn, apply a fallback or
//! refuse to launch.

use serde::{Deserialize, Serialize};

use crate::Profile;

/// What a [`ProfileWarning`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningKind {
    /// The profile was tuned on different hardware.
    FingerprintMismatch,
    /// The expected RAM peak does not fit in the reported available RAM.
    RamInsufficient,
}

/// How strongly the caller should react. Validation itself never fails, so
/// [`WarningSeverity::Error`] means "do not launch without a fallback".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningSeverity {
    Warning,
    Error,
}

/// One non-fatal validation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileWarning {
    pub severity: WarningSeverity,
    pub kind: WarningKind,
    pub message: String,
}

/// Check a profile against the current machine.
///
/// * A non-empty fingerprint that differs from `current_fingerprint` warns and
///   points at [`crate::Fallbacks::if_ram_insufficient`].
/// * An available RAM below `expected_performance.ram_peak_gb` emits an
///   error-severity warning suggesting a lower `n_ctx` (the fallback, when
///   present) or a smaller quant.
///
/// An empty fingerprint or a zero `ram_avail_bytes` means "unknown", so that
/// check is skipped rather than producing a false alarm.
pub fn validate(
    profile: &Profile,
    current_fingerprint: &str,
    ram_avail_bytes: u64,
) -> Vec<ProfileWarning> {
    let mut warnings = Vec::new();

    let stored_fingerprint = profile.hardware_fingerprint.trim();
    let current_fingerprint = current_fingerprint.trim();
    if !stored_fingerprint.is_empty()
        && !current_fingerprint.is_empty()
        && stored_fingerprint != current_fingerprint
    {
        let fallback = profile
            .fallbacks
            .if_ram_insufficient
            .as_ref()
            .map(|fallback| format!("fallbacks.if_ram_insufficient.n_ctx={}", fallback.n_ctx))
            .unwrap_or_else(|| "no RAM fallback recorded".to_string());
        warnings.push(ProfileWarning {
            severity: WarningSeverity::Warning,
            kind: WarningKind::FingerprintMismatch,
            message: format!(
                "hardware fingerprint mismatch (profile '{stored_fingerprint}', current \
                 '{current_fingerprint}'); tune again for this machine or apply {fallback}"
            ),
        });
    }

    let peak_gb = profile.expected_performance.ram_peak_gb;
    if ram_avail_bytes > 0 && peak_gb.is_finite() && peak_gb > 0.0 {
        let peak_bytes = peak_gb * 1e9;
        if (ram_avail_bytes as f64) < peak_bytes {
            let fallback = profile
                .fallbacks
                .if_ram_insufficient
                .as_ref()
                .map(|fallback| format!("n_ctx={}", fallback.n_ctx))
                .unwrap_or_else(|| "a lower n_ctx".to_string());
            warnings.push(ProfileWarning {
                severity: WarningSeverity::Error,
                kind: WarningKind::RamInsufficient,
                message: format!(
                    "available RAM {:.1} GB is below the expected peak {:.1} GB; try {fallback} \
                     or a smaller quant",
                    ram_avail_bytes as f64 / 1e9,
                    peak_gb
                ),
            });
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ExpectedPerformance, Fallbacks, Profile, RamFallback};

    fn profile(fingerprint: &str, peak_gb: f64, fallback_ctx: Option<u32>) -> Profile {
        Profile {
            hardware_fingerprint: fingerprint.to_string(),
            expected_performance: ExpectedPerformance {
                ram_peak_gb: peak_gb,
                ..ExpectedPerformance::default()
            },
            fallbacks: Fallbacks {
                if_ram_insufficient: fallback_ctx.map(|n_ctx| Box::new(RamFallback { n_ctx })),
            },
            ..Profile::default()
        }
    }

    #[test]
    fn matching_fingerprint_and_enough_ram_is_clean() {
        let profile = profile("cpu|8|avx2|16000000000", 4.0, Some(2_048));
        let warnings = validate(&profile, "cpu|8|avx2|16000000000", 16_000_000_000);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }

    #[test]
    fn fingerprint_mismatch_warns_and_points_at_the_fallback() {
        let profile = profile("old-cpu|4|sse|8000000000", 4.0, Some(1_024));
        let warnings = validate(&profile, "new-cpu|8|avx2|16000000000", 16_000_000_000);

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::FingerprintMismatch);
        assert_eq!(warnings[0].severity, WarningSeverity::Warning);
        assert!(
            warnings[0].message.contains("n_ctx=1024"),
            "must point at the fallback: {}",
            warnings[0].message
        );
    }

    #[test]
    fn low_ram_warns_without_failing_and_names_both_knobs() {
        let profile = profile("cpu|8|avx2|16000000000", 8.0, Some(1_024));
        let warnings = validate(&profile, "cpu|8|avx2|16000000000", 2_000_000_000);

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::RamInsufficient);
        assert_eq!(warnings[0].severity, WarningSeverity::Error);
        assert!(
            warnings[0].message.contains("n_ctx=1024") && warnings[0].message.contains("quant"),
            "must name both knobs: {}",
            warnings[0].message
        );
    }

    #[test]
    fn unknown_ram_or_fingerprint_is_skipped() {
        let profile = profile("cpu|8|avx2|16000000000", 8.0, None);
        assert!(validate(&profile, "", 16_000_000_000).is_empty());
        assert!(validate(&profile, "cpu|8|avx2|16000000000", 0).is_empty());
    }

    #[test]
    fn fingerprint_mismatch_without_a_fallback_still_warns() {
        let profile = profile("old-cpu|4|sse|8000000000", 0.0, None);
        let warnings = validate(&profile, "new-cpu|8|avx2|16000000000", 16_000_000_000);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("no RAM fallback recorded"));
    }
}
