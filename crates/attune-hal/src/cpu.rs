//! CPU detection (M1: real CPUID-based, graceful fallback on missing data).
//!
//! Source of truth on x86/x86_64: `raw-cpuid` (CPUID leaves 0x00, 0x01, 0x07,
//! 0x8000_0002..4, 0x04/0x8000_001D). On other targets (or when CPUID leaves
//! are not supported by the chip) we fall back to `std::thread::available_parallelism`
//! and `sysinfo::System::physical_core_count`.
//!
//! Per AGENTS.md §3.2 we never `.unwrap()` in non-test code — every query is
//! best-effort with a graceful fallback string.

use crate::HalError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub vendor: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    /// Human-readable SIMD string, e.g. `"SSE4.2, AVX, AVX2, AVX-512F"`.
    /// `"unknown"` when no SIMD info could be gathered.
    pub simd: String,
    /// Structured list of detected SIMD instruction sets (machine-readable).
    /// `serde(default)` keeps it optional for backward compatibility with the
    /// M1 stub JSON shape.
    #[serde(default)]
    pub simd_flags: Vec<String>,
    /// Best-effort estimate of theoretical FP32 peak in decimal GFLOPS,
    /// derived from the detected SIMD FMA path × physical cores (with a
    /// 1 GHz baseline — multiply by clock GHz for the real peak). 0.0 when
    /// no FMA path (SSE-only, no AVX) is detected, or `physical_cores == 0`.
    /// `serde(default)` keeps old JSON without the field parseable.
    /// TODO(M2+): plumb CPU base/boost clocks for a tighter estimate.
    #[serde(default)]
    pub theoretical_fp32_gflops: f64,
}

/// Cross-platform logical/physical core count.
fn core_counts() -> (u32, u32) {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    let physical = sysinfo::System::physical_core_count()
        .map(|n| n as u32)
        .unwrap_or(logical);
    (physical, logical)
}

pub fn detect() -> Result<CpuInfo, HalError> {
    let (physical_cores, logical_cores) = core_counts();

    // Brand + vendor + SIMD via raw-cpuid on x86; otherwise generic fallback.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let (brand, vendor, simd, simd_flags) = {
        let info = detect_x86_cpuid();
        let brand = info
            .brand
            .unwrap_or_else(|| format!("{} (generic)", std::env::consts::ARCH));
        let vendor = info.vendor.unwrap_or_else(|| "unknown".to_string());
        let simd_flags = info.simd_flags;
        let simd = if simd_flags.is_empty() {
            "unknown".to_string()
        } else {
            simd_flags.join(", ")
        };
        (brand, vendor, simd, simd_flags)
    };

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let (brand, vendor, simd, simd_flags) = (
        format!("{} (generic)", std::env::consts::ARCH),
        "unknown".to_string(),
        "unknown (non-x86 target)".to_string(),
        Vec::new(),
    );

    let theoretical_fp32_gflops = theoretical_fp32_gflops(physical_cores, &simd_flags);

    Ok(CpuInfo {
        brand,
        vendor,
        physical_cores,
        logical_cores,
        simd,
        simd_flags,
        theoretical_fp32_gflops,
    })
}

/// Pure: estimate theoretical FP32 peak in decimal GFLOPS for `cores` and a
/// fixed vocabulary of SIMD flag strings. 0.0 when no FMA path is detected
/// (caller can render this as `"theoretical: unknown"`).
///
/// Math: 1 FMA / cycle/core × `lanes` × 2 ops/FMA, with cycles counted as
/// 1 ns (i.e. 1 GHz baseline — multiply by clock GHz for the real peak).
/// Both AVX2 and AVX-512F land at 32 FLOPS/cycle/core: AVX2 uses 2 FMA
/// ports × 8 lanes; AVX-512F uses 1 FMA port × 16 lanes (Intel disables
/// one of the two FMA ports when AVX-512 is in flight on consumer cores).
pub fn theoretical_fp32_gflops(physical_cores: u32, simd_flags: &[String]) -> f64 {
    if physical_cores == 0 {
        return 0.0;
    }
    let per_core: f64 = if simd_flags.iter().any(|f| f == "AVX-512F") {
        // 1 FMA unit × 16 lanes × 2 ops/FMA.
        32.0
    } else if simd_flags.iter().any(|f| f == "AVX2") {
        // 2 FMA units × 8 lanes × 2 ops/FMA.
        32.0
    } else {
        // No FMA path detected — peak is unknown, leave it to the caller.
        return 0.0;
    };
    f64::from(physical_cores) * per_core
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
struct X86CpuId {
    brand: Option<String>,
    vendor: Option<String>,
    simd_flags: Vec<String>,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86_cpuid() -> X86CpuId {
    use raw_cpuid::CpuId;

    let cpuid = CpuId::new();

    let vendor = cpuid.get_vendor_info().map(|v| v.as_str().to_string());

    let brand = cpuid
        .get_processor_brand_string()
        .map(|b| b.as_str().trim().to_string())
        // Brand string sometimes comes back as empty/all-space on VMs.
        .filter(|s| !s.is_empty());

    // Build SIMD flag list (most-significant first).
    let mut simd_flags: Vec<String> = Vec::new();

    if let Some(feat) = cpuid.get_feature_info() {
        if feat.has_sse42() {
            simd_flags.push("SSE4.2".to_string());
        }
        if feat.has_avx() {
            simd_flags.push("AVX".to_string());
        }
    }

    if let Some(ext) = cpuid.get_extended_feature_info() {
        if ext.has_avx2() {
            simd_flags.push("AVX2".to_string());
        }
        if ext.has_avx512f() {
            simd_flags.push("AVX-512F".to_string());
        }
        if ext.has_amx_tile() || ext.has_amx_bf16() || ext.has_amx_int8() {
            simd_flags.push("AMX".to_string());
        }
    }

    X86CpuId {
        brand,
        vendor,
        simd_flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_has_cores() {
        let cpu = detect().expect("cpu detect");
        assert!(cpu.logical_cores >= 1);
        assert!(cpu.physical_cores >= 1);
        assert!(!cpu.brand.is_empty());
    }

    #[test]
    fn simd_flag_strings_are_known() {
        let cpu = detect().expect("cpu detect");
        for flag in &cpu.simd_flags {
            // We only emit a fixed vocabulary — keeps JSON stable for downstream.
            assert!(
                ["SSE4.2", "AVX", "AVX2", "AVX-512F", "AMX"].contains(&flag.as_str()),
                "unexpected SIMD flag: {flag}"
            );
        }
    }

    #[test]
    fn cpu_deserializes_old_m2pre_json_without_theoretical() {
        // Pre-slice-3 JSON does not include `theoretical_fp32_gflops`; the
        // field must default to 0.0 so old profiles keep loading.
        let old_json = r#"{
            "brand": "Example CPU",
            "vendor": "GenuineIntel",
            "physical_cores": 8,
            "logical_cores": 16,
            "simd": "SSE4.2, AVX, AVX2",
            "simd_flags": ["SSE4.2", "AVX", "AVX2"]
        }"#;
        let cpu: CpuInfo = serde_json::from_str(old_json).expect("deserialize old CpuInfo JSON");
        assert_eq!(cpu.physical_cores, 8);
        assert_eq!(cpu.logical_cores, 16);
        assert_eq!(cpu.theoretical_fp32_gflops, 0.0);
    }

    #[test]
    fn theoretical_fp32_gflops_for_known_paths() {
        // 8 cores × 32 GFLOPS/core on any FMA path = 256 GFLOPS.
        let avx2 = vec!["AVX2".to_string()];
        assert_eq!(theoretical_fp32_gflops(8, &avx2), 256.0);
        let avx512 = vec!["AVX-512F".to_string()];
        assert_eq!(theoretical_fp32_gflops(8, &avx512), 256.0);
        // AVX-512F beats AVX2 on a tie-break (consumer cores have 1 FMA port
        // × 16 lanes vs 2 × 8 — same FLOPS/cycle, but the lane width is
        // higher so any FMA-heavy roofline prefers AVX-512F).
        let both = vec!["AVX2".to_string(), "AVX-512F".to_string()];
        assert_eq!(theoretical_fp32_gflops(8, &both), 256.0);
    }

    #[test]
    fn theoretical_fp32_gflops_unknown_for_sse_only_or_no_cores() {
        let sse = vec!["SSE4.2".to_string()];
        assert_eq!(theoretical_fp32_gflops(8, &sse), 0.0);
        assert_eq!(theoretical_fp32_gflops(0, &["AVX2".to_string()]), 0.0);
        assert_eq!(theoretical_fp32_gflops(8, &[]), 0.0);
    }
}
