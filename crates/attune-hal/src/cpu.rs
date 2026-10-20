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

    Ok(CpuInfo {
        brand,
        vendor,
        physical_cores,
        logical_cores,
        simd,
        simd_flags,
    })
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
}
