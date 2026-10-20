//! attune-hal — hardware detection (M1).
//!
//! Public surface:
//! - `cpu::CpuInfo`        — brand, vendor, cores, SIMD (via raw-cpuid on x86)
//! - `memory::MemoryInfo`  — total/available RAM (via sysinfo)
//! - `cache::CacheInfo`    — L1d/L1i/L2/L3 + line size (via raw-cpuid on x86)
//! - `NumaInfo` / `GpuInfo` — stubs for M1, real implementation lands in M2+
//!
//! Cross-platform: x86/x86_64 get full detail; other targets compile and
//! gracefully degrade to `"unknown"` placeholders.

pub mod cache;
pub mod cpu;
pub mod memory;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HalError {
    #[error("detection failed: {0}")]
    Detection(String),
}

/// M1 stub. Real NUMA topology detection (hwloc / Windows groups) is planned
/// in M2. We expose the field now so the JSON contract is stable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaInfo {
    pub nodes: u32,
    pub note: String,
}

/// M1 stub. GPU enumeration (NVIDIA/AMD/Intel via vendor APIs) is planned
/// for M3+ — `count == 0` is the safe placeholder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub count: u32,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: cpu::CpuInfo,
    pub memory: memory::MemoryInfo,
    pub cache: cache::CacheInfo,
    pub numa: NumaInfo,
    pub gpu: GpuInfo,
    pub os: String,
}

impl HardwareInfo {
    pub fn detect() -> Result<Self, HalError> {
        Ok(Self {
            cpu: cpu::detect()?,
            memory: memory::detect()?,
            cache: cache::detect()?,
            // M1 stub: most consumer hardware reports a single NUMA node.
            // Real detection (hwloc on Linux, NUMA APIs on Windows) is M2.
            numa: NumaInfo {
                nodes: 1,
                note: "stub M1 — hwloc lands in M2".to_string(),
            },
            // M1 stub: no GPU enumeration yet. M3 will probe NVML + ADL +
            // Windows Display Driver APIs.
            gpu: GpuInfo {
                count: 0,
                note: "stub M1 — vendor APIs land in M3".to_string(),
            },
            os: std::env::consts::OS.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_info() {
        let hw = HardwareInfo::detect().expect("detect should succeed");
        assert!(!hw.cpu.brand.is_empty());
        assert!(!hw.cpu.vendor.is_empty());
        assert!(hw.cpu.logical_cores >= 1);
        assert!(hw.cpu.physical_cores >= 1);
        assert!(hw.memory.total_bytes > 0);
        assert_eq!(hw.numa.nodes, 1); // M1 stub invariant
        assert_eq!(hw.gpu.count, 0); // M1 stub invariant
    }

    #[test]
    fn json_roundtrip() {
        let hw = HardwareInfo::detect().expect("detect should succeed");
        let json = serde_json::to_string(&hw).expect("serialize");
        let back: HardwareInfo = serde_json::from_str(&json).expect("deserialize");
        // Backward-compat: pre-cache field must roundtrip identically.
        assert_eq!(back.cpu.logical_cores, hw.cpu.logical_cores);
        assert_eq!(back.cpu.physical_cores, hw.cpu.physical_cores);
        assert_eq!(back.cpu.brand, hw.cpu.brand);
        assert_eq!(back.memory.total_bytes, hw.memory.total_bytes);
        assert_eq!(back.os, hw.os);
    }

    #[test]
    fn json_contains_new_fields() {
        let hw = HardwareInfo::detect().expect("detect should succeed");
        let json = serde_json::to_string(&hw).expect("serialize");
        // New M1 fields must be present in the JSON contract.
        assert!(json.contains("\"cache\""), "missing cache: {json}");
        assert!(json.contains("\"numa\""), "missing numa: {json}");
        assert!(json.contains("\"gpu\""), "missing gpu: {json}");
        assert!(json.contains("\"vendor\""), "missing cpu.vendor: {json}");
    }
}
