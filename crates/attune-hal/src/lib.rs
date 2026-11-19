//! attune-hal — hardware detection (M1).
//!
//! Public surface:
//! - `cpu::CpuInfo`        — brand, vendor, cores, SIMD (via raw-cpuid on x86)
//! - `memory::MemoryInfo`  — total/available RAM (via sysinfo)
//! - `cache::CacheInfo`    — L1d/L1i/L2/L3 + line size (via raw-cpuid on x86)
//! - `NumaInfo` / `GpuInfo` — placeholders; no real NUMA/GPU probing yet
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

/// Placeholder: real NUMA topology detection is not implemented. The field
/// exists so the JSON contract is stable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaInfo {
    pub nodes: u32,
    pub note: String,
}

/// Placeholder: GPU enumeration is not implemented;
/// `count == 0` is the safe default.
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
            // Most consumer hardware reports a single NUMA node.
            numa: NumaInfo {
                nodes: 1,
                note: "not detected; single node assumed".to_string(),
            },
            gpu: GpuInfo {
                count: 0,
                note: "GPU detection not implemented".to_string(),
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
        assert_eq!(hw.numa.nodes, 1); // placeholder invariant
        assert_eq!(hw.gpu.count, 0); // placeholder invariant
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
