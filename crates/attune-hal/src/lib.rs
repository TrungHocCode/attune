//! attune-hal — hardware detection (M1 skeleton).
//!
//! Commit 1: minimal `HardwareInfo` to unblock `attune detect`.
//! Real SIMD/cache/NUMA detection (raw-cpuid, hwloc) lands in commit 2+.

pub mod cpu;
pub mod memory;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HalError {
    #[error("detection failed: {0}")]
    Detection(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: cpu::CpuInfo,
    pub memory: memory::MemoryInfo,
    pub os: String,
}

impl HardwareInfo {
    pub fn detect() -> Result<Self, HalError> {
        Ok(Self {
            cpu: cpu::detect()?,
            memory: memory::detect()?,
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
        assert!(hw.cpu.logical_cores >= 1);
        assert!(hw.memory.total_bytes > 0);
    }

    #[test]
    fn json_roundtrip() {
        let hw = HardwareInfo::detect().expect("detect should succeed");
        let json = serde_json::to_string(&hw).expect("serialize");
        let back: HardwareInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.cpu.logical_cores, hw.cpu.logical_cores);
    }
}
