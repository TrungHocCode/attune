//! Minimal CPU detection (stub for commit 1).
//!
//! TODO(commit 2): use `raw-cpuid` for brand / SIMD (AVX2/AVX-512/AMX)
//! and cache topology. Keep std-only here so M1 stays green cross-platform.

use crate::HalError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    /// Placeholder — real SIMD string comes in commit 2.
    pub simd: String,
}

pub fn detect() -> Result<CpuInfo, HalError> {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);

    // std has no stable brand string; use arch as placeholder until raw-cpuid lands.
    let brand = format!("{} (generic)", std::env::consts::ARCH);

    Ok(CpuInfo {
        brand,
        physical_cores: logical, // TODO: picks real physical count via hwloc
        logical_cores: logical,
        simd: "unknown (TODO raw-cpuid)".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_has_cores() {
        let cpu = detect().expect("cpu detect");
        assert!(cpu.logical_cores >= 1);
    }
}
