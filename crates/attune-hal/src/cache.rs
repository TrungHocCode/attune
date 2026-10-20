//! CPU cache topology via `raw-cpuid` (LEAF=0x04 on Intel, 0x8000_001D on AMD).
//!
//! TODO(M2): hwloc gives cross-platform topology on Linux. For M1 we keep it
//! x86-only and graceful — if the leaf is unsupported or empty, every level
//! stays at 0 and `line_size_bytes` at 0, with no panic. AGENTS.md §3.2 says
//! no `.unwrap()` in non-test code, so we collect into a small struct first
//! and fold explicitly.

use crate::HalError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheInfo {
    pub l1d_bytes: u64,
    pub l1i_bytes: u64,
    pub l2_bytes: u64,
    pub l3_bytes: u64,
    pub line_size_bytes: u32,
}

pub fn detect() -> Result<CacheInfo, HalError> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Ok(detect_x86_cpuid())
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        Ok(CacheInfo::default())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86_cpuid() -> CacheInfo {
    use raw_cpuid::{CacheType, CpuId};

    let cpuid = CpuId::new();
    let mut info = CacheInfo::default();

    let Some(iter) = cpuid.get_cache_parameters() else {
        // Leaf unsupported (e.g. older AMD without 0x8000_001D).
        return info;
    };

    // The iterator calls CPUID on every next(), so collect to avoid re-querying.
    let params: Vec<_> = iter.collect();

    for cache in &params {
        // Skip terminator / reserved entries (Null means "no more caches").
        match cache.cache_type() {
            CacheType::Null | CacheType::Reserved => continue,
            _ => {}
        }

        let size = cache.associativity()
            * cache.physical_line_partitions()
            * cache.coherency_line_size()
            * cache.sets();

        if size == 0 {
            continue;
        }

        let line = cache.coherency_line_size() as u32;
        if line > info.line_size_bytes {
            info.line_size_bytes = line;
        }

        match (cache.level(), cache.cache_type()) {
            (1, CacheType::Data) => info.l1d_bytes = size as u64,
            (1, CacheType::Instruction) => info.l1i_bytes = size as u64,
            // Modern CPUs report L1 as Unified — attribute to l1d as a
            // best-effort split. We don't have separate d/i sizes then.
            (1, CacheType::Unified) => {
                if info.l1d_bytes == 0 {
                    info.l1d_bytes = size as u64;
                } else if info.l1i_bytes == 0 {
                    info.l1i_bytes = size as u64;
                }
            }
            (2, _) => info.l2_bytes = size as u64,
            (3, _) => info.l3_bytes = size as u64,
            _ => {
                // L4 or unknown — ignore for M1.
            }
        }
    }

    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_parse_does_not_panic() {
        // Just exercise the parser — on VMs without CPUID leaf 0x04 every
        // level will be 0, which is fine. We only assert non-negative.
        let info = detect().expect("cache detect");
        assert!(info.l1d_bytes < u64::MAX);
        assert!(info.l2_bytes < u64::MAX);
        assert!(info.l3_bytes < u64::MAX);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn cache_levels_are_consistent() {
        let info = detect().expect("cache detect");
        // Standard x86 hierarchy: L1 < L2 < L3 when all present. Some CPUs
        // report only L2 + L3 (e.g. when leaf 0x04 isn't exposed); in that
        // case l1d == 0 == l1i and the assertion trivially holds.
        if info.l1d_bytes > 0 && info.l2_bytes > 0 {
            assert!(info.l1d_bytes < info.l2_bytes);
        }
        if info.l2_bytes > 0 && info.l3_bytes > 0 {
            assert!(info.l2_bytes < info.l3_bytes);
        }
    }
}
