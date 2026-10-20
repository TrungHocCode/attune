//! Minimal memory detection via `sysinfo`.
//!
//! Justification for dep: plan §6 mandates `sysinfo` for RAM/CPU.
//! Cross-platform (Linux + Windows), graceful fallback to 0 on error.

use crate::HalError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub total_gb: f64,
}

pub fn detect() -> Result<MemoryInfo, HalError> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_memory();
    let total = sys.total_memory(); // bytes in sysinfo 0.36

    if total == 0 {
        return Err(HalError::Detection("sysinfo returned 0 RAM".to_string()));
    }

    Ok(MemoryInfo {
        total_bytes: total,
        total_gb: total as f64 / 1024.0 / 1024.0 / 1024.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_nonzero() {
        let mem = detect().expect("memory detect");
        assert!(mem.total_bytes > 0);
    }
}
