//! Memory detection via `sysinfo` (cross-platform) plus best-effort DRAM
//! speed / channel count for theoretical-bandwidth context.
//!
//! Justification for dep: plan §6 mandates `sysinfo` for RAM/CPU.
//! Cross-platform (Linux + Windows), graceful fallback to 0 on error.

use crate::HalError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub total_gb: f64,
    /// Best-effort available memory (matches `sysinfo::System::available_memory`).
    /// On Windows this is the same value as `free_memory` (Windows does not
    /// report a distinct "free" pool). 0 when sysinfo cannot read it.
    #[serde(default)]
    pub available_bytes: u64,
    #[serde(default)]
    pub available_gb: f64,
    /// DRAM transfer rate in MT/s (megatransfers per second).
    /// 0 when the OS does not expose it or when the probe failed.
    /// Populated best-effort on Windows via CIM
    /// (`Get-CimInstance Win32_PhysicalMemory`); left as 0 elsewhere for
    /// now (Linux uses `/sys/class/dmi/id/...` in a later slice).
    #[serde(default)]
    pub speed_mts: u64,
    /// Populated DIMM count, used as the channel-count proxy (a single DIMM
    /// per channel is the dominant topology for consumer boards). 0 when
    /// unknown.
    #[serde(default)]
    pub channels: u32,
    /// Derived peak DRAM bandwidth in decimal GB/s. 0 when either
    /// `speed_mts` or `channels` is 0. See [`theoretical_gbps`].
    #[serde(default)]
    pub theoretical_bandwidth_gbps: f64,
}

/// Pure helper: theoretical DRAM bandwidth in decimal GB/s.
///
/// `speed_mts × 8` converts MT/s → MB/s (one transfer = one byte on a
/// 64-bit DDR bus), multiplied by `channels` for the parallel aggregate,
/// then divided by `1000` to land in **decimal** GB/s so the number is
/// directly comparable to vendor figures and to the measured
/// `attune-bench` GB/s. Returns `0.0` for any zero / unknown input so the
/// caller can use the value as-is.
pub fn theoretical_gbps(speed_mts: u64, channels: u32) -> f64 {
    if speed_mts == 0 || channels == 0 {
        return 0.0;
    }
    // Saturating cast on `channels`; the value is tiny in practice (≤8).
    let ch = u64::from(channels);
    speed_mts.saturating_mul(8).saturating_mul(ch) as f64 / 1000.0
}

pub fn detect() -> Result<MemoryInfo, HalError> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_memory();
    let total = sys.total_memory(); // bytes in sysinfo 0.36
    let available = sys.available_memory();

    if total == 0 {
        return Err(HalError::Detection("sysinfo returned 0 RAM".to_string()));
    }

    let (speed_mts, channels) = detect_dram_speed_channels();
    let theoretical_bandwidth_gbps = theoretical_gbps(speed_mts, channels);

    Ok(MemoryInfo {
        total_bytes: total,
        total_gb: total as f64 / 1024.0 / 1024.0 / 1024.0,
        available_bytes: available,
        available_gb: available as f64 / 1024.0 / 1024.0 / 1024.0,
        speed_mts,
        channels,
        theoretical_bandwidth_gbps,
    })
}

/// Best-effort probe for DRAM speed (MT/s) and DIMM count.
///
/// On Windows we shell out to PowerShell and parse
/// `Get-CimInstance Win32_PhysicalMemory`. The command is run with a short
/// timeout so a stuck or absent `powershell.exe` can never block `detect`.
/// On Linux we currently return `(0, 0)` — populating these via
/// `/sys/class/dmi/id/...` and `dmidecode` is M2+ territory. On any other
/// platform or on any error path we also fall back to `(0, 0)`.
fn detect_dram_speed_channels() -> (u64, u32) {
    #[cfg(target_os = "windows")]
    {
        windows_dram_speed_channels()
    }
    #[cfg(not(target_os = "windows"))]
    {
        // TODO(M2+): parse `/sys/class/dmi/id/...` on Linux to populate
        // `speed_mts` and a `channels` count derived from populated DIMM
        // slots. For now callers see "theoretical: unknown".
        let _ = 0u32; // silence the unused-warning on platforms with no impl
        (0, 0)
    }
}

/// Windows-side DRAM probe. Returns `(0, 0)` on any failure; the caller
/// never sees an `Err` from this path because surfacing a hard error
/// would break `detect` on systems where WMI/CIM is locked down.
#[cfg(target_os = "windows")]
fn windows_dram_speed_channels() -> (u64, u32) {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    // Wrap in `@(...)` so the result is always an array — even when there is
    // exactly zero or one DIMM, so `.Count` is defined downstream.
    let script = "$cim = @(Get-CimInstance Win32_PhysicalMemory -ErrorAction SilentlyContinue); \
                  if ($cim.Count -gt 0) { \
                    $speed = 0; \
                    if ($null -ne $cim[0].Speed) { $speed = [int64]$cim[0].Speed }; \
                    Write-Output \"$speed|$($cim.Count)\" \
                  }";

    let child = match Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return (0, 0),
    };

    let timeout = Duration::from_secs(3);
    let deadline = Instant::now() + timeout;
    let mut child = child;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let output = match child.wait_with_output() {
                    Ok(o) => o,
                    Err(_) => return (0, 0),
                };
                let stdout = match String::from_utf8(output.stdout) {
                    Ok(s) => s,
                    Err(_) => return (0, 0),
                };
                return parse_speed_channels(&stdout);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (0, 0);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return (0, 0);
            }
        }
    }
}

/// Pure parser for the `"<speed>|<count>"` line emitted by the Windows
/// PowerShell probe. Kept platform-independent so its unit tests run on
/// every OS; the only production caller is Windows-gated, hence the
/// `allow(dead_code)` on other platforms (plain `cargo clippy` without
/// `--all-targets` does not count `#[cfg(test)]` uses).
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn parse_speed_channels(s: &str) -> (u64, u32) {
    let trimmed = s.trim();
    let mut parts = trimmed.split('|');
    let speed_str = match parts.next() {
        Some(p) => p.trim(),
        None => return (0, 0),
    };
    let count_str = match parts.next() {
        Some(p) => p.trim(),
        None => return (0, 0),
    };
    let speed = match speed_str.parse::<u64>() {
        Ok(n) => n,
        Err(_) => return (0, 0),
    };
    let count = match count_str.parse::<u32>() {
        Ok(n) => n,
        Err(_) => return (0, 0),
    };
    (speed, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_nonzero() {
        let mem = detect().expect("memory detect");
        assert!(mem.total_bytes > 0);
    }

    #[test]
    fn memory_available_within_bounds() {
        let mem = detect().expect("memory detect");
        // available <= total (sysinfo invariant); allow equal when total == available.
        assert!(mem.available_bytes <= mem.total_bytes);
    }

    #[test]
    fn theoretical_ddr4_dual_channel() {
        // DDR4-3200 dual channel: 3200 * 8 * 2 / 1000 = 51.2 GB/s
        assert!((theoretical_gbps(3200, 2) - 51.2).abs() < 1e-9);
    }

    #[test]
    fn theoretical_ddr5_quad_channel() {
        // DDR5-4800 quad channel: 4800 * 8 * 4 / 1000 = 153.6 GB/s
        assert!((theoretical_gbps(4800, 4) - 153.6).abs() < 1e-9);
    }

    #[test]
    fn theoretical_zero_inputs() {
        assert_eq!(theoretical_gbps(0, 4), 0.0);
        assert_eq!(theoretical_gbps(3200, 0), 0.0);
        assert_eq!(theoretical_gbps(0, 0), 0.0);
    }

    #[test]
    fn memory_json_roundtrip_includes_new_fields() {
        let mem = detect().expect("memory detect");
        let json = serde_json::to_string(&mem).expect("serialize MemoryInfo");
        // New M2 fields must appear in the JSON contract.
        assert!(json.contains("\"speed_mts\""), "missing speed_mts: {json}");
        assert!(json.contains("\"channels\""), "missing channels: {json}");
        assert!(
            json.contains("\"theoretical_bandwidth_gbps\""),
            "missing theoretical_bandwidth_gbps: {json}"
        );

        let back: MemoryInfo = serde_json::from_str(&json).expect("deserialize MemoryInfo");
        assert_eq!(back.speed_mts, mem.speed_mts);
        assert_eq!(back.channels, mem.channels);
        assert_eq!(
            back.theoretical_bandwidth_gbps,
            mem.theoretical_bandwidth_gbps
        );
    }

    #[test]
    fn memory_deserializes_old_m1_json_without_new_fields() {
        // Old M1 JSON does not include speed_mts / channels /
        // theoretical_bandwidth_gbps. The new fields must default to 0 so
        // profiles shipped before this slice still load.
        let old_json = r#"{
            "total_bytes": 17179869184,
            "total_gb": 16.0,
            "available_bytes": 8589934592,
            "available_gb": 8.0
        }"#;
        let mem: MemoryInfo =
            serde_json::from_str(old_json).expect("deserialize old M1 MemoryInfo JSON");
        assert_eq!(mem.total_bytes, 17_179_869_184);
        assert_eq!(mem.available_bytes, 8_589_934_592);
        assert_eq!(mem.speed_mts, 0);
        assert_eq!(mem.channels, 0);
        assert_eq!(mem.theoretical_bandwidth_gbps, 0.0);
        // Derivation matches the helper:
        assert_eq!(
            mem.theoretical_bandwidth_gbps,
            theoretical_gbps(mem.speed_mts, mem.channels)
        );
    }

    #[test]
    fn parse_speed_channels_handles_clean_input() {
        assert_eq!(parse_speed_channels("3200|2"), (3200, 2));
    }

    #[test]
    fn parse_speed_channels_handles_trailing_whitespace() {
        assert_eq!(parse_speed_channels("  2666 | 4\n"), (2666, 4));
    }

    #[test]
    fn parse_speed_channels_returns_zero_on_garbage() {
        assert_eq!(parse_speed_channels(""), (0, 0));
        assert_eq!(parse_speed_channels("nope"), (0, 0));
        assert_eq!(parse_speed_channels("3200|"), (0, 0));
        assert_eq!(parse_speed_channels("|4"), (0, 0));
        assert_eq!(parse_speed_channels("oops|four"), (0, 0));
    }
}
