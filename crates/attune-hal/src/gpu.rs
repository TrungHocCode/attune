//! GPU detection (M7 slice 1): shell out to vendor tools, never link an SDK.
//!
//! Detection order:
//! 1. `nvidia-smi` on every platform (authoritative name / VRAM / driver).
//! 2. On Windows only, CIM `Win32_VideoController` for adapters `nvidia-smi`
//!    cannot see (e.g. an Intel iGPU), or as the sole fallback when
//!    `nvidia-smi` is missing. `AdapterRAM` caps at 4 GB, so VRAM is left
//!    unknown for CIM-sourced devices.
//!
//! Every probe is best-effort with a 5 s timeout: a missing or hung tool must
//! never block `attune detect`. The parsers are pure so CI (no GPU) can test
//! them against recorded output.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub const SOURCE_NVIDIA_SMI: &str = "nvidia-smi";
pub const SOURCE_CIM: &str = "cim";

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const MIB: u64 = 1024 * 1024;

/// One physical GPU adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// `"nvidia"`, `"amd"`, `"intel"` or `"other"`.
    pub vendor: String,
    pub name: String,
    #[serde(default)]
    pub vram_total_bytes: Option<u64>,
    #[serde(default)]
    pub vram_free_bytes: Option<u64>,
    #[serde(default)]
    pub driver: Option<String>,
    /// `"nvidia-smi"` or `"cim"`.
    pub source: String,
}

/// GPU inventory. `count` and `note` are kept from the M1 shape; `devices` is
/// additive so pre-M7 JSON still deserializes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub count: u32,
    pub note: String,
    #[serde(default)]
    pub devices: Vec<GpuDevice>,
}

/// Detect every GPU we can see. Never fails: an empty `devices` list with
/// `count == 0` is the safe result when no vendor tool is available.
pub fn detect() -> GpuInfo {
    #[cfg(target_os = "windows")]
    let devices = {
        let (nvidia, cim) = std::thread::scope(|scope| {
            let nvidia = scope.spawn(probe_nvidia);
            let cim = scope.spawn(probe_cim);
            (
                nvidia.join().unwrap_or_default(),
                cim.join().unwrap_or_default(),
            )
        });
        merge_devices(nvidia, cim)
    };
    #[cfg(not(target_os = "windows"))]
    let devices = probe_nvidia();

    let note = if devices.is_empty() {
        "no GPU found (nvidia-smi not available)".to_string()
    } else {
        summarize(&devices)
    };

    GpuInfo {
        count: devices.len() as u32,
        note,
        devices,
    }
}

/// Run the `nvidia-smi` probe and parse it. A missing or failed tool yields no
/// devices rather than an error.
fn probe_nvidia() -> Vec<GpuDevice> {
    run_nvidia_smi()
        .map(|text| parse_nvidia_smi(&text))
        .unwrap_or_default()
}

/// Run the CIM probe and parse it (Windows only), falling back to no devices.
#[cfg(target_os = "windows")]
fn probe_cim() -> Vec<GpuDevice> {
    run_cim().map(|text| parse_cim(&text)).unwrap_or_default()
}

/// Run `nvidia-smi`, returning its stdout only on a successful exit.
fn run_nvidia_smi() -> Option<String> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]);
    let captured = run_captured(&mut cmd, PROBE_TIMEOUT)?;
    if captured.success {
        Some(captured.stdout)
    } else {
        None
    }
}

/// Windows-only CIM fallback. Returns stdout only on a successful exit.
#[cfg(target_os = "windows")]
fn run_cim() -> Option<String> {
    let mut cmd = Command::new("powershell");
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }",
    ]);
    let captured = run_captured(&mut cmd, PROBE_TIMEOUT)?;
    if captured.success {
        Some(captured.stdout)
    } else {
        None
    }
}

struct Captured {
    success: bool,
    stdout: String,
}

/// Spawn `cmd`, wait at most `timeout`, then collect stdout. Returns `None`
/// when the process cannot start, errors, or is killed after the timeout.
/// Windows spawns use `CREATE_NO_WINDOW` so no console flashes.
fn run_captured(cmd: &mut Command, timeout: Duration) -> Option<Captured> {
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }

    let mut child = cmd.spawn().ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child.wait_with_output().ok()?;
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                return Some(Captured {
                    success: status.success(),
                    stdout,
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
/// Memory columns are in MiB; malformed lines and `[N/A]` values degrade to
/// `None` rather than failing the whole probe.
pub fn parse_nvidia_smi(text: &str) -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        let name = parts[0];
        if name.is_empty() {
            continue;
        }
        devices.push(GpuDevice {
            vendor: "nvidia".to_string(),
            name: name.to_string(),
            vram_total_bytes: parse_mib(parts[1]),
            vram_free_bytes: parse_mib(parts[2]),
            driver: normalize_opt(parts[3]),
            source: SOURCE_NVIDIA_SMI.to_string(),
        });
    }
    devices
}

/// Parse `Win32_VideoController` lines of the form `Name|DriverVersion`.
/// VRAM is intentionally unknown (CIM's `AdapterRAM` caps at 4 GB).
pub fn parse_cim(text: &str) -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || is_ignored_adapter(line) {
            continue;
        }
        let mut parts = line.splitn(2, '|');
        let name = match parts.next() {
            Some(name) => name.trim(),
            None => continue,
        };
        if name.is_empty() {
            continue;
        }
        let driver = parts.next().and_then(normalize_opt);
        devices.push(GpuDevice {
            vendor: classify_vendor(name),
            name: name.to_string(),
            vram_total_bytes: None,
            vram_free_bytes: None,
            driver,
            source: SOURCE_CIM.to_string(),
        });
    }
    devices
}

/// Add CIM adapters that `nvidia-smi` did not already report. When the primary
/// list already has an NVIDIA device, NVIDIA entries from CIM are dropped
/// (nvidia-smi is authoritative); everything else is kept unless its name was
/// already listed.
pub fn merge_devices(primary: Vec<GpuDevice>, extra: Vec<GpuDevice>) -> Vec<GpuDevice> {
    let mut out = primary;
    let has_primary_nvidia = out.iter().any(|d| d.vendor == "nvidia");
    for dev in extra {
        if has_primary_nvidia && dev.vendor == "nvidia" {
            continue;
        }
        if out.iter().any(|d| d.name.eq_ignore_ascii_case(&dev.name)) {
            continue;
        }
        out.push(dev);
    }
    out
}

/// Map an adapter name to a normalized vendor string.
pub fn classify_vendor(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia") {
        "nvidia".to_string()
    } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("ati ") {
        "amd".to_string()
    } else if lower.contains("intel") {
        "intel".to_string()
    } else {
        "other".to_string()
    }
}

/// Adapters that are never a usable GPU: the software fallback and virtual
/// display adapters (e.g. a USB monitor bridge) that CIM lists like a GPU.
fn is_ignored_adapter(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("microsoft basic display")
        || lower.contains("remote display")
        || lower.contains("virtual display")
}

/// MiB string -> bytes. Missing, `[N/A]`, non-numeric or zero values -> `None`.
fn parse_mib(raw: &str) -> Option<u64> {
    let mib: u64 = raw.trim().parse().ok()?;
    if mib == 0 {
        return None;
    }
    Some(mib.saturating_mul(MIB))
}

fn normalize_opt(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("[n/a]")
        || trimmed.eq_ignore_ascii_case("n/a")
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Short human summary, e.g. `1 NVIDIA GPU via nvidia-smi`.
fn summarize(devices: &[GpuDevice]) -> String {
    if let [only] = devices {
        return format!("1 {} GPU via {}", vendor_label(&only.vendor), only.source);
    }

    let (mut nvidia, mut amd, mut intel, mut other) = (0, 0, 0, 0);
    for dev in devices {
        match dev.vendor.as_str() {
            "nvidia" => nvidia += 1,
            "amd" => amd += 1,
            "intel" => intel += 1,
            _ => other += 1,
        }
    }
    let mut parts = Vec::new();
    if nvidia > 0 {
        parts.push(format!("{nvidia} NVIDIA"));
    }
    if amd > 0 {
        parts.push(format!("{amd} AMD"));
    }
    if intel > 0 {
        parts.push(format!("{intel} Intel"));
    }
    if other > 0 {
        parts.push(format!("{other} other"));
    }

    let has_smi = devices.iter().any(|d| d.source == SOURCE_NVIDIA_SMI);
    let has_cim = devices.iter().any(|d| d.source == SOURCE_CIM);
    let sources = match (has_smi, has_cim) {
        (true, true) => "nvidia-smi + cim",
        (true, false) => "nvidia-smi",
        _ => "cim",
    };
    format!(
        "{} GPU{} ({}) via {sources}",
        devices.len(),
        if devices.len() == 1 { "" } else { "s" },
        parts.join(", "),
    )
}

fn vendor_label(vendor: &str) -> &'static str {
    match vendor {
        "nvidia" => "NVIDIA",
        "amd" => "AMD",
        "intel" => "Intel",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_single_gpu_converts_mib_to_bytes() {
        let devices = parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 5921, 561.00\n");
        assert_eq!(devices.len(), 1);
        let dev = &devices[0];
        assert_eq!(dev.vendor, "nvidia");
        assert_eq!(dev.name, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(dev.vram_total_bytes, Some(6_439_305_216));
        assert_eq!(dev.vram_free_bytes, Some(5_921 * MIB));
        assert_eq!(dev.driver.as_deref(), Some("561.00"));
        assert_eq!(dev.source, SOURCE_NVIDIA_SMI);
    }

    #[test]
    fn parse_nvidia_two_gpus_and_blank_lines() {
        let text = "\nNVIDIA GeForce RTX 4090, 24564, 24000, 560.94\n\n\
                    NVIDIA GeForce RTX 3060, 12288, 12000, 552.22\n\n";
        let devices = parse_nvidia_smi(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devices[1].name, "NVIDIA GeForce RTX 3060");
        assert_eq!(devices[1].vram_total_bytes, Some(12_288 * MIB));
    }

    #[test]
    fn parse_nvidia_skips_malformed_line() {
        let text = "NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 5921, 561.00\n\
                    this line is not csv\n";
        let devices = parse_nvidia_smi(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4050 Laptop GPU");
    }

    #[test]
    fn parse_nvidia_na_memory_is_none() {
        let devices = parse_nvidia_smi("NVIDIA A100, [N/A], [N/A], 550.54\n");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vram_total_bytes, None);
        assert_eq!(devices[0].vram_free_bytes, None);
        assert_eq!(devices[0].driver.as_deref(), Some("550.54"));
    }

    #[test]
    fn parse_cim_vendors_and_ignores_basic_adapter() {
        let text = "NVIDIA GeForce RTX 4050 Laptop GPU|32.0.15.6100\n\
                    Intel(R) Graphics|32.0.101.6078\n\
                    Microsoft Basic Display Adapter|10.0\n";
        let devices = parse_cim(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].vendor, "nvidia");
        assert_eq!(devices[0].driver.as_deref(), Some("32.0.15.6100"));
        assert_eq!(devices[1].vendor, "intel");
        assert_eq!(devices[1].name, "Intel(R) Graphics");
        assert_eq!(devices[1].vram_total_bytes, None);
        assert_eq!(devices[1].source, SOURCE_CIM);
    }

    #[test]
    fn parse_cim_classifies_amd() {
        let devices = parse_cim("AMD Radeon RX 7800 XT|31.0.24027.1012\n");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor, "amd");
    }

    #[test]
    fn parse_cim_skips_virtual_display_adapters() {
        let text = "USB Mobile Monitor Virtual Display|2.0.0.1\n\
                    Intel(R) Graphics|32.0.101.7076\n";
        let devices = parse_cim(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "Intel(R) Graphics");
    }

    #[test]
    fn merge_keeps_intel_and_drops_duplicate_nvidia() {
        let nvidia = parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 5921, 561.00\n");
        let cim = parse_cim(
            "NVIDIA GeForce RTX 4050 Laptop GPU|32.0.15.6100\n\
             Intel(R) Graphics|32.0.101.6078\n",
        );
        let merged = merge_devices(nvidia, cim);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].vendor, "nvidia");
        assert_eq!(merged[1].vendor, "intel");
        assert_eq!(merged[1].name, "Intel(R) Graphics");
    }

    #[test]
    fn merge_without_primary_keeps_cim_nvidia() {
        let cim = parse_cim("NVIDIA GeForce GTX 1650|31.0.15.3683\n");
        let merged = merge_devices(Vec::new(), cim);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].vendor, "nvidia");
        assert_eq!(merged[0].source, SOURCE_CIM);
    }

    #[test]
    fn empty_inputs_yield_no_devices() {
        assert!(parse_nvidia_smi("").is_empty());
        assert!(parse_cim("").is_empty());
    }

    #[test]
    fn gpu_info_deserializes_old_json_without_devices() {
        let old = r#"{"count":0,"note":"GPU detection not implemented"}"#;
        let info: GpuInfo = serde_json::from_str(old).expect("old GpuInfo JSON should parse");
        assert_eq!(info.count, 0);
        assert!(info.devices.is_empty());
    }

    #[test]
    fn gpu_info_json_roundtrip_includes_devices() {
        let info = GpuInfo {
            count: 1,
            note: "1 NVIDIA GPU via nvidia-smi".to_string(),
            devices: parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 5921, 561.00\n"),
        };
        let json = serde_json::to_string(&info).expect("serialize GpuInfo");
        assert!(json.contains("\"devices\""), "missing devices: {json}");
        assert!(
            json.contains("\"vram_total_bytes\""),
            "missing vram: {json}"
        );
        let back: GpuInfo = serde_json::from_str(&json).expect("deserialize GpuInfo");
        assert_eq!(back.devices.len(), 1);
        assert_eq!(back.devices[0].vram_total_bytes, Some(6_439_305_216));
    }
}
