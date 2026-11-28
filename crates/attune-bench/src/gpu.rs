//! GPU device discovery and the measured-VRAM-bandwidth block (M7 slice 3B).
//!
//! llama.cpp builds with a GPU backend can list their devices
//! (`llama-cli --list-devices`). The output is a handful of lines such as
//!
//! ```text
//!   Vulkan0: Intel(R) Graphics (RPL-P) (5934 MiB, 5150 MiB free)
//!   Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5921 MiB, 5153 MiB free)
//! ```
//!
//! The device id (`Vulkan1`) is what `-dev` expects, so [`pick_device`] picks
//! one deliberately instead of letting llama.cpp default to Vulkan0 (which on
//! a laptop can be the integrated GPU listed first).
//!
//! Everything that reads text is pure ([`parse_list_devices`],
//! [`pick_device`]) and unit-tested against canned output; only
//! [`list_devices`] spawns a process, reusing the bounded runner in
//! [`crate::llama`].

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::e2e::E2eResult;
use crate::llama::{run_once, stderr_tail};
use crate::BenchError;

/// Deadline for `--list-devices`. The probe only enumerates adapters, so a
/// healthy build answers in well under a second.
pub const LIST_DEVICES_TIMEOUT: Duration = Duration::from_secs(10);

/// One device as listed by llama.cpp: `(id, name, total MiB)`.
///
/// `id` is the `-dev` value (`Vulkan0`, `CUDA1`, …). `total` is the total
/// memory in MiB when the build reports it; `None` when the parenthesis is
/// absent or unparseable.
pub type DeviceEntry = (String, String, Option<u64>);

/// Parse `llama.cpp --list-devices` output into `(id, name, total MiB)`.
///
/// Unrecognised lines (headers, empty lines, chatter) are ignored rather than
/// failing the parse. The total is the *first* `N MiB` value inside the
/// trailing parenthesis; the free value that follows it is ignored.
pub fn parse_list_devices(text: &str) -> Vec<DeviceEntry> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((id, rest)) = line.split_once(':') else {
            continue;
        };
        let id = id.trim();
        if !is_device_id(id) {
            continue;
        }
        let rest = rest.trim();
        if rest.is_empty() {
            continue;
        }
        let (name, params) = match rest.rfind('(') {
            Some(idx) => (rest[..idx].trim(), Some(&rest[idx + 1..])),
            None => (rest, None),
        };
        if name.is_empty() {
            continue;
        }
        out.push((
            id.to_string(),
            name.to_string(),
            params.and_then(parse_total_mib),
        ));
    }
    out
}

/// A device id is one token of ASCII alphanumerics that starts with a letter:
/// `Vulkan1`, `CUDA0`, `Metal`, `BLAS`. That rejects `Available devices:` and
/// log prefixes like `ggml_vulkan:` without a hard-coded list of backends.
fn is_device_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric())
}

/// First `N MiB` value in a parenthesis body, e.g. `5921 MiB, 5153 MiB free`.
fn parse_total_mib(params: &str) -> Option<u64> {
    let fields: Vec<&str> = params.split_whitespace().collect();
    for pair in fields.windows(2) {
        if pair[1].trim_end_matches(',') != "MiB" {
            continue;
        }
        if let Ok(mib) = pair[0].trim_end_matches(',').parse::<u64>() {
            return Some(mib);
        }
    }
    None
}

/// True for adapters we do not want to run a bench on when a discrete GPU is
/// present: Intel graphics and anything labelled integrated.
fn is_integrated(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("intel")
        || lower.contains("integrated")
        || lower.contains("uhd graphics")
        || lower.contains("iris")
        || lower.contains("radeon graphics")
}

/// Pick the device id to pass to `-dev`.
///
/// A `preferred_name` (the NVIDIA/AMD adapter from `attune detect`) wins when
/// some listed name contains it (case-insensitive). Otherwise the discrete
/// adapter with the most reported memory wins, skipping Intel/integrated
/// devices; if every listed device is integrated, the largest is still used
/// rather than returning nothing.
pub fn pick_device(devices: &[DeviceEntry], preferred_name: Option<&str>) -> Option<String> {
    if let Some(preferred) = preferred_name.map(str::trim).filter(|s| !s.is_empty()) {
        let needle = preferred.to_ascii_lowercase();
        if let Some((id, _, _)) = devices
            .iter()
            .find(|(_, name, _)| name.to_ascii_lowercase().contains(&needle))
        {
            return Some(id.clone());
        }
    }

    let discrete: Vec<&DeviceEntry> = devices
        .iter()
        .filter(|(_, name, _)| !is_integrated(name))
        .collect();
    let pool: Vec<&DeviceEntry> = if discrete.is_empty() {
        devices.iter().collect()
    } else {
        discrete
    };
    pool.into_iter()
        .max_by_key(|(_, _, total)| total.unwrap_or(0))
        .map(|(id, _, _)| id.clone())
}

/// Run `<llama_bin> --list-devices` and parse the result.
///
/// A non-zero exit with no parseable device names is reported as a
/// [`BenchError::LlamaFailed`] carrying the stderr tail, so a wrong binary or
/// a missing GPU runtime surfaces as a clear error instead of an empty list.
pub fn list_devices(llama_bin: &Path) -> Result<Vec<DeviceEntry>, BenchError> {
    let output = run_once(llama_bin, &["--list-devices"], LIST_DEVICES_TIMEOUT)?;
    let mut devices = parse_list_devices(&output.stdout);
    if devices.is_empty() {
        devices = parse_list_devices(&output.stderr);
    }
    if devices.is_empty() && !output.success() {
        return Err(BenchError::LlamaFailed {
            cli: llama_bin.display().to_string(),
            args: "--list-devices".to_string(),
            code: output.exit_code,
            stderr: stderr_tail(&output.stderr),
        });
    }
    Ok(devices)
}

/// Measurement block for a full-offload GPU e2e run.
///
/// `effective_vram_gbps` is the decode throughput multiplied by the model
/// weight bytes, i.e. the bandwidth the decode actually streamed. The
/// theoretical value and eta are optional because `attune-bench` does not
/// depend on `attune-estimate`; the CLI fills them from the GPU name when the
/// table knows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuE2eResult {
    /// Backend device id (`Vulkan1`, `CUDA0`, …).
    pub device: String,
    /// Human-readable device name, when the listing carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    /// Installed build variant that ran: `"vulkan"` or `"cuda12"`.
    pub build: String,
    /// Decode throughput in tokens per second.
    pub decode_tps: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tps: f64,
    /// Model weight footprint in bytes (the GGUF file size).
    pub weight_bytes: u64,
    /// Effective VRAM bandwidth in decimal GB/s: `decode_tps * weight_bytes`.
    pub effective_vram_gbps: f64,
    /// Theoretical VRAM bandwidth in GB/s, when the GPU name is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theoretical_vram_gbps: Option<f64>,
    /// `effective_vram_gbps / theoretical_vram_gbps`, when both are known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_gpu: Option<f64>,
}

impl GpuE2eResult {
    /// Fold a completed e2e result into the GPU measurement block.
    pub fn from_e2e(
        e2e: &E2eResult,
        device: String,
        device_name: Option<String>,
        build: &str,
        weight_bytes: u64,
        theoretical_vram_gbps: Option<f64>,
    ) -> Self {
        let decode_tps = e2e.tok_s;
        let prefill_tps = if e2e.ttft_ms > 0.0 && e2e.ttft_ms.is_finite() {
            let seconds = e2e.ttft_ms / 1_000.0;
            let tps = e2e.prompt_tokens as f64 / seconds;
            if tps.is_finite() {
                tps
            } else {
                0.0
            }
        } else {
            0.0
        };
        let effective_vram_gbps = decode_tps * weight_bytes as f64 / 1e9;
        let eta_gpu = theoretical_vram_gbps
            .filter(|theoretical| *theoretical > 0.0 && effective_vram_gbps.is_finite())
            .map(|theoretical| effective_vram_gbps / theoretical);
        Self {
            device,
            device_name,
            build: build.to_string(),
            decode_tps,
            prefill_tps,
            weight_bytes,
            effective_vram_gbps,
            theoretical_vram_gbps,
            eta_gpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_list_devices -----------------------------------------------

    #[test]
    fn parses_vulkan_with_integrated_and_discrete_gpus() {
        let text = "\
Available devices:
  Vulkan0: Intel(R) Graphics (RPL-P) (5934 MiB, 5150 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5921 MiB, 5153 MiB free)
";
        let devices = parse_list_devices(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].0, "Vulkan0");
        assert_eq!(devices[0].1, "Intel(R) Graphics (RPL-P)");
        assert_eq!(devices[0].2, Some(5934));
        assert_eq!(devices[1].0, "Vulkan1");
        assert_eq!(devices[1].1, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(devices[1].2, Some(5921));
    }

    #[test]
    fn parses_a_single_cuda_device() {
        let text = "  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU (6140 MiB, 5075 MiB free)\n";
        let devices = parse_list_devices(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].0, "CUDA0");
        assert_eq!(devices[0].1, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(devices[0].2, Some(6140));
    }

    #[test]
    fn parse_ignores_garbage_lines() {
        let text = "\
Available devices:
not a device line
ggml_vulkan: backend loaded
  : missing id
  Vulkan1: GPU without memory
  Vulkan2: GPU with bad memory (not a number MiB)
";
        let devices = parse_list_devices(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(
            devices[0],
            ("Vulkan1".into(), "GPU without memory".into(), None)
        );
        assert_eq!(
            devices[1],
            ("Vulkan2".into(), "GPU with bad memory".into(), None)
        );
    }

    #[test]
    fn parse_returns_empty_for_empty_or_unrelated_output() {
        assert!(parse_list_devices("").is_empty());
        assert!(parse_list_devices("llama-cli: unrecognized option '--list-devices'\n").is_empty());
    }

    // ---- pick_device -------------------------------------------------------

    fn vulkan_pair() -> Vec<DeviceEntry> {
        vec![
            (
                "Vulkan0".into(),
                "Intel(R) Graphics (RPL-P)".into(),
                Some(5934),
            ),
            (
                "Vulkan1".into(),
                "NVIDIA GeForce RTX 4050 Laptop GPU".into(),
                Some(5921),
            ),
        ]
    }

    #[test]
    fn pick_prefers_the_named_discrete_gpu() {
        let picked = pick_device(&vulkan_pair(), Some("NVIDIA GeForce RTX 4050 Laptop GPU"));
        assert_eq!(picked.as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_matches_the_preferred_name_case_insensitively() {
        let picked = pick_device(&vulkan_pair(), Some("nvidia geforce rtx 4050"));
        assert_eq!(picked.as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_skips_intel_and_chooses_the_largest_discrete_adapter() {
        let devices = vec![
            ("Vulkan0".into(), "Intel(R) Graphics".into(), Some(8192)),
            (
                "Vulkan1".into(),
                "AMD Radeon RX 7800 XT".into(),
                Some(16384),
            ),
            (
                "Vulkan2".into(),
                "NVIDIA GeForce RTX 4060".into(),
                Some(8192),
            ),
        ];
        // No preferred name: the Intel adapter has tied/large memory but must
        // not win against a real discrete card.
        assert_eq!(pick_device(&devices, None).as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_falls_back_to_the_largest_when_everything_is_integrated() {
        let devices = vec![
            ("Vulkan0".into(), "Intel UHD Graphics".into(), Some(4096)),
            (
                "Vulkan1".into(),
                "AMD Radeon(TM) Graphics".into(),
                Some(8192),
            ),
        ];
        assert_eq!(pick_device(&devices, None).as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_returns_none_for_no_devices() {
        assert_eq!(pick_device(&[], None), None);
        assert_eq!(pick_device(&[], Some("NVIDIA")), None);
    }

    // ---- GpuE2eResult ------------------------------------------------------

    fn sample_e2e(tok_s: f64, ttft_ms: f64) -> E2eResult {
        E2eResult {
            engine: "11804".into(),
            model: "/models/qwen3-4b.gguf".into(),
            prompt_tokens: 512,
            eval_tokens: 128,
            ttft_ms,
            tpot_ms: if tok_s > 0.0 { 1_000.0 / tok_s } else { 0.0 },
            tok_s,
            ram_peak_bytes: 0,
            cpu_avg_pct: 0.0,
            power_avg_watts: None,
            elapsed_ms: 0.0,
        }
    }

    #[test]
    fn from_e2e_derives_effective_bandwidth_and_eta() {
        // 61.5 tok/s * 2.5 GB = 153.75 GB/s; 153.75 / 192 = 0.80.
        let e2e = sample_e2e(61.5, 250.0);
        let gpu = GpuE2eResult::from_e2e(
            &e2e,
            "Vulkan1".into(),
            Some("NVIDIA GeForce RTX 4050 Laptop GPU".into()),
            "vulkan",
            2_500_000_000,
            Some(192.0),
        );
        assert_eq!(gpu.device, "Vulkan1");
        assert_eq!(gpu.build, "vulkan");
        assert!((gpu.decode_tps - 61.5).abs() < 1e-9);
        assert!((gpu.prefill_tps - 2048.0).abs() < 1e-9);
        assert_eq!(gpu.weight_bytes, 2_500_000_000);
        assert!((gpu.effective_vram_gbps - 153.75).abs() < 1e-9);
        assert!((gpu.eta_gpu.expect("eta known") - 0.800_781_25).abs() < 1e-9);
    }

    #[test]
    fn from_e2e_keeps_eta_unknown_when_the_gpu_is_unknown() {
        let gpu = GpuE2eResult::from_e2e(
            &sample_e2e(10.0, 100.0),
            "Vulkan1".into(),
            None,
            "vulkan",
            1_000_000_000,
            None,
        );
        assert_eq!(gpu.theoretical_vram_gbps, None);
        assert_eq!(gpu.eta_gpu, None);
    }
}
