//! Which llama.cpp build(s) `attune setup --gpu` should install (M7).
//!
//! The CPU build is always kept so `--device cpu` keeps working. The decision
//! is pure: it only looks at the detected GPUs and the catalog pin, never at
//! the filesystem or the network.

use std::str::FromStr;

use attune_estimate::model::registry::LlamaPin;
use attune_hal::GpuDevice;

use crate::BackendError;

/// The `--gpu` value the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRequest {
    /// GPU build when a discrete GPU is present, CPU otherwise.
    Auto,
    /// CPU build only.
    Cpu,
    /// CPU + Vulkan builds.
    Vulkan,
    /// CPU + CUDA 12 builds, when the NVIDIA driver is new enough.
    Cuda,
}

impl FromStr for GpuRequest {
    type Err = BackendError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(GpuRequest::Auto),
            "cpu" => Ok(GpuRequest::Cpu),
            "vulkan" => Ok(GpuRequest::Vulkan),
            "cuda" => Ok(GpuRequest::Cuda),
            other => Err(BackendError::InvalidConfig(format!(
                "unknown --gpu value `{other}`: use auto, cpu, vulkan or cuda"
            ))),
        }
    }
}

/// The variants to install, always starting with `cpu`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantPlan {
    /// Build variants in install order; the first entry is always `cpu`.
    pub variants: Vec<String>,
    /// Why the plan chose what it chose, when it is worth telling the user.
    pub note: Option<String>,
}

fn has_asset(pin: &LlamaPin, os: &str, arch: &str, variant: &str) -> bool {
    !pin.assets_for(os, arch, variant).is_empty()
}

fn is_vendor(device: &GpuDevice, vendor: &str) -> bool {
    device.vendor.eq_ignore_ascii_case(vendor)
}

fn is_discrete(device: &GpuDevice) -> bool {
    is_vendor(device, "nvidia") || is_vendor(device, "amd")
}

/// A dotted numeric version (`551.61`) as comparable integer segments.
fn parse_version(value: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = value.trim().split('.').collect();
    if parts.is_empty() {
        return None;
    }
    parts.iter().map(|part| part.parse::<u64>().ok()).collect()
}

/// `Some(true)` when `found >= required`, `None` when either is unparseable.
///
/// Missing segments count as zero, so `551.6 == 551.60` and
/// `550.100 > 550.54`.
fn version_at_least(found: &str, required: &str) -> Option<bool> {
    let found = parse_version(found)?;
    let required = parse_version(required)?;
    let len = found.len().max(required.len());
    for index in 0..len {
        let found = found.get(index).copied().unwrap_or(0);
        let required = required.get(index).copied().unwrap_or(0);
        if found != required {
            return Some(found > required);
        }
    }
    Some(true)
}

fn cpu_plan(note: Option<String>) -> VariantPlan {
    VariantPlan {
        variants: vec!["cpu".to_string()],
        note,
    }
}

fn auto_plan(pin: &LlamaPin, gpus: &[GpuDevice], os: &str, arch: &str) -> VariantPlan {
    if gpus.iter().any(is_discrete) {
        if has_asset(pin, os, arch, "vulkan") {
            return VariantPlan {
                variants: vec!["cpu".to_string(), "vulkan".to_string()],
                note: None,
            };
        }
        return cpu_plan(Some(format!(
            "no Vulkan build for {os}/{arch}, CPU build only"
        )));
    }
    if gpus.iter().any(|device| is_vendor(device, "intel")) {
        return cpu_plan(Some(
            "Intel integrated GPU found, CPU build only".to_string(),
        ));
    }
    cpu_plan(Some("no discrete GPU found, CPU build only".to_string()))
}

fn cuda_plan(
    pin: &LlamaPin,
    gpus: &[GpuDevice],
    os: &str,
    arch: &str,
) -> Result<VariantPlan, BackendError> {
    let device = gpus
        .iter()
        .find(|device| is_vendor(device, "nvidia"))
        .ok_or_else(|| {
            BackendError::InvalidConfig("no NVIDIA GPU found; use --gpu vulkan".to_string())
        })?;
    let assets = pin.assets_for(os, arch, "cuda12");
    if assets.is_empty() {
        return Err(BackendError::InvalidConfig(format!(
            "no llama.cpp {} cuda12 build for {os}/{arch}; use --gpu vulkan",
            pin.tag
        )));
    }
    let required = assets.iter().find_map(|asset| asset.min_driver.as_deref());
    if let Some(required) = required {
        let found = device.driver.as_deref().unwrap_or("");
        match version_at_least(found, required) {
            Some(true) => {}
            Some(false) => {
                return Err(BackendError::InvalidConfig(format!(
                    "NVIDIA driver {found} is older than the CUDA build needs ({required}); use --gpu vulkan"
                )));
            }
            None => {
                return Err(BackendError::InvalidConfig(format!(
                    "could not read the NVIDIA driver version (the CUDA build needs {required}); use --gpu vulkan"
                )));
            }
        }
    }
    Ok(VariantPlan {
        variants: vec!["cpu".to_string(), "cuda12".to_string()],
        note: None,
    })
}

/// Choose the llama.cpp build variants to install.
///
/// The result always starts with `cpu`; the optional `note` explains a fallback.
pub fn choose_variants(
    request: GpuRequest,
    gpus: &[GpuDevice],
    pin: &LlamaPin,
    os: &str,
    arch: &str,
) -> Result<VariantPlan, BackendError> {
    match request {
        GpuRequest::Cpu => Ok(cpu_plan(None)),
        GpuRequest::Vulkan => {
            if !has_asset(pin, os, arch, "vulkan") {
                return Err(BackendError::InvalidConfig(format!(
                    "no llama.cpp {} vulkan build for {os}/{arch}; use --gpu cpu",
                    pin.tag
                )));
            }
            Ok(VariantPlan {
                variants: vec!["cpu".to_string(), "vulkan".to_string()],
                note: None,
            })
        }
        GpuRequest::Cuda => cuda_plan(pin, gpus, os, arch),
        GpuRequest::Auto => Ok(auto_plan(pin, gpus, os, arch)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attune_estimate::model::registry::LlamaAsset;

    fn asset(os: &str, arch: &str, variant: &str, min_driver: Option<&str>) -> LlamaAsset {
        LlamaAsset {
            os: os.to_string(),
            arch: arch.to_string(),
            variant: variant.to_string(),
            url: format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/b1/{variant}.zip"
            ),
            size: 1,
            sha256: "3".repeat(64),
            min_driver: min_driver.map(str::to_string),
        }
    }

    fn pin() -> LlamaPin {
        LlamaPin {
            tag: "b1".to_string(),
            assets: vec![
                asset("windows", "x86_64", "cpu", None),
                asset("windows", "x86_64", "vulkan", None),
                asset("windows", "x86_64", "cuda12", Some("551.61")),
                asset("windows", "x86_64", "cuda12", Some("551.61")),
            ],
        }
    }

    fn gpu(vendor: &str, driver: Option<&str>) -> GpuDevice {
        GpuDevice {
            vendor: vendor.to_string(),
            name: format!("{vendor} card"),
            vram_total_bytes: Some(6_000_000_000),
            vram_free_bytes: Some(5_000_000_000),
            driver: driver.map(str::to_string),
            source: "test".to_string(),
        }
    }

    #[test]
    fn gpu_request_parsing_is_case_insensitive() {
        for (text, expected) in [
            ("AUTO", GpuRequest::Auto),
            ("Cpu", GpuRequest::Cpu),
            (" vulkan ", GpuRequest::Vulkan),
            ("CuDa", GpuRequest::Cuda),
        ] {
            assert_eq!(text.parse::<GpuRequest>().expect("parse"), expected);
        }
    }

    #[test]
    fn unknown_gpu_request_lists_every_value() {
        let err = "tensor"
            .parse::<GpuRequest>()
            .expect_err("unknown")
            .to_string();
        for value in ["auto", "cpu", "vulkan", "cuda"] {
            assert!(err.contains(value), "{err}");
        }
    }

    #[test]
    fn cpu_request_installs_only_the_cpu_build() {
        let plan = choose_variants(
            GpuRequest::Cpu,
            &[gpu("nvidia", Some("561.00"))],
            &pin(),
            "windows",
            "x86_64",
        )
        .expect("plan");
        assert_eq!(plan.variants, vec!["cpu"]);
        assert_eq!(plan.note, None);
    }

    #[test]
    fn vulkan_request_needs_a_vulkan_asset() {
        let plan =
            choose_variants(GpuRequest::Vulkan, &[], &pin(), "windows", "x86_64").expect("plan");
        assert_eq!(plan.variants, vec!["cpu", "vulkan"]);

        let err = choose_variants(GpuRequest::Vulkan, &[], &pin(), "macos", "aarch64")
            .expect_err("no asset")
            .to_string();
        assert!(err.contains("macos/aarch64"), "{err}");
    }

    #[test]
    fn cuda_accepts_a_new_enough_driver() {
        for driver in ["561.00", "551.61", "600.10"] {
            let plan = choose_variants(
                GpuRequest::Cuda,
                &[gpu("nvidia", Some(driver))],
                &pin(),
                "windows",
                "x86_64",
            )
            .expect("plan");
            assert_eq!(plan.variants, vec!["cpu", "cuda12"]);
        }
    }

    #[test]
    fn cuda_rejects_a_driver_below_the_minimum() {
        for driver in ["551.60", "550.100"] {
            let err = choose_variants(
                GpuRequest::Cuda,
                &[gpu("nvidia", Some(driver))],
                &pin(),
                "windows",
                "x86_64",
            )
            .expect_err("too old")
            .to_string();
            assert!(err.contains(driver), "{err}");
            assert!(err.contains("551.61"), "{err}");
            assert!(err.contains("--gpu vulkan"), "{err}");
        }
    }

    #[test]
    fn cuda_orders_dotted_versions_numerically() {
        assert_eq!(version_at_least("561.00", "551.61"), Some(true));
        assert_eq!(version_at_least("550.100", "550.54"), Some(true));
        assert_eq!(version_at_least("550.54", "550.100"), Some(false));
        assert_eq!(version_at_least("551.61.0", "551.61"), Some(true));
        assert_eq!(version_at_least("abc", "551.61"), None);
    }

    #[test]
    fn cuda_needs_an_nvidia_gpu() {
        for gpus in [
            Vec::new(),
            vec![gpu("amd", Some("1.0")), gpu("intel", None)],
        ] {
            let err = choose_variants(GpuRequest::Cuda, &gpus, &pin(), "windows", "x86_64")
                .expect_err("no nvidia")
                .to_string();
            assert!(err.contains("no NVIDIA GPU found"), "{err}");
            assert!(err.contains("--gpu vulkan"), "{err}");
        }
    }

    #[test]
    fn cuda_rejects_an_unknown_driver() {
        let err = choose_variants(
            GpuRequest::Cuda,
            &[gpu("nvidia", None)],
            &pin(),
            "windows",
            "x86_64",
        )
        .expect_err("unknown driver")
        .to_string();
        assert!(err.contains("driver"), "{err}");
        assert!(err.contains("551.61"), "{err}");
        assert!(err.contains("--gpu vulkan"), "{err}");
    }

    #[test]
    fn cuda_without_a_cuda_asset_suggests_vulkan() {
        let pin = LlamaPin {
            tag: "b1".to_string(),
            assets: vec![asset("windows", "x86_64", "vulkan", None)],
        };
        let err = choose_variants(
            GpuRequest::Cuda,
            &[gpu("nvidia", Some("561.00"))],
            &pin,
            "windows",
            "x86_64",
        )
        .expect_err("no cuda")
        .to_string();
        assert!(err.contains("cuda12"), "{err}");
        assert!(err.contains("--gpu vulkan"), "{err}");
    }

    #[test]
    fn cuda_without_a_min_driver_skips_the_check() {
        let pin = LlamaPin {
            tag: "b1".to_string(),
            assets: vec![asset("windows", "x86_64", "cuda12", None)],
        };
        let plan = choose_variants(
            GpuRequest::Cuda,
            &[gpu("nvidia", None)],
            &pin,
            "windows",
            "x86_64",
        )
        .expect("plan");
        assert_eq!(plan.variants, vec!["cpu", "cuda12"]);
    }

    #[test]
    fn auto_picks_vulkan_for_a_discrete_gpu() {
        for vendor in ["nvidia", "amd", "AMD"] {
            let plan = choose_variants(
                GpuRequest::Auto,
                &[gpu(vendor, Some("561.00"))],
                &pin(),
                "windows",
                "x86_64",
            )
            .expect("plan");
            assert_eq!(plan.variants, vec!["cpu", "vulkan"], "{vendor}");
            assert_eq!(plan.note, None);
        }
    }

    #[test]
    fn auto_keeps_cpu_for_intel_only() {
        let plan = choose_variants(
            GpuRequest::Auto,
            &[gpu("intel", None)],
            &pin(),
            "windows",
            "x86_64",
        )
        .expect("plan");
        assert_eq!(plan.variants, vec!["cpu"]);
        let note = plan.note.expect("note");
        assert!(note.contains("Intel"), "{note}");
        assert!(note.contains("CPU build only"), "{note}");
    }

    #[test]
    fn auto_keeps_cpu_without_a_gpu() {
        let plan =
            choose_variants(GpuRequest::Auto, &[], &pin(), "windows", "x86_64").expect("plan");
        assert_eq!(plan.variants, vec!["cpu"]);
        assert_eq!(
            plan.note.as_deref(),
            Some("no discrete GPU found, CPU build only")
        );
    }

    #[test]
    fn auto_falls_back_when_this_platform_has_no_vulkan_asset() {
        let plan = choose_variants(
            GpuRequest::Auto,
            &[gpu("nvidia", Some("561.00"))],
            &pin(),
            "macos",
            "aarch64",
        )
        .expect("plan");
        assert_eq!(plan.variants, vec!["cpu"]);
        assert!(plan.note.expect("note").contains("CPU build only"));
    }
}
