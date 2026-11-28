//! Profile validation, spawn and health-check lifecycle (M5 slice 1).
//!
//! The launcher is deliberately thin around the backend: it turns a profile
//! file plus live hardware facts into a running server, and it owns the policy
//! decisions (warn on fingerprint mismatch, auto-apply the RAM fallback,
//! `--force` skips validation). Everything process-level lives in
//! [`attune_backend`].

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use attune_backend::{
    ensure_model, list_devices, pick_device as pick_gpu_device, resolve_llama_server, Backend,
    LlamaServerBackend, Paths, ServerGuard, ENV_LLAMA_SERVER,
};
use attune_estimate::model::registry::TRUSTED_PUBLISHERS;
use attune_hal::{GpuDevice, HardwareInfo};
use attune_profile::{validate, ModelSource, Profile, WarningKind};
use serde::Serialize;

use crate::server::is_loopback_host;
use crate::ServeError;

/// Where `attune serve` should run a profile.
///
/// A CLI-side mirror of the values accepted by `--device`; a plain enum keeps
/// clap out of this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServeDevice {
    /// Use the device the profile was tuned for.
    #[default]
    Auto,
    /// Force CPU execution (a GPU profile gets `n_gpu_layers = 0`).
    Cpu,
    /// Force GPU execution (a CPU-tuned profile is refused).
    Gpu,
}

/// The build/device decision for one launch, before any I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServePlan {
    /// `true` when llama.cpp runs device layers and needs a `-dev` id.
    pub gpu: bool,
    /// llama.cpp build variant: `cpu`, `vulkan` or `cuda12`.
    pub variant: String,
    /// Emit the "running a GPU-tuned profile on the CPU" note.
    pub cpu_override: bool,
}

/// The note printed when `--device cpu` forces a GPU-tuned profile onto the CPU.
pub const CPU_OVERRIDE_NOTE: &str =
    "running a GPU-tuned profile on the CPU; `attune tune --device cpu --save` gives a CPU-tuned one";

/// Error text when a GPU profile has no GPU llama.cpp build to run on.
pub const NO_GPU_BUILD: &str = "this profile runs on the GPU but no GPU llama.cpp build is \
     installed. Fix: `attune setup --gpu auto`, or serve on the CPU with `--device cpu`";

/// Error text when `--device gpu` is asked of a CPU-tuned profile.
pub const CPU_PROFILE_ON_GPU: &str = "this profile was tuned for the CPU; retune with \
     `attune tune --model <...> --device gpu --save <...>` or `attune run <model> --device gpu`";

impl ServeDevice {
    /// The device this flag asks for, given the profile's own placement.
    fn resolves_to_gpu(self, profile_device: &str) -> bool {
        match self {
            ServeDevice::Auto => profile_device == "gpu",
            ServeDevice::Cpu => false,
            ServeDevice::Gpu => true,
        }
    }
}

/// Pure: decide which build/device a launch uses.
///
/// `profile_device` is [`Profile::device`] (`"gpu"` or `"cpu"`),
/// `profile_variant` is `engine.variant`, and `installed` lists the GPU build
/// variants present in the attune home dir (or just the chosen variant when the
/// user passed an explicit binary). Returns the exact user-facing error text on
/// a refusal so the caller can wrap it in [`ServeError::InvalidConfig`].
pub fn plan_serve(
    profile_device: &str,
    profile_variant: &str,
    requested: ServeDevice,
    installed: &[String],
) -> Result<ServePlan, String> {
    let profile_is_gpu = profile_device == "gpu";
    if !profile_is_gpu && requested.resolves_to_gpu(profile_device) {
        return Err(CPU_PROFILE_ON_GPU.to_string());
    }
    if profile_is_gpu && requested == ServeDevice::Cpu {
        return Ok(ServePlan {
            gpu: false,
            variant: "cpu".to_string(),
            cpu_override: true,
        });
    }
    if !profile_is_gpu {
        return Ok(ServePlan {
            gpu: false,
            variant: "cpu".to_string(),
            cpu_override: false,
        });
    }
    let variant =
        choose_gpu_variant(profile_variant, installed).ok_or_else(|| NO_GPU_BUILD.to_string())?;
    Ok(ServePlan {
        gpu: true,
        variant,
        cpu_override: false,
    })
}

/// Prefer the profile's own build, then the other GPU build.
fn choose_gpu_variant(profile_variant: &str, installed: &[String]) -> Option<String> {
    let has = |variant: &str| installed.iter().any(|entry| entry == variant);
    if has(profile_variant) {
        return Some(profile_variant.to_string());
    }
    let fallbacks: &[&str] = match profile_variant {
        "cuda12" => &["vulkan"],
        "vulkan" => &["cuda12"],
        _ => &["vulkan", "cuda12"],
    };
    fallbacks
        .iter()
        .find(|variant| has(variant))
        .map(|variant| (*variant).to_string())
}

/// The build label reported for an explicit GPU binary, whose real variant we
/// cannot introspect. A GPU profile records `vulkan` or `cuda12`; anything else
/// is reported as `vulkan`.
fn gpu_build_label(profile_variant: &str) -> &str {
    if profile_variant == "cuda12" {
        "cuda12"
    } else {
        "vulkan"
    }
}

/// Parameters for one `attune serve` launch.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Path to the profile JSON file.
    pub profile: PathBuf,
    /// Explicit `llama-server` path (`--llama-server`); `None` falls back to
    /// `ATTUNE_LLAMA_SERVER` and then the profile's build (CPU: `PATH`).
    pub llama_server: Option<PathBuf>,
    /// Where to run the profile (`--device`).
    pub device: ServeDevice,
    /// Interface to bind the upstream server to.
    pub host: String,
    /// Upstream port.
    pub port: u16,
    /// How long to wait for `/health` before giving up.
    pub timeout: Duration,
    /// Skip profile validation (fingerprint/RAM warnings) and launch as-is.
    pub force: bool,
    /// Whether a missing model may be downloaded from `model.source`.
    pub download: DownloadConsent,
    /// Download progress `(bytes_done, bytes_total)`.
    pub progress: fn(u64, u64),
}

/// May `serve` download a profile's missing model?
#[derive(Debug, Clone, Copy)]
pub enum DownloadConsent {
    /// `--yes`: download without asking.
    Yes,
    /// Non-interactive without `--yes`: never download, explain how.
    No,
    /// Interactive: ask with `(size_bytes, description)`.
    Ask(fn(u64, &str) -> bool),
}

/// True when `path` is relative to the attune home dir.
fn needs_home(path: &str) -> bool {
    path.starts_with(attune_profile::HOME_PREFIX)
}

/// Point `profile.model.path` at a real file: expand a `~/.attune/` path
/// against `home` and, when the file is missing and the profile has a
/// `source`, download it under `consent`.
///
/// A profile's `model.path` is only ever read. Downloads always go to
/// `<home>/models/<repo>/<file>` from the pinned source, so a profile cannot
/// make `serve` write outside the attune home.
pub fn resolve_model_file(
    profile: &mut Profile,
    home: &Path,
    consent: DownloadConsent,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), ServeError> {
    let path = expand_model_path(&profile.model.path, home)?;
    match profile.model.source.clone() {
        Some(source) => {
            validate_model_source(&source)?;
            if std::fs::metadata(&path).is_ok_and(|m| m.len() == source.size) {
                // A local file the profile points at is read in place and is
                // never replaced by a download.
                profile.model.path = path.display().to_string();
                return Ok(());
            }

            let download_home = if home.as_os_str().is_empty() {
                attune_fetch::home_dir()?
            } else {
                home.to_path_buf()
            };
            let dest = download_destination(&source, &download_home)?;
            if std::fs::metadata(&dest).is_ok_and(|m| m.len() == source.size) {
                profile.model.path = dest.display().to_string();
                return Ok(());
            }

            let size_gb = source.size as f64 / 1e9;
            let url = source.url();
            let what = format!(
                "{} ({size_gb:.1} GB) from {url} to {}",
                source.file,
                dest.display()
            );
            if matches!(consent, DownloadConsent::No) {
                return Err(ServeError::InvalidConfig(format!(
                    "model file `{}` is missing or has the wrong size; pass --yes to download \
                     {size_gb:.1} GB from {url} to {}",
                    path.display(),
                    dest.display()
                )));
            }
            let mut confirm = |size: u64| match consent {
                // `--yes` still shows exactly what will be fetched and where.
                DownloadConsent::Yes => {
                    eprintln!("Download {what}?");
                    true
                }
                DownloadConsent::No => false,
                DownloadConsent::Ask(ask) => ask(size, &what),
            };
            ensure_model(&dest, &source, &mut confirm, progress)?;
            profile.model.path = dest.display().to_string();
        }
        None if !path.exists() => {
            return Err(ServeError::InvalidConfig(format!(
                "model file `{}` not found",
                path.display()
            )));
        }
        None => {
            profile.model.path = path.display().to_string();
        }
    }
    Ok(())
}

/// Expand `model.path`, rejecting parent/absolute components inside the
/// portable `~/.attune/` form. Absolute local paths are read-only inputs and
/// are kept as-is.
fn expand_model_path(path: &str, home: &Path) -> Result<PathBuf, ServeError> {
    if let Some(rest) = path.strip_prefix(attune_profile::HOME_PREFIX) {
        let unsafe_component = Path::new(rest).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
        if unsafe_component {
            return Err(ServeError::InvalidConfig(format!(
                "model.path `{path}` must stay under `~/.attune/` (no `..` or absolute components)"
            )));
        }
    }
    Ok(attune_profile::expand_home(path, home))
}

/// Validate the pinned source: exactly `owner/name`, a safe file name and a
/// publisher on the catalog's trusted list.
fn validate_model_source(source: &ModelSource) -> Result<(), ServeError> {
    let (owner, name) = source.repo.split_once('/').ok_or_else(|| {
        ServeError::InvalidConfig(format!(
            "model source repo `{}` is invalid: expected exactly `owner/name`",
            source.repo
        ))
    })?;
    if name.contains('/') || !safe_component(owner) || !safe_component(name) {
        return Err(ServeError::InvalidConfig(format!(
            "model source repo `{}` is invalid: expected exactly `owner/name`",
            source.repo
        )));
    }
    if !safe_file_name(&source.file) {
        return Err(ServeError::InvalidConfig(format!(
            "model source file `{}` is invalid: expected a single safe file name",
            source.file
        )));
    }
    if !TRUSTED_PUBLISHERS.contains(&owner) {
        return Err(ServeError::InvalidConfig(format!(
            "model source repo `{}` comes from untrusted publisher `{owner}`; refusing to download",
            source.repo
        )));
    }
    Ok(())
}

fn safe_component(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && !part.starts_with('.')
        && !part.contains('/')
        && !part.contains('\\')
        && !part.contains(':')
        && !part.contains('\0')
}

fn safe_file_name(name: &str) -> bool {
    safe_component(name)
}

/// The only writable destination: `<home>/models/<repo>/<file>`.
fn download_destination(source: &ModelSource, home: &Path) -> Result<PathBuf, ServeError> {
    validate_model_source(source)?;
    let models = home.join("models");
    let dest = attune_profile::expand_home(&source.home_path(), home);
    let models_canon = canonical_with_missing_tail(&models)?;
    let dest_canon = canonical_with_missing_tail(&dest)?;
    if !dest_canon.starts_with(&models_canon) {
        return Err(ServeError::InvalidConfig(format!(
            "refusing to download to `{}`: it is outside `{}`",
            dest.display(),
            models.display()
        )));
    }
    Ok(dest)
}

/// Canonicalize the existing prefix of `path` and append the missing tail, so
/// the containment check also catches a symlinked parent.
fn canonical_with_missing_tail(path: &Path) -> Result<PathBuf, ServeError> {
    let mut current = path.to_path_buf();
    let mut tail = Vec::new();
    loop {
        match std::fs::canonicalize(&current) {
            Ok(mut base) => {
                for part in tail.iter().rev() {
                    base.push(part);
                }
                return Ok(base);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(name) = current.file_name().map(std::ffi::OsString::from) else {
                    return Err(ServeError::InvalidConfig(format!(
                        "cannot resolve `{}`: {error}",
                        path.display()
                    )));
                };
                tail.push(name);
                if !current.pop() {
                    return Err(ServeError::InvalidConfig(format!(
                        "cannot resolve `{}`: {error}",
                        path.display()
                    )));
                }
            }
            Err(error) => {
                return Err(ServeError::InvalidConfig(format!(
                    "cannot resolve `{}`: {error}",
                    path.display()
                )))
            }
        }
    }
}

/// A launched server plus the metadata the CLI needs to print.
///
/// Dropping this value drops the [`ServerGuard`], which kills and reaps the
/// child.
pub struct Serving {
    /// RAII handle to the running `llama-server`.
    pub guard: ServerGuard,
    /// The profile that was actually used (after any fallback was applied).
    pub profile: Profile,
    /// `host:port` of the upstream server.
    pub upstream: String,
    /// Full `http://.../health` URL that was polled.
    pub health_url: String,
    /// How long the health wait took.
    pub health_elapsed: Duration,
    /// Validation and argv warnings to surface to the user.
    pub warnings: Vec<String>,
    /// `cpu` or `gpu`, matching the device that actually ran.
    pub device: String,
    /// llama.cpp build variant that ran (`cpu`, `vulkan`, `cuda12`).
    pub build: String,
    /// llama.cpp device id passed to `-dev` (`Vulkan1`, `CUDA0`), when GPU.
    pub device_id: Option<String>,
    /// Human-readable GPU name from `--list-devices`, when GPU.
    pub device_name: Option<String>,
}

/// One-line JSON ready message (`serve --json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadyReport {
    /// Always `ready`; a failed launch returns an error instead.
    pub status: &'static str,
    /// `host:port` of the running upstream.
    pub upstream: String,
    /// Health endpoint that answered.
    pub health_url: String,
    /// Health wait duration in milliseconds.
    pub health_ms: u64,
    /// Model path recorded in the profile.
    pub model: String,
    /// Effective context window after any RAM fallback.
    pub n_ctx: u32,
    /// Non-fatal warnings emitted during preparation.
    pub warnings: Vec<String>,
    /// `cpu` or `gpu`.
    pub device: String,
    /// llama.cpp build variant that ran (`cpu`, `vulkan`, `cuda12`).
    pub build: String,
    /// llama.cpp device id passed to `-dev`, when running on the GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_device: Option<String>,
    /// Human-readable GPU name, when running on the GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_name: Option<String>,
}

impl Serving {
    /// Build the stable JSON shape printed by `attune serve --json`.
    pub fn ready_report(&self) -> ReadyReport {
        ReadyReport {
            status: "ready",
            upstream: self.upstream.clone(),
            health_url: self.health_url.clone(),
            health_ms: u64::try_from(self.health_elapsed.as_millis()).unwrap_or(u64::MAX),
            model: self.profile.model.path.clone(),
            n_ctx: self.profile.load_config.n_ctx,
            warnings: self.warnings.clone(),
            device: self.device.clone(),
            build: self.build.clone(),
            gpu_device: self.device_id.clone(),
            gpu_name: self.device_name.clone(),
        }
    }

    /// `CPU` or `GPU (<build>, <id> <name>)` for the human ready line.
    pub fn device_label(&self) -> String {
        if self.device != "gpu" {
            return "CPU".to_string();
        }
        let mut detail = format!(
            "{}, {}",
            self.build,
            self.device_id.as_deref().unwrap_or("?")
        );
        if let Some(name) = self.device_name.as_deref() {
            detail.push(' ');
            detail.push_str(name);
        }
        format!("GPU ({detail})")
    }
}

/// The result of validating a profile against the current machine.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedProfile {
    /// The profile to launch (possibly with a smaller `n_ctx`).
    pub profile: Profile,
    /// Human-readable warnings, including the fallback application message.
    pub warnings: Vec<String>,
    /// The `n_ctx` that was applied, when a fallback was used.
    pub applied_fallback_n_ctx: Option<u32>,
}

/// Apply validation policy to a profile.
///
/// * `force` skips validation entirely.
/// * A fingerprint mismatch is only a warning.
/// * A RAM shortfall applies `fallbacks.if_ram_insufficient.n_ctx` when that
///   fallback is smaller than the recorded `n_ctx`; otherwise it stays a
///   warning (the user may free RAM or pick a smaller quant).
///
/// Pure apart from cloning: the CLI logs [`PreparedProfile::warnings`].
pub fn prepare_profile(
    profile: &Profile,
    current_fingerprint: &str,
    ram_avail_bytes: u64,
    force: bool,
) -> PreparedProfile {
    let mut prepared = PreparedProfile {
        profile: profile.clone(),
        warnings: Vec::new(),
        applied_fallback_n_ctx: None,
    };
    if force {
        return prepared;
    }

    for warning in validate(profile, current_fingerprint, ram_avail_bytes) {
        prepared.warnings.push(warning.message);
        if warning.kind != WarningKind::RamInsufficient {
            continue;
        }
        let Some(fallback) = profile.fallbacks.if_ram_insufficient.as_ref() else {
            continue;
        };
        let original = prepared.profile.load_config.n_ctx;
        if fallback.n_ctx < original {
            prepared.profile.load_config.n_ctx = fallback.n_ctx;
            prepared.applied_fallback_n_ctx = Some(fallback.n_ctx);
            prepared.warnings.push(format!(
                "applied fallbacks.if_ram_insufficient: n_ctx {original} -> {}",
                fallback.n_ctx
            ));
        }
    }
    prepared
}

/// The readable hardware fingerprint for a CPU profile, using the same
/// `brand|physical_cores|simd|total_ram_bytes` format the M4 exporter writes.
pub fn fingerprint_of(
    brand: &str,
    physical_cores: u32,
    simd: &str,
    total_ram_bytes: u64,
) -> String {
    fingerprint_of_gpu(brand, physical_cores, simd, total_ram_bytes, None)
}

/// Like [`fingerprint_of`], but appends the GPU name and rounded VRAM for GPU
/// profiles.
///
/// Delegates to [`attune_tune::export::hardware_fingerprint_with_gpu`] so a
/// profile saved by `attune tune --device gpu` compares byte-for-byte. CPU
/// profiles pass `None` and keep the four-part form unchanged.
pub fn fingerprint_of_gpu(
    brand: &str,
    physical_cores: u32,
    simd: &str,
    total_ram_bytes: u64,
    gpu: Option<(&str, u64)>,
) -> String {
    attune_tune::export::hardware_fingerprint_with_gpu(
        brand,
        physical_cores,
        simd,
        total_ram_bytes,
        gpu,
    )
}

/// The GPU `(name, total VRAM bytes)` a GPU profile was tuned on.
///
/// Matches the profile's recorded `load_config.gpu_name` against the detected
/// adapters (case-insensitive substring, like the tune path); otherwise uses the
/// adapter with the most VRAM. Returns `None` when no GPU is visible, which
/// makes the fingerprint compare as CPU-only and warn (correctly) on a machine
/// without that GPU.
fn gpu_identity<'a>(devices: &'a [GpuDevice], preferred: Option<&str>) -> Option<(&'a str, u64)> {
    if let Some(preferred) = preferred.map(str::trim).filter(|name| !name.is_empty()) {
        let needle = preferred.to_ascii_lowercase();
        if let Some(device) = devices
            .iter()
            .find(|device| device.name.to_ascii_lowercase().contains(&needle))
        {
            return Some((&device.name, device.vram_total_bytes.unwrap_or(0)));
        }
    }
    devices
        .iter()
        .filter_map(|device| {
            device
                .vram_total_bytes
                .map(|vram| (device.name.as_str(), vram))
        })
        .max_by_key(|(_, vram)| *vram)
        .or_else(|| {
            devices
                .first()
                .map(|d| (d.name.as_str(), d.vram_total_bytes.unwrap_or(0)))
        })
}

/// Read, validate, resolve, spawn and health-check a profile.
///
/// On any error after spawn the [`ServerGuard`] is dropped, so a health failure
/// never leaves `llama-server` running.
pub fn launch(options: &ServeOptions) -> Result<Serving, ServeError> {
    let host = options.host.trim();
    if host.is_empty() {
        return Err(ServeError::InvalidConfig(
            "--host must not be empty".to_string(),
        ));
    }
    check_upstream_host(host)?;
    if options.port == 0 {
        return Err(ServeError::InvalidConfig(
            "--port must be between 1 and 65535".to_string(),
        ));
    }
    if options.timeout.is_zero() {
        return Err(ServeError::InvalidConfig(
            "--timeout-s must be greater than zero".to_string(),
        ));
    }

    let raw =
        std::fs::read_to_string(&options.profile).map_err(|source| ServeError::ReadProfile {
            path: options.profile.display().to_string(),
            source,
        })?;
    let mut profile = Profile::from_json_str(&raw)?;
    if profile.model.path.trim().is_empty() {
        return Err(ServeError::InvalidConfig(
            "profile has an empty model.path; run `attune tune --save` first".to_string(),
        ));
    }
    let profile_is_gpu = profile.device() == "gpu";
    // Only a `~/.attune/` path needs the home dir; an absolute path must keep
    // working where HOME/USERPROFILE is unset (services).
    let home = if needs_home(&profile.model.path) {
        attune_fetch::home_dir().map_err(|err| ServeError::InvalidConfig(err.to_string()))?
    } else {
        PathBuf::new()
    };
    let progress = options.progress;
    resolve_model_file(&mut profile, &home, options.download, &mut |done, total| {
        progress(done, total)
    })?;

    let explicit = explicit_llama_server(options);
    let install_home = attune_fetch::home_dir().ok();
    let (plan, binary) = resolve_launch_binary(
        &profile,
        options.device,
        explicit.as_deref(),
        install_home.as_deref(),
    )?;

    apply_device_plan(&mut profile, &plan);

    // `--force` means "do not validate against this machine", so skip the
    // (not free) hardware probe entirely: no fingerprint is needed and a HAL
    // failure must not block a forced launch.
    let (current_fingerprint, ram_available) = if options.force {
        (String::new(), 0)
    } else {
        let hardware = HardwareInfo::detect()?;
        let gpu = if profile_is_gpu {
            gpu_identity(
                &hardware.gpu.devices,
                profile.load_config.gpu_name.as_deref(),
            )
        } else {
            None
        };
        let fingerprint = fingerprint_of_gpu(
            &hardware.cpu.brand,
            hardware.cpu.physical_cores,
            &hardware.cpu.simd,
            hardware.memory.total_bytes,
            gpu,
        );
        (fingerprint, hardware.memory.available_bytes)
    };
    let mut prepared =
        prepare_profile(&profile, &current_fingerprint, ram_available, options.force);
    if plan.cpu_override {
        prepared.warnings.push(CPU_OVERRIDE_NOTE.to_string());
    }

    let (device_id, device_name) = if plan.gpu {
        let devices = list_devices(&binary)?;
        let id = pick_gpu_device(&devices, profile.load_config.gpu_name.as_deref())
            .ok_or_else(|| no_gpu_device_error(&binary))?;
        let name = devices
            .iter()
            .find(|(entry_id, _, _)| *entry_id == id)
            .map(|(_, name, _)| name.clone());
        (Some(id), name)
    } else {
        (None, None)
    };

    let paths = Paths::new(binary);
    let backend = LlamaServerBackend;
    let guard = backend.spawn_with_device(
        &prepared.profile,
        &paths,
        host,
        options.port,
        device_id.as_deref(),
    )?;

    let health_url = health_url_for(host, options.port);
    let health_elapsed = match backend.health_wait(&health_url, options.timeout) {
        Ok(elapsed) => elapsed,
        Err(error) => {
            // Explicit drop keeps the "health failure kills the child"
            // invariant readable at the call site.
            drop(guard);
            return Err(error.into());
        }
    };

    let mut warnings = prepared.warnings;
    warnings.extend_from_slice(guard.warnings());

    Ok(Serving {
        guard,
        profile: prepared.profile,
        upstream: format!("{host}:{}", options.port),
        health_url,
        health_elapsed,
        warnings,
        device: if plan.gpu { "gpu" } else { "cpu" }.to_string(),
        build: plan.variant,
        device_id,
        device_name,
    })
}

/// Apply a plan to an in-memory profile.
///
/// A GPU-tuned profile forced onto the CPU loses its offload, so the emitted
/// argv is CPU-only (`-ngl 0`, no `--fit`/`-np`/`-dev`).
fn apply_device_plan(profile: &mut Profile, plan: &ServePlan) {
    if plan.cpu_override {
        profile.load_config.n_gpu_layers = 0;
    }
}

/// The explicit binary from `--llama-server` or `ATTUNE_LLAMA_SERVER`, blank
/// values treated as unset.
fn explicit_llama_server(options: &ServeOptions) -> Option<PathBuf> {
    let usable =
        |path: &PathBuf| !path.as_os_str().is_empty() && !path.to_string_lossy().trim().is_empty();
    options
        .llama_server
        .as_ref()
        .filter(|path| usable(path))
        .cloned()
        .or_else(|| {
            std::env::var_os(ENV_LLAMA_SERVER)
                .map(PathBuf::from)
                .filter(|path| usable(path))
        })
}

/// GPU build variants installed in the attune home, in preference order.
fn installed_gpu_builds(home: &std::path::Path) -> Vec<(String, PathBuf)> {
    let mut builds = Vec::new();
    for variant in ["vulkan", "cuda12"] {
        if let Some(path) = attune_fetch::find_installed_variant(home, "llama-server", variant) {
            builds.push((variant.to_string(), path));
        }
    }
    builds
}

/// Pick the binary and the build/device plan for one launch.
///
/// An explicit flag/env binary always wins (the user chose it); otherwise a GPU
/// profile uses its own installed build, then the other GPU build, then fails
/// with [`NO_GPU_BUILD`]. CPU profiles keep the M5 resolution untouched.
fn resolve_launch_binary(
    profile: &Profile,
    requested: ServeDevice,
    explicit: Option<&std::path::Path>,
    home: Option<&std::path::Path>,
) -> Result<(ServePlan, PathBuf), ServeError> {
    let profile_device = profile.device();

    if let Some(binary) = explicit {
        let installed = if profile_device == "gpu" {
            vec![gpu_build_label(&profile.engine.variant).to_string()]
        } else {
            Vec::new()
        };
        let plan = plan_serve(
            profile_device,
            &profile.engine.variant,
            requested,
            &installed,
        )
        .map_err(ServeError::InvalidConfig)?;
        let binary = resolve_llama_server(Some(binary))?;
        return Ok((plan, binary));
    }

    // CPU profiles (and a GPU profile forced to the CPU) use the CPU build and
    // the unchanged `--llama-server`/env/PATH/installed precedence.
    if profile_device != "gpu" || requested == ServeDevice::Cpu {
        let plan = plan_serve(profile_device, &profile.engine.variant, requested, &[])
            .map_err(ServeError::InvalidConfig)?;
        let binary = resolve_llama_server(None)?;
        return Ok((plan, binary));
    }

    let builds = home.map(installed_gpu_builds).unwrap_or_default();
    let variants: Vec<String> = builds.iter().map(|(variant, _)| variant.clone()).collect();
    let plan = plan_serve(
        profile_device,
        &profile.engine.variant,
        requested,
        &variants,
    )
    .map_err(ServeError::InvalidConfig)?;
    let binary = builds
        .iter()
        .find(|(variant, _)| *variant == plan.variant)
        .map(|(_, path)| path.clone())
        .ok_or_else(|| {
            ServeError::InvalidConfig(format!(
                "the `{}` llama.cpp build vanished while preparing the launch; rerun \
                 `attune setup --gpu auto`",
                plan.variant
            ))
        })?;
    Ok((plan, binary))
}

/// Error text for a GPU build that lists no device.
fn no_gpu_device_error(binary: &std::path::Path) -> ServeError {
    ServeError::InvalidConfig(format!(
        "this profile runs on the GPU but `{}` reports no GPU device. Fix: `attune setup --gpu \
         auto`, or serve on the CPU with `--device cpu`",
        binary.display()
    ))
}

/// llama-server has no authentication, so its bind address must stay local.
/// Remote clients belong behind the proxy's `--listen` + `--api-key`.
fn check_upstream_host(host: &str) -> Result<(), ServeError> {
    if is_loopback_host(host) {
        return Ok(());
    }
    Err(ServeError::InvalidConfig(format!(
        "--host {host} is not a loopback address: llama-server has no authentication; remote \
         clients must use --listen with --api-key"
    )))
}

/// The `/health` URL for a bind address.
fn health_url_for(host: &str, port: u16) -> String {
    format!("http://{host}:{port}/health")
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use attune_backend::parse_list_devices;
    use attune_profile::{ExpectedPerformance, Fallbacks, RamFallback};

    fn profile(fingerprint: &str, peak_gb: f64, n_ctx: u32, fallback: Option<u32>) -> Profile {
        Profile {
            hardware_fingerprint: fingerprint.to_string(),
            load_config: attune_profile::LoadConfig {
                n_ctx,
                ..attune_profile::LoadConfig::default()
            },
            expected_performance: ExpectedPerformance {
                ram_peak_gb: peak_gb,
                ..ExpectedPerformance::default()
            },
            fallbacks: Fallbacks {
                if_ram_insufficient: fallback.map(|n_ctx| Box::new(RamFallback { n_ctx })),
            },
            ..Profile::default()
        }
    }

    #[test]
    fn matching_hardware_and_enough_ram_is_clean() {
        let profile = profile("cpu|8|avx2|16000000000", 1.0, 2_048, Some(1_024));
        let prepared = prepare_profile(&profile, "cpu|8|avx2|16000000000", 16_000_000_000, false);
        assert!(prepared.warnings.is_empty(), "{:?}", prepared.warnings);
        assert_eq!(prepared.profile.load_config.n_ctx, 2_048);
        assert_eq!(prepared.applied_fallback_n_ctx, None);
    }

    #[test]
    fn fingerprint_mismatch_warns_but_still_launches() {
        let profile = profile("old-cpu|4|sse|8000000000", 1.0, 2_048, Some(1_024));
        let prepared = prepare_profile(
            &profile,
            "new-cpu|8|avx2|16000000000",
            16_000_000_000,
            false,
        );
        assert_eq!(prepared.warnings.len(), 1);
        assert!(
            prepared.warnings[0].contains("fingerprint mismatch"),
            "{:?}",
            prepared.warnings
        );
        assert_eq!(prepared.profile.load_config.n_ctx, 2_048);
    }

    #[test]
    fn low_ram_applies_the_recorded_fallback() {
        let profile = profile("cpu|8|avx2|16000000000", 8.0, 4_096, Some(1_024));
        let prepared = prepare_profile(&profile, "cpu|8|avx2|16000000000", 2_000_000_000, false);
        assert_eq!(prepared.profile.load_config.n_ctx, 1_024);
        assert_eq!(prepared.applied_fallback_n_ctx, Some(1_024));
        assert!(
            prepared
                .warnings
                .iter()
                .any(|warning| warning.contains("n_ctx 4096 -> 1024")),
            "{:?}",
            prepared.warnings
        );
    }

    #[test]
    fn low_ram_without_a_fallback_only_warns() {
        let profile = profile("cpu|8|avx2|16000000000", 8.0, 4_096, None);
        let prepared = prepare_profile(&profile, "cpu|8|avx2|16000000000", 2_000_000_000, false);
        assert_eq!(prepared.profile.load_config.n_ctx, 4_096);
        assert_eq!(prepared.applied_fallback_n_ctx, None);
        assert_eq!(prepared.warnings.len(), 1);
        assert!(prepared.warnings[0].contains("smaller quant"));
    }

    #[test]
    fn force_skips_validation_and_keeps_the_profile_intact() {
        let profile = profile("old-cpu|4|sse|8000000000", 8.0, 4_096, Some(1_024));
        let prepared = prepare_profile(&profile, "new-cpu|8|avx2|16000000000", 1_000_000_000, true);
        assert!(prepared.warnings.is_empty());
        assert_eq!(prepared.profile.load_config.n_ctx, 4_096);
        assert_eq!(prepared.applied_fallback_n_ctx, None);
    }

    #[test]
    fn a_fallback_that_is_not_smaller_is_not_applied() {
        let profile = profile("cpu|8|avx2|16000000000", 8.0, 2_048, Some(2_048));
        let prepared = prepare_profile(&profile, "cpu|8|avx2|16000000000", 1_000_000_000, false);
        assert_eq!(prepared.profile.load_config.n_ctx, 2_048);
        assert_eq!(prepared.applied_fallback_n_ctx, None);
        assert_eq!(prepared.warnings.len(), 1);
    }

    #[test]
    fn fingerprint_matches_the_m4_exporter_format() {
        assert_eq!(
            fingerprint_of(
                " AMD Ryzen 9 7950X ",
                16,
                " AVX2, AVX-512F ",
                33_500_000_000
            ),
            "AMD Ryzen 9 7950X|16|AVX2, AVX-512F|33500000000"
        );
    }

    #[test]
    fn health_url_for_brackets_nothing_for_ipv4() {
        assert_eq!(
            health_url_for("127.0.0.1", 8081),
            "http://127.0.0.1:8081/health"
        );
    }

    fn download_profile(path: &str, size: Option<u64>) -> Profile {
        let mut profile = Profile::default();
        profile.model.path = path.to_string();
        profile.model.source = size.map(|size| attune_profile::ModelSource {
            repo: "Qwen/repo-GGUF".to_string(),
            file: "m.gguf".to_string(),
            revision: "rev".to_string(),
            sha256: "a".repeat(64),
            size,
        });
        profile
    }

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn non_tty_without_yes_errors_clearly() {
        let home = temp_home("notty");
        let mut profile =
            download_profile("~/.attune/models/org/repo-GGUF/m.gguf", Some(2_500_000_000));
        let err = resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
            .expect_err("must not download");
        assert!(
            err.to_string().contains("pass --yes to download 2.5 GB"),
            "{err}"
        );
    }

    #[test]
    fn profile_without_source_and_missing_file_errors_without_download() {
        let home = temp_home("nosource");
        let mut profile = download_profile("F:/nowhere/missing.gguf", None);
        let err = resolve_model_file(&mut profile, &home, DownloadConsent::Yes, &mut |_, _| {})
            .expect_err("missing");
        let text = err.to_string();
        assert!(text.contains("not found"), "{text}");
        assert!(!text.contains("--yes"), "{text}");
    }

    #[test]
    fn home_relative_path_is_expanded_for_llama_server() {
        let home = temp_home("expand");
        let file = home.join("models/org/repo-GGUF/m.gguf");
        std::fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        std::fs::write(&file, b"12345").expect("write");
        let mut profile = download_profile("~/.attune/models/org/repo-GGUF/m.gguf", Some(5));
        resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
            .expect("present");
        assert_eq!(PathBuf::from(&profile.model.path), file);
    }

    #[test]
    fn absolute_model_path_needs_no_home_dir() {
        let home = temp_home("abs");
        let file = home.join("m.gguf");
        std::fs::write(&file, b"x").expect("write");
        assert!(!needs_home(&file.display().to_string()));
        assert!(needs_home("~/.attune/models/a/b.gguf"));
    }

    #[test]
    fn upstream_host_must_be_loopback() {
        for host in ["127.0.0.1", "localhost", "::1", "[::1]"] {
            assert!(check_upstream_host(host).is_ok(), "{host}");
        }
        for host in ["0.0.0.0", "192.168.1.5", "::", "example.com"] {
            let error = check_upstream_host(host).expect_err("must refuse");
            assert!(error.to_string().contains("--listen"), "{error}");
            assert!(error.to_string().contains("--api-key"), "{error}");
        }
    }

    #[test]
    fn home_relative_model_path_rejects_parent_components() {
        let home = temp_home("parent");
        let mut profile = download_profile("~/.attune/models/../../secret.gguf", None);
        let err = resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
            .expect_err("parent traversal must be rejected");
        assert!(err.to_string().contains("no `..`"), "{err}");
    }

    #[test]
    fn source_rejects_an_untrusted_publisher() {
        let home = temp_home("untrusted");
        let mut profile = download_profile("~/.attune/models/evilcorp/repo/m.gguf", Some(5));
        profile.model.source.as_mut().expect("source").repo = "evilcorp/repo".to_string();
        let err = resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
            .expect_err("untrusted owner must be refused");
        assert!(err.to_string().contains("evilcorp"), "{err}");
        assert!(err.to_string().contains("untrusted publisher"), "{err}");
    }

    #[test]
    fn source_rejects_traversal_repo_and_file() {
        let home = temp_home("source-paths");
        for (repo, file, needle) in [
            ("../../evil", "m.gguf", "expected exactly"),
            ("Qwen/repo/extra", "m.gguf", "expected exactly"),
            ("Qwen/repo", "../../evil.gguf", "single safe file name"),
            ("Qwen/repo", ".hidden.gguf", "single safe file name"),
        ] {
            let mut profile = download_profile("~/.attune/models/Qwen/repo/m.gguf", Some(5));
            let source = profile.model.source.as_mut().expect("source");
            source.repo = repo.to_string();
            source.file = file.to_string();
            let err = resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
                .expect_err("unsafe source must be refused");
            assert!(err.to_string().contains(needle), "{repo}/{file}: {err}");
        }
    }

    #[test]
    fn download_destination_always_uses_home_models() {
        let home = temp_home("dest");
        let source = download_profile("nowhere", Some(5))
            .model
            .source
            .expect("source");
        let dest = download_destination(&source, &home).expect("dest");
        assert_eq!(dest, home.join("models/Qwen/repo-GGUF/m.gguf"));
    }

    #[test]
    fn existing_absolute_model_is_read_without_a_download() {
        let home = temp_home("read-local");
        let file = home.join("local.gguf");
        std::fs::write(&file, b"12345").expect("write");
        let mut profile = download_profile(&file.display().to_string(), Some(5));
        resolve_model_file(
            &mut profile,
            &home,
            DownloadConsent::Ask(|_, _| -> bool {
                panic!("an existing local file must not trigger a download")
            }),
            &mut |_, _| {},
        )
        .expect("local file");
        assert_eq!(PathBuf::from(&profile.model.path), file);
    }

    #[test]
    fn a_wrong_size_absolute_model_is_never_overwritten() {
        let home = temp_home("wrong-size");
        let file = home.join("local.gguf");
        std::fs::write(&file, b"x").expect("write");
        let mut profile = download_profile(&file.display().to_string(), Some(2_500_000_000));
        let err = resolve_model_file(&mut profile, &home, DownloadConsent::No, &mut |_, _| {})
            .expect_err("must require consent for the home copy");
        let text = err.to_string();
        assert!(text.contains("pass --yes to download 2.5 GB"), "{text}");
        assert!(
            text.contains(
                &home
                    .join("models/Qwen/repo-GGUF/m.gguf")
                    .display()
                    .to_string()
            ),
            "{text}"
        );
        assert_eq!(std::fs::read(&file).expect("read"), b"x");
    }

    // ---- device / build resolution ----------------------------------------

    fn variants(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn gpu_profile_auto_prefers_its_own_build() {
        let plan = plan_serve(
            "gpu",
            "vulkan",
            ServeDevice::Auto,
            &variants(&["vulkan", "cuda12"]),
        )
        .expect("plan");
        assert_eq!(
            plan,
            ServePlan {
                gpu: true,
                variant: "vulkan".to_string(),
                cpu_override: false,
            }
        );
        // `--device gpu` on a GPU profile is the same as auto.
        assert_eq!(
            plan_serve("gpu", "vulkan", ServeDevice::Gpu, &variants(&["vulkan"]))
                .expect("plan")
                .variant,
            "vulkan"
        );
    }

    #[test]
    fn gpu_profile_falls_back_to_the_other_gpu_build() {
        let plan =
            plan_serve("gpu", "cuda12", ServeDevice::Auto, &variants(&["vulkan"])).expect("plan");
        assert_eq!(plan.variant, "vulkan");
        assert!(plan.gpu);

        let plan =
            plan_serve("gpu", "vulkan", ServeDevice::Auto, &variants(&["cuda12"])).expect("plan");
        assert_eq!(plan.variant, "cuda12");
    }

    #[test]
    fn gpu_profile_without_a_gpu_build_reports_the_fix() {
        let error = plan_serve("gpu", "vulkan", ServeDevice::Auto, &[]).expect_err("no build");
        assert_eq!(error, NO_GPU_BUILD);
        assert!(error.contains("attune setup --gpu auto"), "{error}");
        assert!(error.contains("--device cpu"), "{error}");
    }

    #[test]
    fn cpu_profile_refuses_device_gpu_with_a_retune_hint() {
        let error = plan_serve("cpu", "cpu", ServeDevice::Gpu, &[]).expect_err("cpu profile");
        assert_eq!(error, CPU_PROFILE_ON_GPU);
        assert!(error.contains("attune tune --model"), "{error}");
        assert!(error.contains("--device gpu"), "{error}");
    }

    #[test]
    fn cpu_profile_auto_and_cpu_stay_on_the_cpu() {
        for requested in [ServeDevice::Auto, ServeDevice::Cpu] {
            let plan = plan_serve("cpu", "cpu", requested, &[]).expect("plan");
            assert!(!plan.gpu);
            assert_eq!(plan.variant, "cpu");
            assert!(!plan.cpu_override);
        }
    }

    fn has_pair(args: &[String], flag: &str, value: &str) -> bool {
        args.windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    #[test]
    fn cpu_override_zeroes_layers_and_emits_cpu_args() {
        let mut profile = Profile::default();
        profile.load_config.n_gpu_layers = 20;
        profile.load_config.gpu_name = Some("NVIDIA GeForce RTX 4050 Laptop GPU".to_string());
        profile.engine.variant = "vulkan".to_string();

        let plan = plan_serve("gpu", "vulkan", ServeDevice::Cpu, &[]).expect("plan");
        assert!(plan.cpu_override);
        assert!(!plan.gpu);
        assert_eq!(plan.variant, "cpu");

        apply_device_plan(&mut profile, &plan);
        assert_eq!(profile.load_config.n_gpu_layers, 0);
        assert_eq!(profile.device(), "cpu");

        let args = attune_profile::to_llama_args_with_device(&profile, Some("Vulkan1"));
        assert!(has_pair(&args, "-ngl", "0"));
        assert!(!args.iter().any(|arg| arg == "--fit"));
        assert!(!args.iter().any(|arg| arg == "-np"));
        assert!(!args.iter().any(|arg| arg == "-dev"));
    }

    #[test]
    fn device_id_prefers_the_discrete_gpu_in_a_recorded_listing() {
        let text = "\
Available devices:
  Vulkan0: Intel(R) Graphics (RPL-P) (5934 MiB, 5150 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5921 MiB, 5153 MiB free)
";
        let devices = parse_list_devices(text);
        let picked = pick_gpu_device(&devices, Some("NVIDIA GeForce RTX 4050 Laptop GPU"))
            .expect("a GPU device");
        assert_eq!(picked, "Vulkan1");
    }

    fn nvidia_device() -> GpuDevice {
        GpuDevice {
            vendor: "nvidia".to_string(),
            name: "NVIDIA GeForce RTX 4050 Laptop GPU".to_string(),
            vram_total_bytes: Some(6_442_450_944),
            vram_free_bytes: Some(6_200_000_000),
            driver: Some("561.00".to_string()),
            source: "test".to_string(),
        }
    }

    #[test]
    fn gpu_fingerprint_matches_the_tune_export_format() {
        let devices = vec![nvidia_device()];
        let gpu = gpu_identity(&devices, Some("nvidia geforce rtx 4050")).expect("gpu");
        assert_eq!(gpu.0, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(gpu.1, 6_442_450_944);
        assert_eq!(
            fingerprint_of_gpu("AMD Ryzen 9 7950X", 16, "AVX2", 33_500_000_000, Some(gpu)),
            "AMD Ryzen 9 7950X|16|AVX2|33500000000|NVIDIA GeForce RTX 4050 Laptop GPU|6"
        );
    }

    #[test]
    fn gpu_identity_falls_back_to_the_largest_adapter() {
        let devices = vec![
            GpuDevice {
                vendor: "intel".to_string(),
                name: "Intel(R) Graphics".to_string(),
                vram_total_bytes: Some(2_000_000_000),
                vram_free_bytes: None,
                driver: None,
                source: "test".to_string(),
            },
            nvidia_device(),
        ];
        let gpu = gpu_identity(&devices, None).expect("gpu");
        assert_eq!(gpu.0, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(gpu_identity(&devices, Some("missing adapter")), Some(gpu));
    }

    #[test]
    fn cpu_fingerprint_stays_four_parts() {
        assert_eq!(
            fingerprint_of("AMD Ryzen 9 7950X", 16, "AVX2", 33_500_000_000),
            "AMD Ryzen 9 7950X|16|AVX2|33500000000"
        );
    }
}
