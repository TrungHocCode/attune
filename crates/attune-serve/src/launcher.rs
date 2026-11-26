//! Profile validation, spawn and health-check lifecycle (M5 slice 1).
//!
//! The launcher is deliberately thin around the backend: it turns a profile
//! file plus live hardware facts into a running server, and it owns the policy
//! decisions (warn on fingerprint mismatch, auto-apply the RAM fallback,
//! `--force` skips validation). Everything process-level lives in
//! [`attune_backend`].

use std::path::PathBuf;
use std::time::Duration;

use attune_backend::{
    ensure_model, resolve_llama_server, Backend, LlamaServerBackend, Paths, ServerGuard,
};
use attune_hal::HardwareInfo;
use attune_profile::{validate, Profile, WarningKind};
use serde::Serialize;

use crate::ServeError;

/// Parameters for one `attune serve` launch.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Path to the profile JSON file.
    pub profile: PathBuf,
    /// Explicit `llama-server` path (`--llama-server`); `None` falls back to
    /// `ATTUNE_LLAMA_SERVER` and then `PATH`.
    pub llama_server: Option<PathBuf>,
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

/// Point `profile.model.path` at a real file: expand a `~/.attune/` path
/// against `home` and, when the file is missing and the profile has a
/// `source`, download it under `consent`.
pub fn resolve_model_file(
    profile: &mut Profile,
    home: &std::path::Path,
    consent: DownloadConsent,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), ServeError> {
    let path = attune_profile::expand_home(&profile.model.path, home);
    match profile.model.source.clone() {
        Some(source) => {
            let present = std::fs::metadata(&path).is_ok_and(|m| m.len() == source.size);
            let size_gb = source.size as f64 / 1e9;
            if !present && matches!(consent, DownloadConsent::No) {
                return Err(ServeError::InvalidConfig(format!(
                    "model file `{}` is missing; pass --yes to download {size_gb:.1} GB from {}",
                    path.display(),
                    source.repo
                )));
            }
            let what = format!("{} ({size_gb:.1} GB) from {}", source.file, source.repo);
            let mut confirm = |size: u64| match consent {
                DownloadConsent::Yes => true,
                DownloadConsent::No => false,
                DownloadConsent::Ask(ask) => ask(size, &what),
            };
            ensure_model(&path, &source, &mut confirm, progress)?;
        }
        None if !path.exists() => {
            return Err(ServeError::InvalidConfig(format!(
                "model file `{}` not found",
                path.display()
            )));
        }
        None => {}
    }
    profile.model.path = path.display().to_string();
    Ok(())
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
        }
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

/// The readable hardware fingerprint, using the same
/// `brand|physical_cores|simd|total_ram_bytes` format as
/// `attune_tune::hardware_fingerprint`.
///
/// Duplicated on purpose: adding `attune-tune` here would drag in
/// `attune-estimate` and the whole tuning sweep. Serve only needs to *compare*
/// the fingerprint, never to produce a new profile. Keep the format in sync if
/// the M4 exporter changes it.
pub fn fingerprint_of(
    brand: &str,
    physical_cores: u32,
    simd: &str,
    total_ram_bytes: u64,
) -> String {
    format!(
        "{}|{}|{}|{}",
        brand.trim(),
        physical_cores,
        simd.trim(),
        total_ram_bytes
    )
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
    let home =
        attune_fetch::home_dir().map_err(|err| ServeError::InvalidConfig(err.to_string()))?;
    let progress = options.progress;
    resolve_model_file(&mut profile, &home, options.download, &mut |done, total| {
        progress(done, total)
    })?;

    // `--force` means "do not validate against this machine", so skip the
    // (not free) hardware probe entirely: no fingerprint is needed and a HAL
    // failure must not block a forced launch.
    let (current_fingerprint, ram_available) = if options.force {
        (String::new(), 0)
    } else {
        let hardware = HardwareInfo::detect()?;
        let fingerprint = fingerprint_of(
            &hardware.cpu.brand,
            hardware.cpu.physical_cores,
            &hardware.cpu.simd,
            hardware.memory.total_bytes,
        );
        (fingerprint, hardware.memory.available_bytes)
    };
    let prepared = prepare_profile(&profile, &current_fingerprint, ram_available, options.force);

    let binary = resolve_llama_server(options.llama_server.as_deref())?;
    let paths = Paths::new(binary);
    let backend = LlamaServerBackend;
    let guard = backend.spawn(&prepared.profile, &paths, host, options.port)?;

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
    })
}

/// The `/health` URL for a bind address.
fn health_url_for(host: &str, port: u16) -> String {
    format!("http://{host}:{port}/health")
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
            repo: "org/repo-GGUF".to_string(),
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
}
