//! Export the winning [`TuneConfig`] as an `attune-profile` document
//! (M4 slice 3).
//!
//! `attune-tune` owns this adapter because it already knows the swept config
//! and its predicted numbers; `attune-profile` stays a leaf and owns only the
//! schema. The CLI supplies the live inputs (fingerprint, model identity,
//! engine version) and owns the file write.
//!
//! Note: `Profile` is defined in `attune-profile`, and Rust forbids an inherent
//! `impl Profile` in a downstream crate (orphan rule). The plan's
//! `Profile::from_best` is therefore exposed as the free function
//! [`from_best`].

use attune_estimate::Quant;
use attune_profile::{now_unix, Fallbacks, LoadConfig, Profile, RamFallback, PROFILE_VERSION};

use crate::{tpot_ms_from_tps, TuneConfig};

/// Inputs the CLI supplies that do not come from the swept [`TuneConfig`].
#[derive(Debug, Clone, Copy)]
pub struct ProfileInputs<'a> {
    /// Profile name recorded in the document.
    pub name: &'a str,
    /// Readable hardware fingerprint, see [`hardware_fingerprint`].
    pub fingerprint: &'a str,
    /// Engine version/build, or an honest `unknown` marker when no llama.cpp
    /// binary was inspected.
    pub engine_version: &'a str,
    /// Model path or catalog label.
    pub model_path: &'a str,
    /// Weight format of the winning config.
    pub quant: Quant,
    /// Weight size in bytes (weights only).
    pub size_bytes: u64,
    /// Expected peak RAM in decimal GB (weights only in slice 3).
    pub ram_peak_gb: f64,
    /// Confidence in the prediction, `0.0..=1.0`.
    pub confidence: f64,
}

/// Build a versioned profile from the winning config and its prediction.
///
/// `decode_tps` and `ttft_ms` come straight from the best [`crate::RankedConfig`].
/// The profile records a halved context as the RAM fallback and no draft model
/// (speculative decoding is M6).
pub fn from_best(
    config: &TuneConfig,
    decode_tps: f64,
    ttft_ms: f64,
    inputs: &ProfileInputs<'_>,
) -> Profile {
    let fallback_n_ctx = (config.n_ctx / 2).max(256);

    Profile {
        profile_version: PROFILE_VERSION,
        name: inputs.name.to_string(),
        created_at: now_unix(),
        hardware_fingerprint: inputs.fingerprint.to_string(),
        engine: attune_profile::Engine {
            name: "llama.cpp".to_string(),
            version: inputs.engine_version.to_string(),
        },
        model: attune_profile::ModelConfig {
            path: inputs.model_path.to_string(),
            quant: inputs.quant.as_str().to_string(),
            size_bytes: inputs.size_bytes,
        },
        load_config: LoadConfig {
            n_ctx: config.n_ctx,
            n_batch: config.n_batch,
            n_ubatch: config.n_ubatch,
            n_gpu_layers: config.n_gpu_layers,
            mmap: config.mmap,
            mlock: config.mlock,
            flash_attn: config.flash_attn,
            cache_type_k: config.cache_type_k.clone(),
            cache_type_v: config.cache_type_v.clone(),
            numa: config.numa.clone(),
        },
        compute_config: attune_profile::ComputeConfig {
            n_threads: config.n_threads,
            // The slice-2 grid only sweeps one thread count; mirror it for the
            // prompt batch so the profile does not claim a second value the
            // tuner never measured.
            n_threads_batch: config.n_threads,
            cpu_affinity: config.cpu_affinity.clone(),
        },
        sampling_defaults: attune_profile::SamplingDefaults::default(),
        expected_performance: attune_profile::ExpectedPerformance {
            ttft_ms,
            tpot_ms: tpot_ms_from_tps(decode_tps),
            tok_s: decode_tps,
            ram_peak_gb: inputs.ram_peak_gb,
            confidence: inputs.confidence,
        },
        fallbacks: Fallbacks {
            if_ram_insufficient: Some(Box::new(RamFallback {
                n_ctx: fallback_n_ctx,
            })),
        },
        draft_model: None,
    }
}

/// Build a readable, non-cryptographic hardware fingerprint.
///
/// Format: `brand|physical_cores|simd|total_ram_bytes`, e.g.
/// `AMD Ryzen 9 7950X|16|AVX2, AVX-512F|33500000000`. This is deliberately not
/// a hash: it needs no extra dependency and a human can spot why two machines
/// differ. Treat it as an identity hint, not a security token.
pub fn hardware_fingerprint(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TuneConfig {
        TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16)
    }

    fn inputs() -> ProfileInputs<'static> {
        ProfileInputs {
            name: "test-profile",
            fingerprint: "cpu|8|avx2|16000000000",
            engine_version: "unknown (estimate-only)",
            model_path: "models/tiny.gguf",
            quant: Quant::Q4_K_M,
            size_bytes: 82_600_000,
            ram_peak_gb: 0.0826,
            confidence: 0.5,
        }
    }

    #[test]
    fn from_best_maps_the_winning_config() {
        let profile = from_best(&config(), 250.0, 120.0, &inputs());

        assert_eq!(profile.profile_version, PROFILE_VERSION);
        assert_eq!(profile.name, "test-profile");
        assert_eq!(profile.model.quant, "q4_k_m");
        assert_eq!(profile.model.size_bytes, 82_600_000);
        assert_eq!(profile.load_config.n_ctx, 2_048);
        assert_eq!(profile.load_config.n_batch, 512);
        assert_eq!(profile.load_config.n_ubatch, 128);
        assert_eq!(profile.load_config.cache_type_k, "q8_0");
        assert_eq!(profile.compute_config.n_threads, 8);
        assert_eq!(profile.compute_config.n_threads_batch, 8);
        assert!((profile.expected_performance.tok_s - 250.0).abs() < 1e-9);
        assert!((profile.expected_performance.tpot_ms - 4.0).abs() < 1e-9);
        assert!((profile.expected_performance.ttft_ms - 120.0).abs() < 1e-9);
        assert!(profile.draft_model.is_none());
        let fallback = profile
            .fallbacks
            .if_ram_insufficient
            .as_ref()
            .expect("fallback recorded");
        assert_eq!(fallback.n_ctx, 1_024);
    }

    #[test]
    fn from_best_roundtrips_through_json() {
        let profile = from_best(&config(), 250.0, 120.0, &inputs());
        let json = serde_json::to_string(&profile).expect("serialize profile");
        let back: Profile = serde_json::from_str(&json).expect("parse profile");
        assert_eq!(back.model, profile.model);
        assert_eq!(back.load_config, profile.load_config);
        assert_eq!(back.fallbacks, profile.fallbacks);
        assert!((back.expected_performance.tok_s - 250.0).abs() < 1e-9);
    }

    #[test]
    fn fingerprint_is_stable_for_the_same_input() {
        let first = hardware_fingerprint(
            " AMD Ryzen 9 7950X ",
            16,
            " AVX2, AVX-512F ",
            33_500_000_000,
        );
        let second = hardware_fingerprint(
            " AMD Ryzen 9 7950X ",
            16,
            " AVX2, AVX-512F ",
            33_500_000_000,
        );
        assert_eq!(first, second);
        assert_eq!(first, "AMD Ryzen 9 7950X|16|AVX2, AVX-512F|33500000000");
    }
}
