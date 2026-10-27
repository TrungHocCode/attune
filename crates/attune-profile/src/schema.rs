//! The versioned profile JSON contract (M4 slice 3).
//!
//! Every struct uses `#[serde(default)]` so a document written by a newer
//! Attune still parses here (unknown fields are ignored by serde) and a
//! document written by an older one gains the new defaults. The defaults are
//! documented llama.cpp values, not silent zeroes, wherever a zero would be
//! unusable.
//!
//! `created_at` is a Unix epoch second rather than an ISO string so the crate
//! needs no `chrono`/time dependency. [`now_unix`] supplies it.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{ProfileError, PROFILE_VERSION};

/// Default context window when a profile omits `load_config.n_ctx`.
pub const DEFAULT_N_CTX: u32 = 2_048;
/// Default prompt batch size when a profile omits `load_config.n_batch`.
pub const DEFAULT_N_BATCH: u32 = 512;
/// Default micro-batch size when a profile omits `load_config.n_ubatch`.
pub const DEFAULT_N_UBATCH: u32 = 128;
/// Default fallback context when an old profile has no fallbacks block.
pub const DEFAULT_FALLBACK_N_CTX: u32 = 2_048;

/// Current Unix epoch time in whole seconds.
///
/// A clock before the epoch (or an unusable system clock) yields `0` rather
/// than panicking; the field is metadata, not a validation input.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// One reusable inference profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    /// Always [`PROFILE_VERSION`] for documents written by this crate.
    pub profile_version: u32,
    /// Human-readable profile name.
    pub name: String,
    /// Creation time as a Unix epoch second.
    pub created_at: u64,
    /// Readable, non-cryptographic hardware fingerprint. See
    /// `attune_tune::export::hardware_fingerprint`.
    pub hardware_fingerprint: String,
    /// Inference engine and its version.
    pub engine: Engine,
    /// Weights the profile was tuned for.
    pub model: ModelConfig,
    /// Flags applied when the model is loaded.
    pub load_config: LoadConfig,
    /// Flags that can be changed without a model reload.
    pub compute_config: ComputeConfig,
    /// Default sampling parameters for requests.
    pub sampling_defaults: SamplingDefaults,
    /// Predicted performance this profile was ranked on.
    pub expected_performance: ExpectedPerformance,
    /// Degraded modes to try when validation fails.
    pub fallbacks: Fallbacks,
    /// Optional draft model for speculative decoding (M6). Omitted from JSON
    /// when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<DraftConfig>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            profile_version: PROFILE_VERSION,
            name: String::new(),
            created_at: 0,
            hardware_fingerprint: String::new(),
            engine: Engine::default(),
            model: ModelConfig::default(),
            load_config: LoadConfig::default(),
            compute_config: ComputeConfig::default(),
            sampling_defaults: SamplingDefaults::default(),
            expected_performance: ExpectedPerformance::default(),
            fallbacks: Fallbacks::default(),
            draft_model: None,
        }
    }
}

impl Profile {
    /// Parse a profile document. Missing fields fall back to the struct
    /// defaults; use [`crate::migrate_value`] first when the document may be
    /// older than [`PROFILE_VERSION`].
    pub fn from_json_str(input: &str) -> Result<Self, ProfileError> {
        Ok(serde_json::from_str(input)?)
    }

    /// Serialize the profile as pretty JSON.
    pub fn to_json_string(&self) -> Result<String, ProfileError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Inference engine identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Engine {
    /// Engine name, e.g. `llama.cpp`.
    pub name: String,
    /// Engine version/build, or an explicit `unknown` marker when the
    /// estimate-only path did not inspect a binary.
    pub version: String,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            name: "llama.cpp".to_string(),
            version: "unknown".to_string(),
        }
    }
}

/// Weights the profile was tuned for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path or catalog label of the weights.
    pub path: String,
    /// Weight format name (`q4_k_m`, `q8_0`, ...).
    pub quant: String,
    /// Weight file size in bytes (weights only; no KV cache).
    pub size_bytes: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            quant: "q4_k_m".to_string(),
            size_bytes: 0,
        }
    }
}

/// Load-time llama.cpp flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadConfig {
    /// Context window in tokens (`-c`).
    pub n_ctx: u32,
    /// Prompt batch size (`-b`).
    pub n_batch: u32,
    /// Physical micro-batch size (`--ubatch-size`).
    pub n_ubatch: u32,
    /// Layers offloaded to the GPU (`-ngl`).
    pub n_gpu_layers: u32,
    /// Memory-map the weights (`--no-mmap` when false).
    pub mmap: bool,
    /// Lock weights in RAM (`--mlock`).
    pub mlock: bool,
    /// Enable flash attention (`-fa`).
    pub flash_attn: bool,
    /// KV K-cache format (`--cache-type-k`).
    pub cache_type_k: String,
    /// KV V-cache format (`--cache-type-v`).
    pub cache_type_v: String,
    /// NUMA policy (`--numa`).
    pub numa: String,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            n_ctx: DEFAULT_N_CTX,
            n_batch: DEFAULT_N_BATCH,
            n_ubatch: DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            mmap: true,
            mlock: false,
            flash_attn: true,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            numa: "local".to_string(),
        }
    }
}

/// Flags that do not require a model reload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeConfig {
    /// Worker threads (`-t`).
    pub n_threads: u32,
    /// Threads used by the prompt batch (`-tb`).
    pub n_threads_batch: u32,
    /// CPU affinity mask/range, when a launcher should pin threads. Omitted
    /// from JSON when absent; the M4 args mapper leaves pinning to M5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_affinity: Option<String>,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            n_threads: 1,
            n_threads_batch: 1,
            cpu_affinity: None,
        }
    }
}

/// Request sampling defaults (llama.cpp defaults).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingDefaults {
    /// Sampling temperature.
    pub temperature: f64,
    /// Nucleus sampling probability.
    pub top_p: f64,
    /// Top-k cutoff.
    pub top_k: u32,
    /// Repetition penalty.
    pub repeat_penalty: f64,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

/// Predicted performance the profile was ranked on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExpectedPerformance {
    /// Time to first token in milliseconds.
    pub ttft_ms: f64,
    /// Time per output token in milliseconds (`1000 / tok_s`).
    pub tpot_ms: f64,
    /// Decode throughput in tokens/second.
    pub tok_s: f64,
    /// Expected peak RAM in decimal GB (weights only in slice 3).
    pub ram_peak_gb: f64,
    /// Confidence in the prediction, `0.0..=1.0`.
    pub confidence: f64,
}

impl Default for ExpectedPerformance {
    fn default() -> Self {
        Self {
            ttft_ms: 0.0,
            tpot_ms: 0.0,
            tok_s: 0.0,
            ram_peak_gb: 0.0,
            confidence: 0.0,
        }
    }
}

/// Degraded modes to try when validation warns.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Fallbacks {
    /// Smaller context to try when the expected peak does not fit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub if_ram_insufficient: Option<Box<RamFallback>>,
}

/// The only fallback slice 3 records: a smaller context window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RamFallback {
    /// Context window to retry with.
    pub n_ctx: u32,
}

impl Default for RamFallback {
    fn default() -> Self {
        Self {
            n_ctx: DEFAULT_FALLBACK_N_CTX,
        }
    }
}

/// Draft model for speculative decoding (reserved for M6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DraftConfig {
    /// Draft model path or catalog label.
    pub model: String,
    /// Number of draft tokens per step.
    pub draft_k: u32,
    /// Measured acceptance rate, `0.0..=1.0`.
    pub acceptance_rate: f64,
}

impl Default for DraftConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            draft_k: 4,
            acceptance_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Profile {
        Profile {
            name: "test-profile".to_string(),
            created_at: 1_700_000_000,
            hardware_fingerprint: "cpu|8|avx2|16000000000".to_string(),
            model: ModelConfig {
                path: "models/tiny.gguf".to_string(),
                quant: "q4_k_m".to_string(),
                size_bytes: 82_600_000,
            },
            load_config: LoadConfig {
                n_ctx: 2_048,
                n_batch: 512,
                n_ubatch: 128,
                n_gpu_layers: 0,
                mmap: true,
                mlock: false,
                flash_attn: true,
                cache_type_k: "q8_0".to_string(),
                cache_type_v: "q8_0".to_string(),
                numa: "local".to_string(),
            },
            compute_config: ComputeConfig {
                n_threads: 4,
                n_threads_batch: 4,
                cpu_affinity: None,
            },
            expected_performance: ExpectedPerformance {
                ttft_ms: 120.0,
                tpot_ms: 4.0,
                tok_s: 250.0,
                ram_peak_gb: 0.08,
                confidence: 0.5,
            },
            fallbacks: Fallbacks {
                if_ram_insufficient: Some(Box::new(RamFallback { n_ctx: 1_024 })),
            },
            ..Profile::default()
        }
    }

    #[test]
    fn default_is_versioned_and_uses_llama_cpp_defaults() {
        let profile = Profile::default();
        assert_eq!(profile.profile_version, PROFILE_VERSION);
        assert_eq!(profile.engine.name, "llama.cpp");
        assert_eq!(profile.load_config.n_ctx, 2_048);
        assert_eq!(profile.load_config.n_batch, 512);
        assert_eq!(profile.load_config.n_ubatch, 128);
        assert!(profile.load_config.mmap);
        assert!(profile.load_config.flash_attn);
        assert_eq!(profile.sampling_defaults.temperature, 0.8);
        assert_eq!(profile.sampling_defaults.top_p, 0.95);
        assert_eq!(profile.sampling_defaults.top_k, 40);
        assert_eq!(profile.sampling_defaults.repeat_penalty, 1.1);
        assert!(profile.draft_model.is_none());
    }

    #[test]
    fn now_unix_is_a_plain_epoch_second() {
        // The M4 contract is a Unix second, not an ISO string and not millis.
        // Headroom: a live clock can only be checked against a floor.
        assert!(now_unix() > 1_600_000_000);
    }

    #[test]
    fn roundtrip_omits_an_absent_draft_model() {
        let profile = sample();
        let json = profile.to_json_string().expect("serialize profile");
        assert!(
            !json.contains("\"draft_model\""),
            "absent draft must be omitted: {json}"
        );

        let back = Profile::from_json_str(&json).expect("parse profile");
        assert_eq!(back.profile_version, PROFILE_VERSION);
        assert_eq!(back.name, profile.name);
        assert_eq!(back.created_at, profile.created_at);
        assert_eq!(back.hardware_fingerprint, profile.hardware_fingerprint);
        assert_eq!(back.engine, profile.engine);
        assert_eq!(back.model, profile.model);
        assert_eq!(back.load_config, profile.load_config);
        assert_eq!(back.compute_config, profile.compute_config);
        assert_eq!(back.fallbacks, profile.fallbacks);
        assert!(back.draft_model.is_none());
        // Float JSON roundtrips are compared with tolerance (AGENTS §6).
        assert!((back.sampling_defaults.temperature - 0.8).abs() < 1e-12);
        assert!(
            (back.expected_performance.tok_s - profile.expected_performance.tok_s).abs() < 1e-9
        );
    }

    #[test]
    fn roundtrip_keeps_a_present_draft_model() {
        let mut profile = sample();
        profile.draft_model = Some(DraftConfig {
            model: "models/draft.gguf".to_string(),
            draft_k: 4,
            acceptance_rate: 0.72,
        });
        let json = profile.to_json_string().expect("serialize profile");
        assert!(
            json.contains("\"draft_model\""),
            "present draft must be serialized: {json}"
        );

        let back = Profile::from_json_str(&json).expect("parse profile");
        let draft = back.draft_model.expect("draft parsed");
        assert_eq!(draft.model, "models/draft.gguf");
        assert_eq!(draft.draft_k, 4);
        assert!((draft.acceptance_rate - 0.72).abs() < 1e-12);
    }

    #[test]
    fn missing_fields_use_defaults_for_forward_compat() {
        let json = r#"{
            "profile_version": 1,
            "name": "partial",
            "model": { "path": "models/tiny.gguf" }
        }"#;
        let profile = Profile::from_json_str(json).expect("parse partial profile");
        assert_eq!(profile.name, "partial");
        assert_eq!(profile.model.quant, "q4_k_m");
        assert_eq!(profile.load_config.n_ctx, 2_048);
        assert_eq!(profile.sampling_defaults.top_k, 40);
        assert!(profile.draft_model.is_none());
    }
}
