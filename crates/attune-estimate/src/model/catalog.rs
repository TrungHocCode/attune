//! Verified model catalog backing `estimate --model <name>`.
//!
//! Each entry carries the parameter count and the KV-cache shape, so one flag
//! replaces the `--params` + `--arch` pair. The numbers below are pinned:
//! `n_layer` / `n_kv_heads` / `head_dim` were checked against each model's
//! published `config.json` (or the vendor card when the config is gated or
//! omits `head_dim`), and the unit tests mirror every field. Change a number
//! only together with a source update in the comment and in the test.
//!
//! `ctx_default` is the model's documented context window. The CLI uses it
//! only when the user did not pass `--ctx`; an explicit `--ctx` always wins.
//! It is stored and pinned here so the resolution rule stays testable.

use super::{Arch, Quant};
use crate::EstimateError;

/// One verified model and the inputs `estimate` needs to size it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelEntry {
    /// Publishing organization (e.g. `meta`, `qwen`).
    pub provider: &'static str,
    /// Catalog name accepted by `--model` (case-insensitive).
    pub name: &'static str,
    /// Total parameter count used for the weight footprint.
    pub params: u64,
    /// KV-cache shape (layers / KV heads / head dim).
    pub arch: Arch,
    /// Formats this model is commonly distributed in.
    pub quants: &'static [Quant],
    /// Documented default context window. Used by `estimate` only when
    /// `--ctx` is not passed explicitly.
    pub ctx_default: u32,
}

/// Every catalog entry ships the same four GGUF formats today.
const ALL_QUANTS: &[Quant] = &[Quant::Fp16, Quant::Q8_0, Quant::Q4_0, Quant::Q4_K_M];

/// The verified catalog. Keep it grouped by provider so the unknown-name
/// error lists names in a predictable order.
pub const MODEL_CATALOG: &[ModelEntry] = &[
    ModelEntry {
        provider: "google",
        name: "gemma-2-9b",
        // Google Gemma 2 9B model card: 9.24B parameters.
        params: 9_240_000_000,
        arch: Arch {
            name: "gemma-2-9b",
            // google/gemma-2-9b config.json: num_hidden_layers=42,
            // num_key_value_heads=8, head_dim=256.
            n_layer: 42,
            n_kv_heads: 8,
            head_dim: 256,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=8192.
        ctx_default: 8_192,
    },
    ModelEntry {
        provider: "huggingface",
        name: "smollm2-135m",
        // HuggingFaceTB/SmolLM2-135M model card: 135M parameters.
        params: 135_000_000,
        arch: Arch {
            name: "smollm2-135m",
            // HuggingFaceTB/SmolLM2-135M config.json: num_hidden_layers=30,
            // num_key_value_heads=3. head_dim is hidden_size (576) /
            // num_attention_heads (9) = 64.
            n_layer: 30,
            n_kv_heads: 3,
            head_dim: 64,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=8192.
        ctx_default: 8_192,
    },
    ModelEntry {
        provider: "huggingface",
        name: "smollm2-360m",
        // HuggingFaceTB/SmolLM2-360M model card: 360M parameters.
        params: 360_000_000,
        arch: Arch {
            name: "smollm2-360m",
            // HuggingFaceTB/SmolLM2-360M config.json: num_hidden_layers=32,
            // num_key_value_heads=5. head_dim is hidden_size (960) /
            // num_attention_heads (15) = 64.
            n_layer: 32,
            n_kv_heads: 5,
            head_dim: 64,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=8192.
        ctx_default: 8_192,
    },
    ModelEntry {
        provider: "meta",
        name: "llama-3.1-8b",
        // Meta Llama 3.1 8B model card: 8.03B parameters.
        params: 8_030_000_000,
        arch: Arch {
            name: "llama-3.1-8b",
            // meta-llama/Llama-3.1-8B config.json: num_hidden_layers=32,
            // num_key_value_heads=8, head_dim=128.
            n_layer: 32,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=131072.
        ctx_default: 131_072,
    },
    ModelEntry {
        provider: "meta",
        name: "llama-3.1-70b",
        // Meta Llama 3.1 70B model card: 70.6B parameters.
        params: 70_600_000_000,
        arch: Arch {
            name: "llama-3.1-70b",
            // meta-llama/Llama-3.1-70B config.json: num_hidden_layers=80,
            // num_key_value_heads=8, head_dim=128.
            n_layer: 80,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=131072.
        ctx_default: 131_072,
    },
    ModelEntry {
        provider: "microsoft",
        name: "phi-4-mini",
        // microsoft/Phi-4-mini-instruct model card: 3.8B parameters.
        params: 3_800_000_000,
        arch: Arch {
            name: "phi-4-mini",
            // microsoft/Phi-4-mini-instruct config.json: num_hidden_layers=32,
            // num_key_value_heads=8. head_dim is hidden_size (3072) /
            // num_attention_heads (24) = 128.
            n_layer: 32,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=131072.
        ctx_default: 131_072,
    },
    ModelEntry {
        provider: "mistralai",
        name: "mistral-7b",
        // Mistral 7B Instruct v0.3 model card: 7.25B parameters.
        params: 7_250_000_000,
        arch: Arch {
            name: "mistral-7b",
            // mistralai/Mistral-7B-Instruct-v0.3 config.json:
            // num_hidden_layers=32, num_key_value_heads=8, head_dim=128.
            n_layer: 32,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // config.json max_position_embeddings=32768.
        ctx_default: 32_768,
    },
    ModelEntry {
        provider: "qwen",
        name: "qwen2.5-7b",
        // Qwen2.5-7B model card: 7.6B parameters.
        params: 7_600_000_000,
        arch: Arch {
            name: "qwen2.5-7b",
            // Qwen/Qwen2.5-7B-Instruct config.json: num_hidden_layers=28,
            // num_key_value_heads=4, head_dim=128.
            n_layer: 28,
            n_kv_heads: 4,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // Qwen2.5 card: 131072 tokens with YaRN (config.json native max is
        // 32768).
        ctx_default: 131_072,
    },
    ModelEntry {
        provider: "qwen",
        name: "qwen2.5-14b",
        // Qwen2.5-14B model card: 14.8B parameters.
        params: 14_800_000_000,
        arch: Arch {
            name: "qwen2.5-14b",
            // Qwen/Qwen2.5-14B-Instruct config.json: num_hidden_layers=48,
            // num_key_value_heads=8, head_dim=128.
            n_layer: 48,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // Qwen2.5 card: 131072 tokens with YaRN (config.json native max is
        // 32768).
        ctx_default: 131_072,
    },
    ModelEntry {
        provider: "qwen",
        name: "qwen3-8b",
        // Qwen3-8B model card: 8.2B parameters.
        params: 8_200_000_000,
        arch: Arch {
            name: "qwen3-8b",
            // Qwen/Qwen3-8B config.json: num_hidden_layers=36,
            // num_key_value_heads=8, head_dim=128.
            n_layer: 36,
            n_kv_heads: 8,
            head_dim: 128,
        },
        quants: ALL_QUANTS,
        // Qwen3-8B card: 32768-token native context (131072 with YaRN).
        ctx_default: 32_768,
    },
];

/// Resolve a `--model` catalog name, case-insensitively.
///
/// The error lists the accepted names so a typo is actionable; the CLI adds
/// the `--params` fallback suggestion for models outside the catalog.
pub fn parse_model(input: &str) -> Result<&'static ModelEntry, EstimateError> {
    let name = input.trim();
    MODEL_CATALOG
        .iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let known = MODEL_CATALOG
                .iter()
                .map(|entry| entry.name)
                .collect::<Vec<_>>()
                .join(", ");
            EstimateError::InvalidModelName(format!("{name} (known: {known})"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kv_bytes_per_token, parse_arch, parse_params, predict_decode_tps_with_kv, HwInputs,
    };

    /// Independent mirror of the catalog. Every value here is the one verified
    /// against `config.json` / the vendor card; an edit to `MODEL_CATALOG`
    /// must be matched here too.
    const EXPECTED: &[(&str, &str, u64, u32, u32, u32, u32)] = &[
        ("gemma-2-9b", "google", 9_240_000_000, 42, 8, 256, 8_192),
        ("smollm2-135m", "huggingface", 135_000_000, 30, 3, 64, 8_192),
        ("smollm2-360m", "huggingface", 360_000_000, 32, 5, 64, 8_192),
        ("llama-3.1-8b", "meta", 8_030_000_000, 32, 8, 128, 131_072),
        ("llama-3.1-70b", "meta", 70_600_000_000, 80, 8, 128, 131_072),
        (
            "phi-4-mini",
            "microsoft",
            3_800_000_000,
            32,
            8,
            128,
            131_072,
        ),
        ("mistral-7b", "mistralai", 7_250_000_000, 32, 8, 128, 32_768),
        ("qwen2.5-7b", "qwen", 7_600_000_000, 28, 4, 128, 131_072),
        ("qwen2.5-14b", "qwen", 14_800_000_000, 48, 8, 128, 131_072),
        ("qwen3-8b", "qwen", 8_200_000_000, 36, 8, 128, 32_768),
    ];

    #[test]
    fn catalog_pins_the_verified_names_params_and_arch() {
        assert_eq!(MODEL_CATALOG.len(), EXPECTED.len());
        for (entry, &(name, provider, params, n_layer, n_kv_heads, head_dim, ctx_default)) in
            MODEL_CATALOG.iter().zip(EXPECTED)
        {
            assert_eq!(entry.name, name);
            assert_eq!(entry.provider, provider);
            assert_eq!(entry.params, params);
            assert_eq!(entry.arch.name, name);
            assert_eq!(entry.arch.n_layer, n_layer);
            assert_eq!(entry.arch.n_kv_heads, n_kv_heads);
            assert_eq!(entry.arch.head_dim, head_dim);
            assert_eq!(entry.ctx_default, ctx_default);
            assert_eq!(entry.quants, ALL_QUANTS);
            assert_eq!(parse_model(name).expect("catalog lookup").name, name);
        }
    }

    #[test]
    fn catalog_arch_matches_the_existing_arch_presets() {
        // Overlapping entries must stay byte-identical to `--arch` presets,
        // otherwise the two CLI paths would disagree.
        for name in ["llama-3.1-8b", "llama-3.1-70b", "qwen2.5-7b"] {
            let entry = parse_model(name).expect("catalog entry");
            let preset = parse_arch(name).expect("arch preset");
            assert_eq!(entry.arch, *preset, "catalog and preset drift for {name}");
        }
    }

    #[test]
    fn model_lookup_is_case_insensitive_and_rejects_unknown_names() {
        assert_eq!(
            parse_model("LLAMA-3.1-8B").expect("mixed case").name,
            "llama-3.1-8b"
        );
        assert_eq!(
            parse_model("  Qwen3-8B  ").expect("trimmed").name,
            "qwen3-8b"
        );

        let err = parse_model("not-a-model").expect_err("unknown model");
        assert!(
            matches!(err, EstimateError::InvalidModelName(_)),
            "got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains("llama-3.1-8b"), "message: {message}");
        assert!(message.contains("smollm2-135m"), "message: {message}");
    }

    #[test]
    fn catalog_and_params_arch_paths_predict_the_same_decode_rate() {
        // Acceptance for `--model`: the catalog entry and the explicit
        // `--params 8.03B --arch llama-3.1-8b` pair resolve to the same
        // roofline inputs and therefore the same decode prediction.
        let entry = parse_model("llama-3.1-8b").expect("catalog model");
        let params = parse_params("8.03B").expect("explicit size");
        let arch = parse_arch("llama-3.1-8b").expect("arch preset");
        assert_eq!(entry.params, params);
        assert_eq!(entry.arch, *arch);

        let hw = HwInputs {
            bw_gbps: 20.0,
            flops_gflops: 50.0,
        };
        let bytes_per_param = Quant::Q4_K_M.bytes_per_param();
        let via_model = hw
            .predict_decode_tps_with_kv(
                entry.params,
                bytes_per_param,
                kv_bytes_per_token(&entry.arch),
                2048,
            )
            .expect("catalog prediction");
        let via_flags = hw
            .predict_decode_tps_with_kv(params, bytes_per_param, kv_bytes_per_token(arch), 2048)
            .expect("flag prediction");
        assert_eq!(via_model, via_flags);
        // Explicit free-function form stays identical as well.
        assert_eq!(
            via_model,
            predict_decode_tps_with_kv(
                entry.params,
                bytes_per_param,
                20.0,
                kv_bytes_per_token(&entry.arch),
                2048,
            )
            .expect("free-function prediction")
        );
    }
}
