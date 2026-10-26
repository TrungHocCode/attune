//! Model-size and quantization parsing.
//!
//! Everything here is pure: no allocation beyond the error strings, no I/O,
//! no hardware access. The CLI feeds user text in and the roofline module
//! consumes the resulting [`ModelSpec`].

use crate::EstimateError;

/// Supported weight formats. The JSON names are pinned explicitly so a
/// future variant rename cannot silently change the report contract.
// The variant names mirror the GGUF quant labels users type (`q4_k_m`); the
// extra underscores are intentional and part of the public API.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Quant {
    /// Only used when an old/partial report omits `quant`; fp16 is the
    /// conservative (largest) weight size.
    #[default]
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "q8_0")]
    Q8_0,
    #[serde(rename = "q4_0")]
    Q4_0,
    /// Mixed 4-bit / 6-bit blocks. Its effective rate is 4.8944 bits per
    /// weight (see [`BITS_PER_WEIGHT`]), not the round `0.5` a naive
    /// "4-bit" label suggests.
    #[serde(rename = "q4_k_m")]
    Q4_K_M,
}

/// Effective storage cost of every supported weight format, in bits per
/// parameter.
///
/// These are the real llama.cpp block layouts (`ggml-common.h` /
/// `k_quants.h`, and the `llama-quantize` table), not rounded label values:
///
/// | format   | bits/weight | bytes/param |
/// |----------|-------------|-------------|
/// | `fp16`   | 16.0        | 2.0         |
/// | `q8_0`   | 8.5008      | 1.0626      |
/// | `q4_0`   | 4.5         | 0.5625      |
/// | `q4_k_m` | 4.8944      | 0.6118      |
///
/// Q8_0 packs 32 weights plus one f16 scale into 34 bytes; Q4_0 packs 32
/// nibbles plus one f16 scale into 18 bytes; Q4_K_M is a mixed 4/6-bit
/// k-quant whose measured effective rate is 4.8944 bits/weight.
///
/// **Scope:** [`Quant::bytes_per_param`] is weights only. Decode also streams
/// the KV cache, which grows with context size. Pass an [`Arch`] to
/// [`crate::predict_decode_tps_with_kv`] to include it; when the architecture
/// is unknown the KV term is omitted and the prediction is optimistic at
/// large context sizes. `eta_mem` still absorbs kernel overhead and any
/// residual per-token traffic.
pub const BITS_PER_WEIGHT: [(Quant, f64); 4] = [
    (Quant::Fp16, 16.0),
    (Quant::Q8_0, 8.5008),
    (Quant::Q4_0, 4.5),
    (Quant::Q4_K_M, 4.8944),
];

impl Quant {
    /// Effective bits stored per parameter for this format.
    pub fn bits_per_weight(self) -> f64 {
        BITS_PER_WEIGHT
            .iter()
            .find_map(|(quant, bits)| (*quant == self).then_some(*bits))
            // Every enum variant is in the table above; this fallback keeps
            // the function total without a panic path.
            .unwrap_or(16.0)
    }

    /// Bytes stored per parameter for this format (`bits / 8`).
    ///
    /// FP16 = 2.0, Q8_0 = 1.0626, Q4_0 = 0.5625 and Q4_K_M = 0.6118. See
    /// [`BITS_PER_WEIGHT`] for the source of each value and for the
    /// weights-only scope note.
    pub fn bytes_per_param(self) -> f64 {
        self.bits_per_weight() / 8.0
    }

    /// Canonical lowercase name (also the JSON value).
    pub fn as_str(self) -> &'static str {
        match self {
            Quant::Fp16 => "fp16",
            Quant::Q8_0 => "q8_0",
            Quant::Q4_0 => "q4_0",
            Quant::Q4_K_M => "q4_k_m",
        }
    }
}

impl std::fmt::Display for Quant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Quant {
    type Err = EstimateError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "fp16" => Ok(Self::Fp16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            "q4_k_m" => Ok(Self::Q4_K_M),
            other => Err(EstimateError::InvalidQuant(other.to_string())),
        }
    }
}

/// KV-cache shape of a known transformer architecture.
///
/// Only the fields that size the KV cache are carried: per layer the cache
/// stores one K and one V tensor, each `n_kv_heads × head_dim` values. Values
/// are pinned in [`ARCH_PRESETS`] and unit-tested, so a typo cannot silently
/// skew the decode estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    /// Canonical preset name accepted by `--arch` (case-insensitive).
    pub name: &'static str,
    /// Number of transformer layers.
    pub n_layer: u32,
    /// Number of key/value heads (GQA/MQA), not the attention-head count.
    pub n_kv_heads: u32,
    /// Per-head dimension of K/V.
    pub head_dim: u32,
}

/// Documented architecture presets.
///
/// Keep this list short and certain: every entry below is pinned by a unit
/// test. To add one, verify all three values against the model's published
/// `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim`),
/// then extend this table and its test. The KV math is
/// [`kv_bytes_per_token`].
pub const ARCH_PRESETS: &[Arch] = &[
    Arch {
        name: "llama-3.1-8b",
        // Llama-3.1-8B config.json: 32 layers, 8 KV heads, head_dim 128.
        n_layer: 32,
        n_kv_heads: 8,
        head_dim: 128,
    },
    Arch {
        name: "llama-3.1-70b",
        // Llama-3.1-70B config.json: 80 layers, 8 KV heads, head_dim 128.
        n_layer: 80,
        n_kv_heads: 8,
        head_dim: 128,
    },
    Arch {
        name: "qwen2.5-7b",
        // Qwen2.5-7B config.json: 28 layers, 4 KV heads, head_dim 128.
        n_layer: 28,
        n_kv_heads: 4,
        head_dim: 128,
    },
];

/// Resolve an `--arch` preset name, case-insensitively.
///
/// The error names the accepted presets so a typo is actionable.
pub fn parse_arch(input: &str) -> Result<&'static Arch, EstimateError> {
    let name = input.trim();
    ARCH_PRESETS
        .iter()
        .find(|arch| arch.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let known = ARCH_PRESETS
                .iter()
                .map(|arch| arch.name)
                .collect::<Vec<_>>()
                .join(", ");
            EstimateError::InvalidArch(format!("{name} (known: {known})"))
        })
}

/// Bytes of KV cache read per generated token for `arch`, assuming f16 K/V.
///
/// ```text
/// 2 (K + V) × n_layer × n_kv_heads × head_dim × 2 bytes (f16)
/// ```
///
/// This is the per-token KV *read* that decode adds on top of the weights;
/// multiply by the context length with [`kv_total_bytes`].
pub fn kv_bytes_per_token(arch: &Arch) -> f64 {
    2.0 * arch.n_layer as f64 * arch.n_kv_heads as f64 * arch.head_dim as f64 * 2.0
}

/// Total KV-cache bytes for a context of `ctx` tokens: per-token × ctx.
pub fn kv_total_bytes(arch: &Arch, ctx: u32) -> f64 {
    kv_bytes_per_token(arch) * ctx as f64
}

/// A parsed model description: parameter count plus weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub params: u64,
    pub quant: Quant,
}

impl ModelSpec {
    /// Total weight footprint in bytes (parameters × bytes per parameter).
    pub fn weight_bytes(&self) -> f64 {
        self.params as f64 * self.quant.bytes_per_param()
    }
}

/// Parse a human model size: `"135M"`, `"1.2B"`, or a plain integer such as
/// `"7000000"`. Suffixes are case-insensitive and decimal values are allowed
/// (`"0.5B"`). Zero, negatives, NaN/Inf and garbage are rejected instead of
/// being coerced to a bogus size.
/// Parse a comma-separated quant list such as `"fp16,q8_0,q4_k_m"`.
///
/// Entries are trimmed and parsed case-insensitively; duplicates are dropped
/// in first-seen order. An empty list or an empty entry is rejected instead of
/// silently ignored.
pub fn parse_quants(input: &str) -> Result<Vec<Quant>, EstimateError> {
    let mut quants = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(EstimateError::InvalidQuant(input.trim().to_string()));
        }
        let quant: Quant = part.parse()?;
        if !quants.contains(&quant) {
            quants.push(quant);
        }
    }

    if quants.is_empty() {
        Err(EstimateError::InvalidQuant(input.trim().to_string()))
    } else {
        Ok(quants)
    }
}

pub fn parse_params(input: &str) -> Result<u64, EstimateError> {
    let text = input.trim();
    if text.is_empty() {
        return Err(EstimateError::InvalidModel(
            "empty size (expected e.g. 135M, 1.2B or 7000000)".to_string(),
        ));
    }

    let (number, multiplier) = match text.as_bytes().last().copied() {
        Some(b'm' | b'M') => (&text[..text.len() - 1], 1_000_000.0),
        Some(b'b' | b'B') => (&text[..text.len() - 1], 1_000_000_000.0),
        _ => (text, 1.0),
    };
    let number = number.trim();
    if number.is_empty() {
        return Err(EstimateError::InvalidModel(input.to_string()));
    }

    // Plain integers stay exact (no f64 round-trip above 2^53).
    if multiplier == 1.0 {
        let params: u64 = number
            .parse()
            .map_err(|_| EstimateError::InvalidModel(input.to_string()))?;
        return if params == 0 {
            Err(EstimateError::InvalidModel(input.to_string()))
        } else {
            Ok(params)
        };
    }

    let value: f64 = number
        .parse()
        .map_err(|_| EstimateError::InvalidModel(input.to_string()))?;
    let scaled = value * multiplier;
    if !value.is_finite() || value <= 0.0 || !scaled.is_finite() || scaled >= u64::MAX as f64 {
        return Err(EstimateError::InvalidModel(input.to_string()));
    }

    let params = scaled.round() as u64;
    if params == 0 {
        Err(EstimateError::InvalidModel(input.to_string()))
    } else {
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_millions() {
        assert_eq!(parse_params("135M").expect("135M"), 135_000_000);
        assert_eq!(parse_params("135m").expect("lowercase suffix"), 135_000_000);
        assert_eq!(parse_params(" 135M ").expect("whitespace"), 135_000_000);
    }

    #[test]
    fn parses_billions_with_decimals() {
        assert_eq!(parse_params("1.2B").expect("1.2B"), 1_200_000_000);
        assert_eq!(parse_params("0.5b").expect("0.5b"), 500_000_000);
        assert_eq!(parse_params("7B").expect("7B"), 7_000_000_000);
    }

    #[test]
    fn parses_plain_integer_exactly() {
        assert_eq!(parse_params("7000000").expect("plain"), 7_000_000);
        assert_eq!(
            parse_params("9007199254740993").expect("above f64 precision"),
            9_007_199_254_740_993
        );
    }

    #[test]
    fn rejects_bad_sizes() {
        for input in ["", "M", "B", "abc", "NaN", "inf", "-5M", "0", "0.0B"] {
            assert!(parse_params(input).is_err(), "{input:?} should be rejected");
        }
    }

    #[test]
    fn parses_comma_separated_quants() {
        assert_eq!(
            parse_quants("fp16,q8_0,q4_k_m").expect("list"),
            vec![Quant::Fp16, Quant::Q8_0, Quant::Q4_K_M]
        );
        assert_eq!(
            parse_quants(" FP16 , q4_0 ").expect("trimmed"),
            vec![Quant::Fp16, Quant::Q4_0]
        );
        assert_eq!(
            parse_quants("q8_0,q8_0").expect("deduped"),
            vec![Quant::Q8_0]
        );
    }

    #[test]
    fn rejects_empty_or_invalid_quant_lists() {
        for input in ["", ",", "fp16,", ",q8_0", "fp16,,q8_0", "int3"] {
            assert!(parse_quants(input).is_err(), "{input:?} should be rejected");
        }
    }

    #[test]
    fn quant_parses_case_insensitively() {
        assert_eq!("FP16".parse::<Quant>().expect("FP16"), Quant::Fp16);
        assert_eq!("Q8_0".parse::<Quant>().expect("Q8_0"), Quant::Q8_0);
        assert_eq!("q4_0".parse::<Quant>().expect("q4_0"), Quant::Q4_0);
        assert_eq!("Q4_K_M".parse::<Quant>().expect("Q4_K_M"), Quant::Q4_K_M);
        assert_eq!(" q4_k_m ".parse::<Quant>().expect("trimmed"), Quant::Q4_K_M);
    }

    #[test]
    fn quant_rejects_unknown_names() {
        assert!(matches!(
            "int3".parse::<Quant>(),
            Err(EstimateError::InvalidQuant(_))
        ));
    }

    #[test]
    fn bits_per_weight_matches_the_documented_table() {
        assert_eq!(Quant::Fp16.bits_per_weight(), 16.0);
        assert_eq!(Quant::Q8_0.bits_per_weight(), 8.5008);
        assert_eq!(Quant::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(Quant::Q4_K_M.bits_per_weight(), 4.8944);
    }

    #[test]
    fn bytes_per_param_matches_the_documented_table() {
        assert_eq!(Quant::Fp16.bytes_per_param(), 2.0);
        assert!((Quant::Q8_0.bytes_per_param() - 1.0626).abs() < 1e-12);
        assert!((Quant::Q4_0.bytes_per_param() - 0.5625).abs() < 1e-12);
        assert!((Quant::Q4_K_M.bytes_per_param() - 0.6118).abs() < 1e-9);
    }

    #[test]
    fn quant_names_roundtrip_through_display() {
        for quant in [Quant::Fp16, Quant::Q8_0, Quant::Q4_0, Quant::Q4_K_M] {
            assert_eq!(
                quant.to_string().parse::<Quant>().expect("roundtrip"),
                quant
            );
        }
    }

    #[test]
    fn model_spec_computes_weight_bytes() {
        let spec = ModelSpec {
            params: 7_000_000_000,
            quant: Quant::Fp16,
        };
        assert_eq!(spec.weight_bytes(), 14_000_000_000.0);
    }

    #[test]
    fn arch_presets_pin_the_published_shapes() {
        let llama_8b = parse_arch("llama-3.1-8b").expect("llama-3.1-8b");
        assert_eq!(
            (llama_8b.n_layer, llama_8b.n_kv_heads, llama_8b.head_dim),
            (32, 8, 128)
        );

        let llama_70b = parse_arch("llama-3.1-70b").expect("llama-3.1-70b");
        assert_eq!(
            (llama_70b.n_layer, llama_70b.n_kv_heads, llama_70b.head_dim),
            (80, 8, 128)
        );

        let qwen = parse_arch("qwen2.5-7b").expect("qwen2.5-7b");
        assert_eq!((qwen.n_layer, qwen.n_kv_heads, qwen.head_dim), (28, 4, 128));
    }

    #[test]
    fn arch_lookup_is_case_insensitive_and_rejects_unknown_names() {
        assert_eq!(
            parse_arch("Llama-3.1-70B").expect("mixed case").name,
            "llama-3.1-70b"
        );
        assert_eq!(
            parse_arch("  qwen2.5-7b  ").expect("trimmed").name,
            "qwen2.5-7b"
        );
        let err = parse_arch("smollm2-135m").expect_err("unknown arch");
        assert!(matches!(err, EstimateError::InvalidArch(_)), "got {err:?}");
        let message = err.to_string();
        assert!(message.contains("llama-3.1-8b"), "message: {message}");
    }

    #[test]
    fn kv_bytes_follow_the_documented_formula() {
        let llama_8b = parse_arch("llama-3.1-8b").expect("preset");
        // 2 (K+V) × 32 × 8 × 128 × 2 bytes = 131_072 B/token (128 KiB).
        assert_eq!(kv_bytes_per_token(llama_8b), 131_072.0);
        assert_eq!(kv_total_bytes(llama_8b, 2048), 268_435_456.0);

        let llama_70b = parse_arch("llama-3.1-70b").expect("preset");
        // 2 × 80 × 8 × 128 × 2 = 327_680 B/token.
        assert_eq!(kv_bytes_per_token(llama_70b), 327_680.0);

        let qwen = parse_arch("qwen2.5-7b").expect("preset");
        // 2 × 28 × 4 × 128 × 2 = 57_344 B/token.
        assert_eq!(kv_bytes_per_token(qwen), 57_344.0);
        assert_eq!(kv_total_bytes(qwen, 0), 0.0);
    }
}
