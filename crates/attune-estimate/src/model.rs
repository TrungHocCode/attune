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
    /// Mixed 4-bit / 6-bit blocks. `≈0.6` bytes per parameter is a temporary
    /// constant; a real GGUF parser (M4) will read the exact block layout.
    #[serde(rename = "q4_k_m")]
    Q4_K_M,
}

impl Quant {
    /// Bytes stored per parameter for this format.
    ///
    /// FP16 = 2.0, Q8_0 = 1.0, Q4_0 = 0.5 and Q4_K_M ≈ 0.6. `Q4_K_M` is an
    /// approximation until the GGUF parser lands in M4; the decode math only
    /// needs the order of magnitude for slice 1.
    pub fn bytes_per_param(self) -> f64 {
        match self {
            Quant::Fp16 => 2.0,
            Quant::Q8_0 => 1.0,
            Quant::Q4_0 => 0.5,
            Quant::Q4_K_M => 0.6,
        }
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
    fn bytes_per_param_matches_the_documented_table() {
        assert_eq!(Quant::Fp16.bytes_per_param(), 2.0);
        assert_eq!(Quant::Q8_0.bytes_per_param(), 1.0);
        assert_eq!(Quant::Q4_0.bytes_per_param(), 0.5);
        assert_eq!(Quant::Q4_K_M.bytes_per_param(), 0.6);
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
}
