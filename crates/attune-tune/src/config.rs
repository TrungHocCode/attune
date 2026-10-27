//! One point in the tuning grid.

use attune_estimate::Quant;
use serde::{Deserialize, Serialize};

use crate::TuneError;

/// A candidate inference configuration.
///
/// `quant` is the weight format and `cache_kv` the KV-cache format; both reuse
/// [`Quant`] so the sweep cannot invent a second format vocabulary. Slice 1
/// ranks with the weights-only roofline, so `n_threads`, `n_batch` and
/// `cache_kv` are carried through to the report but do not change the score
/// yet — the fields are here so the grid/JSON shape is stable when later
/// slices model them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneConfig {
    /// Weight format.
    #[serde(default)]
    pub quant: Quant,
    /// Context window in tokens. Also used as the slice-1 prefill length for
    /// TTFT (see [`crate::rank_by_estimate`]).
    #[serde(default)]
    pub n_ctx: u32,
    /// Worker thread count.
    #[serde(default)]
    pub n_threads: u32,
    /// Prompt batch size.
    #[serde(default)]
    pub n_batch: u32,
    /// KV-cache format.
    #[serde(default)]
    pub cache_kv: Quant,
}

impl TuneConfig {
    /// Build one grid point.
    pub fn new(quant: Quant, n_ctx: u32, n_threads: u32, n_batch: u32, cache_kv: Quant) -> Self {
        Self {
            quant,
            n_ctx,
            n_threads,
            n_batch,
            cache_kv,
        }
    }

    /// Reject zero-sized fields before they reach the roofline math.
    ///
    /// A zero context cannot size a prefill, and zero threads/batches cannot
    /// describe a real llama.cpp run; failing loudly beats silently ranking a
    /// nonsense point.
    pub fn validate(&self) -> Result<(), TuneError> {
        if self.n_ctx == 0 {
            return Err(TuneError::InvalidConfig("n_ctx must be > 0".to_string()));
        }
        if self.n_threads == 0 {
            return Err(TuneError::InvalidConfig(
                "n_threads must be > 0".to_string(),
            ));
        }
        if self.n_batch == 0 {
            return Err(TuneError::InvalidConfig("n_batch must be > 0".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> TuneConfig {
        TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16)
    }

    #[test]
    fn validate_accepts_a_positive_config() {
        assert!(valid().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_fields() {
        for config in [
            TuneConfig {
                n_ctx: 0,
                ..valid()
            },
            TuneConfig {
                n_threads: 0,
                ..valid()
            },
            TuneConfig {
                n_batch: 0,
                ..valid()
            },
        ] {
            assert!(
                matches!(config.validate(), Err(TuneError::InvalidConfig(_))),
                "{config:?} should be rejected"
            );
        }
    }
}
