//! One point in the tuning grid.

use attune_estimate::Quant;
use serde::{Deserialize, Serialize};

use crate::TuneError;

/// Maximum number of points [`expand_full_grid`] keeps. A 4-format ×
/// 2-context × 2-thread × 2-batch × 2-KV grid is exactly 64 points, so this
/// guard only bites when the axes grow. The CLI applies the user-facing
/// `--max-configs` cap again in [`crate::prune`].
pub const FULL_GRID_CAP: usize = 64;

/// A candidate inference configuration.
///
/// `quant` is the weight format and `cache_kv` the KV-cache format; both reuse
/// [`Quant`] so the sweep cannot invent a second format vocabulary. Slice 1
/// ranks with the weights-only roofline, so `n_threads`, `n_batch` and
/// `cache_kv` are carried through to the report but do not change the score
/// yet — the fields are here so the grid/JSON shape is stable when later
/// slices model them. The llama.cpp launch fields (`n_ubatch`, `n_gpu_layers`,
/// `mmap`, ...) are likewise carried through unchanged for the M4 backend
/// slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Physical micro-batch size (`llama.cpp -ub`).
    #[serde(default = "default_n_ubatch")]
    pub n_ubatch: u32,
    /// Number of layers offloaded to the GPU (`llama.cpp -ngl`).
    #[serde(default)]
    pub n_gpu_layers: u32,
    /// Memory-map weights instead of reading them eagerly.
    #[serde(default = "default_true")]
    pub mmap: bool,
    /// Lock weights in RAM (prevents swapping).
    #[serde(default)]
    pub mlock: bool,
    /// Enable flash attention.
    #[serde(default = "default_true")]
    pub flash_attn: bool,
    /// NUMA policy passed to llama.cpp (`local`, `distribute`, ...).
    #[serde(default = "default_numa")]
    pub numa: String,
    /// KV-cache format.
    #[serde(default)]
    pub cache_kv: Quant,
    /// Explicit K-cache format (`llama.cpp --cache-type-k`).
    #[serde(default = "default_cache_type")]
    pub cache_type_k: String,
    /// Explicit V-cache format (`llama.cpp --cache-type-v`).
    #[serde(default = "default_cache_type")]
    pub cache_type_v: String,
    /// CPU affinity mask/range, when the launcher should pin threads.
    #[serde(default)]
    pub cpu_affinity: Option<String>,
}

fn default_n_ubatch() -> u32 {
    128
}

fn default_true() -> bool {
    true
}

fn default_numa() -> String {
    "local".to_string()
}

fn default_cache_type() -> String {
    "q8_0".to_string()
}

impl TuneConfig {
    /// Build one grid point, filling the launch fields with safe defaults.
    pub fn new(quant: Quant, n_ctx: u32, n_threads: u32, n_batch: u32, cache_kv: Quant) -> Self {
        Self {
            quant,
            n_ctx,
            n_threads,
            n_batch,
            n_ubatch: default_n_ubatch(),
            n_gpu_layers: 0,
            mmap: default_true(),
            mlock: false,
            flash_attn: default_true(),
            numa: default_numa(),
            cache_kv,
            cache_type_k: default_cache_type(),
            cache_type_v: default_cache_type(),
            cpu_affinity: None,
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

/// Expand the full slice-2 grid for every supported weight format.
///
/// Axis values are fixed by the M4 plan: threads
/// `{max(1, cores/2), cores}`, contexts `{2048, 4096}`, prompt batches
/// `{128, 512}` and KV formats `{fp16, q8_0}`. The loop order — quant, context,
/// thread, batch, KV (innermost) — is deterministic, and the list is truncated
/// at [`FULL_GRID_CAP`]. The second return value is the number of points
/// dropped by that cap.
pub fn expand_full_grid(physical_cores: u32) -> (Vec<TuneConfig>, usize) {
    expand_full_grid_with_quants(
        physical_cores,
        &[Quant::Fp16, Quant::Q8_0, Quant::Q4_0, Quant::Q4_K_M],
    )
}

/// Same as [`expand_full_grid`] but restricted to the caller's weight formats.
///
/// The CLI uses this so `--quant q4_k_m,q8_0` expands only the requested
/// formats while still sharing the plan's thread/context/batch/KV axes and the
/// deterministic cap.
pub fn expand_full_grid_with_quants(
    physical_cores: u32,
    quants: &[Quant],
) -> (Vec<TuneConfig>, usize) {
    let cores = physical_cores.max(1);
    let half = (cores / 2).max(1);
    let thread_axis = if half == cores {
        vec![cores]
    } else {
        vec![half, cores]
    };
    let ctx_axis = [2_048_u32, 4_096];
    let batch_axis = [128_u32, 512];
    let kv_axis = [Quant::Fp16, Quant::Q8_0];

    let total =
        quants.len() * ctx_axis.len() * thread_axis.len() * batch_axis.len() * kv_axis.len();
    let mut configs = Vec::with_capacity(total.min(FULL_GRID_CAP));

    'outer: for &quant in quants {
        for &n_ctx in &ctx_axis {
            for &n_threads in &thread_axis {
                for &n_batch in &batch_axis {
                    for &cache_kv in &kv_axis {
                        if configs.len() == FULL_GRID_CAP {
                            break 'outer;
                        }
                        configs.push(TuneConfig::new(quant, n_ctx, n_threads, n_batch, cache_kv));
                    }
                }
            }
        }
    }

    let pruned_by_cap = total.saturating_sub(configs.len());
    (configs, pruned_by_cap)
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

    #[test]
    fn new_fills_the_launch_defaults() {
        let config = valid();
        assert_eq!(config.n_ubatch, 128);
        assert_eq!(config.n_gpu_layers, 0);
        assert!(config.mmap);
        assert!(!config.mlock);
        assert!(config.flash_attn);
        assert_eq!(config.numa, "local");
        assert_eq!(config.cache_type_k, "q8_0");
        assert_eq!(config.cache_type_v, "q8_0");
        assert_eq!(config.cpu_affinity, None);
    }

    #[test]
    fn full_grid_is_deterministic_and_respects_the_axes() {
        let (first, pruned) = expand_full_grid(8);
        let (second, pruned_again) = expand_full_grid(8);
        assert_eq!(first, second);
        assert_eq!(pruned, 0);
        assert_eq!(pruned_again, 0);
        // 4 formats × 2 ctx × 2 threads × 2 batches × 2 KV = 64.
        assert_eq!(first.len(), 64);
        assert_eq!(first[0].quant, Quant::Fp16);
        assert_eq!(first[0].n_ctx, 2_048);
        assert_eq!(first[0].n_threads, 4);
        assert_eq!(first[0].n_batch, 128);
        assert_eq!(first[0].cache_kv, Quant::Fp16);
        assert!(first.iter().all(|config| config.n_ctx <= 4_096));
        assert!(first
            .iter()
            .all(|config| config.n_threads == 4 || config.n_threads == 8));
        assert!(first
            .iter()
            .all(|config| config.n_batch == 128 || config.n_batch == 512));
        assert!(first
            .iter()
            .all(|config| config.cache_kv == Quant::Fp16 || config.cache_kv == Quant::Q8_0));
    }

    #[test]
    fn full_grid_dedupes_the_thread_axis_on_a_single_core() {
        let (configs, _) = expand_full_grid(1);
        // 4 × 2 × 1 × 2 × 2 = 32.
        assert_eq!(configs.len(), 32);
        assert!(configs.iter().all(|config| config.n_threads == 1));
    }

    #[test]
    fn full_grid_cap_is_reported_when_formats_exceed_it() {
        let many = [Quant::Fp16; 5];
        let (configs, pruned) = expand_full_grid_with_quants(8, &many);
        let total = 5 * 2 * 2 * 2 * 2;
        assert_eq!(configs.len(), FULL_GRID_CAP);
        assert_eq!(pruned, total - FULL_GRID_CAP);
    }
}
