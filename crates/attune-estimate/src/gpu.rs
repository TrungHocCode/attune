//! GPU + CPU split prediction (M7 slice 3A).
//!
//! A model with `n_gpu_layers` of its layers on the GPU and the rest on the
//! CPU is memory-bound on both devices during decode (see the "M7 GPU spike"
//! in `docs/benchmark-methodology.md`). Per generated token the GPU streams
//! its share of the weights and KV cache from VRAM, and the CPU streams the
//! remaining share from RAM; the two proceed concurrently, so the time per
//! token is the *sum* of the two streaming times:
//!
//! ```text
//! t = (weights + kv) × frac     / vram_bandwidth      (GPU layers)
//!   + (weights + kv) × (1-frac) / ram_bandwidth       (CPU layers)
//! tok/s = 1 / t
//! ```
//!
//! Everything here is pure: it takes plain bandwidth numbers (GB/s) and byte
//! counts. Wiring in the detected GPU, the benchmarked CPU bandwidth and the
//! VRAM footprint is the caller's job (`tune`/`recommend`).

use crate::EstimateError;

/// Share of the quoted VRAM bandwidth llama.cpp reaches during decode.
/// Measured on the reference RTX 4050 Laptop 6 GB (192 GB/s nominal): an 8B
/// Q4_K_M fully offloaded decodes at 34.6 tok/s × 5.03 GB ≈ 174 GB/s (≈0.9),
/// a 4B at 61.5 × 2.5 GB ≈ 154 GB/s (≈0.8). Use the middle: 0.85.
pub const GPU_DECODE_EFFICIENCY: f64 = 0.85;

/// VRAM kept free above the model so the desktop/driver and transient
/// allocations cannot push the split over the edge. The spike showed 30 layers
/// of the 14B left ~280 MB free and misbehaved; ~512 MiB is the safe margin.
pub const VRAM_SAFETY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// Theoretical VRAM bandwidth in GB/s for known GPUs.
///
/// Matched case-insensitively by substring, longest entry first so the laptop
/// SKUs win over their desktop namesakes (`RTX 3060 Laptop` before `RTX 3060`)
/// and the Ti suffix wins over the base card (`RTX 4060 Ti` before `RTX 4060`).
const GPU_BANDWIDTH_TABLE: &[(&str, f64)] = &[
    ("RTX 4050 Laptop", 192.0),
    ("RTX 4060 Laptop", 256.0),
    ("RTX 4070 Laptop", 256.0),
    ("RTX 4080 Laptop", 432.0),
    ("RTX 4090 Laptop", 576.0),
    ("RTX 4060 Ti", 288.0),
    ("RTX 4060", 272.0),
    ("RTX 4070", 504.0),
    ("RTX 4080", 717.0),
    ("RTX 4090", 1008.0),
    ("RTX 3050", 224.0),
    ("RTX 3060 Laptop", 336.0),
    ("RTX 3060", 360.0),
    ("RTX 3070", 448.0),
    ("RTX 3080", 760.0),
    ("RTX 3090", 936.0),
];

/// Look up the theoretical VRAM bandwidth (GB/s) for a GPU name.
///
/// Matching is case-insensitive and by substring, preferring the longest
/// table entry so `RTX 3060 Laptop` is not mistaken for `RTX 3060`. Returns
/// `None` for an unknown name so the caller can fall back to a measured value.
pub fn gpu_bandwidth_gbps(name: &str) -> Option<f64> {
    let needle = name.to_ascii_lowercase();
    let mut best: Option<(&str, f64)> = None;
    for &(label, bw) in GPU_BANDWIDTH_TABLE {
        if needle.contains(&label.to_ascii_lowercase()) {
            let longer = best.is_none_or(|(prev, _)| label.len() > prev.len());
            if longer {
                best = Some((label, bw));
            }
        }
    }
    best.map(|(_, bw)| bw)
}

/// Inputs to a GPU/CPU split decode prediction.
///
/// All bandwidths are *effective* (already multiplied by the device's decode
/// efficiency or calibrated): the GPU side should pass
/// `table_bandwidth * GPU_DECODE_EFFICIENCY`, the CPU side the measured RAM
/// bandwidth from the existing calibration path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitInputs {
    /// Total model weight footprint in bytes (e.g. the GGUF file size).
    pub weight_bytes: f64,
    /// Number of transformer layers in the model.
    pub n_layers: u32,
    /// Number of layers offloaded to the GPU (`0..=n_layers`).
    pub gpu_layers: u32,
    /// KV-cache bytes read per generated token (filled context).
    pub kv_bytes: f64,
    /// Effective VRAM bandwidth in decimal GB/s.
    pub vram_gbps: f64,
    /// Effective system-RAM bandwidth in decimal GB/s.
    pub ram_gbps: f64,
}

/// Predict decode throughput (tok/s) for a GPU/CPU layer split.
///
/// GPU and CPU layers stream concurrently during decode, so the time per
/// token is the sum of the two streaming times. With `gpu_layers == 0` this
/// reduces exactly to the CPU-only roofline for the same bytes.
pub fn predict_split_decode_tps(inputs: &SplitInputs) -> Result<f64, EstimateError> {
    if inputs.n_layers == 0 {
        return Err(EstimateError::InvalidModel(
            "layer count must be > 0".to_string(),
        ));
    }
    if !inputs.weight_bytes.is_finite() || inputs.weight_bytes <= 0.0 {
        return Err(EstimateError::InvalidModel(
            "model weight footprint must be finite and > 0".to_string(),
        ));
    }
    if !inputs.kv_bytes.is_finite() || inputs.kv_bytes < 0.0 {
        return Err(EstimateError::BadHw(format!(
            "KV bytes must be finite and >= 0, got {}",
            inputs.kv_bytes
        )));
    }
    if !inputs.vram_gbps.is_finite() || inputs.vram_gbps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "VRAM bandwidth must be finite and > 0 GB/s, got {}",
            inputs.vram_gbps
        )));
    }
    if !inputs.ram_gbps.is_finite() || inputs.ram_gbps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "RAM bandwidth must be finite and > 0 GB/s, got {}",
            inputs.ram_gbps
        )));
    }

    let frac = split_fraction(inputs.gpu_layers, inputs.n_layers);
    let total_bytes = inputs.weight_bytes + inputs.kv_bytes;
    let vram_secs = total_bytes * frac / (inputs.vram_gbps * 1e9);
    let ram_secs = total_bytes * (1.0 - frac) / (inputs.ram_gbps * 1e9);
    let secs = vram_secs + ram_secs;

    if !secs.is_finite() || secs <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "prediction is not a finite positive time, got {secs}"
        )));
    }
    let tps = 1.0 / secs;
    if !tps.is_finite() || tps <= 0.0 {
        return Err(EstimateError::BadHw(format!(
            "prediction is not a finite positive number, got {tps}"
        )));
    }
    Ok(tps)
}

/// Fraction of layers offloaded to the GPU, clamped to `0..=1`.
fn split_fraction(gpu_layers: u32, n_layers: u32) -> f64 {
    if n_layers == 0 {
        return 0.0;
    }
    gpu_layers.min(n_layers) as f64 / n_layers as f64
}

/// VRAM needed to hold `gpu_layers` of the model: the offloaded share of the
/// weights and the KV cache plus the fixed compute buffer.
pub fn vram_needed_bytes(
    weight_bytes: f64,
    n_layers: u32,
    gpu_layers: u32,
    kv_total_bytes: f64,
    compute_buffer_bytes: f64,
) -> f64 {
    let frac = split_fraction(gpu_layers, n_layers);
    (weight_bytes + kv_total_bytes) * frac + compute_buffer_bytes
}

/// Estimated llama.cpp compute buffer for a context of `n_ctx` tokens:
/// 300 MiB base plus 0.1 MiB per context token.
///
/// Measured in the GPU spike as ~0.6–0.8 GB of VRAM used beyond the weights
/// and KV cache at a 4096-token context.
pub fn default_compute_buffer_bytes(n_ctx: u32) -> f64 {
    const MIB: f64 = 1024.0 * 1024.0;
    300.0 * MIB + 0.1 * MIB * n_ctx as f64
}

/// Largest number of GPU layers (`0..=n_layers`) whose VRAM footprint fits in
/// `vram_free_bytes` minus [`VRAM_SAFETY_MARGIN_BYTES`].
///
/// Returns `0` if even a single layer does not fit. Weights and the KV cache
/// scale with the offloaded fraction; the compute buffer does not.
pub fn max_gpu_layers(
    weight_bytes: f64,
    n_layers: u32,
    kv_total_bytes: f64,
    compute_buffer_bytes: f64,
    vram_free_bytes: u64,
) -> u32 {
    if n_layers == 0 {
        return 0;
    }
    let budget = vram_free_bytes as f64 - VRAM_SAFETY_MARGIN_BYTES as f64;
    if !budget.is_finite() || budget <= 0.0 {
        return 0;
    }
    let total_offload = weight_bytes + kv_total_bytes;
    if !total_offload.is_finite() || total_offload <= 0.0 {
        return 0;
    }

    // needed(g) = total_offload × g/n_layers + compute_buffer <= budget
    let per_layer = total_offload / n_layers as f64;
    let headroom = budget - compute_buffer_bytes;
    if !per_layer.is_finite() || per_layer <= 0.0 {
        return 0;
    }
    if headroom <= 0.0 {
        return 0;
    }
    let fit = (headroom / per_layer).floor();
    if !fit.is_finite() || fit <= 0.0 {
        return 0;
    }
    (fit as u32).min(n_layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RTX_4050_EFFECTIVE: f64 = 192.0 * GPU_DECODE_EFFICIENCY;
    const RAM_GBPS: f64 = 36.0;

    fn within(got: f64, want: f64, pct: f64) -> bool {
        (got - want).abs() / want <= pct
    }

    #[test]
    fn bandwidth_table_matches_known_gpus() {
        assert_eq!(
            gpu_bandwidth_gbps("NVIDIA GeForce RTX 4050 Laptop GPU"),
            Some(192.0)
        );
        assert_eq!(gpu_bandwidth_gbps("rtx 4090"), Some(1008.0));
        assert_eq!(
            gpu_bandwidth_gbps("NVIDIA RTX 3060 Laptop GPU"),
            Some(336.0)
        );
    }

    #[test]
    fn laptop_and_ti_names_win_over_desktop_base() {
        // Laptop SKU must not be read as its desktop namesake.
        assert_eq!(gpu_bandwidth_gbps("RTX 4060 Laptop GPU"), Some(256.0));
        assert_eq!(gpu_bandwidth_gbps("RTX 4060"), Some(272.0));
        // Ti suffix must win over the base card.
        assert_eq!(gpu_bandwidth_gbps("RTX 4060 Ti"), Some(288.0));
    }

    #[test]
    fn unknown_gpu_is_none() {
        assert_eq!(gpu_bandwidth_gbps("Intel Arc A770"), None);
        assert_eq!(gpu_bandwidth_gbps(""), None);
    }

    #[test]
    fn zero_gpu_layers_matches_cpu_only_roofline() {
        let inputs = SplitInputs {
            weight_bytes: 8_988_110_976.0,
            n_layers: 48,
            gpu_layers: 0,
            kv_bytes: 805_306_368.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        let split = predict_split_decode_tps(&inputs).expect("split");
        let cpu_only =
            crate::predict_decode_tps_from_bytes(inputs.weight_bytes, RAM_GBPS, inputs.kv_bytes, 1)
                .expect("cpu roofline");
        assert!(
            (split - cpu_only).abs() < 1e-9,
            "split {split} vs cpu {cpu_only}"
        );
    }

    #[test]
    fn qwen25_14b_28_layers_matches_spike() {
        // 7.4 tok/s measured (CUDA) / 7.0 (Vulkan); additive model -> 7.35.
        let inputs = SplitInputs {
            weight_bytes: 8_988_110_976.0,
            n_layers: 48,
            gpu_layers: 28,
            kv_bytes: 0.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        let tps = predict_split_decode_tps(&inputs).expect("predict");
        assert!(within(tps, 7.4, 0.15), "got {tps} tok/s, want ~7.4");
    }

    #[test]
    fn qwen25_14b_20_layers_matches_spike() {
        // 4.9 tok/s measured. The purely additive model lands ~5.9: real
        // partial splits pay extra CPU/GPU overlap overhead, so allow 25%.
        let inputs = SplitInputs {
            weight_bytes: 8_988_110_976.0,
            n_layers: 48,
            gpu_layers: 20,
            kv_bytes: 0.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        let tps = predict_split_decode_tps(&inputs).expect("predict");
        assert!(within(tps, 4.9, 0.25), "got {tps} tok/s, want ~4.9");
    }

    #[test]
    fn qwen3_8b_full_offload_matches_spike() {
        // 34.6 tok/s measured (CUDA); additive model -> 32.5.
        let inputs = SplitInputs {
            weight_bytes: 5_027_783_488.0,
            n_layers: 36,
            gpu_layers: 36,
            kv_bytes: 0.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        let tps = predict_split_decode_tps(&inputs).expect("predict");
        assert!(within(tps, 34.6, 0.15), "got {tps} tok/s, want ~34.6");
    }

    #[test]
    fn qwen3_4b_full_offload_matches_spike() {
        // 61.4 tok/s measured (CUDA); additive model -> 65.4.
        let inputs = SplitInputs {
            weight_bytes: 2_497_281_120.0,
            n_layers: 36,
            gpu_layers: 36,
            kv_bytes: 0.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        let tps = predict_split_decode_tps(&inputs).expect("predict");
        assert!(within(tps, 61.4, 0.25), "got {tps} tok/s, want ~61.4");
    }

    #[test]
    fn rejects_bad_split_inputs() {
        let good = SplitInputs {
            weight_bytes: 8_988_110_976.0,
            n_layers: 48,
            gpu_layers: 28,
            kv_bytes: 0.0,
            vram_gbps: RTX_4050_EFFECTIVE,
            ram_gbps: RAM_GBPS,
        };
        for bad in [
            SplitInputs {
                n_layers: 0,
                ..good
            },
            SplitInputs {
                weight_bytes: 0.0,
                ..good
            },
            SplitInputs {
                weight_bytes: f64::NAN,
                ..good
            },
            SplitInputs {
                vram_gbps: 0.0,
                ..good
            },
            SplitInputs {
                ram_gbps: -1.0,
                ..good
            },
            SplitInputs {
                kv_bytes: -1.0,
                ..good
            },
        ] {
            assert!(
                predict_split_decode_tps(&bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn compute_buffer_scales_with_context() {
        let at_4096 = default_compute_buffer_bytes(4096);
        assert!((at_4096 - (300.0 + 0.1 * 4096.0) * 1024.0 * 1024.0).abs() < 1e-6);
        assert!(default_compute_buffer_bytes(8192) > at_4096);
    }

    #[test]
    fn vram_needed_scales_weights_and_kv_with_fraction() {
        let needed = vram_needed_bytes(6_000.0, 48, 24, 1_000.0, 500.0);
        // (6000 + 1000) * 0.5 + 500 = 4000
        assert!((needed - 4_000.0).abs() < 1e-9, "got {needed}");
    }

    #[test]
    fn max_layers_8b_fits_all() {
        // Weights + Qwen3 8B KV at ctx 4096 (~576 MiB) fit the ~5.4 GiB free
        // the spike reported, so every layer is offloaded.
        let kv = 36.0 * 8.0 * 128.0 * 2.0 * 2.0 * 4096.0;
        assert_eq!(kv, 603_979_776.0);
        let g = max_gpu_layers(5_027_783_488.0, 36, kv, 0.0, 5_921 * 1024 * 1024);
        assert_eq!(g, 36);
    }

    #[test]
    fn max_layers_14b_lands_in_spike_range() {
        // Qwen2.5-14B at ctx 4096: 48 × 8 × 128 × 2 × 2 × 4096 = 805 MB KV.
        let kv = 48.0 * 8.0 * 128.0 * 2.0 * 2.0 * 4096.0;
        assert_eq!(kv, 805_306_368.0);
        let g = max_gpu_layers(
            8_988_110_976.0,
            48,
            kv,
            default_compute_buffer_bytes(4096),
            5_921 * 1024 * 1024,
        );
        assert!((24..=29).contains(&g), "got {g}, want 24..=29");
    }

    #[test]
    fn max_layers_zero_when_nothing_fits() {
        let g = max_gpu_layers(8_988_110_976.0, 48, 0.0, 0.0, VRAM_SAFETY_MARGIN_BYTES);
        assert_eq!(g, 0);
    }

    #[test]
    fn max_layers_respects_safety_margin() {
        // Only 2 bytes over the safety margin are free, less than one
        // 10-byte layer, so nothing can be offloaded.
        let free = VRAM_SAFETY_MARGIN_BYTES + 2;
        let g = max_gpu_layers(10.0, 1, 0.0, 0.0, free);
        assert_eq!(g, 0, "2 bytes of headroom cannot hold a 10-byte layer");
    }
}
