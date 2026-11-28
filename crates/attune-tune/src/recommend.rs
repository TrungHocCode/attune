//! Recommend mode: rank catalog models by fit and predicted speed (M6).
//!
//! Pure: the same roofline maths as `tune`/`estimate`, applied to every
//! catalog model × file instead of one model's config grid.

use crate::sweep::{Device, GpuTarget};
use attune_estimate::model::registry::{CatalogFile, CatalogModel};
use attune_estimate::{
    default_compute_buffer_bytes, host_resident_bytes, kv_bytes_per_token, kv_filled_tokens,
    max_gpu_layers, predict_decode_tps_from_bytes, predict_split_decode_tps, HwInputs, Quant,
    SplitInputs,
};
use serde::Serialize;

/// PCIe throughput assumed when the CPU-resident weights have to stream to a
/// GPU-side compute buffer once per prompt batch (decimal GB/s, from the M7
/// spike). Matches the `tune` GPU TTFT model.
const GPU_PREFILL_PCIE_GBPS: f64 = 12e9;
/// Prompt batch the GPU TTFT model assumes (llama.cpp's default `n_batch`).
const GPU_PREFILL_BATCH: u64 = 512;

/// A fitting row needs this much spare RAM to be ✅ rather than ⚠.
pub const HEADROOM: f64 = 0.15;
/// Slowest decode a pick may have (tok/s): readable streaming speed.
pub const MIN_PICK_TPS: f64 = 5.0;
/// Longest first reply a pick may have (ms).
pub const MAX_PICK_TTFT_MS: f64 = 15_000.0;
/// Models within this params ratio of the biggest candidate count as the
/// same size; the newer one wins.
pub const SAME_SIZE: f64 = 0.15;

/// Size tier by parameter count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Tiny,
    Small,
    Medium,
    Large,
}

impl Tier {
    /// tiny < 2B, small 2–5B, medium 5–10B, large > 10B.
    pub fn of(params: u64) -> Self {
        match params {
            p if p < 2_000_000_000 => Self::Tiny,
            p if p < 5_000_000_000 => Self::Small,
            p if p < 10_000_000_000 => Self::Medium,
            _ => Self::Large,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}

/// Does the model fit the RAM budget?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Verdict {
    /// Fits with at least [`HEADROOM`] spare.
    Fits,
    /// Fits, but with less than [`HEADROOM`] spare.
    Tight,
    /// Does not fit; this many more bytes are needed.
    NeedsMore { short_bytes: u64 },
    /// `--device gpu` asked for a GPU, but not even one layer fits in VRAM.
    DoesNotFitVram,
}

impl Verdict {
    /// Higher is better; used to decide whether a GPU estimate beats the CPU
    /// one in `--device auto` mode.
    fn rank(self) -> u8 {
        match self {
            Verdict::Fits => 3,
            Verdict::Tight => 2,
            Verdict::NeedsMore { .. } => 1,
            Verdict::DoesNotFitVram => 0,
        }
    }
}

/// One catalog model × file, predicted on this machine.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecommendRow {
    pub model_id: String,
    pub display: String,
    pub quant: Quant,
    pub download_bytes: u64,
    pub ram_bytes: u64,
    pub decode_tps: f64,
    pub ttft_ms: f64,
    pub verdict: Verdict,
    pub tier: Tier,
    /// The recommended row of its tier.
    pub pick: bool,
    /// The overall recommendation (largest tier with a pick).
    pub star: bool,
    /// Where this row runs: `cpu`, `gpu` (full offload) or `gpu+cpu` (split).
    pub device: String,
    /// Layers offloaded to the GPU, when this row uses one.
    pub gpu_layers: Option<u32>,
    /// The model's layer count, so a split row can be shown as `g/n`.
    pub n_layers: Option<u32>,
}

/// Inputs that are not per model.
#[derive(Debug, Clone, PartialEq)]
pub struct RecommendOptions {
    /// Available RAM, or the user's `--max-ram` budget.
    pub budget_bytes: f64,
    /// Context to size the KV cache at, clamped to each model's maximum.
    pub n_ctx: u32,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    /// Which device each model should be judged on.
    pub device: Device,
    /// The GPU to plan against, when one is usable.
    pub gpu: Option<GpuTarget>,
}

/// Rank every non-gated catalog model × file. Output order: the star, the
/// other tier picks (large → tiny), then the rest by tier and decode speed.
///
/// Each row is judged on the device the caller asked for: `cpu` keeps the
/// pre-M7 RAM roofline, `gpu` plans a full or partial offload and never falls
/// back, and `auto` takes the GPU row only when it both fits and decodes
/// faster than the CPU row.
pub fn recommend(
    models: &[CatalogModel],
    hw: HwInputs,
    opts: &RecommendOptions,
) -> Vec<RecommendRow> {
    let mut rows: Vec<(RecommendRow, &CatalogModel)> = Vec::new();
    for model in models.iter().filter(|m| !m.gated) {
        let kv = kv_bytes_per_token(&model.arch.to_arch());
        let n_ctx = opts.n_ctx.min(model.ctx_default.max(1));
        for file in &model.files {
            let Some(cpu) = cpu_row(model, file, hw, opts, n_ctx, kv) else {
                continue;
            };
            let gpu = if opts.device == Device::Cpu {
                None
            } else {
                gpu_row(model, file, hw, opts, n_ctx, kv)
            };
            let row = match opts.device {
                Device::Cpu => cpu,
                Device::Gpu => gpu.unwrap_or_else(|| no_vram_row(&cpu, model)),
                Device::Auto => match gpu {
                    Some(gpu)
                        if gpu.verdict.rank() >= cpu.verdict.rank()
                            && gpu.decode_tps > cpu.decode_tps =>
                    {
                        gpu
                    }
                    _ => cpu,
                },
            };
            rows.push((row, model));
        }
    }
    mark_picks(&mut rows);

    let mut rows: Vec<RecommendRow> = rows.into_iter().map(|(row, _)| row).collect();
    rows.sort_by(|a, b| {
        b.star
            .cmp(&a.star)
            .then(b.pick.cmp(&a.pick))
            .then(b.tier.cmp(&a.tier))
            .then(b.decode_tps.total_cmp(&a.decode_tps))
    });
    rows
}

/// The CPU roofline row: unchanged from before M7 (a `gpu` target is ignored).
fn cpu_row(
    model: &CatalogModel,
    file: &CatalogFile,
    hw: HwInputs,
    opts: &RecommendOptions,
    n_ctx: u32,
    kv: f64,
) -> Option<RecommendRow> {
    let decode_tps = predict_decode_tps_from_bytes(
        file.size as f64,
        hw.bw_gbps,
        kv,
        kv_filled_tokens(opts.prompt_tokens, opts.gen_tokens, n_ctx),
    )
    .ok()?;
    let ttft_ms = hw
        .predict_prefill_ms(
            model.params,
            opts.prompt_tokens.min(u64::from(n_ctx)),
            file.quant,
        )
        .ok()?;
    let ram_bytes = file.size + (kv * f64::from(n_ctx)) as u64;
    Some(RecommendRow {
        model_id: model.id.clone(),
        display: model.display.clone(),
        quant: file.quant,
        download_bytes: file.size,
        ram_bytes,
        decode_tps,
        ttft_ms,
        verdict: verdict(ram_bytes as f64, opts.budget_bytes),
        tier: Tier::of(model.params),
        pick: false,
        star: false,
        device: "cpu".to_string(),
        gpu_layers: None,
        n_layers: Some(model.arch.n_layer),
    })
}

/// The GPU row for a full or partial offload, or `None` when nothing fits in
/// VRAM (or the GPU/layer data is unusable).
fn gpu_row(
    model: &CatalogModel,
    file: &CatalogFile,
    hw: HwInputs,
    opts: &RecommendOptions,
    n_ctx: u32,
    kv: f64,
) -> Option<RecommendRow> {
    let gpu = opts.gpu.as_ref()?;
    let n_layers = model.arch.n_layer;
    if n_layers == 0 {
        return None;
    }
    if !gpu.vram_gbps.is_finite() || gpu.vram_gbps <= 0.0 {
        return None;
    }
    if !gpu.prefill_tflops.is_finite() || gpu.prefill_tflops <= 0.0 {
        return None;
    }
    let kv_total = kv * f64::from(n_ctx);
    let gpu_layers = max_gpu_layers(
        file.size as f64,
        n_layers,
        kv_total,
        default_compute_buffer_bytes(n_ctx),
        gpu.vram_free_bytes,
    );
    if gpu_layers == 0 {
        return None;
    }
    let frac = f64::from(gpu_layers.min(n_layers)) / f64::from(n_layers);
    let decode_tps = predict_split_decode_tps(&SplitInputs {
        weight_bytes: file.size as f64,
        n_layers,
        gpu_layers,
        kv_bytes: kv * f64::from(kv_filled_tokens(opts.prompt_tokens, opts.gen_tokens, n_ctx)),
        vram_gbps: gpu.vram_gbps,
        ram_gbps: hw.bw_gbps,
    })
    .ok()?;

    // Prefill: GPU layers compute at `prefill_tflops`; the CPU-resident weights
    // stream over PCIe once per prompt batch (same model as `tune`).
    let prompt = opts.prompt_tokens.min(u64::from(n_ctx));
    let cpu_bytes = file.size as f64 * (1.0 - frac);
    let prefill_secs = 2.0 * model.params as f64 * prompt as f64 / (gpu.prefill_tflops * 1e12)
        + cpu_bytes / GPU_PREFILL_PCIE_GBPS * (prompt as f64 / GPU_PREFILL_BATCH as f64).ceil();
    let ttft_ms = prefill_secs * 1000.0;
    if !ttft_ms.is_finite() {
        return None;
    }

    // Only the host-resident share of weights and KV stays in system RAM:
    // the embedding table plus the CPU layers, and the un-offloaded KV.
    let ram_bytes =
        host_resident_bytes(file.size as f64, n_layers, gpu_layers, kv_total).round() as u64;
    Some(RecommendRow {
        model_id: model.id.clone(),
        display: model.display.clone(),
        quant: file.quant,
        download_bytes: file.size,
        ram_bytes,
        decode_tps,
        ttft_ms,
        verdict: verdict(ram_bytes as f64, opts.budget_bytes),
        tier: Tier::of(model.params),
        pick: false,
        star: false,
        device: if gpu_layers >= n_layers {
            "gpu".to_string()
        } else {
            "gpu+cpu".to_string()
        },
        gpu_layers: Some(gpu_layers),
        n_layers: Some(n_layers),
    })
}

/// `--device gpu` row for a model that cannot place even one layer in VRAM.
fn no_vram_row(cpu: &RecommendRow, model: &CatalogModel) -> RecommendRow {
    RecommendRow {
        ram_bytes: cpu.ram_bytes,
        decode_tps: 0.0,
        ttft_ms: 0.0,
        verdict: Verdict::DoesNotFitVram,
        device: "gpu".to_string(),
        gpu_layers: None,
        n_layers: Some(model.arch.n_layer),
        ..cpu.to_owned()
    }
}

fn verdict(ram: f64, budget: f64) -> Verdict {
    if budget >= ram * (1.0 + HEADROOM) {
        Verdict::Fits
    } else if budget >= ram {
        Verdict::Tight
    } else {
        Verdict::NeedsMore {
            short_bytes: (ram - budget).ceil() as u64,
        }
    }
}

/// Per tier: among models with a fitting row that meets the speed bar, take
/// the most params; a newer model within [`SAME_SIZE`] of it wins instead.
/// The pick is that model's smallest passing file. The star is the largest
/// tier that has a pick.
fn mark_picks(rows: &mut [(RecommendRow, &CatalogModel)]) {
    let passes = |row: &RecommendRow| {
        row.verdict == Verdict::Fits
            && row.decode_tps >= MIN_PICK_TPS
            && row.ttft_ms <= MAX_PICK_TTFT_MS
    };
    let mut best_tier: Option<Tier> = None;
    for tier in [Tier::Tiny, Tier::Small, Tier::Medium, Tier::Large] {
        // Smallest passing file per model in this tier.
        let mut candidates: Vec<usize> = Vec::new();
        for (i, (row, model)) in rows.iter().enumerate() {
            if row.tier != tier || !passes(row) {
                continue;
            }
            match candidates.iter().position(|&j| rows[j].1.id == model.id) {
                Some(k) if rows[candidates[k]].0.download_bytes > row.download_bytes => {
                    candidates[k] = i;
                }
                Some(_) => {}
                None => candidates.push(i),
            }
        }
        let Some(&biggest) = candidates.iter().max_by_key(|&&i| rows[i].1.params) else {
            continue;
        };
        let floor = rows[biggest].1.params as f64 / (1.0 + SAME_SIZE);
        let chosen = candidates
            .iter()
            .copied()
            .filter(|&i| rows[i].1.params as f64 >= floor)
            .max_by(|&a, &b| {
                let (ma, mb) = (rows[a].1, rows[b].1);
                ma.released
                    .cmp(&mb.released)
                    .then(ma.params.cmp(&mb.params))
            })
            .unwrap_or(biggest);
        rows[chosen].0.pick = true;
        best_tier = Some(tier);
    }
    if let Some(tier) = best_tier {
        if let Some((row, _)) = rows.iter_mut().find(|(r, _)| r.pick && r.tier == tier) {
            row.star = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attune_estimate::model::registry::{CatalogArch, CatalogFile, CatalogModel};

    const GB: u64 = 1_000_000_000;

    fn hw() -> HwInputs {
        HwInputs {
            bw_gbps: 32.0,
            flops_gflops: 563.0,
            int8_gops: Some(1700.0),
        }
    }

    fn file(quant: Quant, size: u64) -> CatalogFile {
        CatalogFile {
            quant,
            repo: "org/repo".to_string(),
            file: format!("m-{}.gguf", quant.as_str()),
            revision: "rev".to_string(),
            size,
            sha256: "a".repeat(64),
        }
    }

    fn model(
        id: &str,
        params: u64,
        released: Option<&str>,
        files: Vec<CatalogFile>,
    ) -> CatalogModel {
        CatalogModel {
            id: id.to_string(),
            display: id.to_string(),
            publisher: "org".to_string(),
            license: "apache-2.0".to_string(),
            released: released.map(str::to_string),
            gated: false,
            params,
            arch: CatalogArch {
                n_layer: 32,
                n_kv_heads: 8,
                head_dim: 128,
            },
            ctx_default: 32_768,
            files,
        }
    }

    fn opts_for(
        device: Device,
        gpu: Option<GpuTarget>,
        budget: f64,
        n_ctx: u32,
    ) -> RecommendOptions {
        RecommendOptions {
            budget_bytes: budget,
            n_ctx,
            prompt_tokens: 512,
            gen_tokens: 128,
            device,
            gpu,
        }
    }

    fn opts(budget: f64) -> RecommendOptions {
        opts_for(Device::Cpu, None, budget, 4096)
    }

    fn rtx_4050() -> GpuTarget {
        GpuTarget {
            name: "RTX 4050 Laptop".to_string(),
            vram_free_bytes: 5_921 * 1024 * 1024,
            vram_gbps: 163.0,
            prefill_tflops: 15.0,
        }
    }

    fn qwen3_8b() -> CatalogModel {
        let mut m = model(
            "qwen3-8b",
            8_190_735_360,
            None,
            vec![file(Quant::Q4_K_M, 5_027_783_488)],
        );
        m.arch = CatalogArch {
            n_layer: 36,
            n_kv_heads: 8,
            head_dim: 128,
        };
        m
    }

    fn qwen25_14b() -> CatalogModel {
        let mut m = model(
            "qwen2.5-14b",
            14_000_000_000,
            None,
            vec![file(Quant::Q4_K_M, 8_988_110_976)],
        );
        m.arch = CatalogArch {
            n_layer: 48,
            n_kv_heads: 8,
            head_dim: 128,
        };
        m
    }

    fn row<'a>(rows: &'a [RecommendRow], id: &str) -> &'a RecommendRow {
        rows.iter().find(|r| r.model_id == id).expect("row")
    }

    #[test]
    fn ram_is_weights_plus_kv_at_n_ctx_clamped() {
        let mut small_ctx = model("short", 1_000_000_000, None, vec![file(Quant::Q4_K_M, GB)]);
        small_ctx.ctx_default = 2048;
        let long_ctx = model("long", 1_000_000_000, None, vec![file(Quant::Q4_K_M, GB)]);
        let rows = recommend(&[small_ctx, long_ctx], hw(), &opts(100e9));
        let kv = 2.0 * 32.0 * 8.0 * 128.0 * 2.0;
        assert_eq!(row(&rows, "short").ram_bytes, GB + (kv * 2048.0) as u64);
        assert_eq!(row(&rows, "long").ram_bytes, GB + (kv * 4096.0) as u64);
    }

    #[test]
    fn verdicts_follow_headroom() {
        let m = model("m", 1_000_000_000, None, vec![file(Quant::Q4_K_M, GB)]);
        let ram = recommend(std::slice::from_ref(&m), hw(), &opts(100e9))[0].ram_bytes as f64;
        let at =
            |factor: f64| recommend(std::slice::from_ref(&m), hw(), &opts(ram * factor))[0].verdict;
        assert_eq!(at(1.20), Verdict::Fits);
        assert_eq!(at(1.05), Verdict::Tight);
        match at(0.8) {
            Verdict::NeedsMore { short_bytes } => {
                assert!(
                    (short_bytes as f64 - 0.2 * ram).abs() < 2.0,
                    "{short_bytes}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pick_is_most_params_meeting_the_bar_per_tier() {
        let rows = recommend(
            &[
                model(
                    "small-4b",
                    4 * GB,
                    None,
                    vec![file(Quant::Q4_K_M, 2_500_000_000)],
                ),
                model(
                    "small-3b",
                    3 * GB,
                    None,
                    vec![file(Quant::Q4_K_M, 1_900_000_000)],
                ),
                // 32 GB/s / 7.1 GB ≈ 4.5 tok/s: below the 5 tok/s bar.
                model(
                    "medium-9b",
                    9 * GB,
                    None,
                    vec![file(Quant::Q4_K_M, 7_100_000_000)],
                ),
            ],
            hw(),
            &opts(100e9),
        );
        assert!(row(&rows, "small-4b").pick);
        assert!(!row(&rows, "small-3b").pick);
        assert!(!row(&rows, "medium-9b").pick);
        assert_eq!(row(&rows, "small-4b").tier, Tier::Small);
    }

    #[test]
    fn date_breaks_ties_within_15_percent_params() {
        let rows = recommend(
            &[
                model(
                    "old-7.6b",
                    7_600_000_000,
                    Some("2024-01-01"),
                    vec![file(Quant::Q4_K_M, 4 * GB)],
                ),
                model(
                    "new-7.2b",
                    7_200_000_000,
                    Some("2025-06-01"),
                    vec![file(Quant::Q4_K_M, 4 * GB)],
                ),
            ],
            hw(),
            &opts(100e9),
        );
        assert!(row(&rows, "new-7.2b").pick);
        assert!(!row(&rows, "old-7.6b").pick);

        let rows = recommend(
            &[
                model(
                    "old-8b",
                    8 * GB,
                    Some("2023-01-01"),
                    vec![file(Quant::Q4_K_M, 4 * GB)],
                ),
                model(
                    "new-6b",
                    6 * GB,
                    Some("2025-06-01"),
                    vec![file(Quant::Q4_K_M, 3 * GB)],
                ),
            ],
            hw(),
            &opts(100e9),
        );
        assert!(row(&rows, "old-8b").pick);
        assert!(!row(&rows, "new-6b").pick);
    }

    #[test]
    fn star_is_the_largest_tier_with_a_pick() {
        let rows = recommend(
            &[
                model(
                    "tiny",
                    500_000_000,
                    None,
                    vec![file(Quant::Q4_K_M, 400_000_000)],
                ),
                model(
                    "small",
                    4 * GB,
                    None,
                    vec![file(Quant::Q4_K_M, 2_500_000_000)],
                ),
                // Does not fit the budget below, so medium has no pick.
                model("medium", 8 * GB, None, vec![file(Quant::Q4_K_M, 5 * GB)]),
            ],
            hw(),
            &opts(4e9),
        );
        let stars: Vec<&str> = rows
            .iter()
            .filter(|r| r.star)
            .map(|r| r.model_id.as_str())
            .collect();
        assert_eq!(stars, ["small"]);
        assert_eq!(rows[0].model_id, "small");
        assert!(row(&rows, "tiny").pick);
    }

    #[test]
    fn gated_or_missing_quant_is_never_picked() {
        let mut gated = model(
            "gated-4b",
            4 * GB,
            None,
            vec![file(Quant::Q4_K_M, 2_500_000_000)],
        );
        gated.gated = true;
        // Only a q8_0 file, too slow at 7.5 GB: no other quant to fall back on.
        let only_q8 = model(
            "q8-only-8b",
            8 * GB,
            None,
            vec![file(Quant::Q8_0, 7_500_000_000)],
        );
        let rows = recommend(&[gated, only_q8], hw(), &opts(100e9));
        assert!(rows.iter().all(|r| r.model_id != "gated-4b"));
        assert!(!row(&rows, "q8-only-8b").pick);
    }

    #[test]
    fn smallest_passing_quant_is_the_pick() {
        let rows = recommend(
            &[model(
                "m4b",
                4 * GB,
                None,
                vec![
                    file(Quant::Q8_0, 4_300_000_000),
                    file(Quant::Q4_K_M, 2_500_000_000),
                ],
            )],
            hw(),
            &opts(100e9),
        );
        let picks: Vec<Quant> = rows.iter().filter(|r| r.pick).map(|r| r.quant).collect();
        assert_eq!(picks, [Quant::Q4_K_M]);
    }

    #[test]
    fn predictions_match_estimate() {
        let m = model("m", 4 * GB, None, vec![file(Quant::Q4_K_M, 2_500_000_000)]);
        let r = &recommend(&[m], hw(), &opts(100e9))[0];
        let kv = 2.0 * 32.0 * 8.0 * 128.0 * 2.0;
        let decode = predict_decode_tps_from_bytes(
            2_500_000_000.0,
            32.0,
            kv,
            kv_filled_tokens(512, 128, 4096),
        )
        .expect("decode");
        let ttft = hw()
            .predict_prefill_ms(4 * GB, 512, Quant::Q4_K_M)
            .expect("ttft");
        assert!((r.decode_tps - decode).abs() < 1e-9);
        assert!((r.ttft_ms - ttft).abs() < 1e-9);
    }

    #[test]
    fn qwen3_8b_fully_offloads_when_it_fits_and_beats_the_cpu_by_3x() {
        let m = qwen3_8b();
        let cpu = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Cpu, None, 6e9, 1024),
        );
        let gpu = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Auto, Some(rtx_4050()), 6e9, 1024),
        );
        assert_eq!(gpu[0].device, "gpu", "{:?}", gpu[0]);
        assert_eq!(gpu[0].gpu_layers, Some(36));
        assert!(
            gpu[0].decode_tps > 3.0 * cpu[0].decode_tps,
            "GPU {:.1} tok/s must beat CPU {:.1} tok/s by 3x",
            gpu[0].decode_tps,
            cpu[0].decode_tps
        );
    }

    #[test]
    fn qwen3_8b_at_the_default_context_still_fully_offloads_and_beats_the_cpu_by_3x() {
        // At 4096 tokens the smaller compute buffer and host-resident
        // embeddings leave room for all 36 layers.
        let m = qwen3_8b();
        let cpu = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Cpu, None, 6e9, 4096),
        );
        let gpu = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Auto, Some(rtx_4050()), 6e9, 4096),
        );
        assert_eq!(gpu[0].device, "gpu", "{:?}", gpu[0]);
        assert_eq!(gpu[0].gpu_layers, Some(36));
        assert!(
            gpu[0].decode_tps > 3.0 * cpu[0].decode_tps,
            "GPU {:.1} tok/s must beat CPU {:.1} tok/s by 3x",
            gpu[0].decode_tps,
            cpu[0].decode_tps
        );
    }

    #[test]
    fn full_gpu_rows_report_the_host_resident_ram() {
        // A fully offloaded model still needs RAM for the token-embedding
        // table left on the host: ~8% of the file, not 0.
        let m = qwen3_8b();
        let rows = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Gpu, Some(rtx_4050()), 6e9, 4096),
        );
        assert_eq!(rows[0].device, "gpu", "{:?}", rows[0]);
        assert_eq!(rows[0].gpu_layers, Some(36));
        let expected = (5_027_783_488.0 * (1.0 - attune_estimate::GPU_RESIDENT_WEIGHT_FRACTION))
            .round() as u64;
        assert_eq!(rows[0].ram_bytes, expected);
        assert!(rows[0].ram_bytes > 0);
    }

    #[test]
    fn a_14b_q4_only_partially_fits_the_gpu() {
        let rows = recommend(
            &[qwen25_14b()],
            hw(),
            &opts_for(Device::Auto, Some(rtx_4050()), 6e9, 4096),
        );
        assert_eq!(rows[0].device, "gpu+cpu", "{:?}", rows[0]);
        let layers = rows[0].gpu_layers.expect("split offload");
        assert!((1..48).contains(&layers), "{layers}");
        assert_eq!(rows[0].verdict, Verdict::Fits);
    }

    #[test]
    fn cpu_device_rows_are_unchanged_by_a_gpu_target() {
        let m = qwen3_8b();
        let without = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Cpu, None, 6e9, 4096),
        );
        let with = recommend(
            std::slice::from_ref(&m),
            hw(),
            &opts_for(Device::Cpu, Some(rtx_4050()), 6e9, 4096),
        );
        assert_eq!(without, with);
        assert_eq!(with[0].device, "cpu");
        assert_eq!(with[0].gpu_layers, None);
    }

    #[test]
    fn gpu_device_without_a_fit_is_marked_and_never_picked() {
        // No GPU at all: only the "does not fit in VRAM" rows are emitted.
        let rows = recommend(&[qwen3_8b()], hw(), &opts_for(Device::Gpu, None, 6e9, 4096));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].verdict, Verdict::DoesNotFitVram);
        assert_eq!(rows[0].device, "gpu");
        assert_eq!(rows[0].gpu_layers, None);
        assert!(!rows[0].pick && !rows[0].star);

        // A model too large for any layer gets the same marker.
        let mut huge = qwen3_8b();
        huge.files[0].size = 400_000_000_000;
        let rows = recommend(
            &[huge],
            hw(),
            &opts_for(Device::Gpu, Some(rtx_4050()), 6e9, 4096),
        );
        assert_eq!(rows[0].verdict, Verdict::DoesNotFitVram);
    }

    #[test]
    fn gpu_fields_serialize_additively() {
        let rows = recommend(
            &[qwen3_8b()],
            hw(),
            &opts_for(Device::Auto, Some(rtx_4050()), 6e9, 1024),
        );
        let json = serde_json::to_value(&rows[0]).expect("serialize");
        assert_eq!(json["device"], "gpu");
        assert_eq!(json["gpu_layers"], 36);
        assert_eq!(json["n_layers"], 36);
    }
}
