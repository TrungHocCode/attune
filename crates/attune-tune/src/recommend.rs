//! Recommend mode: rank catalog models by fit and predicted speed (M6).
//!
//! Pure: the same roofline maths as `tune`/`estimate`, applied to every
//! catalog model × file instead of one model's config grid.

use attune_estimate::model::registry::CatalogModel;
use attune_estimate::{
    kv_bytes_per_token, kv_filled_tokens, predict_decode_tps_from_bytes, HwInputs, Quant,
};
use serde::Serialize;

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
}

/// Inputs that are not per model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecommendOptions {
    /// Available RAM, or the user's `--max-ram` budget.
    pub budget_bytes: f64,
    /// Context to size the KV cache at, clamped to each model's maximum.
    pub n_ctx: u32,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
}

/// Rank every non-gated catalog model × file. Output order: the star, the
/// other tier picks (large → tiny), then the rest by tier and decode speed.
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
            let decode_tps = predict_decode_tps_from_bytes(
                file.size as f64,
                hw.bw_gbps,
                kv,
                kv_filled_tokens(opts.prompt_tokens, opts.gen_tokens, n_ctx),
            );
            let ttft_ms = hw.predict_prefill_ms(
                model.params,
                opts.prompt_tokens.min(u64::from(n_ctx)),
                file.quant,
            );
            let (Ok(decode_tps), Ok(ttft_ms)) = (decode_tps, ttft_ms) else {
                continue;
            };
            let ram_bytes = file.size + (kv * f64::from(n_ctx)) as u64;
            rows.push((
                RecommendRow {
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
                },
                model,
            ));
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
            int8_gops: Some(600.0),
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

    fn opts(budget: f64) -> RecommendOptions {
        RecommendOptions {
            budget_bytes: budget,
            n_ctx: 4096,
            prompt_tokens: 512,
            gen_tokens: 128,
        }
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
}
