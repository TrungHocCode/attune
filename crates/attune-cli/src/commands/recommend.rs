//! `attune tune` without `--model`: recommend catalog models that suit this
//! machine (M6). Measurement mirrors `tune`; ranking is the pure
//! [`attune_tune::recommend`]; this module only renders it.

use std::fmt::Write as _;

use anyhow::{anyhow, Result};
use attune_estimate::model::registry::{load_catalog, Catalog};
use attune_estimate::{HwInputs, Quant};
use attune_tune::recommend::{recommend, RecommendOptions, RecommendRow, Verdict};
use attune_tune::{Device, GpuTarget};

/// Context the KV cache is sized at for recommendations.
const RECOMMEND_CTX: u32 = 4096;
const PROMPT_TOKENS: u64 = 512;
const GEN_TOKENS: u64 = 128;
/// Non-pick rows shown per tier without `--all`.
const EXTRA_PER_TIER: usize = 2;

/// Parse `--max-ram` (`4G`, `4GB`, `512M`; decimal units).
pub fn parse_size(input: &str) -> Result<f64> {
    let text = input.trim().to_ascii_lowercase();
    let text = text.strip_suffix('b').unwrap_or(&text);
    let (number, scale) = match text.chars().last() {
        Some('g') => (&text[..text.len() - 1], 1e9),
        Some('m') => (&text[..text.len() - 1], 1e6),
        _ => (text, 1.0),
    };
    let value: f64 = number
        .trim()
        .parse()
        .map_err(|_| anyhow!("invalid size `{input}`; use e.g. 4G or 512M"))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(anyhow!("size `{input}` must be > 0"));
    }
    Ok(value * scale)
}

/// Rank the catalog against `max_ram` when given, else available RAM, judging
/// each model on the requested device.
pub fn recommend_report(
    catalog: &Catalog,
    hw: HwInputs,
    avail_bytes: f64,
    max_ram: Option<f64>,
    device: Device,
    gpu: Option<&GpuTarget>,
) -> Vec<RecommendRow> {
    let opts = RecommendOptions {
        budget_bytes: max_ram.unwrap_or(avail_bytes),
        n_ctx: RECOMMEND_CTX,
        prompt_tokens: PROMPT_TOKENS,
        gen_tokens: GEN_TOKENS,
        device,
        gpu: gpu.cloned(),
    };
    recommend(&catalog.models, hw, &opts)
}

fn gb(bytes: f64) -> String {
    format!("{:.1} GB", bytes / 1e9)
}

fn fit_label(verdict: Verdict) -> String {
    match verdict {
        Verdict::Fits => "✅".to_string(),
        Verdict::Tight => "⚠ tight".to_string(),
        Verdict::NeedsMore { short_bytes } => format!("❌ needs {} more", gb(short_bytes as f64)),
        Verdict::DoesNotFitVram => "❌ doesn't fit VRAM".to_string(),
    }
}

/// The `Runs on` cell: `CPU`, `GPU`, or `GPU+CPU <g>/<n>` for a split, where
/// `n` is the offloadable layer count (`n_layers + 1`: llama.cpp can also
/// offload the output layer, so a full offload reads `37/37`).
fn runs_on(row: &RecommendRow) -> String {
    match row.device.as_str() {
        "gpu" => match row.gpu_layers {
            Some(_) => "GPU".to_string(),
            // `--device gpu` asked for a GPU but not one layer fit VRAM.
            None => "-".to_string(),
        },
        "gpu+cpu" => match (row.gpu_layers, row.n_layers) {
            (Some(layers), Some(total)) => format!("GPU+CPU {layers}/{total}"),
            (Some(layers), None) => format!("GPU+CPU {layers}"),
            _ => "GPU+CPU".to_string(),
        },
        _ => "CPU".to_string(),
    }
}

/// The human table, footer included.
pub fn render_recommend(
    rows: &[RecommendRow],
    hw: HwInputs,
    budget_bytes: f64,
    origin: &str,
    date: &str,
    all: bool,
    gpu: Option<&GpuTarget>,
) -> String {
    let mut out = String::new();
    let mut machine = format!(
        "Machine: RAM {:.1} GB/s, {} for models · catalog {date} ({origin})",
        hw.bw_gbps,
        gb(budget_bytes)
    );
    if let Some(gpu) = gpu.filter(|_| rows.iter().any(|row| row.gpu_layers.is_some())) {
        let _ = write!(
            machine,
            " · GPU {} {:.1} GB",
            gpu.name,
            gpu.vram_free_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    let _ = writeln!(out, "{machine}\n");
    let _ = writeln!(
        out,
        "  {:<26} {:<7} {:>9} {:>8} {:>10} {:>12}  {:<14} Fit",
        "Model", "Quant", "Download", "RAM", "Speed", "First reply", "Runs on"
    );
    let mut shown_per_tier = [0_usize; 4];
    for row in rows {
        let slot = &mut shown_per_tier[row.tier as usize];
        if !all && !row.pick {
            if *slot >= EXTRA_PER_TIER {
                continue;
            }
            *slot += 1;
        }
        let mark = if row.star {
            '★'
        } else if row.pick {
            '•'
        } else {
            ' '
        };
        let _ = writeln!(
            out,
            "{mark} {:<26} {:<7} {:>9} {:>8} {:>6.1} tok/s {:>10.1} s  {:<14} {}",
            row.display,
            row.quant.as_str(),
            gb(row.download_bytes as f64),
            gb(row.ram_bytes as f64),
            row.decode_tps,
            row.ttft_ms / 1000.0,
            runs_on(row),
            fit_label(row.verdict),
        );
    }
    let _ = writeln!(out);
    match rows.iter().find(|r| r.star) {
        Some(star) => {
            let mut hint = format!("Next: attune run {}", star.model_id);
            if star.quant != Quant::Q4_K_M {
                hint.push_str(" --quant ");
                hint.push_str(star.quant.as_str());
            }
            let _ = writeln!(out, "{hint}");
        }
        None => {
            let smallest = rows.iter().map(|r| r.ram_bytes).min().unwrap_or(0) as f64;
            let _ = writeln!(
                out,
                "Nothing fits comfortably in {}: the smallest model needs {}. Close apps or pass --max-ram.",
                gb(budget_bytes),
                gb(smallest)
            );
        }
    }
    let _ = writeln!(
        out,
        "★ recommended · • best of its size · speed is limited by memory bandwidth, so smaller files are faster.\nPredictions are uncalibrated (about ±25%); `attune bench --model <file> --save` then `--calibrate` tightens them."
    );
    if !all {
        let _ = writeln!(out, "--all shows every model and quant.");
    }
    out
}

/// Recommend mode entry point, called by `tune::run` when `--model` and
/// `--params` are both absent.
#[allow(clippy::too_many_arguments)]
pub fn run(
    hw: HwInputs,
    avail_bytes: f64,
    max_ram: Option<f64>,
    all: bool,
    refresh_catalog: bool,
    json: bool,
    device: Device,
    gpu: Option<&GpuTarget>,
) -> Result<()> {
    let loaded = load_catalog(refresh_catalog);
    let rows = recommend_report(&loaded.catalog, hw, avail_bytes, max_ram, device, gpu);
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    for warning in &loaded.warnings {
        eprintln!("warning: {warning}");
    }
    print!(
        "{}",
        render_recommend(
            &rows,
            hw,
            max_ram.unwrap_or(avail_bytes),
            loaded.origin.label(),
            &loaded.date,
            all,
            gpu,
        )
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use attune_estimate::model::registry::{parse_catalog, SNAPSHOT};
    use attune_estimate::Quant;
    use attune_tune::recommend::Tier;

    fn hw() -> HwInputs {
        HwInputs {
            bw_gbps: 32.0,
            flops_gflops: 563.0,
            int8_gops: Some(1700.0),
        }
    }

    fn snapshot() -> attune_estimate::model::registry::Catalog {
        parse_catalog(SNAPSHOT).expect("snapshot")
    }

    #[test]
    fn parse_size_accepts_g_m_and_gb() {
        assert_eq!(parse_size("2G").expect("2G"), 2e9);
        assert_eq!(parse_size("4GB").expect("4GB"), 4e9);
        assert_eq!(parse_size("512m").expect("512m"), 512e6);
        assert_eq!(parse_size("1.5g").expect("1.5g"), 1.5e9);
        assert!(parse_size("lots").is_err());
        assert!(parse_size("0G").is_err());
    }

    #[test]
    fn tune_without_model_prints_recommend_table() {
        let rows = recommend_report(&snapshot(), hw(), 9e9, None, Device::Cpu, None);
        let text = render_recommend(
            &rows,
            hw(),
            9e9,
            "built-in, offline",
            "2026-11-26",
            false,
            None,
        );
        assert!(text.contains("Model"), "{text}");
        assert!(text.contains("Fit"), "{text}");
        assert!(text.lines().any(|line| line.starts_with('★')), "{text}");
        assert!(text.contains("Next: attune run"), "{text}");
        assert!(text.contains("uncalibrated"), "{text}");
        assert!(
            text.contains("catalog 2026-11-26 (built-in, offline)"),
            "{text}"
        );
    }

    #[test]
    fn max_ram_overrides_available() {
        let catalog = snapshot();
        let free = recommend_report(&catalog, hw(), 64e9, None, Device::Cpu, None);
        let capped = recommend_report(&catalog, hw(), 64e9, Some(2e9), Device::Cpu, None);
        let id = "qwen3-4b-instruct-2507";
        let row = |rows: &[RecommendRow]| {
            rows.iter()
                .find(|r| r.model_id == id && r.quant == Quant::Q4_K_M)
                .cloned()
                .expect("row")
        };
        assert_eq!(row(&free).verdict, Verdict::Fits);
        let Verdict::NeedsMore { short_bytes } = row(&capped).verdict else {
            panic!("{:?}", row(&capped).verdict)
        };
        let text = render_recommend(&capped, hw(), 2e9, "cached", "2026-11-26", true, None);
        let shown = format!("needs {:.1} GB more", short_bytes as f64 / 1e9);
        assert!(text.contains(&shown), "{shown}\n{text}");
    }

    #[test]
    fn default_table_shows_picks_plus_two_per_tier() {
        let rows = recommend_report(&snapshot(), hw(), 64e9, None, Device::Cpu, None);
        let body = |all: bool| {
            render_recommend(&rows, hw(), 64e9, "cached", "2026-11-26", all, None)
                .lines()
                .filter(|line| line.contains(" tok/s"))
                .count()
        };
        assert_eq!(body(true), rows.len());
        let expected: usize = [Tier::Tiny, Tier::Small, Tier::Medium, Tier::Large]
            .iter()
            .map(|&tier| {
                let picks = rows.iter().filter(|r| r.tier == tier && r.pick).count();
                let rest = rows.iter().filter(|r| r.tier == tier && !r.pick).count();
                picks + rest.min(2)
            })
            .sum();
        assert_eq!(body(false), expected);
        assert!(body(false) < body(true));
    }

    #[test]
    fn next_hint_uses_run_and_names_a_non_default_quant() {
        fn row(quant: Quant) -> RecommendRow {
            RecommendRow {
                model_id: "qwen3-4b-instruct-2507".to_string(),
                display: "Qwen/Qwen3 4B".to_string(),
                quant,
                download_bytes: 2_500_000_000,
                ram_bytes: 3_000_000_000,
                decode_tps: 20.0,
                ttft_ms: 500.0,
                verdict: Verdict::Fits,
                tier: Tier::Small,
                pick: true,
                star: true,
                device: "cpu".to_string(),
                gpu_layers: None,
                n_layers: None,
            }
        }

        let default = render_recommend(
            &[row(Quant::Q4_K_M)],
            hw(),
            9e9,
            "cached",
            "2026-11-26",
            false,
            None,
        );
        assert!(
            default.contains("Next: attune run qwen3-4b-instruct-2507\n"),
            "{default}"
        );

        let q8 = render_recommend(
            &[row(Quant::Q8_0)],
            hw(),
            9e9,
            "cached",
            "2026-11-26",
            false,
            None,
        );
        assert!(
            q8.contains("Next: attune run qwen3-4b-instruct-2507 --quant q8_0"),
            "{q8}"
        );
    }

    #[test]
    fn nothing_fits_says_how_much_to_free() {
        let rows = recommend_report(&snapshot(), hw(), 0.05e9, None, Device::Cpu, None);
        let text = render_recommend(&rows, hw(), 0.05e9, "cached", "2026-11-26", false, None);
        assert!(!text.contains("Next: attune tune --model"), "{text}");
        assert!(text.contains("Nothing fits"), "{text}");
    }

    fn row_for(device: &str, gpu_layers: Option<u32>, n_layers: Option<u32>) -> RecommendRow {
        RecommendRow {
            model_id: "qwen3-4b-instruct-2507".to_string(),
            display: "Qwen/Qwen3 4B".to_string(),
            quant: Quant::Q4_K_M,
            download_bytes: 2_500_000_000,
            ram_bytes: 3_000_000_000,
            decode_tps: 20.0,
            ttft_ms: 500.0,
            verdict: Verdict::Fits,
            tier: Tier::Small,
            pick: true,
            star: true,
            device: device.to_string(),
            gpu_layers,
            n_layers,
        }
    }

    #[test]
    fn runs_on_column_and_header_name_the_gpu_when_used() {
        let gpu = GpuTarget {
            name: "RTX 4050 Laptop".to_string(),
            vram_free_bytes: 5_921 * 1024 * 1024,
            vram_gbps: 163.0,
            prefill_tflops: 15.0,
        };
        let rows = [
            row_for("gpu", Some(37), Some(37)),
            row_for("gpu+cpu", Some(30), Some(49)),
            row_for("cpu", None, Some(32)),
        ];
        let text = render_recommend(&rows, hw(), 6e9, "cached", "2026-11-26", true, Some(&gpu));
        assert!(text.contains("Runs on"), "{text}");
        assert!(text.contains("GPU RTX 4050 Laptop 5.8 GB"), "{text}");
        assert!(text.contains("GPU+CPU 30/49"), "{text}");
        // Full offload of all 37 layers (36 blocks + the output layer).
        assert_eq!(runs_on(&row_for("gpu", Some(37), Some(37))), "GPU");
        assert_eq!(runs_on(&row_for("gpu", None, Some(37))), "-");
    }

    #[test]
    fn cpu_only_rows_never_name_a_gpu() {
        let gpu = GpuTarget {
            name: "RTX 4050 Laptop".to_string(),
            vram_free_bytes: 5_921 * 1024 * 1024,
            vram_gbps: 163.0,
            prefill_tflops: 15.0,
        };
        let text = render_recommend(
            &[row_for("cpu", None, Some(32))],
            hw(),
            6e9,
            "cached",
            "2026-11-26",
            true,
            Some(&gpu),
        );
        assert!(!text.contains("GPU RTX"), "{text}");
        assert!(text.contains("CPU"), "{text}");
    }
}
