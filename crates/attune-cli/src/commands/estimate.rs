//! `attune estimate` — roofline prediction (M3 slices 1-2).
//!
//! Thin CLI: parse the model/quant strings, resolve the hardware inputs, call
//! the library, and render either the JSON contract or the human lines. With
//! no overrides the bandwidth comes from the same quick probe as
//! `attune bench --quick` and the FP32 peak comes from `attune-hal`; if the
//! CPU exposes no usable theoretical peak the CLI asks for `--gflops` instead
//! of guessing.
//!
//! All math (decode, prefill, ridge, classification) lives in
//! `attune-estimate`; this file only wires flags to it.

use anyhow::{anyhow, Context, Result};
use attune_bench::{run_quick, BandwidthReport};
use attune_estimate::{
    classify, decode_ai, parse_params, predict_prefill_ms, ridge_ai, EstimateInputs,
    EstimateReport, HwInputs, Quant,
};
use attune_hal::HardwareInfo;

/// Resolve the FP32 peak: explicit override first, then the HAL's theoretical
/// estimate, then a clean error that names the fix.
fn resolve_gflops(gflops: Option<f64>) -> Result<f64> {
    match gflops {
        Some(value) => Ok(value),
        None => {
            let hw = HardwareInfo::detect()?;
            let theoretical = hw.cpu.theoretical_fp32_gflops;
            if theoretical.is_finite() && theoretical > 0.0 {
                Ok(theoretical)
            } else {
                Err(anyhow!(
                    "this CPU exposes no theoretical FP32 peak; pass --gflops <GFLOPS>"
                ))
            }
        }
    }
}

/// Resolve bandwidth: explicit override first, then the M2 quick probe.
fn resolve_bandwidth(bandwidth_gbps: Option<f64>) -> Result<f64> {
    match bandwidth_gbps {
        Some(value) => Ok(value),
        None => {
            let report = run_quick().context("failed to measure memory bandwidth")?;
            Ok(match report.bandwidth {
                BandwidthReport::Quick(read) => read.read_gbps,
                BandwidthReport::Full(full) => full.read.median_gbps,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    params: &str,
    quant: &str,
    bandwidth_gbps: Option<f64>,
    gflops: Option<f64>,
    prompt_tokens: u64,
    _gen_tokens: u64,
    _ctx: u32,
    json: bool,
) -> Result<()> {
    let params_count = parse_params(params)?;
    let quant: Quant = quant.parse()?;

    // Resolve the compute peak before the ~1 s bandwidth probe so a missing
    // theoretical peak fails fast.
    let flops_gflops = resolve_gflops(gflops)?;
    let bw_gbps = resolve_bandwidth(bandwidth_gbps)?;

    let hw = HwInputs {
        bw_gbps,
        flops_gflops,
    };
    let decode_tps = hw.predict_decode_tps(params_count, quant.bytes_per_param())?;
    let prefill_ms_ttft = predict_prefill_ms(params_count, prompt_tokens, flops_gflops)?;
    let ai_star = ridge_ai(flops_gflops, bw_gbps)?;
    let workload_ai = decode_ai(quant.bytes_per_param())?;
    let (bottleneck, margin_pct) = classify(workload_ai, ai_star);

    let report = EstimateReport::new(params.trim(), quant, decode_tps)
        .with_inputs(EstimateInputs {
            bw_gbps,
            gflops: flops_gflops,
        })
        .with_prefill_ms(prefill_ms_ttft)
        .with_ridge(ai_star, margin_pct)
        .with_bottleneck(bottleneck);

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Decode:      {:.1} tok/s ({} {})",
            report.decode_tps, report.model, report.quant,
        );
        println!(
            "TTFT:        {:.1} ms (prefill {} tok)",
            prefill_ms_ttft, prompt_tokens
        );
        println!(
            "Bottleneck:  {} (decode AI {:.2} vs ridge {:.2} FLOP/byte, {:.1}% off)",
            report.bottleneck, workload_ai, ai_star, margin_pct,
        );
        println!(
            "Ridge:       {:.2} FLOP/byte ({:.1} GFLOPS / {:.1} GB/s)",
            ai_star, flops_gflops, bw_gbps,
        );
    }
    Ok(())
}
