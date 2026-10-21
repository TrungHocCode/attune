//! `attune bench` — micro-benchmarks.
//!
//! Quick mode (`--quick`) returns the best-of-reps read bandwidth number
//! from M2 slice 1. Full mode (the default now that slice 3 is shipped)
//! runs the read / write / copy kernels back-to-back, then an FP32 FMA
//! throughput sweep, and decorates the human output with theoretical-bandwidth
//! and theoretical-FLOPS context. Both modes accept `--json` and print the
//! same machine-readable shape the human output is rendered from (no surprises).

use anyhow::Result;
use attune_bench::bandwidth::{quick_read, FullBandwidthResult};
use attune_bench::flops::{FlopsResult, WARMUP_PASSES as FLOPS_WARMUP_PASSES};
use attune_bench::monitor::ResourceStats;
use attune_bench::quick_full_with_flops;
use attune_hal::HardwareInfo;

fn print_pretty_quick(result: &attune_bench::bandwidth::ReadResult) {
    println!(
        "Memory bandwidth (read): {:.1} GB/s (buffer {} MiB, {} threads, {} reps)",
        result.read_gbps, result.buffer_mib, result.threads, result.reps,
    );
}

/// Human-readable bytes: pick KB / MB / GB based on magnitude. Used only in
/// informational footers (not in the JSON contract).
fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else if bytes > 0 {
        format!("{} B", bytes)
    } else {
        "unknown".to_string()
    }
}

fn pct_of(measured: f64, theoretical: f64) -> String {
    if !measured.is_finite() || !theoretical.is_finite() || theoretical <= 0.0 {
        return String::new();
    }
    let pct = measured / theoretical * 100.0;
    format!(" ({:.0}% of theoretical {:.1})", pct, theoretical)
}

/// One line of full-mode human output. We keep the line *prefix* identical
/// across kernels so `grep`-ing for `read` / `write` / `copy` works.
fn print_full_line(label: &str, median_gbps: f64, theoretical: f64) {
    if theoretical > 0.0 {
        println!(
            "Memory bandwidth ({}):  {:.1} GB/s{}",
            label,
            median_gbps,
            pct_of(median_gbps, theoretical),
        );
    } else {
        println!(
            "Memory bandwidth ({}):  {:.1} GB/s (theoretical: unknown)",
            label, median_gbps,
        );
    }
}

/// One FP32 FLOPS line for full-mode human output. Same decoration pattern
/// as the bandwidth lines: either `XX% of theoretical YY.Y` or
/// `(theoretical: unknown)` depending on whether the peak was quoteable.
fn print_full_flops_line(flops: &FlopsResult) {
    if flops.theoretical_gflops > 0.0 {
        println!(
            "FP32 FLOPS:             {:.1} GFLOPS (path: {}{})",
            flops.fp32_gflops,
            flops.path.as_str(),
            pct_of(flops.fp32_gflops, flops.theoretical_gflops),
        );
    } else {
        println!(
            "FP32 FLOPS:             {:.1} GFLOPS (path: {}, theoretical: unknown)",
            flops.fp32_gflops,
            flops.path.as_str(),
        );
    }
}

/// One resource-usage line for full-mode human output. Power is Linux-only
/// (RAPL), so the absence of a reading is rendered as text, never as a gap.
fn print_resources_line(res: &ResourceStats) {
    let power = match res.power_avg_watts {
        Some(watts) => format!("{watts:.1} W avg"),
        None => "power unavailable (no counter)".to_string(),
    };
    println!(
        "Resource usage:          RAM peak {}, CPU avg {:.1}% / peak {:.1}%, {} ({} samples @ {} ms)",
        human_bytes(res.ram_peak_bytes),
        res.cpu_avg_pct,
        res.cpu_peak_pct,
        power,
        res.samples,
        res.interval_ms,
    );
}

fn print_pretty_full(result: &FullBandwidthResult, hw: &HardwareInfo) {
    let theoretical_bw = hw.memory.theoretical_bandwidth_gbps;
    print_full_line("read", result.read.median_gbps, theoretical_bw);
    print_full_line("write", result.write.median_gbps, theoretical_bw);
    print_full_line("copy", result.copy.median_gbps, theoretical_bw);
    println!(
        "Working set: {} MiB, {} worker threads, {} timed passes (1 warm-up discarded) per bandwidth kernel",
        result.buffer_mib, result.threads, result.reps,
    );

    if let Some(flops) = result.flops.as_ref() {
        print_full_flops_line(flops);
        println!(
            "FMA kernel: {} threads, {} timed passes ({} warm-up discarded)",
            flops.threads, flops.reps, FLOPS_WARMUP_PASSES,
        );
    } else {
        println!("FP32 FLOPS: skipped (no flops block in report)");
    }

    match result.resources.as_ref() {
        Some(res) => print_resources_line(res),
        None => println!("Resource usage:          not monitored"),
    }

    println!(
        "DRAM (best-effort): speed {} MT/s, {} DIMMs → {} peak",
        hw.memory.speed_mts,
        hw.memory.channels,
        if hw.memory.theoretical_bandwidth_gbps > 0.0 {
            format!("{:.1} GB/s", hw.memory.theoretical_bandwidth_gbps)
        } else {
            "unknown".to_string()
        },
    );
    println!("Total RAM: {}", human_bytes(hw.memory.total_bytes));
}

pub fn run(quick: bool, json: bool) -> Result<()> {
    if quick {
        let result = quick_read()?;
        if json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            print_pretty_quick(&result);
        }
        return Ok(());
    }

    // Full mode: detect hardware so we can decorate the human output with
    // theoretical-bandwidth context (AGENTS.md §6 — JSON is the contract).
    let hw = HardwareInfo::detect()?;
    let result = quick_full_with_flops()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_pretty_full(&result, &hw);
    }
    Ok(())
}
