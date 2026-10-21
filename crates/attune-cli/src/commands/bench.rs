//! `attune bench` — micro-benchmarks.
//!
//! Quick mode (`--quick`) returns the best-of-reps read bandwidth number from
//! M2 slice 1. Full mode (the default now that slice 2 is shipped) runs the
//! read / write / copy kernels in series and reports the median of `reps`
//! timed passes for each. Both modes accept `--json` and print the same
//! machine-readable shape the human output is rendered from (no surprises).

use anyhow::Result;
use attune_bench::bandwidth::{quick_full, quick_read, FullBandwidthResult};
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

fn print_pretty_full(result: &FullBandwidthResult, hw: &HardwareInfo) {
    let theoretical = hw.memory.theoretical_bandwidth_gbps;
    print_full_line("read", result.read.median_gbps, theoretical);
    print_full_line("write", result.write.median_gbps, theoretical);
    print_full_line("copy", result.copy.median_gbps, theoretical);
    println!(
        "Working set: {} MiB, {} worker threads, {} timed passes (1 warm-up discarded) per kernel",
        result.buffer_mib, result.threads, result.reps,
    );
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
    let result = quick_full()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_pretty_full(&result, &hw);
    }
    Ok(())
}
