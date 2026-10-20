//! `attune bench` — micro-benchmarks (M2 slice 1: read bandwidth only).

use anyhow::{bail, Result};
use attune_bench::bandwidth::{quick_read, ReadResult};

fn print_pretty(result: &ReadResult) {
    println!(
        "Memory bandwidth (read): {:.1} GB/s (buffer {} MiB, {} threads, {} reps)",
        result.read_gbps, result.buffer_mib, result.threads, result.reps,
    );
}

pub fn run(quick: bool, json: bool) -> Result<()> {
    if !quick {
        // Phase discipline (AGENTS.md §4): write/copy kernels, FLOPS, power
        // monitoring and llama.cpp e2e ship in later M2 slices.
        bail!("`attune bench` without --quick is not implemented yet — use `attune bench --quick` (read bandwidth).");
    }

    let result = quick_read()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_pretty(&result);
    }
    Ok(())
}
