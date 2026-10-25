//! `attune bench` — micro-benchmarks.
//!
//! Quick mode (`--quick`) returns the best-of-reps read bandwidth number
//! from M2 slice 1. Full mode (the default now that slice 3 is shipped)
//! runs the read / write / copy kernels back-to-back, then an FP32 FMA
//! throughput sweep, and decorates the human output with theoretical-bandwidth
//! and theoretical-FLOPS context. Both modes accept `--json` and print the
//! same machine-readable shape the human output is rendered from (no surprises).
//!
//! `--model <PATH>` adds llama.cpp. In quick mode that is a 1-token load
//! probe (M2 slice 5). In full mode it is the end-to-end decode run (slice
//! 6): it loads the model through `llama-cli`, scrapes the `llama_perf`
//! block, and reports TTFT / decode tok/s / RAM peak / CPU avg. The full
//! mode does not also run the probe, so the model loads once (#31); a
//! millisecond pre-flight (binary resolves, model readable, version parses)
//! runs before the kernels so a misconfigured backend still fails fast. JSON
//! output nests the results as a `"llama"` object plus an `"e2e"` key; in
//! full mode the `"llama"` object is derived from the e2e run. Without
//! `--model` nothing here changes.
//!
//! `--save <PATH>` writes the JSON report to a file in addition to
//! printing it; the file is the JSON contract verbatim and lives
//! wherever the user points it (typically `%TEMP%/attune/...` — never
//! the repo). Optional knobs (only meaningful with `--model`):
//!
//! * `--prompt-tokens <N>` — size of the synthetic prefill (`-p`).
//!   Default 512.
//! * `--gen-tokens <N>` — number of decode tokens (`-n`). Default 128.
//! * `--ctx <N>` — context window (`-c`). Default 2048.
//! * `--threads <N>` — thread pin (`-t`); omit to let llama.cpp pick.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use attune_bench::bandwidth::{FullBandwidthResult, ReadResult};
use attune_bench::e2e::{preflight, E2eConfig, E2eResult};
use attune_bench::flops::{FlopsResult, WARMUP_PASSES as FLOPS_WARMUP_PASSES};
use attune_bench::llama::{probe_model, resolve_llama_cli, LlamaProbeReport, LOAD_TIMEOUT};
use attune_bench::monitor::ResourceStats;
use attune_bench::{run_full, run_quick, BandwidthReport, BenchReport};
use attune_hal::HardwareInfo;

fn print_pretty_quick(result: &ReadResult) {
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

fn print_pretty_full(
    bandwidth: &FullBandwidthResult,
    flops: Option<&FlopsResult>,
    res: Option<&ResourceStats>,
    hw: &HardwareInfo,
) {
    let theoretical_bw = hw.memory.theoretical_bandwidth_gbps;
    print_full_line("read", bandwidth.read.median_gbps, theoretical_bw);
    print_full_line("write", bandwidth.write.median_gbps, theoretical_bw);
    print_full_line("copy", bandwidth.copy.median_gbps, theoretical_bw);
    println!(
        "Working set: {} MiB, {} worker threads, {} timed passes (1 warm-up discarded) per bandwidth kernel",
        bandwidth.buffer_mib, bandwidth.threads, bandwidth.reps,
    );
    if let Some(flops) = flops {
        print_full_flops_line(flops);
        println!(
            "FMA kernel: {} threads, {} timed passes ({} warm-up discarded)",
            flops.threads, flops.reps, FLOPS_WARMUP_PASSES,
        );
    } else {
        println!("FP32 FLOPS: skipped (no flops block in report)");
    }

    match res {
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

// ---- llama.cpp probe + e2e (`--model`) -----------------------------------

/// One-line summary of the slice-5 llama.cpp probe (1-token model load).
fn print_pretty_llama(report: &LlamaProbeReport) {
    println!(
        "llama.cpp ({}) — model load OK ({}, {:.1} s)",
        report.version,
        report.model,
        report.elapsed_ms / 1_000.0,
    );
    if let Some(perf) = report.perf.as_ref() {
        println!(
            "llama.cpp timings:     decode {:.1} tok/s ({} tokens), prefill {:.1} tok/s ({} tokens)",
            perf.eval_tps(),
            perf.eval_tokens,
            perf.prompt_tps(),
            perf.prompt_tokens,
        );
    }
}

/// The slice-6 e2e block in human mode. Two lines: the inference summary
/// (TTFT, decode tok/s, token counts) and the resource readings.
fn print_pretty_e2e(e2e: &E2eResult) {
    if e2e.is_complete() {
        println!(
            "llama.cpp e2e ({}):     TTFT {:.1} ms, decode {:.1} tok/s ({} prompt / {} gen tokens)",
            e2e.engine, e2e.ttft_ms, e2e.tok_s, e2e.prompt_tokens, e2e.eval_tokens,
        );
    } else {
        println!(
            "llama.cpp e2e ({}):     decode measurement unavailable (no perf block)",
            e2e.engine,
        );
    }
    let power = match e2e.power_avg_watts {
        Some(watts) => format!("{watts:.1} W avg"),
        None => "power unavailable (no counter)".to_string(),
    };
    println!(
        "e2e resource usage:     RAM peak {}, CPU avg {:.1}%, {}",
        human_bytes(e2e.ram_peak_bytes),
        e2e.cpu_avg_pct,
        power,
    );
}

/// Resolve `llama-cli` and load `model` through it (the slice-5 smoke
/// test). Quick mode only: full mode lets the e2e run prove the load.
fn probe_llama(llama_cli: Option<&Path>, model: &Path) -> Result<LlamaProbeReport> {
    let cli = resolve_llama_cli(llama_explicit(llama_cli)?)?;
    // `?` keeps `BenchError`'s Display message as the top-level error and its
    // OS error as the `Caused by:` chain.
    Ok(probe_model(&cli, model, LOAD_TIMEOUT, None)?)
}

/// Same resolution chain as `probe_llama`, but returns a `PathBuf` (which
/// the e2e runner needs to spawn the child). Helper so the resolution
/// detail lives in exactly one place.
fn resolve_e2e_cli(llama_cli: Option<&Path>) -> Result<PathBuf> {
    let cli = resolve_llama_cli(llama_explicit(llama_cli)?)?;
    Ok(cli)
}

/// Clap hands us an OS string, but the resolver takes `&str` (paths that reach
/// the JSON contract as text). A non-UTF-8 path is a user mistake worth naming
/// rather than silently ignoring.
fn llama_explicit(path: Option<&Path>) -> Result<Option<&str>> {
    match path {
        None => Ok(None),
        Some(p) => p
            .to_str()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--llama-cli path is not valid UTF-8: {}",
                    p.to_string_lossy()
                )
            })
            .map(Some),
    }
}

/// Build the [`E2eConfig`] from CLI overrides. The model path is required
/// here; the four numeric knobs stay at their [`e2e`] defaults when the
/// caller did not override them.
fn make_e2e_config(
    model: &Path,
    prompt_tokens: Option<usize>,
    gen_tokens: Option<u32>,
    ctx_size: Option<u32>,
    threads: Option<u32>,
) -> E2eConfig {
    let mut cfg = E2eConfig::new(model.to_path_buf());
    if let Some(n) = prompt_tokens {
        cfg.prompt_tokens = n;
    }
    if let Some(n) = gen_tokens {
        cfg.gen_tokens = n;
    }
    if let Some(n) = ctx_size {
        cfg.ctx_size = n;
    }
    if let Some(n) = threads {
        cfg.threads = n;
    }
    cfg
}

// ---- JSON output ---------------------------------------------------------

/// Build the JSON envelope printed by `--model --json` and saved by
/// `--save`. Returning a [`serde_json::Value`] makes print and save share
/// exactly the same shape, so a saved file mirrors what would have been
/// printed to stdout.
///
/// Layout mirrors the slice-5 contract so older consumers still find
/// their `llama` key: the probe sits next to the bench results, and the
/// e2e block lives inside the bench object (`bench.e2e`) for consistency
/// with the standalone [`BenchReport`] shape.
fn json_envelope_with_model(
    probe: &LlamaProbeReport,
    report: &BenchReport,
    bandwidth_value: &serde_json::Value,
) -> Result<serde_json::Value> {
    let mut bench = serde_json::Map::new();
    bench.insert(
        "schema".to_string(),
        serde_json::Value::String(report.schema.clone()),
    );
    bench.insert(
        "mode".to_string(),
        serde_json::Value::String(report.mode.clone()),
    );
    bench.insert("bandwidth".to_string(), bandwidth_value.clone());
    if let Some(flops) = report.flops.as_ref() {
        bench.insert("flops".to_string(), serde_json::to_value(flops)?);
    }
    if let Some(resources) = report.resources.as_ref() {
        bench.insert("resources".to_string(), serde_json::to_value(resources)?);
    }
    if let Some(e2e_block) = report.e2e.as_ref() {
        bench.insert("e2e".to_string(), serde_json::to_value(e2e_block)?);
    }
    Ok(serde_json::json!({
        "llama": probe,
        "bench": bench,
    }))
}

/// Write the JSON document to `path`. Matches what would otherwise have
/// been printed to stdout — a saved report is interchangeable with the
/// printed one. The caller passes an already-built [`serde_json::Value`]
/// so the save path does not duplicate any envelope-building logic.
fn write_json_value(path: &Path, value: &serde_json::Value) -> Result<()> {
    let pretty = serde_json::to_string_pretty(value)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    std::fs::write(path, pretty)
        .with_context(|| format!("failed to write report to {}", path.display()))?;
    Ok(())
}

/// Build the bandwidth-only subshape for the slice-5-compatible JSON
/// envelope. Top-level `read` / `write` / `copy` keys, not the
/// `bandwidth: { … }` wrapper — keeping the slices-1-4 contracts intact.
fn bandwidth_value(bandwidth: &BandwidthReport, flops: &Option<FlopsResult>) -> serde_json::Value {
    let mut value = match bandwidth {
        BandwidthReport::Quick(r) => serde_json::to_value(r).expect("ReadResult serialises"),
        BandwidthReport::Full(r) => serde_json::json!({
            "read":   { "median_gbps": r.read.median_gbps,   "best_gbps": r.read.best_gbps },
            "write":  { "median_gbps": r.write.median_gbps,  "best_gbps": r.write.best_gbps },
            "copy":   { "median_gbps": r.copy.median_gbps,   "best_gbps": r.copy.best_gbps },
            "buffer_mib": r.buffer_mib,
            "threads": r.threads,
            "reps": r.reps,
        }),
    };
    if let (Some(obj), Some(flops)) = (value.as_object_mut(), flops.as_ref()) {
        if let BandwidthReport::Full(_) = bandwidth {
            obj.insert(
                "flops".to_string(),
                serde_json::to_value(flops).expect("FlopsResult serialises"),
            );
        }
    }
    value
}

// ---- entry point ---------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn run(
    quick: bool,
    json: bool,
    model: Option<&Path>,
    llama_cli: Option<&Path>,
    save: Option<&Path>,
    prompt_tokens: Option<usize>,
    gen_tokens: Option<u32>,
    ctx_size: Option<u32>,
    threads: Option<u32>,
) -> Result<()> {
    // Quick mode keeps the 1-token load probe: it is the only llama.cpp step
    // there.
    let probe = match (quick, model) {
        (true, Some(model_path)) => Some(probe_llama(llama_cli, model_path)?),
        _ => None,
    };

    // Full mode skips the probe, because the e2e run loads the model itself
    // (#31). The millisecond pre-flight (llama-cli resolves, model readable,
    // version parses) still runs before the kernels, so a bad setup fails
    // fast without loading the model twice.
    let e2e_cli = match model {
        Some(model_path) if !quick => {
            let cli = resolve_e2e_cli(llama_cli)?;
            preflight(&cli, model_path)?;
            Some(cli)
        }
        _ => None,
    };
    let e2e_cfg = model.map(|m| make_e2e_config(m, prompt_tokens, gen_tokens, ctx_size, threads));

    if quick {
        let report = run_quick()?;
        let bw_value = bandwidth_value(&report.bandwidth, &report.flops);
        if json {
            let value = match probe.as_ref() {
                None => serde_json::to_value(&report)?,
                Some(p) => json_envelope_with_model(p, &report, &bw_value)?,
            };
            println!("{}", serde_json::to_string_pretty(&value)?);
        } else {
            match &report.bandwidth {
                BandwidthReport::Quick(r) => print_pretty_quick(r),
                BandwidthReport::Full(_) => unreachable!("quick mode returns ReadResult"),
            }
            if let Some(p) = probe.as_ref() {
                print_pretty_llama(p);
            }
        }
        if let Some(path) = save {
            let value = match probe.as_ref() {
                None => serde_json::to_value(&report)?,
                Some(p) => json_envelope_with_model(p, &report, &bw_value)?,
            };
            write_json_value(path, &value)?;
        }
        return Ok(());
    }

    // Full mode.
    let hw = HardwareInfo::detect()?;
    let report = run_full(&hw, e2e_cfg.as_ref(), e2e_cli.as_deref())?;
    let bw_value = bandwidth_value(&report.bandwidth, &report.flops);
    // The e2e run stands in for the probe, so the `--model` JSON keeps its
    // `{ "llama": …, "bench": … }` envelope.
    let load = match (report.e2e.as_ref(), e2e_cli.as_deref()) {
        (Some(e2e_block), Some(cli)) => Some(e2e_block.load_report(cli)),
        _ => None,
    };
    if json {
        let value = match load.as_ref() {
            None => serde_json::to_value(&report)?,
            Some(l) => json_envelope_with_model(l, &report, &bw_value)?,
        };
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        print_report_pretty(&report, &hw);
        if let Some(e2e_block) = report.e2e.as_ref() {
            print_pretty_e2e(e2e_block);
        }
    }
    if let Some(path) = save {
        let value = match load.as_ref() {
            None => serde_json::to_value(&report)?,
            Some(l) => json_envelope_with_model(l, &report, &bw_value)?,
        };
        write_json_value(path, &value)?;
    }
    Ok(())
}

/// Pretty-print a full-mode [`BenchReport`]. Mirrors the slice-5 layout
/// so the human output is unchanged for everyone not using `--model` —
/// the e2e block is the only addition. Full mode is the only path that
/// reaches here, so a `BandwidthReport::Full` is the only case worth
/// handling.
fn print_report_pretty(report: &BenchReport, hw: &HardwareInfo) {
    match &report.bandwidth {
        BandwidthReport::Full(r) => {
            print_pretty_full(r, report.flops.as_ref(), report.resources.as_ref(), hw);
        }
        BandwidthReport::Quick(r) => {
            // Quick mode in the CLI goes through `print_pretty_quick` and
            // returns early; this branch only fires when an external
            // caller (a future test or a downstream consumer) hands us a
            // quick-mode report and asks for the long render anyway.
            let theoretical_bw = hw.memory.theoretical_bandwidth_gbps;
            print_full_line("read", r.read_gbps, theoretical_bw);
            println!(
                "Working set: {} MiB, {} worker threads, {} timed passes (1 warm-up discarded)",
                r.buffer_mib, r.threads, r.reps,
            );
            print_resources_or_skip(report.resources.as_ref());
        }
    }
}

fn print_resources_or_skip(res: Option<&ResourceStats>) {
    match res {
        Some(res) => print_resources_line(res),
        None => println!("Resource usage:          not monitored"),
    }
    // Quick-mode report: no FLOPS, no DRAM footer to render. The header
    // already covered the read kernel.
}
