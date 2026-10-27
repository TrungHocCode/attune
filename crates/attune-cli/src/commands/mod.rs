//! Command dispatch — every subcommand is a thin call into a library
//! (AGENTS.md §1: no logic in `main.rs`, all of it in the crates).

mod bench;
mod detect;
mod estimate;

use anyhow::Result;
use clap::{ArgGroup, Subcommand};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Command {
    /// Detect hardware (M1: CPU/SIMD/cache/RAM via raw-cpuid + sysinfo)
    Detect {
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
    },
    /// Benchmark this machine (M2). With `--model` (M2 slice 5 + 6) also
    /// drives `llama-cli` for a llama.cpp load smoke-test and a full
    /// end-to-end decode run.
    Bench {
        /// Quick memory-bandwidth sweep (just the read kernel)
        #[arg(long)]
        quick: bool,
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
        /// GGUF model to load-test with llama.cpp (skips the llama.cpp
        /// probe when omitted)
        #[arg(long, value_name = "PATH")]
        model: Option<PathBuf>,
        /// Path to the llama.cpp `llama-cli` binary (default: `--llama-cli`,
        /// then $ATTUNE_LLAMA_CLI, then a PATH search for `llama-cli`/`main`)
        #[arg(long, value_name = "PATH")]
        llama_cli: Option<PathBuf>,
        /// Save the JSON report to this file (in addition to printing it).
        /// With `--model`, the saved file mirrors the printed envelope.
        #[arg(long, value_name = "PATH")]
        save: Option<PathBuf>,
        /// Approximate prefill token count for the e2e run (default 512).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        prompt_tokens: Option<usize>,
        /// Number of decode tokens for the e2e run (default 128).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        gen_tokens: Option<u32>,
        /// Context window size passed to llama.cpp (default 2048).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        ctx: Option<u32>,
        /// Pin llama.cpp to this thread count (-t). Omit to let llama.cpp
        /// decide. Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        threads: Option<u32>,
    },
    /// Predict decode throughput, prefill/TTFT, the roofline bottleneck and
    /// confidence intervals (M3 slices 1-4). Without overrides, bandwidth
    /// comes from a quick bench and the FP32 peak from `attune detect`.
    /// Comma-separated `--quant` compares several weight formats.
    #[command(group(
        ArgGroup::new("model_source")
            .required(true)
            .args(["model", "params"])
    ))]
    Estimate {
        /// Model source, in priority order: an existing `.gguf` file path
        /// (probed header), an `org/model` Hugging Face id (config.json with
        /// a 30-day cache), or a verified catalog name (e.g. `llama-3.1-8b`).
        /// Mutually exclusive with `--params` / `--arch`.
        #[arg(long, value_name = "NAME_OR_PATH", conflicts_with = "arch")]
        model: Option<String>,
        /// Model size, e.g. `135M`, `1.2B` or `7000000`. Alternative to
        /// `--model` for models outside the catalog.
        #[arg(long, value_name = "SIZE")]
        params: Option<String>,
        /// Weight format(s): `fp16`, `q8_0`, `q4_0` or `q4_k_m`.
        /// Comma-separate to compare configs, e.g. `fp16,q8_0,q4_k_m`
        #[arg(long, value_name = "QUANT[,QUANT...]")]
        quant: String,
        /// Override measured memory bandwidth (decimal GB/s); skips the bench
        #[arg(long, value_name = "GBPS")]
        bandwidth_gbps: Option<f64>,
        /// Explicit half-spread percentage for the confidence interval
        /// (e.g. `10` for ±10%). Overrides the spread from `--bench-json`;
        /// needed when bandwidth is overridden or quick-probed.
        #[arg(long, value_name = "PCT")]
        spread_pct: Option<f64>,
        /// Override the FP32 peak (decimal GFLOPS); skips the HAL estimate
        #[arg(long, value_name = "GFLOPS")]
        gflops: Option<f64>,
        /// Architecture preset for KV-cache sizing: `llama-3.1-8b`,
        /// `llama-3.1-70b` or `qwen2.5-7b`. Omit to exclude KV from the
        /// decode prediction (a warning says so). Not used with `--model`.
        #[arg(long, value_name = "NAME")]
        arch: Option<String>,
        /// Prompt/prefill token count used for TTFT (default 512)
        #[arg(long, value_name = "N", default_value_t = 512)]
        prompt_tokens: u64,
        /// Number of decode tokens to estimate (default 128; reserved for
        /// slice 3 calibration output)
        #[arg(long, value_name = "N", default_value_t = 128)]
        gen_tokens: u64,
        /// Context window size used for KV-cache sizing. When omitted, a
        /// built-in model uses its documented default (catalog/config.json),
        /// otherwise 2048. An explicit value always wins.
        #[arg(long, value_name = "N")]
        ctx: Option<u32>,
        /// Saved `attune bench --json` report to calibrate against. Accepts
        /// either a bare report or the `{ "bench": … }` envelope written by
        /// `attune bench --model --save`. A full report also supplies the
        /// read-kernel spread for confidence intervals.
        #[arg(long, value_name = "PATH")]
        bench_json: Option<PathBuf>,
        /// Calibrate the prediction from `--bench-json` (the report must
        /// contain an `e2e` block with a measured tok/s)
        #[arg(long, requires = "bench_json")]
        calibrate: bool,
        /// Weight format the `--bench-json` run measured (e.g. `q4_k_m`).
        /// The bench report carries no quant, so without this flag no
        /// cross-quant warning can be produced. Used only with `--calibrate`.
        #[arg(long, value_name = "QUANT", requires = "bench_json")]
        bench_quant: Option<String>,
        /// Force a refetch of an `org/model` config instead of using the
        /// 30-day cache. A failed refetch still falls back to the cache or the
        /// embedded catalog.
        #[arg(long)]
        refresh_catalog: bool,
        /// Output machine-readable JSON instead of the human lines
        #[arg(long)]
        json: bool,
    },
}

/// Run the selected subcommand. All logic lives in `commands::*`.
pub fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Detect { json } => detect::run(json),
        Command::Bench {
            quick,
            json,
            model,
            llama_cli,
            save,
            prompt_tokens,
            gen_tokens,
            ctx,
            threads,
        } => bench::run(
            quick,
            json,
            model.as_deref(),
            llama_cli.as_deref(),
            save.as_deref(),
            prompt_tokens,
            gen_tokens,
            ctx,
            threads,
        ),
        Command::Estimate {
            model,
            params,
            quant,
            bandwidth_gbps,
            spread_pct,
            gflops,
            arch,
            prompt_tokens,
            gen_tokens,
            ctx,
            bench_json,
            calibrate,
            bench_quant,
            refresh_catalog,
            json,
        } => estimate::run(
            model.as_deref(),
            params.as_deref(),
            &quant,
            bandwidth_gbps,
            spread_pct,
            gflops,
            arch.as_deref(),
            prompt_tokens,
            gen_tokens,
            ctx,
            bench_json.as_deref(),
            bench_quant.as_deref(),
            calibrate,
            refresh_catalog,
            json,
        ),
    }
}
