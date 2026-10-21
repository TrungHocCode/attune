//! Command dispatch — every subcommand is a thin call into a library
//! (AGENTS.md §1: no logic in `main.rs`, all of it in the crates).

mod bench;
mod detect;

use anyhow::Result;
use clap::Subcommand;
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
    }
}
