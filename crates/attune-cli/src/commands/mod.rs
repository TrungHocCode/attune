//! Command dispatch — every subcommand is a thin call into a library
//! (AGENTS.md §1: no logic in `main.rs`, all of it in the crates).

mod bench;
mod detect;

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum Command {
    /// Detect hardware (M1: CPU/SIMD/cache/RAM via raw-cpuid + sysinfo)
    Detect {
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
    },
    /// Benchmark this machine (M2)
    Bench {
        /// Quick memory-bandwidth sweep (default today)
        #[arg(long)]
        quick: bool,
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
    },
}

/// Run the selected subcommand. All logic lives in `commands::*`.
pub fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Detect { json } => detect::run(json),
        Command::Bench { quick, json } => bench::run(quick, json),
    }
}
