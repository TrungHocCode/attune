//! attune CLI — thin wrapper (no logic here, per AGENTS.md).

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "attune",
    version,
    about = "Tune LLM inference to your hardware"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware (M1: stub, M2+: full SIMD/cache/NUMA)
    Detect {
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Detect { json } => {
            let hw = attune_hal::HardwareInfo::detect()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hw)?);
            } else {
                println!(
                    "CPU:        {} ({} logical)",
                    hw.cpu.brand, hw.cpu.logical_cores
                );
                println!("SIMD:       {}", hw.cpu.simd);
                println!("RAM:        {:.1} GB", hw.memory.total_gb);
                println!("OS:         {}", hw.os);
            }
            Ok(())
        }
    }
}
