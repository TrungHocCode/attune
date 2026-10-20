//! attune CLI — thin wrapper (no logic here, per AGENTS.md §1).

mod commands;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "attune",
    version,
    about = "Tune LLM inference to your hardware"
)]
struct Cli {
    #[command(subcommand)]
    command: commands::Command,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    commands::dispatch(cli.command)
}
