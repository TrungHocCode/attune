//! `attune serve` - launch a tuned profile through llama-server (M5 slice 1).
//!
//! Everything that touches a profile, the filesystem or a child process lives
//! in `attune-serve` / `attune-backend`; this module only translates CLI flags,
//! renders warnings/JSON and blocks. Ctrl+C handling arrives with the proxy
//! slice, so for now the process parks until it is interrupted or killed.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use attune_serve::{launch, run_forever, ServeOptions};

#[allow(clippy::too_many_arguments)]
pub fn run(
    profile: &Path,
    llama_server: Option<&Path>,
    host: &str,
    port: u16,
    timeout_s: u64,
    force: bool,
    json: bool,
) -> Result<()> {
    let options = ServeOptions {
        profile: profile.to_path_buf(),
        llama_server: llama_server.map(Path::to_path_buf),
        host: host.to_string(),
        port,
        timeout: Duration::from_secs(timeout_s),
        force,
    };
    let serving = launch(&options)?;

    for warning in &serving.warnings {
        eprintln!("[attune] warning: {warning}");
    }

    if json {
        println!("{}", serde_json::to_string(&serving.ready_report())?);
    } else {
        println!(
            "Serving\u{2026} upstream {} health OK ({:.1}s)",
            serving.upstream,
            serving.health_elapsed.as_secs_f64()
        );
        println!(
            "[attune] Ctrl+C handling lands in slice 2; interrupt or kill this process to stop."
        );
    }
    // The process is about to park: flush so a piped/redirected stdout still
    // shows the ready line before the caller waits.
    std::io::stdout().flush()?;

    run_forever(serving)
}
