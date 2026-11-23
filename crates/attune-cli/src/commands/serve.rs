//! `attune serve` - launch a tuned profile behind the OpenAI-compatible proxy
//! (M5 slices 1-2).
//!
//! Everything that touches a profile, the filesystem, a child process or an
//! HTTP socket lives in `attune-serve` / `attune-backend`; this module only
//! translates CLI flags, renders warnings/JSON and blocks until Ctrl+C.
//! `--api-key` / `ATTUNE_API_KEY` is enforced by the proxy, and a non-loopback
//! `--listen` without a key is refused by `Proxy::bind`.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use attune_serve::{check_listen, install_shutdown, launch, Proxy, ProxyConfig, ServeOptions};

#[allow(clippy::too_many_arguments)]
pub fn run(
    profile: &Path,
    llama_server: Option<&Path>,
    host: &str,
    port: u16,
    listen: &str,
    listen_port: u16,
    api_key: Option<&str>,
    timeout_s: u64,
    force: bool,
    json: bool,
) -> Result<()> {
    // clap's `env` feature is not enabled; read the variable by hand.
    let env_key = std::env::var("ATTUNE_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    let api_key = api_key.filter(|key| !key.is_empty()).or(env_key.as_deref());
    check_listen(listen, api_key)?;
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
    let proxy_config = ProxyConfig {
        api_key: api_key.map(str::to_string),
        listen: listen.to_string(),
        listen_port,
        upstream_base: format!("http://{}", serving.upstream),
        upstream_pid: Some(serving.guard.pid()),
    };
    let proxy = Proxy::bind(&proxy_config)?;
    let proxy_url = format!("http://{}", proxy.local_addr());
    // `--json` keeps stdout to the single ready line; the proxy address goes to
    // stderr so it cannot break a machine-readable pipe.
    if json {
        eprintln!("[attune] proxy listening on {proxy_url}");
    }

    if json {
        println!("{}", serde_json::to_string(&serving.ready_report())?);
    } else {
        println!(
            "Serving\u{2026} upstream {} health OK ({:.1}s)",
            serving.upstream,
            serving.health_elapsed.as_secs_f64()
        );
        println!("[attune] proxy listening on {proxy_url} (OpenAI-compatible)");
        println!("[attune] press Ctrl+C to stop the proxy and llama-server");
    }
    // The process blocks below: flush so a piped/redirected stdout still shows
    // the ready lines before the caller waits.
    std::io::stdout().flush()?;

    let shutdown = install_shutdown(proxy.unblocker())?;
    let served = proxy.run(&shutdown);
    // Explicit ordering: `run` consumed (and closed) the listener, and dropping
    // the guard now kills/reaps llama-server before returning.
    drop(serving);
    served?;
    Ok(())
}
