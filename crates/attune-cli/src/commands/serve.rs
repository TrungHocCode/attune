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

use anyhow::{anyhow, Result};
use attune_serve::{
    check_listen, install_shutdown, launch, resolve_profile, Proxy, ProxyConfig, ReadyReport,
    ServeDevice, ServeError, ServeOptions, Serving, ShutdownSignal,
};

impl From<super::tune::TuneDevice> for ServeDevice {
    fn from(device: super::tune::TuneDevice) -> Self {
        match device {
            super::tune::TuneDevice::Auto => ServeDevice::Auto,
            super::tune::TuneDevice::Cpu => ServeDevice::Cpu,
            super::tune::TuneDevice::Gpu => ServeDevice::Gpu,
        }
    }
}

/// A launched upstream plus a bound proxy, ready to serve.
///
/// Dropping it closes the listener and drops the [`Serving`] guard, which kills
/// and reaps `llama-server`.
pub struct Running {
    serving: Serving,
    proxy: Proxy,
    proxy_url: String,
    shutdown: ShutdownSignal,
    proxy_thread: Option<std::thread::JoinHandle<Result<(), ServeError>>>,
}

impl Running {
    /// The OpenAI-compatible URL clients should use.
    pub fn proxy_url(&self) -> &str {
        &self.proxy_url
    }

    /// `host:port` of the upstream `llama-server`.
    pub fn upstream(&self) -> &str {
        &self.serving.upstream
    }

    /// How long the health wait took.
    pub fn health_elapsed(&self) -> Duration {
        self.serving.health_elapsed
    }

    /// Non-fatal warnings collected while preparing the launch.
    pub fn warnings(&self) -> &[String] {
        &self.serving.warnings
    }

    /// `CPU` or `GPU (<build>, <id> <name>)` for the human ready line.
    pub fn device_label(&self) -> String {
        self.serving.device_label()
    }

    /// The JSON shape printed by `serve --json`.
    pub fn ready_report(&self) -> ReadyReport {
        self.serving.ready_report()
    }

    /// Whether Ctrl+C (or [`Running::stop`]) asked the proxy to shut down.
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown.is_requested()
    }

    /// Serve the proxy on a worker thread so the caller can chat in-process.
    pub fn start_proxy(&mut self) {
        if self.proxy_thread.is_some() {
            return;
        }
        let proxy = self.proxy.clone();
        let shutdown = self.shutdown.clone();
        self.proxy_thread = Some(std::thread::spawn(move || proxy.run(&shutdown)));
    }

    fn stop_thread(&mut self) -> Result<()> {
        self.shutdown.request();
        self.proxy.unblock();
        if let Some(handle) = self.proxy_thread.take() {
            match handle.join() {
                Ok(result) => result.map_err(|error| anyhow!("{error}"))?,
                Err(_) => return Err(anyhow!("the proxy thread panicked")),
            }
        }
        Ok(())
    }

    /// Stop the proxy and kill `llama-server`.
    pub fn stop(mut self) -> Result<()> {
        let result = self.stop_thread();
        drop(self);
        result
    }

    /// Serve on this thread until Ctrl+C, then stop the proxy and llama-server.
    pub fn wait(self) -> Result<()> {
        let result = self.proxy.clone().run(&self.shutdown);
        drop(self);
        result.map_err(|error| anyhow!("{error}"))
    }
}

/// Resolve the profile, spawn `llama-server`, wait for health and bind the
/// proxy. Nothing is printed so `serve` and `run` can render their own lines.
#[allow(clippy::too_many_arguments)]
pub fn launch_running(
    profile: &str,
    llama_server: Option<&Path>,
    device: ServeDevice,
    host: &str,
    port: u16,
    listen: &str,
    listen_port: u16,
    api_key: Option<&str>,
    timeout_s: u64,
    force: bool,
    yes: bool,
    progress: fn(u64, u64),
) -> Result<Running> {
    check_listen(listen, api_key)?;
    let profile = resolve_profile(profile)?;
    let options = ServeOptions {
        profile,
        llama_server: llama_server.map(Path::to_path_buf),
        device,
        host: host.to_string(),
        port,
        timeout: Duration::from_secs(timeout_s),
        force,
        download: super::download::consent_for_stdin(yes),
        progress,
    };
    let serving = launch(&options)?;

    let proxy_config = ProxyConfig {
        api_key: api_key.map(str::to_string),
        listen: listen.to_string(),
        listen_port,
        upstream_base: format!("http://{}", serving.upstream),
        upstream_pid: Some(serving.guard.pid()),
    };
    let proxy = Proxy::bind(&proxy_config)?;
    let proxy_url = format!("http://{}", proxy.local_addr());
    let shutdown = install_shutdown(proxy.unblocker())?;
    Ok(Running {
        serving,
        proxy,
        proxy_url,
        shutdown,
        proxy_thread: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    profile: &str,
    llama_server: Option<&Path>,
    device: ServeDevice,
    host: &str,
    port: u16,
    listen: &str,
    listen_port: u16,
    api_key: Option<&str>,
    timeout_s: u64,
    force: bool,
    yes: bool,
    json: bool,
) -> Result<()> {
    let api_key = super::chat::resolve_api_key(api_key);
    let running = launch_running(
        profile,
        llama_server,
        device,
        host,
        port,
        listen,
        listen_port,
        api_key.as_deref(),
        timeout_s,
        force,
        yes,
        if json {
            super::download::quiet
        } else {
            super::download::progress
        },
    )?;

    for warning in running.warnings() {
        eprintln!("[attune] warning: {warning}");
    }
    // `--json` keeps stdout to the single ready line; the proxy address goes to
    // stderr so it cannot break a machine-readable pipe.
    if json {
        eprintln!("[attune] proxy listening on {}", running.proxy_url());
    }

    if json {
        println!("{}", serde_json::to_string(&running.ready_report())?);
    } else {
        println!(
            "Serving on {} \u{2026} upstream {} health OK ({:.1}s)",
            running.device_label(),
            running.upstream(),
            running.health_elapsed().as_secs_f64()
        );
        println!(
            "[attune] proxy listening on {} (OpenAI-compatible)",
            running.proxy_url()
        );
        println!("[attune] press Ctrl+C to stop the proxy and llama-server");
    }
    // The process blocks below: flush so a piped/redirected stdout still shows
    // the ready lines before the caller waits.
    std::io::stdout().flush()?;

    running.wait()
}
