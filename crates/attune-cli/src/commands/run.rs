//! `attune run <model>` - download, tune and chat in one command.
//!
//! Everything heavy is reused: the estimate sweep and profile export live in
//! `tune`, the install in `setup`, the launch/proxy in `serve`, and the REPL in
//! `chat`. This module only decides whether a saved profile can be reused and
//! wires those pieces together.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Result};
use attune_profile::Profile;
use attune_serve::{profiles_dir, ChatClient};

use super::tune::TuneDevice;
use super::{chat, download, serve, setup, tune};

/// Weight format used when `--quant` is omitted.
const DEFAULT_QUANT: &str = "q4_k_m";

/// `attune run <MODEL> [flags]`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: &str,
    quant: &str,
    yes: bool,
    retune: bool,
    no_chat: bool,
    system: Option<&str>,
    llama_server: Option<&Path>,
    device: TuneDevice,
    port: u16,
    listen: &str,
    listen_port: u16,
    api_key: Option<&str>,
    timeout_s: u64,
) -> Result<()> {
    let quant = if quant.trim().is_empty() {
        DEFAULT_QUANT
    } else {
        quant
    };
    let name = profile_name(model, quant, device);
    let path = profiles_dir()
        .map_err(|error| anyhow!("{error}"))?
        .join(format!("{name}.json"));

    let existing = existing_profile(&path);
    let current_cpu = tune::current_hardware_fingerprint();
    let current_gpu = existing
        .as_ref()
        .filter(|profile| profile.device() == "gpu")
        .and_then(|profile| machine_gpu_fingerprint(profile.load_config.gpu_name.as_deref()));

    // Resolve/install the llama.cpp build(s) the chosen device needs, as
    // `setup` would (the explicit `--llama-server` wins and installs nothing).
    setup::ensure_llama_server_for(device, llama_server, yes)?;

    if needs_tune(
        existing.as_ref(),
        &current_cpu,
        current_gpu.as_deref(),
        retune,
    ) {
        tune::run(
            Some(model),
            None,
            None,
            quant,
            "balanced",
            None,
            None,
            device,
            64,
            Some(&path),
            Some(&name),
            tune::VerifyArgs {
                enabled: false,
                top_k: 2,
                llama_cli: None,
                prompt_tokens: None,
                gen_tokens: None,
                ctx: None,
            },
            None,
            None,
            false,
            tune::RecommendArgs {
                max_ram: None,
                all: false,
                refresh_catalog: false,
            },
            false,
            false,
            true,
        )?;
    } else {
        println!("Using saved profile {name} ({})", path.display());
    }

    let api_key = chat::resolve_api_key(api_key);
    let mut running = serve::launch_running(
        &path.display().to_string(),
        llama_server,
        attune_serve::ServeDevice::Auto,
        "127.0.0.1",
        port,
        listen,
        listen_port,
        api_key.as_deref(),
        timeout_s,
        false,
        yes,
        download::progress,
    )?;
    for warning in running.warnings() {
        eprintln!("[attune] warning: {warning}");
    }
    let proxy_url = running.proxy_url().to_string();
    let ready = running.ready_report();
    let placement = if ready.device == "gpu" {
        ready
            .gpu_name
            .as_deref()
            .map_or_else(|| "GPU".to_string(), |name| format!("GPU: {name}"))
    } else {
        "CPU".to_string()
    };
    println!("Starting {name} on {proxy_url} ({placement}) ...");
    std::io::stdout().flush()?;

    if no_chat {
        println!("Chat from another terminal: attune chat --url {proxy_url}");
        println!("[attune] press Ctrl+C to stop the proxy and llama-server");
        std::io::stdout().flush()?;
        running.wait()?;
        return Ok(());
    }

    running.start_proxy();
    let mut client = ChatClient::new(proxy_url, api_key, system.map(str::to_string));
    let (sender, receiver) = std::sync::mpsc::channel();
    let chat_thread = std::thread::spawn(move || {
        let result = chat::chat_loop(&mut client);
        let _ = sender.send(result);
    });

    loop {
        match receiver.recv_timeout(Duration::from_millis(200)) {
            Ok(result) => {
                let stopped = running.stop();
                let _ = chat_thread.join();
                stopped?;
                return result;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if running.shutdown_requested() {
                    let _ = running.stop();
                    return Ok(());
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = running.stop();
                return Err(anyhow!("the chat loop stopped unexpectedly"));
            }
        }
    }
}

/// A saved profile can be reused only when its fingerprint matches this
/// machine, computed the same way the profile was made: GPU profiles
/// (`n_gpu_layers > 0`) use the GPU-aware fingerprint, CPU profiles the CPU one.
/// `retune` always forces a fresh tune.
pub fn needs_tune(
    existing: Option<&Profile>,
    current_cpu: &str,
    current_gpu: Option<&str>,
    retune: bool,
) -> bool {
    if retune {
        return true;
    }
    let Some(profile) = existing else {
        return true;
    };
    let current = if profile.device() == "gpu" {
        match current_gpu {
            Some(fingerprint) => fingerprint,
            None => return true,
        }
    } else {
        current_cpu
    };
    profile.hardware_fingerprint != current
}

/// Profile name for a device: `<id>-<quant>` on the CPU (unchanged, so saved
/// CPU profiles keep working), `-gpu` for `--device gpu` and `-auto` for
/// `--device auto`. A `.gguf` source contributes its file stem.
pub fn profile_name(model: &str, quant: &str, device: TuneDevice) -> String {
    let text = model.trim();
    let path = Path::new(text);
    let is_gguf = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"));
    let name = if is_gguf {
        // Split on both separators so a Windows path behaves the same on Linux.
        let file = text.rsplit(['/', '\\']).next().unwrap_or(text);
        file.rsplit_once('.')
            .map_or(file, |(stem, _)| stem)
            .to_string()
    } else {
        format!("{text}-{quant}")
    };
    let suffix = match device {
        TuneDevice::Cpu => "",
        TuneDevice::Gpu => "-gpu",
        TuneDevice::Auto => "-auto",
    };
    format!("{name}{suffix}").replace(['/', '\\'], "-")
}

/// The saved profile, or `None` when the file is missing or unreadable (either
/// way a fresh tune is the safe answer).
fn existing_profile(path: &Path) -> Option<Profile> {
    let text = std::fs::read_to_string(path).ok()?;
    Profile::from_json_str(&text).ok()
}

/// The GPU-aware fingerprint for this machine, matching the profile's
/// `gpu_name` when it is still present and otherwise the first NVIDIA/AMD
/// adapter.
fn machine_gpu_fingerprint(preferred: Option<&str>) -> Option<String> {
    let devices = attune_hal::gpu::detect().devices;
    let device = preferred
        .and_then(|name| devices.iter().find(|device| device.name.as_str() == name))
        .or_else(|| {
            devices
                .iter()
                .find(|device| matches!(device.vendor.as_str(), "nvidia" | "amd"))
        })?;
    let vram = device
        .vram_total_bytes
        .or(device.vram_free_bytes)
        .unwrap_or(0);
    Some(tune::current_hardware_fingerprint_with_gpu(Some((
        &device.name,
        vram,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with(fingerprint: &str, gpu_layers: u32) -> Profile {
        let mut profile = Profile {
            hardware_fingerprint: fingerprint.to_string(),
            ..Profile::default()
        };
        profile.load_config.n_gpu_layers = gpu_layers;
        profile
    }

    #[test]
    fn needs_tune_compares_gpu_profiles_against_the_gpu_fingerprint() {
        let cpu = "cpu|8|avx2|16000000000";
        let gpu = "cpu|8|avx2|16000000000|NVIDIA GeForce RTX 4050|6";
        let saved_cpu = profile_with(cpu, 0);
        let saved_gpu = profile_with(gpu, 36);

        assert!(!needs_tune(Some(&saved_cpu), cpu, Some(gpu), false));
        assert!(needs_tune(Some(&saved_cpu), gpu, Some(gpu), false));
        assert!(!needs_tune(Some(&saved_gpu), cpu, Some(gpu), false));
        assert!(needs_tune(Some(&saved_gpu), cpu, Some(cpu), false));
        assert!(needs_tune(Some(&saved_gpu), cpu, None, false));
        assert!(needs_tune(None, cpu, Some(gpu), false));
        assert!(needs_tune(Some(&saved_cpu), cpu, Some(gpu), true));
    }

    #[test]
    fn profile_name_suffix_follows_the_device() {
        assert_eq!(
            profile_name("qwen3-4b", "q4_k_m", TuneDevice::Cpu),
            "qwen3-4b-q4_k_m"
        );
        assert_eq!(
            profile_name("qwen3-4b", "q4_k_m", TuneDevice::Gpu),
            "qwen3-4b-q4_k_m-gpu"
        );
        assert_eq!(
            profile_name("qwen3-4b", "q4_k_m", TuneDevice::Auto),
            "qwen3-4b-q4_k_m-auto"
        );
        assert_eq!(
            profile_name(" qwen3-4b ", "q8_0", TuneDevice::Cpu),
            "qwen3-4b-q8_0"
        );
    }

    #[test]
    fn profile_name_is_the_file_stem_with_the_device_suffix_for_gguf_paths() {
        assert_eq!(
            profile_name("models/smollm2-Q4.gguf", "q4_k_m", TuneDevice::Cpu),
            "smollm2-Q4"
        );
        assert_eq!(
            profile_name("C:\\models\\smollm2-Q4_K_M.GGUF", "q8_0", TuneDevice::Gpu),
            "smollm2-Q4_K_M-gpu"
        );
        assert_eq!(
            profile_name("models/smollm2-Q4.gguf", "q4_k_m", TuneDevice::Auto),
            "smollm2-Q4-auto"
        );
    }
}
