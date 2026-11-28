//! `attune run <model>` - download, tune and chat in one command.
//!
//! Everything heavy is reused: the estimate sweep and profile export live in
//! `tune`, the install in `setup`, the launch/proxy in `serve`, and the REPL in
//! `chat`. This module only decides whether a saved profile can be reused and
//! wires those pieces together.

use std::io::Write;
use std::path::{Path, PathBuf};
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
    let requested_model = local_gguf_path(model);

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
        requested_model.as_deref(),
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
///
/// When the input is a local `.gguf`, the profile's `model.path` must also
/// resolve to the same file: the name alone is not enough, because
/// `attune run D:\b\model.gguf` would otherwise reuse the profile tuned for
/// `C:\a\model.gguf` (same stem, different weights). `retune` always forces a
/// fresh tune.
pub fn needs_tune(
    existing: Option<&Profile>,
    current_cpu: &str,
    current_gpu: Option<&str>,
    requested_model: Option<&Path>,
    retune: bool,
) -> bool {
    if retune {
        return true;
    }
    let Some(profile) = existing else {
        return true;
    };
    if let Some(requested) = requested_model {
        if canonical_or_given(Path::new(&profile.model.path)) != canonical_or_given(requested) {
            return true;
        }
    }
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

/// Whether `path` names a GGUF file (extension match is case-insensitive).
fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

/// Canonical absolute path when the file can be resolved, else the path as
/// given. Two spellings of the same file (`C:\a\x.gguf`, `c:/a/../a/x.gguf`)
/// compare equal; a path that does not exist still compares by text.
fn canonical_or_given(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The canonical path of a local `.gguf` input; `None` for catalog names and
/// other model ids.
fn local_gguf_path(model: &str) -> Option<PathBuf> {
    let path = Path::new(model.trim());
    is_gguf(path).then(|| canonical_or_given(path))
}

/// 8 lowercase hex characters of the FNV-1a (64-bit) hash of `text`.
///
/// `std`'s `DefaultHasher` is explicitly undocumented and may change between
/// Rust releases, so the profile name gets a hash it controls: the same file
/// must keep its profile across toolchain upgrades.
fn path_tag(text: &str) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:08x}", hash as u32)
}

/// Profile name for a device: `<id>-<quant>` on the CPU (unchanged, so saved
/// CPU profiles keep working), `-gpu` for `--device gpu` and `-auto` for
/// `--device auto`.
///
/// A `.gguf` source contributes its file stem plus 8 hex characters of the
/// hash of its canonical path, so two different `model.gguf` files never share
/// a profile just because their names match.
pub fn profile_name(model: &str, quant: &str, device: TuneDevice) -> String {
    let text = model.trim();
    let path = Path::new(text);
    let name = if is_gguf(path) {
        let canonical = canonical_or_given(path);
        let stem = canonical.file_stem().map_or_else(
            || text.to_string(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        format!("{stem}-{}", path_tag(&canonical.to_string_lossy()))
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

        assert!(!needs_tune(Some(&saved_cpu), cpu, Some(gpu), None, false));
        assert!(needs_tune(Some(&saved_cpu), gpu, Some(gpu), None, false));
        assert!(!needs_tune(Some(&saved_gpu), cpu, Some(gpu), None, false));
        assert!(needs_tune(Some(&saved_gpu), cpu, Some(cpu), None, false));
        assert!(needs_tune(Some(&saved_gpu), cpu, None, None, false));
        assert!(needs_tune(None, cpu, Some(gpu), None, false));
        assert!(needs_tune(Some(&saved_cpu), cpu, Some(gpu), None, true));
    }

    /// The 8-hex path tag at the end of a GGUF profile name.
    fn tag(name: &str) -> &str {
        name.rsplit('-').next().expect("tag")
    }

    #[test]
    fn needs_tune_requires_the_same_gguf_file() {
        let cpu = "cpu|8|avx2|16000000000";
        let dir = std::env::temp_dir().join(format!("attune-run-reuse-{}", std::process::id()));
        let (a_dir, b_dir) = (dir.join("a"), dir.join("b"));
        std::fs::create_dir_all(&a_dir).expect("mkdir a");
        std::fs::create_dir_all(&b_dir).expect("mkdir b");
        let a = a_dir.join("model.gguf");
        let b = b_dir.join("model.gguf");
        std::fs::write(&a, b"GGUF").expect("write a");
        std::fs::write(&b, b"GGUF").expect("write b");

        let mut saved = profile_with(cpu, 0);
        saved.model.path = a.to_string_lossy().into_owned();
        // Same file (and same fingerprint): reuse.
        assert!(!needs_tune(Some(&saved), cpu, None, Some(&a), false));
        // Same name, different file: retune.
        assert!(
            needs_tune(Some(&saved), cpu, None, Some(&b), false),
            "a profile for {a:?} must not serve {b:?}"
        );
        // A catalog name skips the path check.
        assert!(!needs_tune(Some(&saved), cpu, None, None, false));

        let _ = std::fs::remove_dir_all(&dir);
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
    fn profile_name_keys_gguf_files_by_their_path() {
        // Not created on disk: the hash covers the path as given.
        let dir = std::env::temp_dir().join("attune-run-name-test");
        let a = dir.join("a").join("model.gguf");
        let b = dir.join("b").join("model.gguf");
        let a_text = a.to_string_lossy();
        let b_text = b.to_string_lossy();

        let a_name = profile_name(&a_text, "q4_k_m", TuneDevice::Cpu);
        let b_name = profile_name(&b_text, "q4_k_m", TuneDevice::Cpu);
        assert!(a_name.starts_with("model-"), "{a_name}");
        assert_ne!(a_name, b_name, "same file name, different paths");
        assert_eq!(tag(&a_name).len(), 8);
        assert!(tag(&a_name).chars().all(|c| c.is_ascii_hexdigit()));
        // The same path keeps its name; quant does not matter for GGUF names,
        // only the device suffix does.
        assert_eq!(a_name, profile_name(&a_text, "q8_0", TuneDevice::Cpu));
        assert_eq!(
            format!("{a_name}-gpu"),
            profile_name(&a_text, "q4_k_m", TuneDevice::Gpu)
        );
        assert_eq!(
            format!("{a_name}-auto"),
            profile_name(&a_text, "q4_k_m", TuneDevice::Auto)
        );
        // Extension matching stays case-insensitive.
        let upper = profile_name(
            &dir.join("smollm2-Q4_K_M.GGUF").to_string_lossy(),
            "",
            TuneDevice::Cpu,
        );
        assert!(upper.starts_with("smollm2-Q4_K_M-"), "{upper}");
        assert!(!upper.contains('/') && !upper.contains('\\'));
    }
}
