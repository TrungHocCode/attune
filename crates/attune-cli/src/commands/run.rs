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
use attune_serve::{profiles_dir, ChatClient};

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
    let name = profile_name(model, quant);
    let path = profiles_dir()
        .map_err(|error| anyhow!("{error}"))?
        .join(format!("{name}.json"));

    let current = tune::current_hardware_fingerprint();
    let existing = existing_fingerprint(&path);

    // Resolve/install llama-server before the profile work, as `setup` would.
    setup::ensure_llama_server(llama_server, yes)?;

    if needs_tune(existing.as_deref(), &current, retune) {
        tune::run(
            Some(model),
            None,
            None,
            quant,
            "balanced",
            None,
            None,
            tune::TuneDevice::Auto,
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
    println!("Starting {name} on {proxy_url} ...");
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

/// A saved profile can be reused only when it is for this machine and the user
/// did not ask for a fresh tune.
pub fn needs_tune(existing_fingerprint: Option<&str>, current: &str, retune: bool) -> bool {
    retune || existing_fingerprint.is_none_or(|fingerprint| fingerprint != current)
}

/// `<id>-<quant>` for a catalog id, or the file stem for a `.gguf` path.
pub fn profile_name(model: &str, quant: &str) -> String {
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
    name.replace(['/', '\\'], "-")
}

/// The fingerprint recorded in an existing profile, or `None` when the file is
/// missing or unreadable (either way a fresh tune is the safe answer).
fn existing_fingerprint(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    attune_profile::Profile::from_json_str(&text)
        .ok()
        .map(|profile| profile.hardware_fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_tune_for_missing_matching_mismatched_and_retune() {
        assert!(needs_tune(None, "cpu|8|avx2|16000000000", false));
        assert!(!needs_tune(
            Some("cpu|8|avx2|16000000000"),
            "cpu|8|avx2|16000000000",
            false
        ));
        assert!(needs_tune(
            Some("old-cpu|4|sse|8000000000"),
            "cpu|8|avx2|16000000000",
            false
        ));
        assert!(needs_tune(
            Some("cpu|8|avx2|16000000000"),
            "cpu|8|avx2|16000000000",
            true
        ));
    }

    #[test]
    fn profile_name_is_id_quant_for_catalog_ids() {
        assert_eq!(profile_name("qwen3-4b", "q4_k_m"), "qwen3-4b-q4_k_m");
        assert_eq!(profile_name(" qwen3-4b ", "q8_0"), "qwen3-4b-q8_0");
    }

    #[test]
    fn profile_name_is_the_file_stem_for_gguf_paths() {
        assert_eq!(
            profile_name("models/smollm2-Q4.gguf", "q4_k_m"),
            "smollm2-Q4"
        );
        assert_eq!(
            profile_name("C:\\models\\smollm2-Q4_K_M.GGUF", "q8_0"),
            "smollm2-Q4_K_M"
        );
    }
}
