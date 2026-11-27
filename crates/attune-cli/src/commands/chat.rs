//! `attune chat` - a small streaming REPL for a running attune proxy.
//!
//! The network and SSE logic lives in [`attune_serve::ChatClient`]; this module
//! only renders the banner, prompt, streamed tokens and the stats line, and
//! handles `/clear` / `/bye`. [`chat_loop`] is public so `attune run` can reuse
//! it in-process.

use std::io::{BufRead, IsTerminal, Write};

use anyhow::Result;
use attune_serve::{ChatClient, ChatDelta, ChatStats};

/// Resolve `--api-key`, falling back to `ATTUNE_API_KEY`.
///
/// clap's `env` feature is not enabled, so the variable is read by hand.
pub fn resolve_api_key(api_key: Option<&str>) -> Option<String> {
    let env_key = std::env::var("ATTUNE_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    api_key
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or(env_key)
}

/// Entry point for `attune chat`.
pub fn run(url: &str, api_key: Option<&str>, system: Option<&str>) -> Result<()> {
    let api_key = resolve_api_key(api_key);
    let mut client = ChatClient::new(url, api_key, system.map(str::to_string));
    chat_loop(&mut client)
}

/// Run the read-eval-print loop until `/bye`, EOF, or a failed first connection.
pub fn chat_loop(client: &mut ChatClient) -> Result<()> {
    println!("Chatting with {}. /clear resets, /bye quits.", client.url());
    std::io::stdout().flush()?;

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut line = String::new();
    let mut first_message = true;

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let message = line.trim();
        match message {
            "" => continue,
            "/bye" => break,
            "/clear" => {
                client.clear();
                continue;
            }
            _ => {}
        }

        let interactive = std::io::stdout().is_terminal();
        let mut thinking = false;
        let result = {
            let mut out = std::io::stdout();
            let mut on_delta = |delta: ChatDelta<'_>| match delta {
                ChatDelta::Thinking(_) => {
                    if interactive && !thinking {
                        thinking = true;
                        let _ = write!(out, "\u{1b}[2mthinking\u{2026}\u{1b}[0m");
                        let _ = out.flush();
                    }
                }
                ChatDelta::Text(piece) => {
                    if thinking {
                        let _ = out.write_all(b"\r\x1b[2K");
                        thinking = false;
                    }
                    let _ = out.write_all(piece.as_bytes());
                    let _ = out.flush();
                }
            };
            let result = client.send(message, &mut on_delta);
            if thinking {
                let _ = out.write_all(b"\r\x1b[2K");
                let _ = out.flush();
            }
            result
        };
        match result {
            Ok(stats) => {
                println!();
                print_stats(stats);
                first_message = false;
            }
            Err(error) => {
                if first_message && error.is_no_server() {
                    return Err(error.into());
                }
                eprintln!("[attune] {error}");
                first_message = false;
            }
        }
    }
    Ok(())
}

fn print_stats(stats: ChatStats) {
    let line = format!("[{} tok, {:.1} tok/s]", stats.tokens, stats.tokens_per_s);
    if std::io::stdout().is_terminal() {
        println!("\u{1b}[2m{line}\u{1b}[0m");
    } else {
        println!("{line}");
    }
}
