//! Command dispatch — every subcommand is a thin call into a library
//! (AGENTS.md §1: no logic in `main.rs`, all of it in the crates).

mod bench;
mod catalog;
mod chat;
mod detect;
mod download;
mod estimate;
mod recommend;
mod run;
mod serve;
mod setup;
mod tune;

use anyhow::Result;
use clap::{ArgGroup, Subcommand};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Command {
    /// Maintainer tooling for the model catalog (used by CI)
    #[command(hide = true, subcommand)]
    Catalog(CatalogCommand),
    /// Install the llama.cpp build attune uses
    Setup {
        /// Download without asking
        #[arg(long)]
        yes: bool,
        /// Re-download the model catalog even if the cached copy is fresh
        #[arg(long)]
        refresh_catalog: bool,
    },
    /// Show this machine's CPU, memory and GPU
    Detect {
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
    },
    /// Measure memory bandwidth and compute speed (add `--model` for a real
    /// llama.cpp run)
    Bench {
        /// Quick memory-bandwidth sweep (just the read kernel)
        #[arg(long)]
        quick: bool,
        /// Output machine-readable JSON instead of pretty table
        #[arg(long)]
        json: bool,
        /// GGUF model to load-test with llama.cpp (skips the llama.cpp
        /// probe when omitted)
        #[arg(long, value_name = "PATH")]
        model: Option<PathBuf>,
        /// Path to the llama.cpp `llama-cli` binary (default: `--llama-cli`,
        /// then $ATTUNE_LLAMA_CLI, then a PATH search for `llama-cli`/`main`)
        #[arg(long, value_name = "PATH")]
        llama_cli: Option<PathBuf>,
        /// Save the JSON report to this file (in addition to printing it).
        /// With `--model`, the saved file mirrors the printed envelope.
        #[arg(long, value_name = "PATH")]
        save: Option<PathBuf>,
        /// Approximate prefill token count for the e2e run (default 512).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        prompt_tokens: Option<usize>,
        /// Number of decode tokens for the e2e run (default 128).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        gen_tokens: Option<u32>,
        /// Context window size passed to llama.cpp (default 2048).
        /// Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        ctx: Option<u32>,
        /// Pin llama.cpp to this thread count (-t). Omit to let llama.cpp
        /// decide. Ignored unless `--model` is also passed.
        #[arg(long, value_name = "N", requires = "model")]
        threads: Option<u32>,
    },
    /// Predict how fast a model will run on this machine
    #[command(group(
        ArgGroup::new("model_source")
            .required(true)
            .args(["model", "params"])
    ))]
    Estimate {
        /// Model source, in priority order: an existing `.gguf` file path
        /// (probed header), an `org/model` Hugging Face id (config.json with
        /// a 30-day cache), or a verified catalog name (e.g. `llama-3.1-8b`).
        /// Mutually exclusive with `--params` / `--arch`.
        #[arg(long, value_name = "NAME_OR_PATH", conflicts_with = "arch")]
        model: Option<String>,
        /// Model size, e.g. `135M`, `1.2B` or `7000000`. Alternative to
        /// `--model` for models outside the catalog.
        #[arg(long, value_name = "SIZE")]
        params: Option<String>,
        /// Weight format(s): `fp16`, `q8_0`, `q4_0` or `q4_k_m`.
        /// Comma-separate to compare configs, e.g. `fp16,q8_0,q4_k_m`
        #[arg(long, value_name = "QUANT[,QUANT...]")]
        quant: String,
        /// Override measured memory bandwidth (decimal GB/s); skips the bench
        #[arg(long, value_name = "GBPS")]
        bandwidth_gbps: Option<f64>,
        /// Explicit half-spread percentage for the confidence interval
        /// (e.g. `10` for ±10%). Overrides the spread from `--bench-json`;
        /// needed when bandwidth is overridden or quick-probed.
        #[arg(long, value_name = "PCT")]
        spread_pct: Option<f64>,
        /// Override the FP32 peak (decimal GFLOPS); skips the HAL estimate
        #[arg(long, value_name = "GFLOPS")]
        gflops: Option<f64>,
        /// Architecture preset for KV-cache sizing: `llama-3.1-8b`,
        /// `llama-3.1-70b` or `qwen2.5-7b`. Omit to exclude KV from the
        /// decode prediction (a warning says so). Not used with `--model`.
        #[arg(long, value_name = "NAME")]
        arch: Option<String>,
        /// Prompt/prefill token count used for TTFT (default 512)
        #[arg(long, value_name = "N", default_value_t = 512)]
        prompt_tokens: u64,
        /// Number of decode tokens to estimate (default 128)
        #[arg(long, value_name = "N", default_value_t = 128)]
        gen_tokens: u64,
        /// Context window size used for KV-cache sizing. When omitted, a
        /// built-in model uses its documented default (catalog/config.json),
        /// otherwise 2048. An explicit value always wins.
        #[arg(long, value_name = "N")]
        ctx: Option<u32>,
        /// Saved `attune bench --json` report to calibrate against. Accepts
        /// either a bare report or the `{ "bench": … }` envelope written by
        /// `attune bench --model --save`. A full report also supplies the
        /// read-kernel spread for confidence intervals.
        #[arg(long, value_name = "PATH")]
        bench_json: Option<PathBuf>,
        /// Calibrate the prediction from `--bench-json` (the report must
        /// contain an `e2e` block with a measured tok/s)
        #[arg(long, requires = "bench_json")]
        calibrate: bool,
        /// Weight format the `--bench-json` run measured (e.g. `q4_k_m`).
        /// The bench report carries no quant, so without this flag no
        /// cross-quant warning can be produced. Used only with `--calibrate`.
        #[arg(long, value_name = "QUANT", requires = "bench_json")]
        bench_quant: Option<String>,
        /// Force a refetch of an `org/model` config instead of using the
        /// 30-day cache. A failed refetch still falls back to the cache or the
        /// embedded catalog.
        #[arg(long)]
        refresh_catalog: bool,
        /// Output machine-readable JSON instead of the human lines
        #[arg(long)]
        json: bool,
    },
    /// Download, tune and chat with a model in one command
    Run {
        /// Catalog model id (same ids as `tune --model`) or a local `.gguf`
        /// path
        #[arg(value_name = "MODEL")]
        model: String,
        /// Weight format to download and tune
        #[arg(long, value_name = "QUANT", default_value = "q4_k_m")]
        quant: String,
        /// Download the model and install llama.cpp without asking
        #[arg(long)]
        yes: bool,
        /// Tune again even when a matching saved profile exists
        #[arg(long)]
        retune: bool,
        /// Start the server and print the URL instead of chatting here
        #[arg(long)]
        no_chat: bool,
        /// System prompt sent before the conversation
        #[arg(long, value_name = "TEXT")]
        system: Option<String>,
        /// Path to the llama.cpp `llama-server` binary (default:
        /// $ATTUNE_LLAMA_SERVER, then a PATH search for `llama-server`)
        #[arg(long, value_name = "PATH")]
        llama_server: Option<PathBuf>,
        /// Port llama-server (the upstream) listens on
        #[arg(long, value_name = "PORT", default_value_t = 8081)]
        port: u16,
        /// Interface the OpenAI-compatible proxy binds to. A non-loopback
        /// host is refused unless `--api-key` (or `ATTUNE_API_KEY`) is set.
        #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
        listen: String,
        /// Port for the OpenAI-compatible proxy
        #[arg(long, value_name = "PORT", default_value_t = 8080)]
        listen_port: u16,
        /// Bearer key clients must send on every route except `/health`.
        /// Also read from `ATTUNE_API_KEY`.
        #[arg(long, value_name = "KEY")]
        api_key: Option<String>,
        /// Seconds to wait for `/health` before giving up
        #[arg(long, value_name = "N", default_value_t = 60)]
        timeout_s: u64,
    },
    /// Run a model from a profile behind an OpenAI-compatible API
    Serve {
        /// Profile file, or the name of a profile saved in ~/.attune/profiles
        #[arg(long, value_name = "PATH_OR_NAME")]
        profile: String,
        /// Path to the llama.cpp `llama-server` binary (default:
        /// $ATTUNE_LLAMA_SERVER, then a PATH search for `llama-server`)
        #[arg(long, value_name = "PATH")]
        llama_server: Option<PathBuf>,
        /// Interface llama-server (the upstream) binds to
        #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
        host: String,
        /// Port llama-server (the upstream) listens on
        #[arg(long, value_name = "PORT", default_value_t = 8081)]
        port: u16,
        /// Interface the OpenAI-compatible proxy binds to. A non-loopback
        /// host is refused unless `--api-key` (or `ATTUNE_API_KEY`) is set.
        #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
        listen: String,
        /// Port for the OpenAI-compatible proxy
        #[arg(long, value_name = "PORT", default_value_t = 8080)]
        listen_port: u16,
        /// Bearer key clients must send (`Authorization: Bearer <KEY>`) on
        /// every route except `/health`. Also read from `ATTUNE_API_KEY`.
        #[arg(long, value_name = "KEY")]
        api_key: Option<String>,
        /// Seconds to wait for `/health` before giving up
        #[arg(long, value_name = "N", default_value_t = 60)]
        timeout_s: u64,
        /// Launch without validating the profile against this machine
        #[arg(long)]
        force: bool,
        /// Download the profile's model without asking when it is missing
        #[arg(long)]
        yes: bool,
        /// Print one machine-readable JSON line once the upstream is healthy
        /// (the proxy address is printed on stderr)
        #[arg(long)]
        json: bool,
    },
    /// Chat with a running attune server
    Chat {
        /// Base URL of the attune proxy
        #[arg(long, value_name = "URL", default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Bearer key the proxy requires (also read from `ATTUNE_API_KEY`)
        #[arg(long, value_name = "KEY")]
        api_key: Option<String>,
        /// System prompt sent before the conversation
        #[arg(long, value_name = "TEXT")]
        system: Option<String>,
    },
    /// List catalog models that fit this machine
    Recommend {
        /// RAM budget to plan against instead of the RAM available right now,
        /// e.g. `4G` (keeps room for other apps)
        #[arg(long, value_name = "SIZE")]
        max_ram: Option<String>,
        /// Show every catalog model and quant
        #[arg(long)]
        all: bool,
        /// Re-download the model catalog even if the cached copy is fresh
        #[arg(long)]
        refresh_catalog: bool,
        /// Output machine-readable JSON instead of the human table
        #[arg(long)]
        json: bool,
    },
    /// Find the fastest llama.cpp settings for a model and save them as a
    /// profile (without `--model`: recommend models)
    #[command(group(
        ArgGroup::new("model_source")
            .required(false)
            .args(["model", "params"])
    ))]
    Tune {
        /// Model source, in priority order: an existing `.gguf` file path
        /// (probed header), an `org/model` Hugging Face id, or a verified
        /// catalog name (e.g. `llama-3.1-8b`). Mutually exclusive with
        /// `--params` / `--arch`.
        #[arg(long, value_name = "NAME_OR_PATH", conflicts_with = "arch")]
        model: Option<String>,
        /// Model size, e.g. `135M`, `1.2B` or `7000000`. Alternative to
        /// `--model` for models outside the catalog.
        #[arg(long, value_name = "SIZE")]
        params: Option<String>,
        /// Architecture preset for the report/display: `llama-3.1-8b`,
        /// `llama-3.1-70b` or `qwen2.5-7b`. Not used with `--model`.
        #[arg(long, value_name = "NAME")]
        arch: Option<String>,
        /// Weight format(s) to rank: `fp16`, `q8_0`, `q4_0` or `q4_k_m`.
        /// Comma-separate to sweep several, e.g. `q4_k_m,q8_0`
        #[arg(long, value_name = "QUANT[,QUANT...]", default_value = "q4_k_m")]
        quant: String,
        /// Ranking objective: `throughput`, `latency` or `balanced`
        #[arg(long, value_name = "NAME", default_value = "balanced")]
        objective: String,
        /// Override measured memory bandwidth (decimal GB/s); skips the bench
        #[arg(long, value_name = "GBPS")]
        bandwidth_gbps: Option<f64>,
        /// Override the FP32 peak (decimal GFLOPS); skips the HAL estimate
        #[arg(long, value_name = "GFLOPS")]
        gflops: Option<f64>,
        /// Keep at most this many candidates after pruning (default 64)
        #[arg(long, value_name = "N", default_value_t = 64)]
        max_configs: usize,
        /// Save the winning config as a versioned profile JSON file
        #[arg(long, value_name = "PATH")]
        save: Option<PathBuf>,
        /// Profile name recorded in the saved file
        /// (default: `<model>-<objective>`)
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
        /// Verify the best candidates with a real llama.cpp e2e run and
        /// re-score them from the measured numbers. Requires `--model` to be
        /// a local `.gguf` file.
        #[arg(long, requires = "model")]
        verify: bool,
        /// How many of the best estimate configs to verify (default 2)
        #[arg(long, value_name = "N", default_value_t = 2, requires = "verify")]
        top_k: usize,
        /// Path to the llama.cpp `llama-cli` binary (default:
        /// $ATTUNE_LLAMA_CLI, then a PATH search for `llama-cli`/`main`)
        #[arg(long, value_name = "PATH", requires = "verify")]
        llama_cli: Option<PathBuf>,
        /// Approximate prefill token count for the verification run
        /// (default 512)
        #[arg(long, value_name = "N", requires = "verify")]
        prompt_tokens: Option<usize>,
        /// Number of decode tokens for the verification run (default 128)
        #[arg(long, value_name = "N", requires = "verify")]
        gen_tokens: Option<u32>,
        /// Context window for the verification run (overrides each config's
        /// `n_ctx`; default 2048)
        #[arg(long, value_name = "N", requires = "verify")]
        ctx: Option<u32>,
        /// Saved `attune bench --json` report to calibrate the sweep against.
        /// Accepts either a bare report or the `{ "bench": … }` envelope
        /// written by `attune bench --model --save`. The report must contain
        /// an `e2e` block when `--calibrate` is used.
        #[arg(long, value_name = "PATH")]
        bench_json: Option<PathBuf>,
        /// Calibrate every ranked config from `--bench-json`. The eta pair is
        /// fitted once against the best estimate config (decode scaled,
        /// prefill divided), so the whole grid uses the same correction.
        #[arg(long, requires = "bench_json")]
        calibrate: bool,
        /// Weight format the `--bench-json` run measured (e.g. `q4_k_m`).
        /// The bench report carries no quant, so without this flag no
        /// cross-quant warning can be produced. Used only with `--calibrate`.
        #[arg(long, value_name = "QUANT", requires = "bench_json")]
        bench_quant: Option<String>,
        /// Recommend mode: RAM budget to plan against instead of the RAM
        /// available right now, e.g. `4G` (keeps room for other apps)
        #[arg(long, value_name = "SIZE")]
        max_ram: Option<String>,
        /// Recommend mode: show every catalog model and quant
        #[arg(long)]
        all: bool,
        /// Re-download the model catalog even if the cached copy is fresh
        #[arg(long)]
        refresh_catalog: bool,
        /// Show every ranked config, the score column and prune details
        #[arg(long)]
        verbose: bool,
        /// Output machine-readable JSON instead of the human table
        #[arg(long)]
        json: bool,
    },
}

/// Run the selected subcommand. All logic lives in `commands::*`.
pub fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Detect { json } => detect::run(json),
        Command::Catalog(CatalogCommand::Build {
            publishers,
            previous,
            out,
        }) => catalog::build(&publishers, previous.as_deref(), &out),
        Command::Setup {
            yes,
            refresh_catalog,
        } => setup::run(yes, refresh_catalog),
        Command::Bench {
            quick,
            json,
            model,
            llama_cli,
            save,
            prompt_tokens,
            gen_tokens,
            ctx,
            threads,
        } => bench::run(
            quick,
            json,
            model.as_deref(),
            llama_cli.as_deref(),
            save.as_deref(),
            prompt_tokens,
            gen_tokens,
            ctx,
            threads,
        ),
        Command::Estimate {
            model,
            params,
            quant,
            bandwidth_gbps,
            spread_pct,
            gflops,
            arch,
            prompt_tokens,
            gen_tokens,
            ctx,
            bench_json,
            calibrate,
            bench_quant,
            refresh_catalog,
            json,
        } => estimate::run(
            model.as_deref(),
            params.as_deref(),
            &quant,
            bandwidth_gbps,
            spread_pct,
            gflops,
            arch.as_deref(),
            prompt_tokens,
            gen_tokens,
            ctx,
            bench_json.as_deref(),
            bench_quant.as_deref(),
            calibrate,
            refresh_catalog,
            json,
        ),
        Command::Serve {
            profile,
            llama_server,
            host,
            port,
            listen,
            listen_port,
            api_key,
            timeout_s,
            force,
            yes,
            json,
        } => serve::run(
            &profile,
            llama_server.as_deref(),
            &host,
            port,
            &listen,
            listen_port,
            api_key.as_deref(),
            timeout_s,
            force,
            yes,
            json,
        ),
        Command::Chat {
            url,
            api_key,
            system,
        } => chat::run(&url, api_key.as_deref(), system.as_deref()),
        Command::Run {
            model,
            quant,
            yes,
            retune,
            no_chat,
            system,
            llama_server,
            port,
            listen,
            listen_port,
            api_key,
            timeout_s,
        } => run::run(
            &model,
            &quant,
            yes,
            retune,
            no_chat,
            system.as_deref(),
            llama_server.as_deref(),
            port,
            &listen,
            listen_port,
            api_key.as_deref(),
            timeout_s,
        ),
        Command::Recommend {
            max_ram,
            all,
            refresh_catalog,
            json,
        } => tune::run_recommend(max_ram.as_deref(), all, refresh_catalog, json),
        Command::Tune {
            model,
            params,
            arch,
            quant,
            objective,
            bandwidth_gbps,
            gflops,
            max_configs,
            save,
            name,
            verify,
            top_k,
            llama_cli,
            prompt_tokens,
            gen_tokens,
            ctx,
            bench_json,
            calibrate,
            bench_quant,
            max_ram,
            all,
            refresh_catalog,
            verbose,
            json,
        } => tune::run(
            model.as_deref(),
            params.as_deref(),
            arch.as_deref(),
            &quant,
            &objective,
            bandwidth_gbps,
            gflops,
            max_configs,
            save.as_deref(),
            name.as_deref(),
            tune::VerifyArgs {
                enabled: verify,
                top_k,
                llama_cli: llama_cli.as_deref(),
                prompt_tokens,
                gen_tokens,
                ctx,
            },
            bench_json.as_deref(),
            bench_quant.as_deref(),
            calibrate,
            tune::RecommendArgs {
                max_ram: max_ram.as_deref(),
                all,
                refresh_catalog,
            },
            json,
            verbose,
            false,
        ),
    }
}

#[derive(Subcommand)]
pub enum CatalogCommand {
    /// Regenerate catalog.json from the Hugging Face API
    Build {
        /// Generator config (allow-lists, quants, llama.cpp pin)
        #[arg(long, default_value = "catalog/publishers.json")]
        publishers: PathBuf,
        /// Yesterday's catalog: reused header facts and the shrink guard
        #[arg(long)]
        previous: Option<PathBuf>,
        /// Where to write the new catalog
        #[arg(long)]
        out: PathBuf,
    },
}

#[cfg(test)]
mod tests {
    use super::Command;
    use clap::{CommandFactory, Parser};

    fn collect_long_help(command: &clap::Command, out: &mut Vec<String>) {
        out.push(command.clone().render_long_help().to_string());
        for sub in command.get_subcommands() {
            collect_long_help(sub, out);
        }
    }

    #[test]
    fn recommend_parses_its_flags() {
        let cli = crate::Cli::try_parse_from([
            "attune",
            "recommend",
            "--max-ram",
            "4G",
            "--all",
            "--json",
        ])
        .expect("recommend should parse");
        match cli.command {
            Command::Recommend {
                max_ram,
                all,
                refresh_catalog,
                json,
            } => {
                assert_eq!(max_ram.as_deref(), Some("4G"));
                assert!(all);
                assert!(!refresh_catalog);
                assert!(json);
            }
            _ => panic!("expected the recommend subcommand"),
        }
    }

    #[test]
    fn run_parses_model_quant_yes_and_no_chat() {
        let cli = crate::Cli::try_parse_from([
            "attune",
            "run",
            "qwen3-4b",
            "--quant",
            "q8_0",
            "--yes",
            "--no-chat",
        ])
        .expect("run should parse");
        match cli.command {
            Command::Run {
                model,
                quant,
                yes,
                no_chat,
                retune,
                ..
            } => {
                assert_eq!(model, "qwen3-4b");
                assert_eq!(quant, "q8_0");
                assert!(yes);
                assert!(no_chat);
                assert!(!retune);
            }
            _ => panic!("expected the run subcommand"),
        }
    }

    #[test]
    fn help_text_has_no_milestone_notes() {
        let root = crate::Cli::command();
        let mut help = Vec::new();
        collect_long_help(&root, &mut help);
        assert!(help.len() > 1, "expected the root and its subcommands");
        for text in help {
            assert!(
                !text.to_ascii_lowercase().contains("slice"),
                "milestone note left in help:\n{text}"
            );
            let bytes = text.as_bytes();
            for window in bytes.windows(3) {
                if window[0] == b'(' && window[1] == b'M' && window[2].is_ascii_digit() {
                    panic!("milestone note left in help:\n{text}");
                }
            }
        }
    }
}
