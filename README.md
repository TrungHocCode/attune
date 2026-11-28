# attune — Tune LLM inference to your hardware

> [!NOTE]
> Beta (v0.6.x). One-line install, `recommend`, `run` and `chat` work on CPU
> (Linux/Windows x86-64). GPU support (M7) is in progress. Not production-ready yet.

[![CI](https://github.com/TrungHocCode/attune/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TrungHocCode/attune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What it does

Running a Large Language Model (LLM) locally usually means guessing
[llama.cpp](https://github.com/ggerganov/llama.cpp) flags and hoping for the
best — performance depends on your exact hardware, and datasheet numbers are
typically only 60–85% of what you really get.

Attune closes that gap in three steps:

1. **Detect** your hardware (CPU, SIMD, cache, RAM).
2. **Measure** its real capability (memory bandwidth, FLOPS, then end-to-end
   inference speed with llama.cpp).
3. **Tune** — pick the best inference config for your machine and export it
   as a reusable JSON profile that `attune serve` can run.

## Quick start

**1. Install** (one line, no Rust needed):

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/TrungHocCode/attune/main/install.ps1 | iex
```

```bash
# Linux x86-64
curl -fsSL https://raw.githubusercontent.com/TrungHocCode/attune/main/install.sh | sh
```

The installer downloads the latest release, checks its SHA-256, puts `attune` in
`~/.attune/bin` and adds that folder to your PATH. Open a new terminal afterwards.
Windows SmartScreen may warn the first time, because the binary is not code-signed yet.
Run the same line again to upgrade.

**2. See which models suit this machine:**

```powershell
attune recommend
```

```
Machine: RAM 33.5 GB/s, 3.2 GB for models · catalog 2026-11-27 (cached)

  Model                      Quant    Download      RAM      Speed  First reply  Fit
★ Qwen3 4B Instruct 2507     q4_k_m     2.5 GB   3.1 GB   13.0 tok/s        6.5 s  ✅
• Qwen3 0.6B                 q8_0       0.6 GB   1.1 GB   47.6 tok/s        1.0 s  ✅
  Mistral 7B Instruct v0.3   q4_k_m     4.4 GB   4.9 GB    7.5 tok/s       11.7 s  ❌ needs 0.2 GB more
  ...
Next: attune run qwen3-4b-instruct-2507
```

**3. Run it and chat:**

```powershell
attune run qwen3-4b-instruct-2507
```

`attune run` handles everything else:
- installs llama.cpp the first time;
- tunes the settings for your hardware and saves them;
- downloads and verifies the model, asking first;
- opens a chat in the terminal. `/bye` quits.

The next `attune run` of the same model reuses the saved settings.

The same server works with any OpenAI-compatible app:
- `attune run <model> --no-chat` serves at `http://127.0.0.1:8080/v1`;
- `attune chat` talks to a running server from another terminal.

Predictions are uncalibrated (about ±25%; within ±10% decode on the reference machine).
`attune recommend --max-ram 4G` plans against a RAM budget instead of the RAM free right now,
and `--all` lists every model and quant.

### Where files go

Everything lives in `~/.attune` (`%USERPROFILE%\.attune` on Windows):

| Folder | What | Size |
|---|---|---|
| `bin/` | the `attune` binary | ~10 MB |
| `llama.cpp/` | the pinned llama.cpp build | ~50 MB |
| `models/` | downloaded GGUF models | 0.5–10 GB each |
| `profiles/` | tuned settings, one file per model | a few KB |

To keep models on another drive, set `ATTUNE_HOME` before installing:
- PowerShell: `$env:ATTUNE_HOME = "D:\attune"`
- Linux: `export ATTUNE_HOME=/data/attune`

To use it permanently, set it as a user environment variable. To uninstall, delete the folder and remove its `bin` from your PATH.

To build from source instead, see [CONTRIBUTING.md](CONTRIBUTING.md#build-from-source).

## Usage

```powershell
attune detect              # pretty hardware table
attune detect --json       # machine-readable JSON (the contract)
attune bench --quick       # fast memory-bandwidth probe
attune bench               # full bandwidth + FLOPS report
attune bench --json        # machine-readable report
attune estimate --model smollm2-135m --quant q4_k_m   # predict before running
attune tune --model smollm2-135m --objective throughput  # rank configs
attune tune --model smollm2-135m --save profile.json     # export best profile
attune recommend --max-ram 4G --all                    # every model under a 4 GB budget
attune setup --yes                                      # install the pinned llama.cpp build
attune serve --profile profile.json                     # serve a profile (file path or saved name)
attune run qwen3-4b-instruct-2507                       # setup + tune + download + serve + chat
attune chat --url http://127.0.0.1:8080                 # chat with a running server
```

While serving, the proxy answers `GET /v1/metrics` (JSON) and `GET /metrics`
(Prometheus text) with measured TTFT, TPOT, tok/s, request counters, and
llama-server RSS.

The proxy listens on `127.0.0.1` by default. Any other `--listen` host is
refused unless `--api-key <KEY>` (or `ATTUNE_API_KEY`) is set; with a key every
route except `/health` needs `Authorization: Bearer <KEY>`. On Windows,
llama-server is killed with attune even on a hard kill.

`estimate --model` accepts a catalog name, an `org/model` Hugging Face id
(30-day cache, offline fallback), or a local GGUF path:

```powershell
attune estimate --model smollm2-135m --quant q4_k_m
attune estimate --model HuggingFaceTB/SmolLM2-135M --quant q4_k_m
attune estimate --model .\models\smollm2-135m-q4_k_m.gguf --quant q4_k_m
attune estimate --params 135M --arch llama-3.1-8b --quant q4_k_m --ctx 2048
```

`--ctx` overrides the model's documented context default; without either, the
historical 2048 is used.

JSON output is the stable contract for scripting; the pretty table is for humans.

## Platform support

| Platform        | Status | Notes                              |
|-----------------|--------|------------------------------------|
| Linux (x86-64)  | ✅ Full | RAPL power readings where permitted |
| Windows (x86-64)| ✅ Full | RAM speed via CIM, no RAPL          |
| macOS / ARM64   | Planned | After MVP                           |

## Roadmap

M1 `detect` ✅ · M2 `bench` ✅ · M3 `estimate` ✅ · M4 `tune` ✅ · M5 `serve` ✅ (MVP) · M6 recommend & downloads ✅ · M7 GPU (in progress)

Details, acceptance criteria, and status live in [ROADMAP.md](ROADMAP.md).
GitHub issues and milestones are the source of truth for what lands next.

## Development

```powershell
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo run -p attune-cli -- detect
```

The workspace is split into focused crates (`attune-hal`, `attune-bench`,
…). See [AGENTS.md](AGENTS.md) for agent rules and architecture constraints.
Minimum supported Rust is stable (edition 2021, no nightly features).

## Contributing

Contributors of all skill levels are welcome! Please read
[CONTRIBUTING.md](CONTRIBUTING.md), open an issue first for large changes,
and note that `main` is protected — all changes go through PRs with green CI
on Ubuntu and Windows.

## License

MIT — see [LICENSE](LICENSE).
