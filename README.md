# attune — Tune LLM inference to your hardware

> [!NOTE]
> Beta (v0.5.x, MVP). `detect`, `bench`, `estimate`, `tune` and `serve`
> work on CPU (Linux/Windows x86-64). GPU support is planned. Not
> production-ready yet.

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

Requires Rust stable ([install](https://www.rust-lang.org/tools/install)).

```powershell
git clone https://github.com/TrungHocCode/attune.git
cd attune
cargo build -p attune-cli
.\target\debug\attune.exe detect
```

```
CPU:        Intel(R) Core(TM) 5 210H (GenuineIntel, 8C/12T)
SIMD:       SSE4.2, AVX, AVX2, AVX-VNNI
Clock:      unknown (best-effort)
Cache:      L1d 32 KB | L1i 64 KB | L2 2.0 MB | L3 12.0 MB | line 64 B
NUMA:       1 node (not detected; single node assumed)
GPU:        none detected (GPU detection not implemented)
RAM:        15.6 GB total / 1.8 GB available
OS:         windows
```

```powershell
.\target\debug\attune.exe bench --quick
```

```
Memory bandwidth (read): 32.2 GB/s (buffer 256 MiB, 8 threads, 20 reps)
```

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
attune serve --profile profile.json                     # run it behind an OpenAI-compatible API
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

M1 `detect` ✅ · M2 `bench` ✅ · M3 `estimate` ✅ · M4 `tune` ✅ · M5 `serve` ✅ (MVP) · M6 GPU (proposed)

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
