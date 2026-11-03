# attune — Tune LLM inference to your hardware

> [!NOTE]
> Beta (v0.4.x). `detect`, `bench`, `estimate` and `tune` work;
> `serve` is in progress. Not production-ready yet.

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
CPU:        Intel(R) Core(TM) 5 210H (GenuineIntel, 8P, 12L)
SIMD:       SSE4.2, AVX, AVX2
Clock:      2.20 GHz max (best-effort)
Cache:      L1d 48 KB | L1i 32 KB | L2 1.2 MB | L3 12.0 MB | line 64 B
RAM:        15.6 GB total / 2.7 GB available
OS:         windows
```

```powershell
.\target\debug\attune.exe bench --quick
```

```
Memory bandwidth (read): 33.2 GB/s (buffer 256 MiB, 8 threads, 3 reps)
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
attune serve --profile profile.json                     # run it (M5, in progress)
```

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

M1 `detect` ✅ · M2 `bench` ✅ · M3 `estimate` ✅ · M4 `tune` ✅ · M5 `serve` 🚧 (MVP)

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
