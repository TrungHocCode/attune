# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-10-21

Closes **M2 — `attune bench`**. New end-to-end llama.cpp decode measurement
on top of the slice-1…5 micro-bench stack.

### Added
- `attune-bench::e2e` — llama.cpp end-to-end decode measurement: TTFT,
  TPOT, decode tok/s, RAM peak, CPU avg, (Linux) power avg. Pure
  derivations from `PerfSummary`, unit-tested on canned input. Manual
  `--ignored` integration test against a real `llama-cli` + GGUF model,
  env-gated (`ATTUNE_LLAMA_CLI` + `ATTUNE_TEST_MODEL`) so CI never runs
  it.
- `attune-bench::BenchReport` — unified report carrying schema,
  mode (`quick`/`full`), bandwidth kernels, FP32 FLOPS, resource usage,
  and the optional e2e block. `schema = "attune-bench/1"`; missing
  fields default via `#[serde(default)]` so older JSON still parses.
- CLI new flags for `--model` users: `--prompt-tokens`, `--gen-tokens`,
  `--ctx`, `--threads`, `--save <PATH>`. `--save` works with or without
  `--model` and writes the same JSON the CLI prints.
- `docs/benchmark-methodology.md` — kernels, working-set sizing, pass
  count, monitor semantics, decode math, and the caveat list (turbo /
  thermal, AVX-512 downclocking, RAPL permission, llama.cpp output
  drift, CI-only smoke sizes).

### Changed
- `attune bench --json` now wraps the bandwidth/flops/resources/e2e
  payload under a top-level `BenchReport` with `schema` + `mode`
  keys (matches the M2 plan's "JSON is the contract" rule; old
  bandwidth-only consumers can still find their `read_gbps` etc. via
  the `bandwidth` sub-key). `--save` mirrors the printed document.
- `attune-cli` thin-CLI rule honoured: argument parsing only, no
  measurement logic in `main.rs` (clap args → `bench::run`).

## [0.1.0] - 2026-10-20
### Added
- Workspace init (`attune-hal`, `attune-cli`) + CI
- `attune detect` stub (CPU logical cores, RAM via sysinfo, `--json`)
