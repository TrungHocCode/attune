# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-10-28

Calibrated sweep plus the M4 acceptance measurement on SmolLM2-135M Q4_K_M.

### Added
- `attune tune --bench-json <PATH> --calibrate [--bench-quant <QUANT>]`:
  fits `eta_mem` / `eta_comp` **once** against the best estimate config and
  applies the same factors to every ranked config (decode scaled, prefill
  divided). Each `RankedConfig` gets an optional `calibration` block with its
  calibrated numbers and cross-workload warnings; `--bench-quant` enables the
  quant-mismatch notice and the bench prompt length drives the prompt-mismatch
  notice.
- `BenchCalibration` and `parse_bench_report` in `attune-estimate`, plus
  `attune_tune::calibrate_ranked`, so `estimate` and `tune` share one
  calibration/parsing path instead of duplicating it.
- `attune tune` human output prints the fitted `eta_mem` / `eta_comp` and the
  bench's measured tok/s / TTFT above the ranking.

### Changed
- `RankedConfig` gains `calibration` (`serde(default)`, omitted when
  uncalibrated, so older JSON still parses).
- `attune-estimate` exports `calibrate_with_bench` (moved out of the CLI) with
  unchanged behavior; `estimate` still fits and applies per report.

### Acceptance (M4)

Measured on SmolLM2-135M Q4_K_M / llama.cpp 0.7.0-dev, Windows x86-64, five
consecutive runs per side:

- llama.cpp defaults (`-m -p <~500-token prompt> -n 32 -c 2048`, no `-t`):
  **125.3 tok/s** median (runs 118.6–136.2).
- `attune tune --objective throughput --verify --top-k 2 --max-configs 8`:
  **157.5 tok/s** median best measured config (runs 141.8–174.5); the winner
  was `q4_k_m ctx=2048 -t 4 n_batch=128`.
- Median-vs-median improvement **25.7%**; best-vs-best **28.1%**. That clears
  the >20% M4 target on these runs, but both sides are noisy on a 135M model,
  so treat the margin as directional rather than a hard guarantee.

Release binary: `attune.exe` = **3,898,368 bytes (3.72 MiB)**, under the
20 MB budget.

## [0.3.2] - 2026-10-27

Model-resolution wiring for `attune estimate` (closes #48). One resolver now
feeds the single- and multi-quant paths, so parameters, KV shape, context and
warnings cannot diverge between them.

### Added
- Resolution order: existing `.gguf` file (header probe) > `org/model`
  Hugging Face id (30-day cache/fetch) > built-in catalog short name >
  manual `--params` (+ optional `--arch`).
- Context resolution: an explicit `--ctx` always wins; otherwise the
  source's documented default (`config.json` `max_position_embeddings` or
  the catalog entry) is used; otherwise the historical 2048.
- `inputs.arch` and `inputs.ctx` in the JSON contract (`serde(default)`;
  older reports still parse), echoing the resolved KV shape and context.
- Pretty `Model: <provider>/<name> (<params>) | Fit: …` line for both the
  single report and the multi-quant table, reusing the existing `RamFit`.

### Changed
- The resolved architecture now drives `kv_per_token`, and the resolved
  context drives both the decode prediction and the RAM-fit check for every
  source — including catalog names and fetched configs, which previously
  ran weights-only.
- `--ctx` is an optional flag instead of a clap default, so omission can be
  told apart from an explicit `2048`.

### Notes
- No new dependencies and no network in tests; fetch/cache policy stays
  covered by the deterministic offline fixtures.

## [0.3.1] - 2026-10-26

Correctness pass over `attune estimate` so M4 tune can safely suggest
models to users. No new commands; predictions get louder about degraded
inputs instead of printing confident wrong numbers.

### Fixed
- Unknown CPU clock falls back to the 1 GHz baseline with an explicit
  warning (pretty + `warnings` report field) instead of refusing to run;
  `--gflops` still overrides everything (#35).
- Decode prediction includes KV-cache bytes (`--arch` presets,
  `--ctx` now honored); a missing `--arch` warns loudly (#37).
- Prefill TTFT scales per quant (`fp16` 1.0, `q8_0` 1.15, `q4` 1.3 as
  initial heuristics) so multi-quant tables rank formats correctly (#38).

### Added
- RAM fit check: `RamFit` block (`weight + kv` vs 90% of available RAM)
  with `OOM: need …, have …` warning instead of tok/s for models that
  cannot load (#36).
- Cross-workload calibration warnings (quant/prompt mismatch, clamped
  eta) plus `--bench-quant` flag; missing `e2e` still reports
  `CalibrationUnavailable` (#39).
- `warnings` report field (omitted when empty; old JSON still parses).

## [0.3.0] - 2026-10-22

Closes **M3 — `attune estimate`**. Roofline decode/prefill prediction,
calibration against a real bench, confidence intervals and multi-config
compares are now in place.

### Added
- `EstimateReport` confidence interval fields: `decode_tps_lo` /
  `decode_tps_hi` and `prefill_ms_lo` / `prefill_ms_hi`, plus a
  `confidence { level, method }` block. Bounds use the symmetric
  `pred × (1 ± spread/100)` formula, where the spread is
  `(best − median) / median` from a full bench read-kernel `KernelStats`.
- `attune-estimate` exports `interval` and `spread_pct_from_stats` as pure,
  unit-tested helpers; `Confidence::measured` / `Confidence::unavailable`
  carry the level (`high`/`medium`/`low`/`unavailable`) and provenance.
- `--quant fp16,q8_0,q4_k_m` multi-config compare: the human output is a
  fastest-first table and `--json` emits `{ schema, model, configs: [...] }`.
  A single `--quant` still emits the existing bare report (backward
  compatible).
- `--spread-pct <PCT>` to supply an explicit half-spread when bandwidth is
  overridden or quick-probed; a full `--bench-json` report supplies the
  measured spread automatically.
- `parse_quants` in `attune-estimate::model` for comma-separated,
  deduplicated weight-format lists.

### Changed
- `--bench-json` is also used as the confidence-spread source; a full report
  gives intervals, a quick report records why they are unavailable.

### Notes
- No new dependencies. Confidence bounds stay `null` when no measured spread
  is available (the quick probe has no median), with the reason in
  `confidence.method`.

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
