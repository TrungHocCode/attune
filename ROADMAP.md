# Roadmap

> Attune learns your hardware, measures its real capability, and finds the
> best local LLM inference config for it. This file tracks **what is done,
> what is next, and how each milestone is accepted**. The detailed task
> breakdown lives in GitHub milestones and issues, which are the source of
> truth — this file is the summary map.

## M1 — Detect ✅ (v0.1.0)

**Goal:** know what the hardware is.

- `attune detect` prints CPU (brand, cores), SIMD flags, L1/L2/L3 cache,
  RAM (total, available, speed), NUMA and GPU stubs.
- Machine-readable `--json` output; the JSON shape is the stable contract.
- Acceptance: output matches OS tooling (`lscpu`/WMI/CIM) on test machines;
  `cargo test` green on Linux and Windows.

## M2 — Measure ✅ (v0.2.0)

**Goal:** know what the hardware can really do (datasheets overpromise).

- `attune bench --quick`: fast multithreaded memory-bandwidth probe.
- `attune bench`: STREAM-style read/write/copy kernels with median/best
  stats and % of theoretical bandwidth, FP32 FLOPS via FMA loops
  (scalar/AVX2/AVX-512F with runtime selection), 100 ms resource monitor
  (RAM peak, CPU, best-effort power), llama.cpp subprocess runner and
  end-to-end TTFT/tok/s measurement with a real model.
- Versioned report schema (`attune-bench/1`); methodology documented in
  `docs/benchmark-methodology.md`.
- Acceptance: repeated runs agree within 5%; e2e numbers match manual
  llama.cpp runs within ±20%.
- Validated on SmolLM2-135M Q4_K_M / llama.cpp b11804 (see #19): decode
  152–199 tok/s attune vs 157–200 manual; TTFT same ~0.2 s band.

## M3 — Estimate ✅ (v0.3.0)

**Goal:** predict performance before running anything.

- Roofline model: decode `tok/s ≈ bandwidth / bytes-per-weight`,
  prefill `time = 2 × params × tokens / FLOPS`.
- Calibration against this machine's real bench numbers (`--bench-json`
  + `--calibrate`, eta fit per decode/prefill).
- Every prediction ships a confidence interval (from measured spread);
  bottleneck analysis names the limiter (memory bandwidth vs compute).
- `attune estimate --params --quant [--quant fp16,q8_0,q4_k_m]` compares
  configs in one table (GGUF parsing deferred to M4).
- Acceptance: uncalibrated error < 25%, calibrated error < 12%.
- Re-validated 2026-11-23 on Qwen3 4B and 8B Q4_K_M (held-out protocol,
  [docs/benchmark-methodology.md](docs/benchmark-methodology.md#acceptance-protocol-m3m4)):
  uncalibrated decode +5.8% (4B) / -4.1% (8B) vs llama-bench, TTFT -12.3% /
  -21.6% (target < 25%: met, 8B TTFT close to the limit); calibrated on one
  run and scored on 4 held-out runs: median decode +0.8% (4B, range
  +0.2..+1.8%) / -2.5% (8B, -4.3..-0.2%), median TTFT -2.8% (4B,
  -5.8..-1.6%) / -0.2% (8B, -1.9..+1.7%) (target < 12%: met).
  Superseded: the in-sample "0% error" on SmolLM2-135M (fit and scored on
  the same report).

## M4 — Tune ✅ (v0.4.0, target revised 2026-11-25)

**Goal:** find the best config automatically, and tell the user what it
will deliver before they commit to it.

- Config-space sweep (quant, threads, batch, KV-cache type, context size)
  with scoring objectives (throughput / latency / balanced); two-tier
  estimate-then-verify-top-K with real e2e runs. Near-ties within the
  estimate's error band go to llama.cpp defaults (#103).
- Export the winner as a versioned JSON profile with validation and
  fallbacks (hardware mismatch warns, low RAM suggests smaller settings).
- Acceptance (revised, #103): (1) **no regression**: the tuned profile is
  never more than 3% slower at decode or 5% slower at prefill than
  llama.cpp defaults; (2) **trustworthy prediction**: after one calibration
  run, predicted decode and TTFT are within 15% of measured; profile
  round-trips through validation.
- Why the target changed: the original "> 20% faster than defaults" cannot
  hold on CPU, where decode is memory-bound and the defaults already sit on
  the bandwidth roofline (2026-11-23 re-run on Qwen3 4B: +2.9%, with prefill
  17% slower). It was only ever met on a 135M model whose runs varied by
  35%. Attune's value is the accurate prediction and a profile that never
  loses to defaults; large speed-ups belong to M7 (GPU layer split).
- Results 2026-11-25 (llama-bench, 2 alternating rounds x 3 reps, see
  [docs/benchmark-methodology.md](docs/benchmark-methodology.md#acceptance-protocol-m3m4)):
  (1) Qwen3 4B prefill tuned 72.6-72.8 vs defaults 63.6-70.0 tok/s, decode
  9.6-11.3 vs 10.7-11.4 (within run noise); Qwen3 8B prefill 41.9-43.2 vs
  41.0-43.3, decode 6.84-6.85 vs 6.05-6.69: **met**. (2) calibrated held-out
  median error decode +0.8% / -2.5%, TTFT -2.8% / -0.2% (4B / 8B, from the
  M3 re-validation): **met**. Uncalibrated (an M3 target, < 25%) is
  sensitive to machine load: 4B decode +25% on 2026-11-25 with 3.4 GB RAM free and a
  busy browser, vs +5.8% on 2026-11-23; 8B -1.5%.

## M5 — Serve ✅ (v0.5.0, MVP)

**Goal:** run the winning profile for real.

- `attune serve --profile <file>`: launch llama-server with profile flags,
  health-check the model load, proxy an OpenAI-compatible API, show live
  metrics (TTFT, tok/s, RAM).
- Metrics API for local + remote observation, measured at the proxy (no
  engine instrumentation): `GET /v1/metrics` (JSON: ttft/tpot/tok_s, RAM,
  uptime, request counts with mean/p50/p95) and `GET /metrics`
  (Prometheus text for Grafana). Binds `127.0.0.1` by default; remote
  needs explicit `--listen` plus `--api-key`.
- Slices: 1 launch + health (#62), 2 OpenAI proxy + shutdown (#63),
  3 metrics (#90), 4 api-key + hardening incl. Windows child cleanup
  (#91, #77).
- Acceptance: install-to-first-inference under 5 minutes; profile apply
  success > 95%.
- Results 2026-11-24 (main `de096b5`, release build, llama.cpp b11804 CPU,
  Windows 11, Core 5 210H): **profile apply 20/20 = 100%** (`tune --save`
  profiles for SmolLM2-135M, Qwen3 0.6B, 4B and 8B Q4_K_M, 5 serve starts
  each: health, one chat completion answered by the right model, hard kill;
  ready in 2.8 / 3.3 / 4.8-6.5 / 13.9-17.0 s, llama-server gone ~1.5 s
  after attune is killed). **Install to first inference 3 min 38 s**: fresh
  clone, `cargo build --release` (201 s), `tune --save`, `serve`, first
  Qwen3 4B reply at 218 s. Excludes the Rust toolchain, llama.cpp and model
  downloads; the cargo registry cache was warm.

## Estimate accuracy fixes (after M5)

Found while re-testing M1–M5 on real models: #71, #72 (weight and KV
bytes), #73, #74 (GGUF facts), #75 (profile metadata), #69 (prefill
ceiling), #70 (bandwidth probe), then #78 re-validates M3/M4 on 4B+ models.
The GPU work below builds on this math, so it lands first.

## M6 — Recommend & downloads ✅ (v0.6.0)

**Goal:** a new user who does not know model names gets a model that suits
their machine running, with no manual downloads.

- `attune tune` without `--model` lists catalog models that fit this machine
  (download size, RAM, predicted speed and first reply, fit verdict, one pick
  per size tier); `--max-ram` keeps a RAM budget.
- Model catalog regenerated daily by a GitHub Action from trusted Hugging
  Face publishers (pinned files, sizes, SHA-256), cached locally, with a
  built-in snapshot for offline use.
- Profiles record the model `source`; `serve` downloads a missing model
  after confirmation. `attune setup` installs the pinned llama.cpp build.
- New leaf crate `attune-fetch` (home dir, resumable verified downloads).
- Acceptance: zero manual downloads from binary to first reply; uncalibrated
  predicted decode within 25% on 3 catalog models of different tiers, and
  every model marked as fitting loads and serves; valid daily catalogs
  (revised 2026-11-28 from "7 consecutive days", see below); works offline; interrupted downloads resume and corrupted files
  are rejected.
- Design: [docs/superpowers/specs/2026-11-26-recommend-design.md](docs/superpowers/specs/2026-11-26-recommend-design.md).
- Results 2026-11-27 (Windows 11, Core 5 210H, llama.cpp b11804 CPU, see
  [docs/benchmark-methodology.md](docs/benchmark-methodology.md#m6-recommend-acceptance-2026-11-27)):
  zero manual downloads, empty home to first reply in 67.9 s including 0.66 GB
  of downloads (**met**); uncalibrated decode +5.1% (Qwen3 0.6B Q8_0), +9.5%
  (4B Q4_K_M), -2.4% (8B Q4_K_M) vs llama-bench (**met**), and the fitting
  pick served; offline recommend + serve (**met**); a download killed at 76 MB
  resumed, a tampered SHA-256 was rejected (**met**).
- Catalog freshness, revised 2026-11-28:
  - The bar was 7 consecutive days, later cut to 2 scheduled runs, and finally closed on the verified manual run (2026-11-27, 66 models) plus monitoring of the scheduled runs.
  - Why: users never depend on a fresh catalog, because loading falls back from a stale cache to the built-in snapshot (now 66 models).
  - A failing daily run is a bug for a patch release, not a blocker.
- Friendly path, added to M6 (design [2026-11-27-friendly-design.md](docs/superpowers/specs/2026-11-27-friendly-design.md)):
  - One-line install and release binaries (#127).
  - `attune recommend` (#121), `attune chat` (#122), and `attune run <model>` (#126).
- Release acceptance 2026-11-28 (same machine, v0.6.0 from GitHub, empty `ATTUNE_HOME`):

  | Step | Time |
  |---|---|
  | `irm …/install.ps1 \| iex` (PATH set, a new shell finds `attune 0.6.0`) | 6.9 s |
  | `attune recommend` | 7.3 s |
  | `attune run qwen3-0.6b --yes` (installs llama.cpp, tunes, downloads 0.4 GB, serves, streams the reply) | 55.6 s |
  | **Total, install to first chat reply, downloads included** | **≈ 70 s** (target: < 5 min excluding the model download, **met**) |

  - `/bye` left no llama-server running.
  - The `release.yml` pipeline passed on its first real tag.
- Found on the way: Windows power throttling slowed every in-process probe
  after the first (34.7 -> 12-17 GB/s), and `HTTP(S)_PROXY` broke `serve`'s
  localhost health check; both fixed. llama.cpp prefill reaches 0.40 of the
  unthrottled int8 peak (was assumed 0.9-0.95).

## M7 — GPU (proposed)

**Goal:** use the GPU the machine already has, and split the model across
GPU + CPU when it does not fit in VRAM.

- Detect GPUs (vendor CLI, no SDK dependency), measure effective VRAM
  bandwidth through llama.cpp, estimate full and partial offload with a VRAM
  fit check, tune `n_gpu_layers`, serve with a GPU llama.cpp build.
- Full plan, non-goals and proposed acceptance: [docs/gpu-plan.md](docs/gpu-plan.md).

## Later

- **V1 (desktop):** Tauri app with Simple/Preset/Advanced modes for
  non-technical users.
- **V2 (advanced):** MoE offloading, speculative decoding, multi-GPU,
  ARM/Apple Silicon.

## How this file stays accurate

- A milestone flips to ✅ only when its issues close **and** its acceptance
  numbers are posted in the closing PR.
- Per-phase task slices are planned just before the phase starts, when
  unknowns are smallest — not speculated quarters ahead.
- Anything marked “planned” has no promised date; dates appear only once
  work starts.
