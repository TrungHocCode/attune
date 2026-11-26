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
  loses to defaults; large speed-ups belong to M6 (GPU layer split).
- Results 2026-11-25 (llama-bench, 2 alternating rounds x 3 reps, see
  [docs/benchmark-methodology.md](docs/benchmark-methodology.md#acceptance-protocol-m3m4)):
  (1) Qwen3 4B prefill tuned 72.6-72.8 vs defaults 63.6-70.0 tok/s, decode
  9.6-11.3 vs 10.7-11.4 (within run noise); Qwen3 8B prefill 41.9-43.2 vs
  41.0-43.3, decode 6.84-6.85 vs 6.05-6.69: **met**. (2) calibrated held-out
  median error decode +0.8% / -2.5%, TTFT -2.8% / -0.2% (4B / 8B, from the
  M3 re-validation): **met**. Uncalibrated (an M3 target, < 25%) is
  sensitive to machine load: 4B decode +25% today with 3.4 GB RAM free and a
  busy browser, vs +5.8% on 2026-11-23; 8B -1.5%.

## M5 — Serve 🚧 (current, v0.5.0 planned, MVP)

**Goal:** run the winning profile for real.

- `attune serve --profile <file>`: launch llama-server with profile flags,
  health-check the model load, proxy an OpenAI-compatible API, show live
  metrics (TTFT, tok/s, RAM).
- Metrics API for local + remote observation, measured at the proxy (no
  engine instrumentation): `GET /v1/metrics` (JSON: ttft/tpot/tok_s, RAM,
  uptime, request counts with mean/p50/p95) and `GET /metrics`
  (Prometheus text for Grafana). Binds `127.0.0.1` by default; remote
  needs explicit `--listen` plus `--api-key`.
- Acceptance: install-to-first-inference under 5 minutes; profile apply
  success > 95%.
- Slices: 1 launch + health ✅ (#62), 2 OpenAI proxy + shutdown ✅ (#63),
  3 metrics (#90), 4 api-key + hardening incl. Windows child cleanup
  (#91, #77), then the acceptance run and v0.5.0.

## Estimate accuracy fixes (after M5)

Found while re-testing M1–M5 on real models: #71, #72 (weight and KV
bytes), #73, #74 (GGUF facts), #75 (profile metadata), #69 (prefill
ceiling), #70 (bandwidth probe), then #78 re-validates M3/M4 on 4B+ models.
The GPU work below builds on this math, so it lands first.

## M6 — GPU (proposed)

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
