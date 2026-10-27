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
- Validated on SmolLM2-135M Q4_K_M: bench measured 148.05 tok/s,
  calibrated estimate 148.1 (eta=0.420) — 0% error.

## M4 — Tune 🚧 (current, v0.4.0 planned)

**Goal:** find the best config automatically.

- Config-space sweep (quant, threads, batch, KV-cache type, context size)
  with scoring objectives (throughput / latency / balanced).
- Export the winner as a versioned JSON profile with validation and
  fallbacks (hardware mismatch warns, low RAM suggests smaller settings).
- Acceptance: > 20% better than llama.cpp defaults on the test machine;
  profile round-trips through validation.

## M5 — Serve (planned, v0.5.0, MVP)

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

## Later

- **V1 (desktop):** Tauri app with Simple/Preset/Advanced modes for
  non-technical users.
- **V2 (advanced):** MoE offloading, speculative decoding, hybrid CPU+GPU,
  ARM/Apple Silicon.

## How this file stays accurate

- A milestone flips to ✅ only when its issues close **and** its acceptance
  numbers are posted in the closing PR.
- Per-phase task slices are planned just before the phase starts, when
  unknowns are smallest — not speculated quarters ahead.
- Anything marked “planned” has no promised date; dates appear only once
  work starts.
