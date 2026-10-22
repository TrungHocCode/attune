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

## M2 — Measure 🚧 (code complete, validation pending — v0.2.0)

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
- **Remaining:** e2e/subprocess acceptance was never run against a real
  model (no `llama-cli` or `.gguf` on the dev machine; live tests are
  env-gated). Tracked in the validation issue — M2 flips to ✅ only when
  measured TTFT/tok/s numbers are posted there.

## M3 — Estimate 🚧 (current, v0.3.0 planned)

**Goal:** predict performance before running anything.

- Roofline model: decode `tok/s ≈ bandwidth / bytes-per-weight`,
  prefill `time = 2 × params × tokens / FLOPS`.
- Calibration against this machine's real bench numbers.
- Every prediction ships a confidence interval; bottleneck analysis names
  the limiter (memory bandwidth vs compute).
- `attune estimate --model <file> --config <...>`: compare configs.
- Acceptance: uncalibrated error < 25%, calibrated error < 12%.

## M4 — Tune (planned, v0.4.0)

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
