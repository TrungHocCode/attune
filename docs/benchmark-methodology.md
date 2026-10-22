# Benchmark methodology

This page documents how `attune bench` actually measures numbers, so a
user can tell which knob moves which figure and where the hidden
ceilings live. Read it before treating a single `attune bench` output as
ground truth on shared hardware.

## Kernels

The M2 benches live in `crates/attune-bench/src/`. Three families:

* **Memory bandwidth** — read / write / copy. STREAM-style: 256 MiB
  working set (raised to ≥ 4× L3, clamped to ≤ ¼ of available RAM),
  spread across the physical core count, 1 warm-up pass discarded,
  5 timed passes for full mode, 3 for `--quick`. Bytes per pass are
  counted directly by the kernel; seconds come from a wall-clock
  `Instant` around the parallel `std::thread::scope`. GB/s is decimal
  (`1e9` bytes), the same convention vendor DRAM figures use.
* **FP32 FMA throughput** — tight FMA loop on per-thread L1-resident
  float buffers (8 KiB each, 16 KiB total per thread — well below L1d).
  Path is decided at runtime via stable `is_x86_feature_detected!` with a
  raw-cpuid fall-back if the OS disabled a feature after boot (e.g.
  AVX-512-downclocking BIOS options). 3 warm-up passes, 3 timed passes
  in full mode.
* **End-to-end llama.cpp decode** — `--model <path>` spawns the user-
  supplied `llama-cli` with portable flags (`-m -p -n -c` plus optional
  `-t`), scrapes the `llama_perf` block from stderr, and folds RAM / CPU
  / (Linux) power readings into the `e2e` block.

## Working-set sizing

`BandwidthConfig::full_from_hardware` chooses a buffer that is:

* at least the larger of `DEFAULT_BUFFER_MIB` (`256` MiB) and `4 × L3`
  (L3 wins on a 96 MiB L3 CPU → 384 MiB working set),
* at most `1/4` of currently-available RAM (so the bench never pushes
  the page cache out and skews itself),
* rounded down to a MiB boundary,
* never below the `MIN_BUFFER_MIB` (`16` MiB) floor.

The full-mode copy kernel allocates 2× that, since it bills both the
read and the write of every element.

## Pass count and stats

`FULL_REPS` is 5 for full mode, `DEFAULT_REPS` is 3 for `--quick`.
Each kernel reports the *median* GB/s and the *best-of* GB/s (the full
sweep reports both; quick reports only the best-of). Median is the
robust figure (a noisy rep cannot drag it), best is the headline (a
rep ran at peak).
Spread (`(max - min) / median × 100 %`) is pure and exposed in
`bandwidth::spread_pct` for callers that want to sanity-check their
own run.

## Resource monitor

100 ms sampling period on a dedicated thread (`attune-resource-monitor`),
clamped to ≥ 10 ms so the per-tick overhead does not dominate the
period. Aggregates RAM peak, total CPU % (across all cores, so a
saturated 16-thread kernel reports ≈ 1600 %, not 100 %), and Linux
RAPL-derived watts. Power is *strictly* best-effort: a non-Linux
platform, a missing `/sys/class/powercap`, or a run too short for one
delta → `power_avg_watts: null`. Monitoring never fails a bench.

## Decode math (`e2e.rs`)

Three pure derivations from the `PerfSummary` scraped out of
`llama_perf`:

* `ttft_ms(perf) = perf.prompt_eval_ms` — the prefill wall clock is the
  time before the first generated token lands, so it stands in for
  time-to-first-token.
* `tpot_ms(perf) = perf.eval_ms / perf.eval_tokens` — milliseconds per
  generated token, the inverse of decode throughput.
* `tok_s(perf) = 1_000 / tpot_ms` — tokens per second over the decode
  phase. `0.0` on any unmeasurable input — never `NaN`, never `Inf`.

The decode row in the human report and the `tok_s` / `tpot_ms` /
`ttft_ms` fields in JSON are computed by these functions, so a canned
`PerfSummary` is enough to verify them — see `e2e::tests`.

The decode-byte figure from the M2 plan formula (decode is
memory-bound, `tok/s ≈ bandwidth / bytes_weights`) is computed by the
later `attune estimate` slice from this bench's `tok_s` plus the
hardware context. M2 deliberately *measures* what the model actually
did; M3 derives a roofline number from it.

## Estimate math & calibration

`attune estimate` turns measured hardware numbers into a prediction:

* **Decode (memory-bound):** `tok/s = bandwidth / (params × bytes_per_param)`.
  `bytes_per_param` comes from the `BITS_PER_WEIGHT` table in
  `crates/attune-estimate/src/model.rs`, which uses the real llama.cpp block
  layouts rather than rounded labels: FP16 16.0, Q8_0 8.5008, Q4_0 4.5,
  Q4_K_M 4.8944 bits per weight (0.6118 B/param for Q4_K_M). The figure is
  **weights only**; the KV cache grows with context and is not modelled yet.
* **Prefill (compute-bound):**
  `ms = 2 × params × prompt_tokens / (GFLOPS × 1e9) × 1000`.
* **Ridge:** `AI* = FLOPS_peak / BW` in FLOP per byte. Decode's arithmetic
  intensity is `1 / bytes_per_param`; below the ridge it is memory-bound,
  at or above it compute-bound.

Calibration (`--bench-json <PATH> --calibrate`) reads a saved
`attune bench --model` report — either the bare `BenchReport` or the
`{ "llama": …, "bench": … }` envelope written by `--save` — and fits two
scalar correction factors:

```text
eta_mem  = measured_tok_s / roofline_tok_s
eta_comp = roofline_prefill_ms / measured_ttft_ms
calibrated_decode_tps = roofline_tok_s × eta_mem
calibrated_ttft_ms    = roofline_prefill_ms / eta_comp
```

Both etas are clamped to `0.05 ..= 1.5`. Decode is a throughput ratio;
prefill *time* scales inversely, so it is applied as a divisor. The ratio
absorbs everything the roofline leaves out for this slice: KV-cache traffic,
dequantisation and attention overhead, scheduling, and the gap between the
quoted FP32 peak and the precision llama.cpp actually runs. A report with no
`e2e` block (or a non-positive `tok_s`) fails cleanly as
`CalibrationUnavailable` instead of guessing. On the canned fixture
(246.9 tok/s predicted vs 55.6 tok/s measured) `eta_mem ≈ 0.225` and the
calibrated figure reproduces the measurement — zero error by construction.
Pass the same `--prompt-tokens` value used for `attune bench --model`, so the
roofline prefill and the measured TTFT cover the same prompt length; otherwise
the TTFT ratio is comparing two different workloads (the eta clamp keeps it
bounded, but it is not meaningful).

**Median choice.** The bandwidth input is the kernel's *median* GB/s (full
mode), so one noisy rep cannot drag the roofline. The e2e input is already a
whole-decode-phase average (`eval_ms / eval_tokens`), which is the steady-state
number a user sees rather than a single first-token sample.

**FP32-vs-inference precision caveat.** `--gflops` (and the HAL theoretical
peak) is FP32. llama.cpp's prefill usually runs FP16 or quantised kernels, so
the quoted peak is an *upper bound*, not a like-for-like measurement. Treat
`eta_comp` as an empirical fudge factor, not as "compute utilisation"; do not
read `eta_comp < 1` as wasted silicon without checking the kernel's actual
precision first.

## Caveats

These are real failure modes the bench will not catch:

* **CPU turbo / thermal throttling.** A kernel that runs for ~5 s
  boots in the turbo bin, then settles to a lower clock as thermals
  trip. The `tok/s` on a long run is the *steady-state* number, not
  the headline. Compare two machines with the same total wall-clock,
  not the same hand-picked first rep.
* **AVX-512 downclocking.** Some Intel CPUs halve the AVX-512 clock
  (sometimes to 0.8× base) when AVX-512 instructions run. The FLOPS
  kernel reports the *achieved* GFLOPS, which already reflects the
  derate — but a naive comparison against `theoretical_gflops` will
  read as 50 % of peak even on a healthy box. The probe reads
  `is_x86_feature_detected!` at runtime, so it can fall back to AVX2
  or scalar if the OS disabled AVX-512 entirely after boot.
* **RAPL permission.** Linux's RAPL energy counters live under
  `/sys/class/powercap` and need read access per-package; on a locked-
  down workstation or a container that drops `/sys`, `power_avg_watts`
  comes back as `null` rather than an estimate. Treat absent power as
  *"we could not measure it here"*, never as zero watts.
* **llama.cpp output drift.** Perf block wording has changed at least
  three times across b-versions (`prompt eval time`, `prompt
  processing time`, …). `parse_perf` is tolerant by design: it
  recognises the markers, *rejects* the `total time` line so it cannot
  be mistaken for the decode half, and reports `None` for any half or
  empty block — the safe failure mode is "no number" rather than "a
  wrong number". Vendors that reformat llama.cpp lines will keep
  working as long as they keep a `… ms / … tokens` shape.
* **CI runs only smoke sizes.** The bandwidth smoke tests use 8 MiB
  buffers and 1 thread; the FLOPS smoke runs a single core. CI cannot
  exercise the full sweep, the e2e path, or any real llama.cpp binary.
  Use the `--ignore`d integration tests against a real build to
  validate accuracy before tagging a release.

## Cross-platform notes

* **Linux:** RAPL power (when permitted), full memory speed from
  sysinfo, sysinfo child-pid monitoring for the e2e phase.
* **Windows:** no RAPL (so `power_avg_watts: null`), memory speed from
  CIM (`crates/attune-hal/src/memory.rs`), sysinfo process monitoring
  for the e2e phase via `Process::memory()`. The e2e thread-pin works
  the same way it does on Linux.
* **macOS / ARM64:** not yet targeted (AGENTS.md §1). The bandwidth /
  FLOPS kernels fall back to scalar + platform-default thread count;
  monitoring returns `power_avg_watts: null`. M3 or later will gate
  ARM NEON paths.
