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

## Int8 dot-product kernel (`int8.rs`)

llama.cpp's quantized prefill runs on int8 dot products, so FP32 FMA is the
wrong compute ceiling for q8_0 / q4_* prefill (issue #69: llama.cpp reached
665 GFLOPS effective on an 8B model, above the 563 GFLOPS FP32 "peak").
`attune bench` therefore also measures `u8 x i8 -> i32` multiply-accumulate
throughput and reports it as `int8` (`int8_gops`, `best_gops`, `path`,
`threads`, `reps`). One multiply-add counts as 2 ops, like FLOPs, and GOPS is
decimal.

* **Paths**: AVX-VNNI `vpdpbusd` (`_mm256_dpbusd_avx_epi32`, stable on
  rustc 1.99, runtime-gated on `avxvnni`), else AVX2 `vpmaddubsw` +
  `vpmaddwd` + `vpaddd`, else scalar.
* **Tile**: a 2 x 4 register tile (6 loads per 8 dot-product instructions)
  over L1-resident buffers with 8 accumulators, like a real int8 GEMM
  micro-kernel. One fresh load pair per instruction is load-port bound and
  under-reports the multiply units.
* **Budget**: about 70 ms per timed pass on AVX-VNNI (fewer iterations on
  slower paths), 1 warm-up + 3 timed passes, one worker per physical core.
* **Where it is used**: `estimate` / `tune` without `--gflops` run this quick
  probe (after the bandwidth probe, never alongside it) or read the `int8`
  block of `--bench-json`, and use it as the prefill ceiling for quantized
  formats. fp16 keeps the FP32 ceiling; an explicit `--gflops` keeps the old
  FP32-for-everything behaviour. The measured ceiling already includes the
  quantized-matmul cost, so the heuristic dequant penalty is not applied on
  top of it (llama.cpp reached 90-95% of the measured int8 ceiling on
  Q4_K_M); the penalty still applies to the FP32 ceiling.
* Old bench JSON without the `int8` field still parses (`serde(default)`).

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

## Read kernel (issue #70)

The read kernel is a wrapping sum of every `u64` word: AVX2 (runtime-detected)
with four independent 256-bit accumulators over 128 B per iteration, scalar
fallback elsewhere. Findings on the reference laptop (Core 5 210H, 1 DIMM
DDR5-5200, 41.6 GB/s theoretical): a threads {4,6,8,12} x buffer {256 MiB,
1 GiB} x reps {3,10} sweep shows the scalar and AVX2 kernels share the same
~35 GB/s ceiling (best pass), thread count and buffer size barely matter, and
llama.cpp decode implies 30-33 GB/s (qwen3 8B/4B Q4_K_M, file size x tg128).
The run-to-run spread comes from short samples (a 256 MiB pass is ~8 ms), so
`--quick` takes best-of 20 passes (~0.2 s) instead of 3, and uses AVX2 loads
so the kernel is never the limiter. Theoretical bandwidth uses the DIMM's
configured speed (`ConfiguredClockSpeed`) when non-zero, else the rated
`Speed`.

## Pass count and stats

`FULL_REPS` is 5 for full mode, `DEFAULT_REPS` is 20 for `--quick`.
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

## Acceptance protocol (M3/M4)

A milestone acceptance number is only published when it comes from this
protocol. The original M3 "0% error" (fit and scored on the same report) and
M4 "+25.7%" (135M model, 5 consecutive runs per side) did not, and are
superseded.

* **Reference models are 4B or larger** (Qwen3 4B Q4_K_M, Qwen3 8B
  Q4_K_M). SmolLM2-135M is a CI smoke size only: its run-to-run spread
  (168.8 vs 227.2 tok/s for the same command in one day) is larger than the
  effects being measured.
* **M3 uncalibrated:** `attune estimate` vs the measured value
  (`llama-bench` tg128 / pp512, `-r 3`, default threads; also the e2e runs).
  Pass `--ctx 2048` to `estimate` so KV sizing matches the bench (a GGUF's
  header context, e.g. 262144 for Qwen3-4B-2507, otherwise sizes the KV cache
  at 24 GB and reports OOM).
* **M3 calibrated:** fit eta on ONE `attune bench --model` run, then score the
  calibrated decode/TTFT against at least 4 other (held-out) runs. Report the
  per-run error, the median and the min-max. Never score against the fit run.
* **M4:** ALTERNATING default/tuned runs (D,T,D,T,...) with the same prompt
  and generation length: at least 5 runs per side, or at least 2 rounds of
  `llama-bench -r 3` per side. Report decode and prefill for each side. Pass
  = tuned is no more than 3% slower at decode and 5% slower at prefill than
  defaults (revised target, #103).
* **Record** power source, power plan, background load, llama.cpp build and
  the attune commit next to every table.

### Results (2026-11-23)

Conditions: Intel Core 5 210H (8C/12T), 15.6 GB RAM (7.6-9.2 GB free), AC
power, Windows "Balanced" plan, browser (Edge) and the Claude desktop app
open but idle (highest CPU-time processes), nothing else started; every
bench/run strictly sequential. llama.cpp b11804 (`0.7.0-dev`, CPU
`alderlake` backend), attune `2bcf0ff`. Default llama.cpp threads = 8.

**M3, uncalibrated** (target < 25%; error = (estimate - measured) / measured,
ctx 2048):

| Model | Metric | Estimate | llama-bench (default `-t 8`) | Error | vs attune e2e runs (llama-cli) |
|---|---|---|---|---|---|
| Qwen3 4B Q4_K_M | decode tok/s | 12.34 | 11.67 ± 0.44 (11.96 ± 0.19 on a second `-t 8` run) | +5.8% (+3.2%) | +14.0% to +16.0% (e2e 10.64-10.83) |
| Qwen3 4B Q4_K_M | TTFT, 512 tok | 6173 ms | 7038 ms (pp512 72.75 ± 0.51; 6484 ms on the second run, pp512 78.97) | -12.3% (-4.8%) | n/a |
| Qwen3 8B Q4_K_M | decode tok/s | 5.87 | 6.12 ± 0.12 | -4.1% | +0.1% to +4.7% (e2e 5.61-5.86) |
| Qwen3 8B Q4_K_M | TTFT, 512 tok | 9702 ms | 12380 ms (pp512 41.36 ± 0.18) | -21.6% | n/a |

Met, but note the 8B TTFT is within 4 points of the limit, and on the 4B the
`llama-cli` decode path that `attune bench`/`tune` drive is ~8% slower than
`llama-bench` (10.8 vs 11.7 tok/s), so the error depends on which baseline is
used. `-t 12` (all logical cores) is slower for decode (11.03 ± 0.11).

**M3, calibrated, held-out** (target < 12%): eta fitted on `attune bench`
e2e run 1 (`--ctx 2048 --bench-quant q4_k_m --calibrate`, prompt 464 tokens),
scored on e2e runs 2-5.

| Model | Metric | Fit run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median error | Min-max |
|---|---|---|---|---|---|---|---|---|
| 4B | decode tok/s measured | 10.83 | 10.81 | 10.70 | 10.64 | 10.78 | | |
| 4B | decode error | (fit) | +0.2% | +1.1% | +1.8% | +0.4% | +0.8% | +0.2% to +1.8% |
| 4B | TTFT ms measured | 6112 | 6486 | 6329 | 6211 | 6245 | | |
| 4B | TTFT error | (fit) | -5.8% | -3.4% | -1.6% | -2.1% | -2.8% | -5.8% to -1.6% |
| 8B | decode tok/s measured | 5.61 | 5.62 | 5.86 | 5.85 | 5.66 | | |
| 8B | decode error | (fit) | -0.2% | -4.3% | -4.1% | -0.8% | -2.5% | -4.3% to -0.2% |
| 8B | TTFT ms measured | 11787 | 12018 | 11812 | 11795 | 11589 | | |
| 8B | TTFT error | (fit) | -1.9% | -0.2% | -0.1% | +1.7% | -0.2% | -1.9% to +1.7% |

Met. This only says the calibration transfers between runs of the same
workload on the same machine; it is not a prediction for a different model or
quant (the uncalibrated table above is that).

**M4, Qwen3 4B Q4_K_M** (target > 20% over llama.cpp defaults):

* Tuned profile from `attune tune --objective throughput --verify --top-k 2
  --max-configs 8`: `-t 8 -tb 8 -c 2048 -b 128 -ub 128 --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0` (tune measured 10.6 tok/s for it
  and 10.2 for the runner-up, `-t 4`).
* Defaults: `llama-cli -m <4B> -p <464-token prompt> -n 128 -c 2048 -st -v`,
  no other flags. Tuned: the same plus the flags above. Same prompt, 5
  alternating D,T,D,T,... runs, decode rate from llama.cpp's own timing line.

| | Runs (tok/s) | Median | Min-max |
|---|---|---|---|
| default | 10.95, 10.73, 10.74, 10.70, 10.77 | 10.74 | 10.70-10.95 |
| tuned | 10.89, 11.06, 11.07, 11.05, 10.85 | 11.05 | 10.85-11.07 |

Median-vs-median **+2.9%**, best-vs-best +1.1%: **target not met**. Decode on
this machine is memory-bound (~33 GB/s against 2.4 GB of weights per token),
so no thread/batch/KV setting can beat the roofline the defaults already sit
on. The tuned profile was also **slower at prefill**: prompt eval median 59.1
vs 70.8 tok/s (TTFT 7856 vs 6551 ms, +19.9%), probably because `-ub 128` shrinks
the prefill micro-batch (not isolated).

Observation (fixed in #102): `tune`'s estimate-only TTFT column showed
22982 ms for every candidate, versus 6173 ms from `attune estimate`, because
it prefilled the whole context window instead of the prompt.

### M4 re-run with the revised target (2026-11-25)

The "> 20% over defaults" target was replaced (#103) by: **(1) no
regression**, meaning tuned is at most 3% slower at decode and 5% slower at
prefill than llama.cpp defaults, and **(2)** calibrated prediction within
15% (already shown by the held-out table above).

Isolating the old regression (Qwen3 4B, `llama-bench -p 512 -n 128 -r 2`):

| Config | pp512 tok/s |
|---|---|
| llama.cpp defaults (`-t 8`, ub 512, f16 KV) | 71.4 |
| `-t 8`, ub 512, q8_0 KV, flash-attn | 64.6 |
| `-t 8`, ub 128, q8_0 KV, flash-attn | 60.2 |
| `-t 4`, ub 128 or 512, q8_0 KV | 42.7-47.7 |

Decode (tg128) ranged 7.3-11.4 tok/s for all of them, which is run noise. The
estimate cannot tell threads, batch or KV type apart, so they tied and grid
order picked the half-thread, batch-128 point. After #103 (`dfb156d`), `tune
--save` picks `-t 8 -b 512 -ub 512`, f16 KV, flash-attn on, for both
models.

Conditions: as 2026-11-23, but 3.4-4.7 GB RAM free and Edge busy. Defaults
vs tuned (`-t 8 -b 512 -ub 512 -fa 1 -ctk f16 -ctv f16`), `llama-bench -p 512
-n 128 -r 3`, alternating D,T,D,T:

| Model | Metric | Defaults (round 1, 2) | Tuned (round 1, 2) | Verdict |
|---|---|---|---|---|
| Qwen3 4B | pp512 tok/s | 63.6, 70.0 | 72.6, 72.8 | no regression (+9%) |
| Qwen3 4B | tg128 tok/s | 11.36, 10.69 | 9.55, 11.30 | within noise |
| Qwen3 8B | pp512 tok/s | 41.0, 43.3 | 43.2, 41.9 | no regression |
| Qwen3 8B | tg128 tok/s | 6.05, 6.69 | 6.85, 6.84 | no regression (+5%) |

Uncalibrated `tune` prediction for the same profiles: 4B decode 13.8 tok/s
vs ~11.0 measured (**+25%**; the probe read 35.2 GB/s, and five `bench
--quick` runs gave 31.3-34.6), TTFT 5864 vs ~7043 ms (-17%); 8B decode 6.6 vs
~6.7 (-1.5%), TTFT 11893 vs ~12047 ms (-1.3%). Under load llama.cpp gets
about 80% of the probe's bandwidth, so the uncalibrated decode error moves
from +6% (2026-11-23) to +25% (today). After one calibration run it stays
under 6%.

Raw per-run JSON/logs are kept outside the repo.

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
