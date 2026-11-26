# GPU Plan (M7 — proposed)

> Status: **proposal, not started.** M5 (serve) closes first, then the open
> estimate bugs (#69–#75), then this. Nothing here is promised until the
> M7 slices are planned as issues.

## Why

Attune started CPU-first: tune local inference for machines without a data
center GPU. In practice many of those machines *do* have a GPU — usually a
small consumer one (laptop RTX 4050/4060 with 6–8 GB VRAM, an iGPU, an older
desktop card). Today attune ignores it:

- `attune detect` always reports `gpu.count = 0` (`crates/attune-hal/src/lib.rs`).
- The profile already carries `n_gpu_layers` and emits `-ngl`, but `tune`
  hard-codes it to `0` (`crates/attune-tune/src/config.rs`).
- `estimate` only models RAM bandwidth and CPU FLOPS.

So on a GPU machine attune recommends a config that is often several times
slower than plain `llama-server -ngl 99`. That makes the tool look wrong on
exactly the hardware people try it on first.

## Positioning

CPU-first stays. The GPU story is **"use the GPU you already have, and split
the model across GPU + CPU when it does not fit"**. The interesting tuning
problem is the small-VRAM case: which layers to offload, where the KV cache
lives, and which context still fits. Large multi-GPU serving is not our
market (vLLM/TGI own it).

## Goals

1. Detect GPUs: vendor, name, VRAM total/free, driver; theoretical memory
   bandwidth where the tooling exposes it.
2. Measure the GPU's real decode capability through llama.cpp itself (no
   CUDA/Vulkan kernels of our own).
3. Estimate decode/prefill for full offload **and** partial offload
   (`n_gpu_layers` between 0 and the layer count), with a VRAM fit check.
4. Tune `n_gpu_layers` (plus KV placement and flash-attn) alongside the
   existing axes, verify top-K with real runs, export to the profile.
5. Serve a GPU profile with a GPU-enabled `llama-server` build, and report
   VRAM in `/v1/metrics`.

## Non-goals (M7)

- Multi-GPU / tensor-parallel split.
- Apple Metal and Apple Silicon unified memory (stays in "Later").
- Our own GPU kernels, NVML/CUDA bindings, or any GPU SDK as a build
  dependency.
- Training or fine-tuning.

## Anti-goals (failure even if it "works")

- The CPU-only path gets slower, noisier, or needs a GPU build to compile.
- `attune` requires a GPU driver/SDK to be installed to build or run.
- A GPU profile that OOMs VRAM at load time on the machine it was tuned on.

## Design

### Detect (`attune-hal`)

- Shell out to the vendor CLI when present, never link to it:
  - NVIDIA: `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,clocks.max.memory,pci.bus_id --format=csv,noheader,nounits`
    (bus width is not exposed by `nvidia-smi`; keep a small table by GPU
    name, else leave theoretical bandwidth `null`).
  - Everything else (AMD/Intel/iGPU): enumerate via the llama.cpp Vulkan
    build's device list (`llama-cli --list-devices`) when available; Windows
    CIM `Win32_VideoController` as a name-only fallback (its `AdapterRAM` is
    capped at 4 GB, so never trust it for VRAM).
- `GpuInfo` grows a `devices: Vec<GpuDevice>`; `count` and `note` stay so the
  JSON contract only adds fields.
- No GPU tool found → today's behavior (`count: 0`, a note saying why).

### Measure (`attune-bench`)

- Effective VRAM bandwidth = `bytes_weights × decode tok/s` from a full
  offload e2e run (`-ngl 99`) of a model that fits. Same idea as the CPU
  calibration, so `eta_mem` gets a GPU twin (`eta_gpu`).
- PCIe/host-transfer cost is not benched directly; partial-offload runs
  calibrate it (see Estimate).
- Reuse the existing llama.cpp runner; the parser already understands
  `offloaded N/M layers to GPU` lines.

### Estimate (`attune-estimate`)

Decode per token with `g` of `L` layers on GPU:

```
t_token ≈ (bytes_gpu / BW_vram) + (bytes_cpu / BW_ram) + overhead_split
bytes_gpu = weights × g/L (+ KV on GPU),  bytes_cpu = the rest
```

- Fit check: `weights_gpu + KV_gpu + compute_buffer ≤ VRAM_free × safety`
  (safety ~0.9; compute buffer from llama.cpp's own log on the first run,
  default estimate otherwise). Configs that do not fit are pruned, the same
  way RAM misfits are today.
- Prefill on GPU is compute-bound on FP16/tensor throughput; use a
  per-device table + calibration rather than inventing a FLOPS formula.
- Depends on #71/#72 being fixed first (true weight bytes, filled-KV
  charging) — the GPU model reuses that math.

### Tune (`attune-tune`)

- New axis `n_gpu_layers`: `{0, max_fit, L}` plus 2–3 points between when
  partial offload is possible; skip the axis entirely with no GPU.
- Fully offloaded configs collapse the thread axis (threads barely matter).
- New launch fields: KV on GPU or host (`--no-kv-offload`), `--flash-attn`
  on GPU. Verify top-K with real runs, as today.

### Profile (`attune-profile`)

- `n_gpu_layers` already exists. Add the GPU to the hardware fingerprint
  (name + VRAM) and a fallback `if_vram_insufficient.n_gpu_layers`.
- New fields are additive with defaults; bump `profile_version` and add the
  step in `migrate.rs` (never break old profiles).

### Serve (`attune-serve`, `attune-backend`)

- Resolve a GPU-capable `llama-server` (CUDA or Vulkan build) when the
  profile has `n_gpu_layers > 0`; refuse with a clear message if only a CPU
  build is found.
- `/v1/metrics` adds VRAM used/free when `nvidia-smi` is available.

## Slices (planned as issues when M7 starts)

1. `detect`: GPU devices via `nvidia-smi` / Vulkan list, JSON additive.
2. `bench`: full-offload e2e → `eta_gpu`; document a GPU llama.cpp build in
   `docs/benchmark-methodology.md`.
3. `estimate`: partial-offload decode model + VRAM fit check.
4. `tune` + `profile`: `n_gpu_layers` axis, KV placement, fingerprint +
   migrate.
5. `serve`: GPU build resolution, VRAM metrics; M7 acceptance.

## Acceptance (proposed)

Test machine: RTX 4050 Laptop 6 GB + Core 5 210H, llama.cpp CUDA build.

- `detect` matches `nvidia-smi` for name and VRAM.
- Full offload (a 1–4B Q4_K_M that fits): calibrated decode estimate within
  12% of measured; uncalibrated within 25% (same bar as M3).
- Partial offload (a model larger than 6 GB, e.g. 8B Q8_0 or 14B Q4_K_M):
  tuned profile beats llama.cpp's default offload by > 20% decode, or the
  report shows why it cannot (posted numbers either way).
- No VRAM OOM when serving any profile `tune` exported on that machine.
- CPU-only machines: all existing tests and the M4 result unchanged.

## Open questions

- Which GPU users matter most first: NVIDIA laptops (CUDA), or "any GPU"
  through Vulkan? This plan picks NVIDIA/CUDA as the measured target and
  Vulkan as best-effort.
- Does llama.cpp's own auto-offload (`--fit` / default `-ngl` in recent
  builds) already pick a good split? Measure it first; if it is good, M7
  shrinks to "estimate + verify + profile" around it.
- Which reference models to publish numbers for (fits / does not fit).
