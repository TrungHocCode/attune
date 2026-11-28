# M7 GPU: use the GPU you already have

Status: proposed 2026-11-28. This builds on the proposal in `docs/gpu-plan.md`, which stays as background. Target release: v0.7.0.

## Why

attune ignores GPUs today:
- `setup` installs the CPU-only llama.cpp build.
- `detect` reports no GPU.
- `tune` hard-codes `n_gpu_layers = 0`.
- `recommend` only checks RAM.

On the reference laptop (RTX 4050 Laptop, 6 GB VRAM, about 190 GB/s, against 33 GB/s RAM), a 4B Q4 model should decode about 4–5× faster on the GPU. So attune gives its worst advice on exactly the machines people try it on first.

## Goals

- **The friendly path uses the GPU with no flags.**
  - `attune setup` and `attune run` install a GPU-capable llama.cpp build when a GPU is present.
  - `recommend` judges fit and speed with VRAM.
  - `run` serves on the GPU.
- **`detect`** reports the GPU: name, VRAM total and free, driver. The JSON only gains fields.
- **`estimate` and `tune`** predict and choose full or partial offload (`n_gpu_layers`) with a VRAM fit check, and verify the choice with real runs as today.
- **A GPU profile never runs out of VRAM** on the machine it was tuned on.

## Non-goals and anti-goals

- **Non-goals:**
  - Multi-GPU and tensor-parallel split.
  - Apple Metal.
  - Our own GPU kernels.
  - Linking any GPU SDK (CUDA, NVML, Vulkan) into attune.
  - Training.
- **Anti-goals:**
  - The CPU-only path gets slower, or needs GPU tooling to build or run.
  - attune requires a GPU driver to be installed.
  - A tuned GPU profile OOMs at load.
  - A silent multi-hundred-MB download: the CUDA build must be asked for, or shown as a choice.

## Approaches considered

1. **Wrap llama.cpp's own auto-offload.** Recent llama.cpp builds fit layers to VRAM by themselves. attune would only detect, install the right build, and predict. Cheapest, but untested: it may leave speed on the table or OOM with long contexts.
2. **Our own split model and tuning** (the `docs/gpu-plan.md` design). attune predicts `t ≈ bytes_gpu/BW_vram + bytes_cpu/BW_ram + overhead`, tunes `n_gpu_layers`, KV placement and flash-attn, and verifies the top-K with real runs. Full control, more work.
3. **Measure first, then decide (chosen).** Slice 1 is a measurement spike on the reference laptop that settles two questions with numbers:
   - **(a) Build choice:** Vulkan vs CUDA. Rule: Vulkan is the default unless CUDA decodes more than 15% faster on the reference models. Vulkan is 37 MB and runs on any GPU; CUDA is about 660 MB with its runtime and NVIDIA-only. CUDA stays available either way through `--gpu cuda`.
   - **(b) Auto vs tuned:** does llama.cpp's automatic offload match the best `n_gpu_layers` found by hand? If it's within 5% decode and never OOMs, M7 uses approach 1 for the split decision and keeps our model only for predictions. Otherwise it uses approach 2.

## Design

### Slice 1: detect + measurement spike

- **`attune-hal`:**
  - Shell out to `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits` when it's present.
  - Otherwise, on Windows only, use CIM `Win32_VideoController` for names. VRAM is unknown there, because `AdapterRAM` caps at 4 GB.
  - `GpuInfo` gains `devices: Vec<GpuDevice { vendor, name, vram_total, vram_free, driver }>`. `count` and `note` stay, so the change is additive.
  - Short timeout. A missing tool gives today's output.
- **Spike** (scripted by hand, results in `docs/benchmark-methodology.md`): CPU vs Vulkan vs CUDA 12.4 builds of b11804 with `llama-bench`.
  - **Models:**
    - Qwen3 4B Instruct Q4_K_M: fits, full offload.
    - Qwen3 8B Q4_K_M (5 GB): borderline.
    - Qwen2.5 14B Q4_K_M (9 GB): partial offload.
  - **Settings:** default offload vs a hand sweep of `-ngl`, at ctx 4096.
  - **Recorded:** decode, prefill, and peak VRAM from `nvidia-smi`.
  - **Output:** the two decisions above, written into this spec before slice 2.

### Slice 2: install the right build (`attune-fetch`, catalog, `setup`)

- **Catalog:**
  - `llama_cpp.assets` gains `variant: vulkan | cuda12` entries for Windows and Linux.
  - The CUDA entry lists its companion `cudart` archive as a second asset.
  - The generator pins sizes and SHA-256 as it does today. Same host policy: github.com/ggml-org/llama.cpp releases only.
- **Installs:** each variant lives in `~/.attune/llama.cpp/<tag>-<variant>/`.
- **`setup --gpu auto|cpu|vulkan|cuda` (default `auto`):**
  - `auto` picks the slice 1 default build when a GPU is detected, and `cpu` otherwise.
  - CUDA also needs a driver new enough for its CUDA version; if not, `setup` explains and installs Vulkan.
  - Consent shows the size.
- **`run`** uses the same `auto` choice the first time.
- **Server lookup:** resolving `llama-server` prefers a GPU build when the profile has `n_gpu_layers > 0`.

### Slice 3: measure and estimate (`attune-bench`, `attune-estimate`)

- **`bench --model --gpu`:** a full-offload e2e run gives the effective VRAM bandwidth = weight bytes × decode tok/s (`eta_gpu`, the GPU twin of `eta_mem`). Without a calibration, a theoretical table is used (bandwidth by GPU name, e.g. RTX 4050 Laptop 192 GB/s); if the name is unknown, the bandwidth is null and confidence drops to low.
- **Decode with `g` of `L` layers on the GPU:** `t ≈ weights·g/L / BW_vram + weights·(L−g)/L / BW_ram + KV term on its device`.
- **Prefill:** compute-bound. Use a per-device-class table plus calibration rather than a FLOPS formula.
- **Fit:** `weights_gpu + KV_gpu + compute_buffer ≤ vram_free × 0.9`. The compute buffer comes from llama.cpp's log after the first run; before that, a default.
- Configs that don't fit are pruned, as RAM misfits are today.

### Slice 4: tune, profile, recommend

- **`tune`:**
  - New axes: `n_gpu_layers` ∈ {0, max_fit, L} plus 2 points in between when partial offload is possible, and `--flash-attn` on full offload.
  - KV placement only if slice 1 shows it matters.
  - The thread axis collapses under full offload.
  - With no GPU, these axes are skipped, so CPU results are unchanged.
- **Profile v3:**
  - The hardware fingerprint gains the GPU name and VRAM.
  - New optional `engine.variant` (`cpu`/`vulkan`/`cuda12`).
  - `migrate.rs` step v2 → v3, with defaults that keep old profiles valid.
- **`recommend`:**
  - Fit = the model fits VRAM, or VRAM + RAM with a partial split.
  - The speed column is predicted with the GPU, and the table header shows the GPU.

### Slice 5: serve, run, acceptance, v0.7.0

- `serve` and `run` launch the GPU build for GPU profiles. If only a CPU build is installed, `serve` says how to fix it (`attune setup --gpu auto`) and offers to fall back to CPU.
- `/v1/metrics` adds VRAM used and free when `nvidia-smi` is present.
- Docs, ROADMAP M7 ✅ with numbers, tag v0.7.0.

## Acceptance (reference: RTX 4050 Laptop 6 GB + Core 5 210H)

- `detect` matches `nvidia-smi` for name and VRAM.
- `attune run qwen3-4b-instruct-2507 --yes`, starting from an empty home, serves on the GPU with no flags. Decode is at least 3× the CPU profile on the same machine.
- **Full offload:** calibrated decode estimate within 12% of measured; uncalibrated within 25%.
- **Partial offload** (14B Q4_K_M): the tuned profile matches or beats llama.cpp's default offload. Numbers are posted either way.
- No VRAM OOM when serving any profile `tune` exported on that machine.
- CPU-only machines: all existing tests pass and results are unchanged. CI has no GPU, so the GPU code paths are unit-tested with recorded `nvidia-smi` outputs.

## Left to the builder

- Exact table entries for GPU bandwidth beyond the reference card.
- Parsing details of `nvidia-smi` and CIM output.
- How the spike script is organised. It's a doc artefact, not shipped code.
