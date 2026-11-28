# Recommend and model downloads — design

Status: approved 2026-11-26, implemented in v0.6.0 (M6).

## Intent

A new user should not need to know model names, find a llama.cpp build or
research which GGUF to download. They install attune, ask it what suits their
machine, pick one, and run it. Attune's edge over size-based "will it fit"
tools is that it *measures* the machine and predicts speed. This feature puts
that prediction in front of the model choice, and then removes every manual
download from the path to a running model.

Target flow:

```
attune setup                         # once: download the pinned llama.cpp build
attune tune                          # no --model: list models that suit this machine
attune tune --model qwen3-4b --save qwen4b.json
attune serve --profile qwen4b.json   # downloads the GGUF if missing, then serves
```

"Suits this machine" means hardware fit only: it fits in RAM, it is fast
enough to use, and it saves memory. There are no task categories (coding, chat
...) and no quality scores.

## Goals

1. `attune tune` without `--model` runs the existing estimate over every
   model x quant in the catalog and prints a model table. Each row shows
   download size, RAM needed, predicted decode tok/s, first-reply time, a fit
   verdict, and one recommended pick per size tier.
2. A model catalog that updates daily without an attune release, works
   offline, and is trustworthy (pinned files, exact sizes, checksums).
3. Profiles record where their model came from, so `serve` can download a
   missing model and a profile can be copied to another machine.
4. `attune setup` downloads a pinned llama.cpp build for this OS/CPU, so the
   user never has to look for one.
5. Downloads are safe: size shown before starting, disk space checked,
   resumable, SHA-256 verified, atomic (no half files).

## Non-goals (this phase)

- GPU builds and GPU-aware recommendations (M7, `docs/gpu-plan.md`).
- Live Hugging Face search (`tune --search <name>`): planned follow-up
  (approach C below).
- Gated models (Llama, Gemma, which need HF login and a licence click).
  The catalog generator skips them in v1.
- Quality or benchmark scores, task categories.
- Managing downloaded models (`attune models list/rm`): later.
- Prebuilt attune binaries on GitHub Releases: separate task.
- Signing the catalog file.

## Anti-goals (failure even if it works)

- Recommending a model that then fails to load or swaps to disk.
- Downloading gigabytes without saying how many first.
- Using a file whose checksum was not verified.
- Needing internet for anything already downloaded.
- A list so long or noisy that the user has to research again.

## Approaches considered

- **A. Catalog compiled into the binary.** Works offline, but only changes
  with an attune release. Rejected: open-weight models ship daily.
- **B. Published catalog file, regenerated daily by a GitHub Action
  (chosen).** A scheduled job queries Hugging Face for GGUF repos from an
  allow-list of trusted publishers, reads each file's GGUF header through an
  HTTP range request (attune already parses GGUF headers), and writes one
  `catalog.json`. Attune fetches it when online, caches it, and falls back to
  a snapshot compiled into the binary when offline. The "database" is that
  JSON file, versioned in git: no server, no hosting cost, and every change
  is reviewable. Up to a day behind.
- **C. Live HF search on every run.** Always current, but needs internet on
  every run, is slow, and is noisy (fine-tunes, broken uploads). Kept as a
  later opt-in (`--search`), not the default.

The user picked B, with C later.

## Decided

- **One command:** `tune` without `--model` is recommend mode; `tune --model`
  behaves as today. There is no separate `recommend` subcommand.
- **Catalog = published JSON + cache + built-in snapshot**, regenerated daily
  by a GitHub Action (approach B).
- **`serve` downloads a missing model** recorded in the profile.
- Everything stays CPU-only in this phase.

## Design

### Catalog file (`attune-catalog/1`)

One JSON document:

```json
{
  "schema": "attune-catalog/1",
  "generated_at": "2026-11-26T03:00:00Z",
  "llama_cpp": {
    "tag": "b11804",
    "assets": [
      { "os": "windows", "arch": "x86_64", "variant": "cpu",
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11804/<asset>.zip",
        "size": 12345678, "sha256": "..." }
    ]
  },
  "models": [
    {
      "id": "qwen3-4b-instruct-2507",
      "display": "Qwen3 4B Instruct 2507",
      "publisher": "Qwen",
      "license": "apache-2.0",
      "released": "2025-07-30",
      "params": 4022468096,
      "arch": { "n_layer": 36, "n_kv_heads": 8, "head_dim": 128 },
      "ctx_default": 262144,
      "files": [
        { "quant": "q4_k_m", "repo": "Qwen/Qwen3-4B-Instruct-2507-GGUF",
          "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "revision": "<commit sha>",
          "size": 2497280480, "sha256": "..." }
      ]
    }
  ]
}
```

- `params`, `arch` and `ctx_default` come from the GGUF header, not the
  model card, so estimates use the same facts as a local file.
- `size`, `sha256` and `revision` come from the HF API (`siblings[].lfs`
  plus the repo commit), so a download is pinned to an exact file.
- URLs are only allowed under `huggingface.co` and
  `github.com/ggml-org/llama.cpp`. Attune refuses any other host even if the
  catalog names one. This limits the damage of a tampered catalog until
  signing exists.
- Quants offered: those the estimator supports (`q4_k_m`, `q4_0`, `q8_0`,
  `fp16`). The generator keeps `q4_k_m` and `q8_0` per model when they
  exist.

### Catalog generator (GitHub Action)

- A Rust subcommand, hidden from normal help: `attune catalog build
  --publishers <file> --out catalog.json`. No Python, per AGENTS.md.
- Inputs: a checked-in allow-list, `catalog/publishers.toml`, naming the
  trusted orgs and quantizers (initial: Qwen, microsoft, mistralai,
  HuggingFaceTB, ibm-granite, bartowski, unsloth, ggml-org) and the
  `llama_cpp` tag to pin.
- For each publisher it lists GGUF repos (HF API `models?author=&search=gguf`)
  and keeps instruct/chat models only. It skips gated repos and models
  larger than 70B, reads the header of each kept file with a range request
  (first few MB), and fills the schema.
- Daily schedule plus manual dispatch. It commits `catalog.json` to a
  dedicated `catalog` branch. `main` is protected and bots should not open
  daily PRs, so the catalog lives on its own branch with linear history,
  where git diff shows what changed each day.
- A run fails without overwriting the previous catalog if the result is
  empty, shrinks by more than 30%, or fails schema validation.
- Bumping the llama.cpp pin is a human edit to `publishers.toml`, because a
  new llama.cpp can change performance and needs a check.

### Catalog loading in attune

- Location: the user data dir, `~/.attune/` on Linux and
  `%USERPROFILE%\.attune\` on Windows. This replaces today's
  working-directory-relative `.attune/` cache; the HF config cache moves
  there too.
- Order: a fresh cache (< 24 h) → fetch from the `catalog` branch raw URL
  (5 s timeout, the same as `fetch.rs`) → a stale cache → the snapshot
  compiled into the binary at build time. The output always says which one
  was used and its date ("catalog from 2026-11-26, offline").
- `--refresh-catalog` forces a fetch.
- The existing 10-entry `MODEL_CATALOG` becomes the seed of the built-in
  snapshot. Catalog names stay accepted by `--model`.

### `tune` without `--model` (recommend mode)

- Runs the same machine measurement as `tune` (quick bandwidth + int8
  probe, or `--bandwidth-gbps`/`--bench-json`), then for each catalog model
  x quant computes, with today's estimate code, predicted decode, TTFT
  (512-token prompt), and RAM peak (weights + KV at a recommend context of
  4096, clamped to the model's maximum).
- Fit uses available RAM, or `--max-ram <SIZE>` when given (a budget the
  user wants to keep). Verdicts: ✅ fits with ≥ 15% headroom, ⚠ fits
  tightly, ❌ needs X GB more.
- **Recommended pick per tier** (tiny < 2B, small 2–5B, medium 5–10B, large
  > 10B): among fitting rows, the newest model, then the most params, whose
  predicted decode is ≥ 5 tok/s and TTFT ≤ 15 s, at its smallest offered
  quant. One overall ★ is the largest tier that has a pick.
- The default table shows tier picks plus the next 2 per tier (about 10
  rows). `--all` shows everything, and `--json` returns the full ranked
  list.
- A footer notes that the predictions are uncalibrated (about ±25%) and
  calibrating takes one run.
- The bottleneck line from `estimate` appears once, in plain words ("speed
  here is limited by memory bandwidth: smaller files are faster").

Sketch:

```
$ attune tune
Machine: 8 cores (AVX-VNNI), RAM 32 GB/s, 9.1 GB free · catalog 2026-11-26

  Model                  Quant   Download  RAM     Speed     First reply  Fit
★ Qwen3 4B Instruct      Q4_K_M  2.5 GB    2.8 GB  11 tok/s  6 s          ✅
  Qwen3 8B               Q4_K_M  5.0 GB    5.4 GB   6 tok/s  12 s         ✅
  Qwen3 1.7B             Q4_K_M  1.1 GB    1.3 GB  25 tok/s  2 s          ✅ fastest
  Qwen2.5 14B Instruct   Q4_K_M  9.0 GB    9.5 GB   3 tok/s  22 s         ❌ needs 0.4 GB more

Next: attune tune --model qwen3-4b-instruct-2507 --save qwen4b.json
Predictions are uncalibrated (about ±25%).
```

### `tune --model <catalog id>` and the profile `source`

- When `--model` is a catalog id, `tune` uses the catalog facts (exact file
  size, header arch) and does not need the file.
- `--verify` downloads the file first, after asking (see Downloads).
- `--save` writes the profile with a new `model.source`:
  `{ "repo", "file", "revision", "sha256", "size" }`.
- `model.path` holds the expected local path under
  `~/.attune/models/<repo>/<file>`.
- Profile schema: `profile_version` 2 adds `model.source` (optional).
  `migrate.rs` upgrades v1 by adding an empty source. A v1 or hand-made
  profile without a source behaves as today.

### `serve` with a missing model

- If `model.path` does not exist and `model.source` is set: print the model,
  size and destination, ask `Download 2.5 GB? [y/N]` (`--yes` skips the
  prompt), download, verify, then serve.
- No source and no file: today's clear error.
- The path is resolved at run time, and the profile file is not rewritten,
  so the same profile works on another machine with a different home dir.
  A relative `~/.attune/models/...` form is stored for this.

### `attune setup`

- Picks the catalog `llama_cpp` asset for this OS/arch (CPU variant in this
  phase), downloads it to `~/.attune/llama.cpp/<tag>/`, verifies it, and
  unpacks it (zip on Windows; the Linux asset's format is whatever the pinned
  release ships).
- llama-server resolution order becomes: `--llama-server` →
  `ATTUNE_LLAMA_SERVER` → `PATH` → `~/.attune/llama.cpp/<pinned tag>/`.
  When none is found, `serve`/`tune --verify` say "run `attune setup`", or
  offer to run it inline with the same y/N prompt.
- `llama-cli` resolves the same way.

### Downloads (shared)

- One downloader used by models, llama.cpp and the catalog: `ureq` (already
  in the workspace) streaming to `<dest>.part`.
- HTTP `Range` resume when `.part` exists. SHA-256 is computed while
  streaming. Rename to `<dest>` only on a checksum match; on a mismatch
  delete the part and error.
- Before starting, check free disk space ≥ size + 10%.
- Progress on stderr (percentage, MB/s, ETA), quiet under `--json`.
- Allowed hosts only (see Catalog file). No auth headers (no gated models).

### Crates (user decision: new leaf crate)

- **New `attune-fetch`.** It depends on no other attune crate (like
  `attune-hal`) and holds:
  - the home dir (`~/.attune`);
  - the downloader (`ureq` with rustls, `.part` + Range resume, SHA-256,
    disk-space check, host allow-list);
  - archive unpacking;
  - the `~/.attune/llama.cpp/<tag>` install layout and lookup.
- `attune-estimate` uses it for the catalog fetch and cache, and also owns
  the catalog schema, loading and snapshot. The HF config cache moves to the
  home dir.
- `attune-backend` uses it for llama-server lookup and `setup`, and stays
  free of TLS on the serve path: HTTPS lives only in `attune-fetch`, which
  `serve` calls only when a download is needed.
- `attune-bench` keeps depending only on `attune-hal`. The CLI resolves
  `llama-cli` (flag → env → PATH → `attune-fetch` install lookup) and passes
  the path in, as it already does with `--llama-cli`.
- `setup` glue: the CLI reads the `llama_cpp` pin from the catalog
  (estimate) and hands the asset to `attune-fetch` to install.
- Profile `source`: `attune-profile`, which stays dependency-free.
- CLI only parses args and calls these.
- New dependencies, one line each in the PR: `sha2` (checksum), `zip`
  (unpack the Windows llama.cpp asset), `tar` + `flate2` only if the Linux
  asset is a tarball, and a home-dir lookup if std env vars are not enough.

## Calls made after the builder check

These were settled by the designer, not the user; the user can override any
of them.

- **Model identity:** one catalog `id` per base model (e.g.
  `qwen3-4b-instruct-2507`), with `files[]` merged across allow-listed repos.
  Per quant, the repo is chosen by the order in `publishers.toml`: the
  original publisher first, then `ggml-org`, `bartowski`, `unsloth`. Rows are
  per model, never per repo.
- **Choosing a file:** `tune --model <id>` takes `--quant`, with today's
  default `q4_k_m`; `model.source` pins that file. `--model` accepts an exact
  id or a unique prefix (`qwen3-4b` → `qwen3-4b-instruct-2507` if it is the
  only match). An ambiguous prefix errors and lists the matches.
- **Built-in snapshot:** a checked-in `catalog/snapshot.json` on `main`,
  embedded with `include_str!` (no network in `cargo build`). It is
  refreshed by a normal PR, copied from the `catalog` branch, as a step in
  the release checklist. The first snapshot is hand-written with real
  pinned files for the non-gated models we already test (SmolLM2, Qwen3
  0.6B/4B/8B, Qwen2.5 7B/14B, Phi-4-mini, Mistral 7B).
- **Old catalog names:** the existing `MODEL_CATALOG` stays as an
  estimate-only list, so `estimate --model llama-3.1-8b` keeps working.
  Gated models (Llama, Gemma) can be estimated but not downloaded, and
  recommend mode skips them.
- **Ranking without a release date:** the GGUF header has no date, and a
  quantizer repo's date is the upload date, so "newest" uses the
  `createdAt` of the original publisher's base repo (from the GGUF repo's
  `base_model` card field). When that is missing, the model ranks by params
  only. The tier pick is the most params meeting the speed bar; the date
  only breaks ties between models of similar size (within 15%).
- **Linux setup:** in scope. If the pinned Linux asset is a tarball, add
  `tar` + `flate2` (one line each). `serve` sets `LD_LIBRARY_PATH` to the
  install dir for the child process. The acceptance runs on Windows (this
  laptop), and CI covers Linux with a mocked-download smoke test.

## Acceptance (proposed)

1. **Zero manual downloads:** on a machine with no llama.cpp and no models,
   from an `attune.exe` binary to a first chat reply using only `attune setup`,
   `attune tune`, `attune tune --model <pick> --save`, `attune serve`.
   Report the wall time, excluding download time (bandwidth-dependent).
2. **Recommend accuracy:** for 3 catalog models of different tiers, the
   uncalibrated predicted decode is within 25% of the llama-bench measurement
   (the M3 target), and every ✅ model loads and serves without swapping.
3. **Catalog freshness:** the Action has produced a valid catalog on 7
   consecutive days, and a model added upstream appears within 24 h.
4. **Offline:** with the network disabled, `tune` (recommend mode) works from
   the cache or snapshot, and `serve` of an already-downloaded profile works.
5. **Download safety:** an interrupted download resumes, and a corrupted
   file is rejected (tested with a deliberately wrong checksum).

## Left to the builder

- Exact HF API calls and pagination, and the heuristics for "instruct/chat"
  detection.
- Table layout details and plain-language wording.
- Progress bar implementation (no new dependency needed).
- The home-dir lookup approach (std env vars vs a small crate).
- Slice order. Suggested: (1) home dir + downloader, (2) catalog schema +
  snapshot + loading, (3) recommend mode, (4) profile v2 + serve download,
  (5) setup, (6) generator + Action.

## Docs to update when this lands

- `ROADMAP.md`: new **M6 — Recommend & downloads** with these acceptance
  numbers. GPU moves to M7, and `docs/gpu-plan.md` is renamed in the text.
- `AGENTS.md`: current phase line; add `attune-fetch` (leaf crate:
  home dir, downloads, checksums) and `catalog/` to the repo structure in §1.
- `README.md`: Quick start becomes setup → tune → serve.
- `CHANGELOG.md`, `docs/benchmark-methodology.md` (recommend accuracy
  protocol).
