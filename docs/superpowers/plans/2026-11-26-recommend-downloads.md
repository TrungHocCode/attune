# M6 Recommend & Downloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `attune tune` with no `--model` recommends catalog models that fit this machine, and setup → tune → serve then runs one with zero manual downloads.

**Architecture:** A new leaf crate `attune-fetch` owns the home dir and verified, resumable downloads. `attune-estimate::model` owns the catalog (schema, load order, built-in snapshot). `attune-tune` gains a pure `recommend` ranking over catalog models using today's estimate maths. The profile gains `model.source` (v2), which `serve` uses to download a missing model, and `attune setup` installs the pinned llama.cpp. A GitHub Action regenerates the catalog daily via a hidden `attune catalog build`.

**Tech Stack:** Rust 2021 stable, clap, `ureq` 3 (rustls), serde/serde_json, `sha2`, `zip`, `sysinfo` (disk space), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-26-recommend-design.md`

## Global Constraints

- Rust stable, edition 2021, no nightly. Libraries return `Result` with `thiserror`. CLI uses `anyhow`. No `.unwrap()` outside tests. No Python.
- Linux x86-64 and Windows x86-64 must compile and pass CI. Platform code is gated with `#[cfg(...)]`.
- `attune-fetch` depends on no other attune crate. `attune-profile` stays dependency-free (only serde/serde_json/thiserror). `attune-bench` depends only on `attune-hal`. The CLI only parses args and calls library functions.
- Each new dependency gets a one-line justification in its commit/PR: `sha2`, `zip`, `sysinfo` in attune-fetch, and `tar` + `flate2` only if the pinned Linux asset is a tarball.
- Download hosts allowed: `huggingface.co` (and its LFS redirect target `cdn-lfs*.huggingface.co` / `*.hf.co`), `github.com/ggml-org/llama.cpp` (and its `objects.githubusercontent.com` / `release-assets.githubusercontent.com` redirect), and the catalog raw URL on `raw.githubusercontent.com/TrungHocCode/attune/catalog/`. Everything else is refused.
- Home dir: `$HOME/.attune` on Linux, `%USERPROFILE%\.attune` on Windows. `ATTUNE_HOME` overrides it (needed by tests).
- Catalog schema string: `attune-catalog/1`. Catalog cache freshness 24 h. Fetch timeout 5 s.
- Recommend context: 4096 tokens, clamped to the model's maximum. Prompt: 512 tokens. Verdicts: ✅ fits with ≥ 15% headroom, ⚠ tight, ❌ needs X GB more. Tiers: tiny < 2B, small 2–5B, medium 5–10B, large > 10B. Pick bar: decode ≥ 5 tok/s and TTFT ≤ 15 s. The date only breaks ties between models within 15% in params.
- Profile `profile_version` becomes 2. v1 files migrate with no `source`.
- Every PR: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo run -p attune-cli -- detect`. Conventional commit titles, English only, PR body written to a file and passed with `--body-file`, `Refs #<issue>`.
- Tests never touch the real network or the real home dir. They use `ATTUNE_HOME` pointing at a temp dir and a local `TcpListener` stub server, permitted through a test-only host policy.

## Review Focus

1. **The server returns 200 instead of 206 on a resume request**: the downloader must restart from zero, not append the full body to the `.part` file. Test in Task 1.
2. **Redirects:** HF and GitHub answer downloads with a 302 to a CDN host. The allow-list must check every hop, accept the known CDN hosts and refuse a redirect to an unlisted host. Test in Task 1.
3. **Cached catalog is corrupt or from a newer schema** (`attune-catalog/2`): loading must fall back to the next source with a warning, never crash. Test in Task 3.
4. **`serve` run non-interactively** (no TTY, e.g. a service) with a missing model and no `--yes`: it must fail with a clear message, not hang on the prompt. Test in Task 6.
5. **Two models tie and one is gated or has no file for the requested quant**: recommend must skip it rather than recommend something that can't be downloaded. Test in Task 4.

---

### Task 1: `attune-fetch` crate — home dir, host policy, verified resumable download

**Files:**
- Create: `crates/attune-fetch/Cargo.toml`, `crates/attune-fetch/src/lib.rs`, `src/home.rs`, `src/policy.rs`, `src/download.rs`, `src/error.rs`
- Modify: `Cargo.toml` (workspace members), `AGENTS.md` §1 (already lists the crate; check it matches)
- Test: unit tests in each file, plus `crates/attune-fetch/tests/download.rs` (stub HTTP server)

**Interfaces:**
- Produces:
  - `pub fn home_dir() -> Result<PathBuf, FetchError>`: `ATTUNE_HOME`, else `$HOME`/`%USERPROFILE%` + `.attune`. Creates the dir.
  - `pub struct HostPolicy`, with `HostPolicy::production()` and `#[doc(hidden)] HostPolicy::allow_local_for_tests()` (adds `127.0.0.1`).
  - `pub fn HostPolicy::check(&self, url: &str) -> Result<(), FetchError>`.
  - `pub struct Download { pub url: String, pub dest: PathBuf, pub size: u64, pub sha256: String }`
  - `pub fn download(d: &Download, policy: &HostPolicy, progress: &mut dyn FnMut(u64, u64)) -> Result<(), FetchError>`
  - `pub fn fetch_text(url: &str, policy: &HostPolicy, timeout: Duration) -> Result<String, FetchError>`
  - `pub fn free_disk_bytes(path: &Path) -> Option<u64>`
  - `FetchError` variants: `Home`, `HostNotAllowed{url}`, `Http{url,detail}`, `ChecksumMismatch{expected,actual}`, `SizeMismatch{expected,actual}`, `DiskSpace{need,free}`, `Io{path,source}`.

- [ ] **Step 1: Write the failing tests**
  - `home_dir_honours_attune_home`: with `ATTUNE_HOME` set to a temp dir, `home_dir()` returns it and the dir exists.
  - `policy_allows_hf_and_llama_releases_only`: `check` is Ok for `https://huggingface.co/Qwen/x/resolve/abc/f.gguf` and `https://github.com/ggml-org/llama.cpp/releases/download/b1/a.zip`. It is Err(`HostNotAllowed`) for `https://evil.example/f.gguf`, for `http://huggingface.co/...` (no TLS), and for `https://github.com/other/repo/releases/...`.
  - In `tests/download.rs`, against a stub server serving a fixed 1 MiB body with known sha256:
    - `fresh_download_verifies_and_renames`: dest exists and no `.part` file remains.
    - `wrong_checksum_deletes_part_and_errors`: Err(`ChecksumMismatch`), with neither dest nor `.part` left on disk.
    - `resume_sends_range_and_completes`: pre-create a `.part` with the first 300 KiB. The stub asserts it got `Range: bytes=307200-` and answers 206. The final file is correct.
    - `server_ignoring_range_restarts_from_zero`: same, but the stub answers 200 with the full body. The final file is correct, not 1.3 MiB. (Review Focus 1)
    - `redirect_to_unlisted_host_is_refused`: the stub answers 302 to `http://10.0.0.1/x`, which gives Err(`HostNotAllowed`). (Review Focus 2)
    - `not_enough_disk_space_errors_before_downloading`: using an injected free-space function returning 10 bytes gives Err(`DiskSpace`), and the stub saw no request. Expose `download_with(d, policy, free: &dyn Fn(&Path)->Option<u64>, progress)` for this; `download` calls it with `free_disk_bytes`.
- [ ] **Step 2: Run** `cargo test -p attune-fetch`. Expected: fails to compile (crate missing).
- [ ] **Step 3: Implement.**
  - Use `ureq` 3 with rustls and redirects disabled. Follow up to 5 redirects manually, calling `policy.check` on each hop.
  - Stream to `<dest>.part` while updating `sha2::Sha256`. On resume, hash the existing part first.
  - Require free space ≥ `size + size/10`. Rename on a match.
  - `free_disk_bytes` uses `sysinfo::Disks`, picking the disk with the longest mount-point prefix of `path`.
- [ ] **Step 4: Run** `cargo test -p attune-fetch` and the 4 gates. Expected: all pass.
- [ ] **Step 5: Commit** `feat(fetch): add attune-fetch with verified resumable downloads`, then open a PR.

### Task 2: Catalog schema, validation and id lookup

**Files:**
- Create: `crates/attune-estimate/src/model/registry.rs` (schema + validate + lookup)
- Modify: `crates/attune-estimate/src/model/mod.rs` (export)
- Test: unit tests in `registry.rs`

**Interfaces:**
- Produces:
  - `pub struct Catalog { pub schema: String, pub generated_at: String, pub llama_cpp: LlamaPin, pub models: Vec<CatalogModel> }`
  - `pub struct LlamaPin { pub tag: String, pub assets: Vec<LlamaAsset> }`
  - `pub struct LlamaAsset { os, arch, variant, url, size: u64, sha256 }`
  - `pub struct CatalogModel { id, display, publisher, license, released: Option<String>, gated: bool, params: u64, arch: Arch, ctx_default: u32, files: Vec<CatalogFile> }`
  - `pub struct CatalogFile { quant: Quant, repo, file, revision, size: u64, sha256 }`
  - `CatalogFile::url(&self) -> String`, which returns `https://huggingface.co/{repo}/resolve/{revision}/{file}`.
  - `pub fn parse_catalog(text: &str) -> Result<Catalog, EstimateError>`: rejects a schema ≠ `attune-catalog/1`, models with no files, duplicate ids and a non-hex sha256 (64 chars).
  - `pub fn Catalog::find(&self, query: &str) -> Result<&CatalogModel, EstimateError>`: exact id, else a unique case-insensitive prefix. An ambiguous prefix errors and lists the ids.
  - `pub fn CatalogModel::file_for(&self, quant: Quant) -> Option<&CatalogFile>`

- [ ] **Step 1: Write failing tests:**
  - `parses_the_spec_example`: the spec's JSON example, with real-looking sha256 values, parses. `models[0].files[0].url()` equals `https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF/resolve/<rev>/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.
  - `rejects_unknown_schema`: `attune-catalog/2` gives Err.
  - `rejects_duplicate_ids_and_bad_sha`
  - `find_exact_then_unique_prefix`: `qwen3-4b` finds `qwen3-4b-instruct-2507` when it is the only match. With `qwen3-4b-thinking` also present, Err lists both ids.
- [ ] **Step 2: Run** `cargo test -p attune-estimate registry`. Expected: FAIL.
- [ ] **Step 3: Implement** with serde derives. `Quant` deserializes from `q4_k_m` etc., reusing the existing `Quant` parsing. `Arch` here carries `n_layer`, `n_kv_heads` and `head_dim`, the same fields as the existing `Arch`; its `name` is the model id.
- [ ] **Step 4: Run** the tests and the 4 gates. Expected: pass.
- [ ] **Step 5: Commit** `feat(estimate): add the attune-catalog/1 schema and id lookup`.

### Task 3: Catalog load order, built-in snapshot, cache in the home dir

**Files:**
- Create: `catalog/snapshot.json`, `catalog/README.md` (two lines: what the files are, how the snapshot is refreshed)
- Modify: `crates/attune-estimate/src/model/registry.rs` (loader), `crates/attune-estimate/src/model/fetch.rs` (`cache_path()` moves to `attune_fetch::home_dir()`), `crates/attune-estimate/Cargo.toml` (depends on `attune-fetch`)
- Test: unit tests in `registry.rs`

**Interfaces:**
- Consumes: Task 1 `home_dir`, `fetch_text`, `HostPolicy`. Task 2 `parse_catalog`.
- Produces:
  - `pub enum CatalogOrigin { FreshCache, Fetched, StaleCache, Snapshot }`
  - `pub struct LoadedCatalog { pub catalog: Catalog, pub origin: CatalogOrigin, pub date: String, pub warnings: Vec<String> }`
  - `pub fn load_catalog(refresh: bool) -> LoadedCatalog`: it never fails, because the snapshot is the floor.
  - `pub const CATALOG_URL: &str = "https://raw.githubusercontent.com/TrungHocCode/attune/catalog/catalog.json"`
  - Internal `load_catalog_at(home, refresh, now, fetch: &dyn Fn()->Result<String,_>)` for tests.

- [ ] **Step 1: Write the snapshot.** Hand-write `catalog/snapshot.json` for the non-gated models named in the spec: SmolLM2 135M/360M Instruct, Qwen3 0.6B/4B-Instruct-2507/8B, Qwen2.5 7B/14B Instruct, Phi-4-mini-instruct, Mistral 7B Instruct v0.3. For each, include `q4_k_m` (and `q8_0` where it exists).
  - Repo, file, revision, size and sha256 come from the HF API: `https://huggingface.co/api/models/<repo>/tree/<revision>` (`lfs.oid` is the sha256).
  - params/arch/ctx come from the GGUF header, read with a range request or a local probe. Use `curl`/`gh api` by hand.
  - `llama_cpp.tag` is `b11804`, with the Windows CPU x64 asset and the Linux CPU x64 asset from that release (size and sha256 from the GitHub release API `digest` field).
  - Add a unit test `snapshot_parses_and_has_every_tier`: `include_str!("../../../../catalog/snapshot.json")` parses, and at least one model falls in each tier from the Global Constraints.
- [ ] **Step 2: Write failing loader tests** (temp home dir, injected fetch and clock):
  - `fresh_cache_is_used_without_fetching`: cache 1 h old gives `FreshCache`, and the fetch closure is never called.
  - `stale_cache_triggers_fetch_and_rewrites_cache`: 30 h old, fetch Ok, gives `Fetched`, and the cache file is updated.
  - `offline_falls_back_to_stale_cache_then_snapshot`: fetch Err with a stale cache gives `StaleCache` plus a warning. Fetch Err with no cache gives `Snapshot`.
  - `corrupt_or_newer_schema_cache_falls_back`: cache text `{oops` or schema `attune-catalog/2`, with fetch Err, gives `Snapshot` plus a warning naming the cache path. (Review Focus 3)
  - `refresh_forces_fetch_even_when_fresh`
  - `hf_config_cache_lives_in_home`: `cache_path()` is under `ATTUNE_HOME`.
- [ ] **Step 3: Run** `cargo test -p attune-estimate registry`. Expected: FAIL.
- [ ] **Step 4: Implement.** The cache file is `<home>/catalog.json`. Its freshness uses the file's `generated_at` plus the fetch time stored in `<home>/catalog.meta.json`.
- [ ] **Step 5: Run** the tests and the 4 gates. Expected: pass.
- [ ] **Step 6: Commit** `feat(estimate): load the model catalog from cache, network or the built-in snapshot`.

### Task 4: Recommend ranking (pure, attune-tune)

**Files:**
- Create: `crates/attune-tune/src/recommend.rs`
- Modify: `crates/attune-tune/src/lib.rs` (export)
- Test: unit tests in `recommend.rs`

**Interfaces:**
- Consumes: Task 2 `CatalogModel`, `CatalogFile`. The existing `HwInputs::predict_prefill_ms`, `predict_decode_tps_from_bytes`, `kv_filled_tokens` and `kv_elem_bytes`.
- Produces:
  - `pub enum Tier { Tiny, Small, Medium, Large }`, with `Tier::of(params: u64) -> Tier`.
  - `pub enum Verdict { Fits, Tight, NeedsMore { short_bytes: u64 } }`
  - `pub struct RecommendRow { pub model_id: String, pub display: String, pub quant: Quant, pub download_bytes: u64, pub ram_bytes: u64, pub decode_tps: f64, pub ttft_ms: f64, pub verdict: Verdict, pub tier: Tier, pub pick: bool, pub star: bool }`
  - `pub struct RecommendOptions { pub budget_bytes: f64, pub n_ctx: u32 /*4096*/, pub prompt_tokens: u64 /*512*/, pub gen_tokens: u64 /*128*/ }`
  - `pub fn recommend(models: &[CatalogModel], hw: HwInputs, opts: &RecommendOptions) -> Vec<RecommendRow>`: one row per model × file (gated models are skipped). The output is sorted: stars first, then picks by tier (large → tiny), then the rest by tier and decode.

- [ ] **Step 1: Write failing tests** (fixture of 6 models across tiers, hw = 32 GB/s, 563 GFLOPS, 600 int8 GOPS):
  - `ram_is_weights_plus_kv_at_n_ctx_clamped`: `ram_bytes` = file size + KV(f16) × min(4096, ctx_default).
  - `verdicts_follow_headroom`: budget = ram × 1.20 gives Fits, ram × 1.05 gives Tight, ram × 0.8 gives NeedsMore with `short_bytes` ≈ 0.2 × ram.
  - `pick_is_most_params_meeting_the_bar_per_tier`: in the small tier, a 4B model at 11 tok/s is picked over a 3B model. A 9B model at 4.5 tok/s (below 5) is not picked in medium.
  - `date_breaks_ties_within_15_percent_params`: 7.6B released 2025 beats 7.2B released 2024. 8B released 2023 beats 6B released 2025 (more than 15% apart).
  - `star_is_the_largest_tier_with_a_pick`
  - `gated_or_missing_quant_is_never_picked`: a gated model, and a model with only `q8_0` when its q8_0 row misses the bar, are not picked. (Review Focus 5)
  - `predictions_match_estimate`: a row's `decode_tps`/`ttft_ms` equal `predict_decode_tps_from_bytes` / `predict_prefill_ms` for the same inputs (float tolerance 1e-9).
- [ ] **Step 2: Run** `cargo test -p attune-tune recommend`. Expected: FAIL.
- [ ] **Step 3: Implement** with the existing estimate functions only. There is no new maths.
- [ ] **Step 4: Run** the tests and the 4 gates. Expected: pass.
- [ ] **Step 5: Commit** `feat(tune): rank catalog models by fit and predicted speed`.

### Task 5: `tune` CLI — recommend mode, catalog ids, profile v2 `source`

**Files:**
- Modify:
  - `crates/attune-cli/src/commands/mod.rs`: drop `.required(true)` on the `model_source` group; add `--max-ram <SIZE>`, `--all` and `--refresh-catalog` to `Tune`.
  - `crates/attune-cli/src/commands/tune.rs`: recommend path, plus catalog-id resolution in `resolve_model`.
  - `crates/attune-profile/src/schema.rs`: `ModelConfig.source: Option<ModelSource>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`; `pub struct ModelSource { repo, file, revision, sha256, size: u64 }`.
  - `crates/attune-profile/src/lib.rs`: `PROFILE_VERSION = 2`.
  - `crates/attune-profile/src/migrate.rs`: v1 → v2 keeps everything and sets no source.
  - `crates/attune-tune/src/export.rs`: `from_best` takes `Option<ModelSource>`.
- Test: unit tests in `tune.rs` (the crate has no integration tests and no `assert_cmd`; follow the existing pattern of testing pure helpers, and split the recommend path into `fn recommend_report(catalog: &Catalog, hw: HwInputs, avail: f64, args: &RecommendArgs) -> Vec<RecommendRow>` plus `fn render_recommend(rows, origin, all: bool) -> String` so both are testable without I/O), `schema.rs`, `migrate.rs`.

**Interfaces:**
- Consumes: Task 3 `load_catalog`, Task 2 `Catalog::find`, Task 4 `recommend`.
- Produces:
  - Profile `model.source`, with `model.path` stored as `~/.attune/models/<repo>/<file>`.
  - `pub fn attune_profile::expand_home(path: &str, home: &Path) -> PathBuf`: expands a leading `~/.attune/` (pure string logic, no deps).

- [ ] **Step 1: Write failing tests:**
  - `v1_profile_migrates_to_v2_without_source`: an existing v1 fixture loads, with `profile_version == 2` and `source == None`.
  - `v2_roundtrips_source`
  - `expand_home_only_touches_the_attune_prefix`: `~/.attune/models/a/b.gguf` expands to `<home>/models/a/b.gguf`. `F:/x.gguf` is unchanged.
  - `tune_without_model_prints_recommend_table`: run `tune` with no model and `--bandwidth-gbps 32` against an injected catalog. Stdout contains the header `Model`, `Fit`, a `★` row and the line `Next: attune tune --model`. `--json` emits an array of `RecommendRow`.
  - `max_ram_overrides_available`: `--max-ram 2G` turns a 2.7 GB row into ❌ `needs 0.7 GB more`.
  - `default_table_shows_picks_plus_two_per_tier`: with 20 fixture rows, `render_recommend(.., all=false)` shows each tier's pick plus at most 2 more per tier; `all=true` shows all 20.
  - `catalog_id_tune_needs_no_file_and_saves_source`: `tune --model qwen3-4b --save p.json` (catalog id, file absent) writes a profile whose `model.source.sha256` equals the catalog's, and whose `model.path` is `~/.attune/models/Qwen/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.
  - `local_gguf_path_still_wins_over_catalog` (existing behaviour kept).
- [ ] **Step 2: Run** `cargo test -p attune-profile -p attune-cli`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Recommend mode reuses tune's machine measurement path (quick bench + int8 probe, or overrides).
  - Available RAM comes from hal unless `--max-ram` is given.
  - The table is printed as in the spec sketch, followed by a footer: catalog origin + date, "Predictions are uncalibrated (about ±25%)", and the plain bottleneck line.
  - Resolution order for `--model`: existing `.gguf` path → HF id → new catalog `find` → old `MODEL_CATALOG` (estimate-only).
  - `tune --verify` with a catalog id downloads first via the Task 6 helper. Keep it Err ("download first with `attune serve` or pass a .gguf path") until Task 6 lands.
- [ ] **Step 4: Run** the tests and the 4 gates. Live check: `cargo run --release -p attune-cli -- tune` prints the table on this machine.
- [ ] **Step 5: Commit** `feat(tune): recommend catalog models when --model is omitted` (the profile v2 change can be its own commit: `feat(profile): record the model source (profile v2)`).

### Task 6: Download a missing model on `serve`

**Files:**
- Create: `crates/attune-backend/src/models.rs` (ensure-model helper)
- Modify:
  - `crates/attune-serve/src/launcher.rs:190-205`: resolve the path with `expand_home`; when missing and `source` is set, call the helper before validation.
  - `crates/attune-cli/src/commands/mod.rs` + `serve.rs`: `--yes` flag; prompt only when stdin is a TTY (`std::io::IsTerminal`).
  - `crates/attune-backend/Cargo.toml`: depends on `attune-fetch`.
- Test: unit tests in `models.rs`, `launcher.rs`

**Interfaces:**
- Consumes: Task 1 `download`, `home_dir`. Task 5 `ModelSource`, `expand_home`.
- Produces: `pub fn ensure_model(path: &Path, source: &ModelSource, confirm: &mut dyn FnMut(u64) -> bool) -> Result<PathBuf, BackendError>`. It returns immediately if the file exists **and** its size matches. Otherwise it calls `confirm(size)`, downloads and verifies.

- [ ] **Step 1: Write failing tests:**
  - `existing_file_with_right_size_skips_download`: confirm is never called.
  - `declined_confirmation_errors_without_network`: the error says `download declined`.
  - `non_tty_without_yes_errors_clearly`: launcher with `interactive=false, yes=false` and a missing model gives an error containing `pass --yes to download 2.5 GB`, and nothing blocks. (Review Focus 4)
  - `profile_without_source_and_missing_file_keeps_todays_error`
- [ ] **Step 2: Run** `cargo test -p attune-backend -p attune-serve`. Expected: FAIL.
- [ ] **Step 3: Implement.** The prompt text is `Download <display size> to <path>? [y/N]`. Progress goes to stderr, and nothing is printed under `--json`. Also enable `tune --verify` with a catalog id through `ensure_model`.
- [ ] **Step 4: Run** the tests and the 4 gates. Live: `serve --profile p.json` from Task 5 downloads Qwen3 0.6B, verifies it, and serves.
- [ ] **Step 5: Commit** `feat(serve): download a profile's missing model after confirmation`.

### Task 7: `attune setup` and llama.cpp install lookup

**Files:**
- Create: `crates/attune-fetch/src/install.rs` (unpack + layout), `crates/attune-cli/src/commands/setup.rs`
- Modify:
  - `crates/attune-backend/src/llama_subprocess.rs`: `select_llama_server` gains an `installed: Option<&Path>` candidate after PATH; spawning sets `LD_LIBRARY_PATH` to the binary's dir on Linux.
  - `crates/attune-cli/src/commands/{mod.rs,bench.rs,tune.rs}`: resolve `llama-cli` with the same fallback and pass the path in.
  - `crates/attune-fetch/Cargo.toml`: `zip`, plus `tar` + `flate2` only if the Linux asset is `.tar.gz`.
- Test: unit tests in `install.rs`, `llama_subprocess.rs`

**Interfaces:**
- Consumes: Task 3 `LoadedCatalog.catalog.llama_cpp`, Task 1 `download`.
- Produces:
  - `pub fn install_dir(home: &Path, tag: &str) -> PathBuf` (`<home>/llama.cpp/<tag>`).
  - `pub fn installed_binary(home: &Path, tag: &str, name: &str) -> Option<PathBuf>`: searches the install dir recursively for `name` / `name.exe`.
  - `pub fn install_archive(archive: &Path, dest: &Path) -> Result<(), FetchError>`: unpacks into `dest.tmp`, then renames, so it is atomic.

- [ ] **Step 1: Write failing tests:**
  - `install_unpacks_zip_atomically`: an in-memory zip with `bin/llama-server.exe` gives `installed_binary` = that path, and no `.tmp` remains. Unpacking a corrupt zip leaves no install dir.
  - `zip_slip_entries_are_rejected`: an entry `../evil` gives Err, and nothing is written outside `dest`.
  - `select_prefers_flag_env_path_then_installed`
  - `setup_skips_when_already_installed`: `setup` reports `already installed` and does not download.
- [ ] **Step 2: Run** `cargo test -p attune-fetch install` and `-p attune-backend`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `attune setup [--yes] [--refresh-catalog]` picks the asset matching `std::env::consts::{OS, ARCH}` and variant `cpu`, shows its size, confirms, downloads to `<home>/downloads/`, installs, then prints the llama-server path.
  - `serve` and `tune --verify` errors for a missing binary now end with `run \`attune setup\``.
- [ ] **Step 4: Run** the tests and the 4 gates. Live (Windows): `attune setup` installs b11804, and `serve` finds it without `--llama-server`.
- [ ] **Step 5: Commit** `feat(cli): add attune setup to install the pinned llama.cpp build`.

### Task 8: Catalog generator and daily GitHub Action

**Files:**
- Create:
  - `catalog/publishers.toml`
  - `crates/attune-estimate/src/model/generate.rs` (pure selection and grouping)
  - `crates/attune-cli/src/commands/catalog.rs`: hidden `attune catalog build --publishers <file> --previous <file> --out <file>`
  - `.github/workflows/catalog.yml`
- Modify: `crates/attune-cli/src/commands/mod.rs` (hidden subcommand)
- Test: unit tests in `generate.rs` with recorded HF API JSON fixtures in `crates/attune-estimate/tests/fixtures/hf/`

**Interfaces:**
- Consumes: Task 2 schema, Task 1 `fetch_text`, the existing GGUF header parser (`parse_metadata` on the first bytes from a range GET).
- Produces: `catalog.json` on the `catalog` branch at `CATALOG_URL`.

`publishers.toml` shape (exact keys):

```toml
llama_cpp_tag = "b11804"
max_params = 70_000_000_000
quants = ["q4_k_m", "q8_0"]
# order = preference when several repos ship the same model
publishers = ["Qwen", "microsoft", "mistralai", "HuggingFaceTB", "ibm-granite", "ggml-org", "bartowski", "unsloth"]
```

- [ ] **Step 1: Write failing tests** (fixtures, no network):
  - `groups_repos_into_one_model_preferring_publisher_order`: `Qwen/Qwen3-4B-Instruct-2507-GGUF` and `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` give one model, whose q4_k_m file comes from the `Qwen` repo.
  - `skips_gated_base_and_too_large`: a gated repo, a base (non-instruct) model, and a 72B model are excluded.
  - `release_date_comes_from_the_base_repo`: `released` is the `createdAt` of the `base_model` repo, or `None` when it is absent.
  - `guard_refuses_empty_or_shrunk_catalog`: the new catalog has 0 models, or fewer than 70% of `--previous`'s, so it errors and writes nothing.
- [ ] **Step 2: Run** `cargo test -p attune-estimate generate`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - The HF calls are `https://huggingface.co/api/models?author=<p>&filter=gguf&full=true`, `/api/models/<repo>` (gated, cardData.base_model, sha) and `/api/models/<repo>/tree/<sha>`. The header comes from a GET with `Range: bytes=0-8388607`.
  - Use `HostPolicy::production()`, and send `HF_TOKEN` as a bearer only if it is set in the Action (rate limits). It is never used for gated access.
- [ ] **Step 4: Write `.github/workflows/catalog.yml`:**
  - `on: schedule: cron "17 3 * * *"` and `workflow_dispatch`.
  - Steps: build release, then check out the `catalog` branch into `out/`, then `attune catalog build --publishers catalog/publishers.toml --previous out/catalog.json --out out/catalog.json`, then commit and push to `catalog` only if the file changed.
  - Create the `catalog` branch once, seeded from `catalog/snapshot.json`. Branch protection stays on `main` only.
- [ ] **Step 5: Run** the tests and the 4 gates. Run the workflow once with `gh workflow run catalog.yml`, then fetch `CATALOG_URL` and check that `parse_catalog` accepts it.
- [ ] **Step 6: Commit** `feat(ci): regenerate the model catalog daily`.

### Task 9: Acceptance run and phase close

**Files:**
- Modify: `ROADMAP.md` (M6 ✅ with numbers), `README.md` (Quick start = setup → tune → serve, re-captured output), `CHANGELOG.md` (`[0.6.0]`), `Cargo.toml` (`version = "0.6.0"`), `AGENTS.md` (current phase), `docs/benchmark-methodology.md` (recommend accuracy protocol + results)

- [ ] **Step 1: Zero manual downloads.** Set `ATTUNE_HOME` to an empty dir and make sure no llama.cpp is on PATH or in env. From the release `attune.exe`, run `setup --yes` → `tune` → `tune --model <★ pick> --save p.json` → `serve --profile p.json --yes`, then send one chat completion. Record the wall time minus download time.
- [ ] **Step 2: Recommend accuracy.** Take 3 catalog models from different tiers. Compare predicted decode with `llama-bench -p 512 -n 128 -r 3` (default threads). Pass: within 25% each. Every ✅ row actually serves.
- [ ] **Step 3: Offline.** With the network disabled, `tune` works from the cache, and `serve` of the Step 1 profile works.
- [ ] **Step 4: Download safety.** Kill a download mid-way, re-run, and it resumes. A catalog entry with a wrong sha256 is rejected.
- [ ] **Step 5: Catalog freshness.** Check that the Action history shows 7 consecutive green daily runs. This gates the ✅, so the phase close waits for it.
- [ ] **Step 6:** Refresh `catalog/snapshot.json` from the `catalog` branch (the release-time snapshot rule from the spec). Do the phase-close checklist (AGENTS.md §4), open a PR with the numbers, merge, then tag `v0.6.0`.
