# AGENTS.md — Rules for AI Agents working on Attune

> Tagline: *Tune LLM inference to your hardware*
> Stack: Rust workspace + clap CLI + llama.cpp backend. Linux/Windows first, macOS later.

## 1. Repo Structure (must follow `attune-plan.md` §7)

```
Cargo.toml (workspace)
crates/
  attune-hal/       # detect CPU/SIMD/cache/RAM/NUMA/GPU — no dependency on other attune crates
  attune-bench/     # measure BW/FLOPS/e2e — depends only on attune-hal
  attune-estimate/  # roofline + calibrate + bottleneck — depends on attune-bench
  attune-tune/      # sweep + score — depends on estimate/bench
  attune-profile/   # schema/validate/migrate — depends on NOTHING (keep it pure)
  attune-backend/   # Backend trait + llama_subprocess (FFI later)
  attune-serve/     # launcher/health/proxy
  attune-cli/       # thin CLI, depends on all, no logic here
docs/ profiles/ models/ tests/ benches/
```

- New code MUST go into the correct crate per table above. CLI only parses args and calls library functions.
- `attune-profile` must stay dependency-free (only `serde`/`serde_json`/`thiserror`).

## 2. Build / Test / Lint (always run before commit)

```powershell
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo run -p attune-cli -- detect
```

- Phase 0 acceptance: `cargo test` must be green on Linux + Windows.
- Core crates target >70% coverage, CI time <10min, binary <20MB.

## 3. Coding Rules

1. **Edition:** Rust stable, edition 2021. No nightly features.
2. **Errors:** libraries return `Result<T, E>` with `thiserror`; `attune-cli` uses `anyhow` for context. No `.unwrap()` in non-test code.
3. **Cross-platform first:** Linux (x86-64) + Windows (x86-64) must compile. Gate platform code with `#[cfg(target_os = ...)]`. Use `raw-cpuid`, `hwloc`, `sysinfo`. RAPL/`/sys/class/powercap` is Linux-only — fallback gracefully.
4. **No new deps without reason:** prefer std. Adding a crate requires 1-line justification in PR/commit.
5. **No Python.** Prototype in Rust directly (plan §6 decided Rust for perf/FFI/single-binary).
6. **Output:** `detect/bench/estimate/tune` must support `--json` (machine) + pretty table (human). JSON is the contract.
7. **Profile schema (§10):** versioned (`profile_version`). Never break compat — add `migrate.rs` instead. Validate: fingerprint match → warn+fallback, RAM check → suggest lower `n_ctx`/quant.

## 4. Phase Discipline (do NOT jump ahead)

Current phase: **PHASE 0 — FOUNDATION (M1: `attune detect`)**.
- Commit 1: workspace + CI + `attune-hal` skeleton + `detect` stub only.
- DO NOT implement `bench/estimate/tune/serve/Tauri` until M1 is done.
- Each task must meet its Acceptance column in plan §8 before marking done.

## 5. Math Reference (plan §14 — use these, don't invent)

- Decode (memory-bound): `tok/s ≈ Bandwidth / bytes_weights` (FP16=2B, INT8=1B, INT4=0.5B per param)
- Prefill (compute-bound): `time = 2 * params * n_tokens / FLOPS_peak`
- Roofline: `Perf = min(FLOPs_peak, AI * BW)`, `AI* = FLOPs_peak / BW`

## 6. What NOT to do

- Don't touch `models/` (gitignored test weights) or `profiles/` output in commits.
- Don't commit `target/`, `*.gguf`, local bench JSON.
- Don't rename binary/crate prefix (`attune`, `attune-*`).
- Don't add Desktop/Tauri code until M5 (serve) is complete.

## 7. Git Flow (strict — solo dev, chuẩn team)

**Branches:**
- `main` — luôn xanh (CI pass), chỉ merge vào, KHÔNG code trực tiếp cho feature.
- `feat/<m1-detect-simd>` / `fix/<…>` / `chore/<…>` / `docs/<…>` — làm việc ở đây.
- `release/m1`, tag `v0.1.0`, `v0.2.0`… cho M1→M5.
- **Ngoại lệ bootstrap:** commit 1-2 đầu (dựng workspace/CI/skeleton, như `chore: init...`) được push thẳng lên `main` để dựng sườn. Từ commit 3 trở đi bắt buộc đi qua branch + PR.

**Workflow bắt buộc cho mọi task:**
1. `git checkout main; git pull; git checkout -b feat/<ten-ngan>`
2. Code theo §1-§5, chạy đủ 4 lệnh §2 (test + clippy + fmt + detect).
3. Commit theo Conventional Commits (xem dưới). 1 commit = 1 ý, message tiếng Anh, mô tả tiếng Việt trong PR nếu cần.
4. `git push -u origin feat/<…>` → mở PR vào `main` → CI xanh → squash/merge → xóa branch.
5. Update `CHANGELOG.md` khi xong milestone.

**Commit style:**
- Format: `type(scope): subject` — ví dụ: `feat(hal): detect SIMD via raw-cpuid`, `fix(cli): handle zero RAM`, `chore: init CI`, `docs: update architecture`.
- Types cho phép: `feat|fix|chore|docs|refactor|test|perf|ci|build`.
- KHÔNG: `git push --force` lên `main`, commit `target/`, `*.gguf`, `profiles/*.json`, secret, `attune-plan.md` (đã gitignored — local only).
- Trước commit luôn chạy: `git status --short` + `git diff --stat` để soát file lọt (plan, secret).

**Release:**
- M1→M5 tag `v0.1.0`→`v0.5.0`, ghi `CHANGELOG.md`, binary <20MB.
