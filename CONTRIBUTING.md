# Contributing to Attune

Thanks for your interest in contributing! This project follows a standard
open-source flow. Please read this guide before opening an issue or PR.

## Code of Conduct

Be respectful and constructive. We welcome beginners and maintainers alike.
Harassment, spam, or low-effort PRs will be closed.

## How to Contribute

1. **Find or open an issue** describing the bug or feature. For large changes,
   discuss the approach first to avoid wasted work.
2. **Fork the repo** (external contributors) or create a branch (maintainers).
3. **Create a focused branch** from `main`:
   - `feat/<short-name>` for features
   - `fix/<short-name>` for bug fixes
   - `docs/<short-name>` for documentation
   - `chore/<short-name>` for tooling/CI
4. **Make your change** following the rules below.
5. **Verify locally** (all four must pass):
   ```powershell
   cargo test --workspace
   cargo clippy --workspace -- -D warnings
   cargo fmt --check
   cargo run -p attune-cli -- detect
   ```
6. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/):
   `type(scope): subject` (e.g. `feat(hal): detect SIMD via raw-cpuid`).
   Allowed types: `feat|fix|chore|docs|refactor|test|perf|ci|build`.
   All commit messages, PR titles, and PR bodies MUST be in English.
7. **Push and open a PR** against `main`. Fill in the PR template, link the
   issue (`Fixes #123`), and describe how you verified the change.
8. **CI must be green** on Ubuntu + Windows. Address review feedback with new
   commits (maintainers will squash on merge).

## Branch Protection

`main` is protected:
- No direct pushes — all changes go through PRs (no exceptions after bootstrap).
- CI status checks (`check (ubuntu-latest)`, `check (windows-latest)`) must pass.
- Linear history is enforced (squash merge, no merge commits).
- Force pushes and branch deletion are disabled.

## Project Rules (from AGENTS.md)

- New code goes into the correct crate (`attune-hal`, `attune-bench`,
  `attune-estimate`, `attune-tune`, `attune-profile`, `attune-backend`,
  `attune-serve`, `attune-cli`). The CLI stays thin.
- Rust stable, edition 2021. Libraries use `thiserror`, CLI uses `anyhow`.
  No `.unwrap()` in non-test code.
- Linux (x86-64) + Windows (x86-64) must compile. Gate platform code with
  `#[cfg(target_os = ...)]`.
- `detect`/`bench`/`estimate`/`tune` must support `--json` plus a human table.
- `attune-profile` stays dependency-free (only `serde`/`serde_json`/`thiserror`).
- Never commit `target/`, `*.gguf`, `profiles/*.json`, secrets, or the
  local-only `attune-plan.md`.

## Reporting Bugs

Use the bug report template. A good report (see issue #13 as an example)
includes: OS, CPU/RAM, Rust version (`rustc --version`), exact command,
expected vs actual behavior, and relevant logs (`--json` output helps).
For performance bugs, add measured numbers: a before/after table and, when
relevant, disassembly or profiler evidence showing the root cause — not just
the symptom.

## Suggesting Features

Use the feature request template. Explain the use case, the proposed
`attune` command/UX, and which milestone it belongs to (M1–M5).

## Writing Quality Issues and PRs

Follow the standard set by issues/PRs #10–#14 in this repo:

- **Issues:** state the problem first with evidence (numbers, logs, exact
  file:line), then propose a concrete solution with acceptance criteria.
  Name alternatives you considered and why you rejected them. Keep the
  title in Conventional Commits style so it reads well on the board.
- **PRs:** fill in every section of the PR template. Summarize the change,
  show verification output (test counts, benchmark tables, CI status), link
  the issue with `Fixes #N`, and add a short “reviewer focus” note pointing
  at the risky or debatable parts of the diff. Never claim “CI green” —
  let the checks speak; leave the CI box unchecked until it is.
- **No placeholders:** no “coming soon”, no TODO sections, no screenshots
  of text. Paste text as text.
