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

Use the bug report template. Include: OS, CPU/RAM, Rust version
(`rustc --version`), exact command, expected vs actual behavior, and relevant
logs (`--json` output helps).

## Suggesting Features

Use the feature request template. Explain the use case, the proposed
`attune` command/UX, and which milestone it belongs to (M1–M5).
