## What does this PR do?

<!-- Brief description. Link the issue: Fixes #123 -->

## How was it verified?

<!-- Check all that apply -->
- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace -- -D warnings`
- [ ] `cargo fmt --check`
- [ ] `cargo run -p attune-cli -- detect` (and related command)
- [ ] CI green on Ubuntu + Windows

## Checklist

- [ ] Conventional Commit title in English (`type(scope): subject`)
- [ ] PR body in English, no secrets or local-only files (`attune-plan.md`, `*.gguf`, `profiles/*.json`)
- [ ] Correct crate for the change, CLI stays thin
- [ ] No `.unwrap()` in non-test code
