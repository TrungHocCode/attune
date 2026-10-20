# attune — Tune LLM inference to your hardware

> `attune detect` → `bench` → `estimate` → `tune` → `serve` (M1 in progress)

## Dev (Phase 0)

```powershell
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo run -p attune-cli -- detect
cargo run -p attune-cli -- detect --json
```

See `AGENTS.md` for agent rules. Internal plan (`attune-plan.md`) is local-only and gitignored.
