# Friendly attune: install, run, chat (M6 completion)

Status: implemented in v0.6.0 (#121, #122, #126, #127, #128; M6 closed in #129).

## Why

A new user needs about 6 steps today: Rust toolchain, clone, build, `setup`, `tune`, `tune --model … --save <path>`, `serve --profile <path>`. After all that there is still no way to talk to the model without `curl` or a third-party app. The output is written for experts: a score column, a "pruned" line, noisy progress. The bar we want is **install with one line, then one command, then chat**.

## Goals

- One-line install on Windows and Linux. No Rust needed.
- `attune recommend` lists the models that suit this machine.
- `attune run <model>` goes from nothing to chatting with a tuned model.
- `attune chat` chats with any running attune server.
- Output a newcomer can read. Expert detail stays available behind a flag.
- Close M6 and tag v0.6.0. The tag publishes the first binary release.

## Non-goals and anti-goals

- **Non-goals:**
  - GPU (M7).
  - macOS.
  - Code signing.
  - winget, scoop or apt packages.
  - A GUI.
- **Anti-goals:**
  - Breaking scripts or JSON that use today's commands. `tune` without `--model` still recommends.
  - Hiding the expert paths: `tune --verify`, `--json` and `bench` stay as they are.
  - Downloading anything without consent. Today's consent rules apply.

## Approaches considered

1. **New top-level commands over the existing library (chosen).** `recommend`, `run` and `chat` are thin CLI commands that call what `tune`, `setup` and `serve` already do. Little new logic; every existing flag keeps working.
2. **Rework `tune` and `serve` into a wizard.** Interactive prompts everywhere. It breaks scripting and duplicates paths. Rejected.
3. **A separate launcher app or GUI.** Too large, and the Tauri desktop app was deferred. Rejected for now.

## Design

### 1. Install (`install.ps1`, `install.sh`, `release.yml`)

- **`release.yml`** runs on a `v*` tag:
  - Builds `attune` for `x86_64-pc-windows-msvc` and `x86_64-unknown-linux-gnu`.
  - Packs `attune-<tag>-windows-x64.zip` and `attune-<tag>-linux-x64.tar.gz`.
  - Writes a `SHA256SUMS` file and publishes a GitHub Release with the CHANGELOG section as notes.
  - Actions are pinned to SHAs; only this job gets `contents: write`.
- **The install scripts** live in the repo root:
  - Resolve the latest release through the GitHub API, download the asset for the OS, and verify it against `SHA256SUMS`.
  - Place the binary in `~/.attune/bin` (`%USERPROFILE%\.attune\bin`).
  - Add that folder to the user PATH: the user `Path` variable on Windows; on Linux, a line in `~/.profile` (plus `~/.bashrc` and `~/.zshrc` if they exist).
  - Print `attune recommend` as the next step.
  - Re-running the script updates attune. `ATTUNE_VERSION=v0.6.0` pins a version.
  - No admin rights are needed.
- **README quick start:**
  ```
  irm https://raw.githubusercontent.com/TrungHocCode/attune/main/install.ps1 | iex
  ```
  Then `attune recommend`, then `attune run <model>`. Build-from-source moves to CONTRIBUTING.
- **SmartScreen:** Windows may warn on the first run of an unsigned exe. The README says so.

### 2. `attune recommend`

- A visible top-level command. It is the same code and flags as `tune` without `--model`: `--max-ram`, `--all`, `--refresh-catalog`, `--json`.
- The table ends with one hint: `Next: attune run <picked id>`.

### 3. Profiles by name

- Profiles saved by attune go to `~/.attune/profiles/<name>.json`.
- `serve --profile` accepts a path, as today, or a bare name, which resolves to that file.
- `tune --save` keeps requiring a path, so existing behaviour is unchanged.

### 4. `attune run <model> [--yes] [--retune] [--no-chat] [serve flags]`

1. Resolve `<model>` against the catalog. The same ids as `tune --model` are accepted, and a local `.gguf` path works too.
2. **llama-server:** resolve it as `serve` does. If none is found, run the `setup` install, after consent.
3. **Profile:**
   - Use `~/.attune/profiles/<id>.json` if it exists and its hardware fingerprint matches.
   - Otherwise tune it (estimate mode, quick bench, no `--verify`) and save it there.
   - `--retune` forces a new tune.
   - Print one line: the chosen config and the predicted speed.
4. **Serve:** download the model if it is missing, after consent, and start the server as `serve` does.
5. **Chat:** once the server is healthy, start `chat` in the same terminal. `/bye` or Ctrl-C stops the chat and the server.
   - With `--no-chat` it just serves, like `serve`, and prints the URL.

### 5. `attune chat [--url URL] [--api-key KEY] [--system TEXT]`

- The default URL is `http://127.0.0.1:8080`.
- A REPL that sends `/v1/chat/completions` with `stream: true` and prints the reply token by token.
- It keeps the conversation history.
- Commands: `/bye` and `/clear`.
- After each reply it prints `[N tok, X tok/s]` dimly.
- It uses `ureq`, which is already a dependency, and parses the SSE stream by hand. No new dependencies.

### 6. Output polish

- Help texts lose their milestone notes ("(M1: …)", "(M5 slices 2-3 …)"); each command gets a one-line plain description.
- **`tune` table:**
  - The default shows the top 5 configs.
  - The score column and the "pruned" line move behind `--verbose`.
  - JSON is unchanged.
- **Download progress:**
  - One line that redraws in place on a terminal; otherwise it prints every 10%.
  - Sizes show in MB below 1 GB, so no "0.00 GB".
- **Wording:** the catalog-refresh 404 warning becomes "Using the built-in model list (online catalog unavailable)".
- **README:** a short "Where files go" section covering `~/.attune`, the `ATTUNE_HOME` override, and how much disk the models take.

### 7. Close M6 (v0.6.0)

- **Revised catalog freshness bar:** the manual run plus **2 consecutive scheduled daily runs** succeed. This replaces the old bar of 7 runs.
  - Why the change is safe: users never depend on a fresh catalog, because loading falls back from a stale cache to the built-in snapshot.
  - A failing daily run is visible in Actions and is fixed as a bug.
- **Close-out steps:**
  - Refresh `catalog/snapshot.json` from the `catalog` branch.
  - Run the phase-close checklist (ROADMAP ✅ with numbers, README, CHANGELOG 0.6.0, AGENTS phase → M7).
  - Bump the workspace version to 0.6.0 and push the annotated tag.
  - The tag runs `release.yml`.
- **Acceptance (executed, numbers posted):** on the reference laptop, as a new Windows user with no Rust in the session:
  - The one-line install succeeds.
  - `attune recommend` works.
  - `attune run qwen3-4b --yes` gets a streamed chat reply.
  - Total time is recorded, excluding the model download. Target: under 5 minutes.
  - Linux: the install script and `attune detect` work on the CI image (smoke test).

## Slices (one PR each, implemented with OpenCode deepseek)

1. Help text cleanup, `attune recommend` with the next-step hint, progress and wording polish, and `tune --verbose`.
2. Profiles by name and `attune chat`.
3. `attune run`.
4. `release.yml`, install scripts, README quick start and "Where files go", CONTRIBUTING build-from-source.
5. M6 close: snapshot refresh, docs, version 0.6.0, tag, then the release and installer acceptance.

After v0.6.0: M7 step-1 design (NVIDIA detect, CUDA llama.cpp build, full offload when the model fits in VRAM, VRAM-aware recommend).

## Left to the builder

- Exact table widths and wording beyond what is fixed above.
- The PATH-editing mechanics per shell.
- How the SSE parser handles keep-alive comments. It must ignore them.
