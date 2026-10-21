//! llama.cpp subprocess runner (M2 slice 5).
//!
//! Everything here drives an external `llama-cli` binary through
//! [`std::process`] — no FFI, no new dependency. The module exists so the rest
//! of the codebase never has to think about *finding* that binary, *waiting*
//! for it without hanging, or *reading* its two very different output streams:
//!
//! * **Find it** — [`resolve_llama_cli`] applies the precedence
//!   `--llama-cli` flag > `ATTUNE_LLAMA_CLI` > `PATH` search, and the search
//!   looks for `llama-cli` before the legacy `main` name (llama.cpp renamed
//!   the binary, so old build trees only have `main`). The ordering itself is
//!   pure ([`select_llama_cli`]) and unit-tested; only the filesystem walk
//!   ([`search_path`]) touches disk.
//! * **Run it without hanging** — [`run_once`] spawns the child, reads stdout
//!   and stderr on their own threads (llama.cpp writes a lot of progress to
//!   stderr — a blocking read would deadlock on a full pipe), and waits with a
//!   deadline. On expiry it kills the child and reaps it through [`ChildGuard`]'s
//!   `Drop`, so a wedged backend can never leave a zombie behind.
//! * **Read it** — llama.cpp has moved its banner text several times, so both
//!   parsers ([`parse_version`], [`parse_perf`]) are tolerant pure functions
//!   unit-tested against canned output from several `b`-versions. They accept
//!   *and ignore* whatever they do not understand; they never panic.
//!
//! **Test policy.** Automated tests never need llama.cpp or a model: the only
//! child processes they spawn are trivial cross-platform sleepers. Tests that
//! want a real build are `#[ignore]`d *and* env-gated (`ATTUNE_LLAMA_CLI` +
//! `ATTUNE_TEST_MODEL`), so CI — which has neither — can never run them. Those
//! runs are manual-only by design (AGENTS.md §4 phase discipline: a real e2e
//! measurement belongs to a later slice, not to a green test suite).
//!
//! No `.unwrap()` in non-test code (AGENTS.md §3.2) — spawn, wait and parse
//! failures all surface as [`BenchError`].

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::BenchError;

/// Environment variable holding an explicit `llama-cli` path. Second in the
/// precedence chain, after the `--llama-cli` flag.
pub const ENV_LLAMA_CLI: &str = "ATTUNE_LLAMA_CLI";

/// Environment variable pointing at a GGUF model. Read **only** by the manual,
/// `#[ignore]`d integration tests in this module — never by production code.
pub const ENV_TEST_MODEL: &str = "ATTUNE_TEST_MODEL";

/// Binary names looked up on `PATH`, in order. llama.cpp renamed `main` to
/// `llama-cli` (b1295-ish), so an old build tree only has `main` and a new one
/// only has `llama-cli` — we accept either, modern name first.
pub const PATH_CANDIDATES: &[&str] = &["llama-cli", "main"];

/// Deadline for the `--version` probe. A healthy build answers in
/// milliseconds; anything past this is a wedged binary, not a slow machine.
pub const VERSION_TIMEOUT: Duration = Duration::from_secs(10);

/// Default deadline for a model load test. Loading a GGUF (mmap + warm-up of
/// every layer) is the slow part and scales with model size, hence minutes
/// rather than seconds. Callers with a size in mind should pass their own.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Polling period for the wait-with-deadline loop. `std::process::Child` has no
/// portable blocking-wait-with-timeout, so we poll `try_wait`; 10 ms costs
/// nothing next to a multi-second child while keeping the kill responsive.
pub const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long to wait for a child's output readers after the child itself is
/// gone. Comfortably longer than the time a dying process needs to flush, and
/// short enough that a grandchild holding the pipe open cannot hang the bench.
/// See [`join_reader`].
const READER_JOIN_BUDGET: Duration = Duration::from_secs(2);

/// Prompt used by the minimal load test. Deliberately trivial: the point is to
/// make llama.cpp mmap the weights and build its KV cache, not to benchmark
/// anything (that is `attune bench`'s job).
pub const LOAD_PROMPT: &str = "hi";

/// Tokens predicted by the minimal load test. One token is enough to prove the
/// whole graph runs; more would just make the smoke test slower.
pub const LOAD_PREDICT: u32 = 1;

/// Context size for the minimal load test. Small on purpose: llama.cpp sizes
/// the KV cache from `n_ctx` up front, and a 1-token probe has no use for a
/// multi-megabyte cache.
pub const LOAD_CTX: u32 = 512;

/// Number of trailing characters of captured stderr kept in an error message.
/// Full stderr can be megabytes (model list, tensor table); an error message
/// only needs the tail, where the actual failure lives.
const STDERR_TAIL_CHARS: usize = 400;

/// Markers that introduce the build number in llama.cpp `--version` output.
/// The wording has drifted across releases — `main: build = 4639 (a3cf7c6)` in
/// the b1xxx–b2xxx era, `version: 3505 (d09f13b4)` since b3xxx — so we probe
/// several shapes and take the first that yields a plausible number.
const VERSION_MARKERS: &[&str] = &["version:", "version =", "build =", "build:", "version "];

/// Where a resolved `llama-cli` came from. Reported in errors so the user knows
/// *which* of the three precedence levels misfired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaSource {
    /// The explicit `--llama-cli` flag.
    Flag,
    /// The `ATTUNE_LLAMA_CLI` environment variable.
    Env,
    /// Found on `PATH` under this binary name.
    Path {
        /// The name that matched, i.e. an entry of [`PATH_CANDIDATES`].
        name: String,
    },
}

impl LlamaSource {
    /// Human-readable origin, used inside error messages.
    pub fn describe(&self) -> String {
        match self {
            LlamaSource::Flag => "`--llama-cli` path".to_string(),
            LlamaSource::Env => format!("`{ENV_LLAMA_CLI}` path"),
            LlamaSource::Path { name } => format!("`{name}` on PATH"),
        }
    }
}

/// One subprocess run: exit status plus both captured streams.
///
/// Not `Serialize` — this is a process-level type, not part of the JSON
/// contract. What reaches the JSON contract is [`LlamaProbeReport`].
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutput {
    /// Everything the child wrote to stdout (decoded leniently).
    pub stdout: String,
    /// Everything the child wrote to stderr (decoded leniently).
    pub stderr: String,
    /// Exit code, or `None` when the process was terminated by a signal (Unix).
    pub exit_code: Option<i32>,
    /// Wall-clock time from spawn to exit.
    pub duration: Duration,
}

impl RunOutput {
    /// True when the child exited with code 0. A signalled process (`None`)
    /// counts as failure — a killed backend did not succeed.
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// Timings scraped from a llama.cpp `llama_perf` block. All four fields come
/// from the same two context lines, so they are always internally consistent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerfSummary {
    /// Prefill wall-clock time in milliseconds.
    pub prompt_eval_ms: f64,
    /// Tokens processed during the prefill.
    pub prompt_tokens: u64,
    /// Decode wall-clock time in milliseconds.
    pub eval_ms: f64,
    /// Tokens generated during the decode (llama.cpp calls these "runs").
    pub eval_tokens: u64,
}

impl PerfSummary {
    /// Prefill throughput in tokens/second (0.0 when no tokens were timed).
    pub fn prompt_tps(&self) -> f64 {
        if self.prompt_eval_ms > 0.0 && self.prompt_tokens > 0 {
            self.prompt_tokens as f64 / (self.prompt_eval_ms / 1_000.0)
        } else {
            0.0
        }
    }

    /// Decode throughput in tokens/second — the number that matters for
    /// interactive latency (0.0 when no tokens were timed).
    pub fn eval_tps(&self) -> f64 {
        if self.eval_ms > 0.0 && self.eval_tokens > 0 {
            self.eval_tokens as f64 / (self.eval_ms / 1_000.0)
        } else {
            0.0
        }
    }
}

/// What `attune bench --model` learned about the local llama.cpp setup. This
/// is the JSON-contract shape of the probe (`bench --model --json` nests it
/// under `"llama"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaProbeReport {
    /// Path of the `llama-cli` binary that was used.
    pub cli: String,
    /// Build number scraped from `--version`.
    pub version: String,
    /// Path of the model that was loaded.
    pub model: String,
    /// Always `true` in a report: a failed load returns an error instead.
    pub load_ok: bool,
    /// Wall-clock duration of the load test, in milliseconds.
    pub elapsed_ms: f64,
    /// Timings from the run, when llama.cpp printed a `llama_perf` block.
    #[serde(default)]
    pub perf: Option<PerfSummary>,
}

// ---- locating the binary ----------------------------------------------------

/// Pure: apply the documented precedence over candidates that were *already*
/// gathered, i.e. no filesystem access here.
///
/// `path_hits` must be ordered the way [`search_path`] produced it (modern
/// `llama-cli` name before the legacy `main` name). An explicit or env value
/// that is blank is treated as unset, because exporting an empty variable is a
/// common accident. Existence is *not* checked here — [`resolve_llama_cli`]
/// does that after this function has picked a winner.
pub fn select_llama_cli(
    explicit: Option<&str>,
    env_value: Option<&str>,
    path_hits: &[(String, PathBuf)],
) -> Result<(LlamaSource, PathBuf), BenchError> {
    if let Some(value) = explicit.map(str::trim).filter(|v| !v.is_empty()) {
        return Ok((LlamaSource::Flag, PathBuf::from(value)));
    }
    if let Some(value) = env_value.map(str::trim).filter(|v| !v.is_empty()) {
        return Ok((LlamaSource::Env, PathBuf::from(value)));
    }
    if let Some((name, path)) = path_hits.first() {
        return Ok((LlamaSource::Path { name: name.clone() }, path.clone()));
    }
    Err(BenchError::LlamaNotFound {
        detail: format!(
            "searched PATH for {}",
            quoted_list(PATH_CANDIDATES.iter().copied())
        ),
    })
}

/// Locate `llama-cli`, honouring the full precedence chain and verifying that
/// the winner is actually a runnable file.
///
/// A missing or unusable binary is a normal, expected condition (llama.cpp is
/// an optional external tool), so it maps to [`BenchError::LlamaNotFound`] with
/// an actionable message rather than an I/O error.
pub fn resolve_llama_cli(explicit: Option<&str>) -> Result<PathBuf, BenchError> {
    let env_value = std::env::var(ENV_LLAMA_CLI).ok();
    let path_hits = search_path();
    let (source, chosen) = select_llama_cli(explicit, env_value.as_deref(), &path_hits)?;
    if !is_runnable_file(&chosen) {
        return Err(BenchError::LlamaNotFound {
            detail: format!(
                "{} `{}` is not a runnable file",
                source.describe(),
                chosen.display()
            ),
        });
    }
    Ok(chosen)
}

/// Walk `PATH` looking for [`PATH_CANDIDATES`], modern name first. The outer
/// loop is over names (not directories) so that `llama-cli` anywhere on `PATH`
/// always beats a `main` in an earlier directory — the documented precedence.
fn search_path() -> Vec<(String, PathBuf)> {
    let Some(path_var) = std::env::var_os("PATH") else {
        return Vec::new();
    };
    let dirs: Vec<PathBuf> = std::env::split_paths(&path_var).collect();
    let mut hits = Vec::new();
    for name in PATH_CANDIDATES {
        for dir in &dirs {
            for candidate in executable_variants(name) {
                let path = dir.join(candidate);
                if is_runnable_file(&path) {
                    hits.push(((*name).to_string(), path));
                }
            }
        }
    }
    hits
}

/// File names to probe for one binary name inside a `PATH` directory.
///
/// Windows resolves extensions through `PATHEXT`; probing `.exe` first covers
/// every realistic llama.cpp build without re-implementing that machinery.
#[cfg(target_os = "windows")]
fn executable_variants(name: &str) -> Vec<String> {
    vec![format!("{name}.exe"), name.to_string()]
}

/// Non-Windows platforms run the bare name.
#[cfg(not(target_os = "windows"))]
fn executable_variants(name: &str) -> Vec<String> {
    vec![name.to_string()]
}

/// True when `path` is a regular file this OS would let us execute.
///
/// Unix also checks the executable bit: a *directory* named `llama-cli`, or a
/// readable-but-not-executable file sitting in `PATH`, must never win the
/// search — picking it would turn a clean "not found" into a confusing spawn
/// error.
#[cfg(unix)]
pub fn is_runnable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

/// Windows has no executable bit — existence as a regular file is enough.
#[cfg(not(unix))]
pub fn is_runnable_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.is_file())
        .unwrap_or(false)
}

// ---- running the binary -----------------------------------------------------

/// Run `cli` with `args`, capture both streams, and give up after `timeout`.
///
/// The child is killed and reaped on expiry ([`BenchError::LlamaTimeout`]),
/// never left running. A non-zero exit is *not* an error here — callers decide
/// what a bad exit means (the version probe does, a load test does, a caller
/// that only wants the output does not).
pub fn run_once(cli: &Path, args: &[&str], timeout: Duration) -> Result<RunOutput, BenchError> {
    let mut guard = ChildGuard::spawn(cli, args)?;
    let started = Instant::now();

    // `None` means the deadline passed. Returning early drops `guard`, whose
    // `Drop` kills the child and waits for it — that is what keeps this from
    // leaking a process when a backend wedges.
    let Some(status) = guard.wait_for(timeout)? else {
        return Err(BenchError::LlamaTimeout {
            cli: cli.display().to_string(),
            args: args.join(" "),
            // Saturating: a caller passing an absurd timeout must not wrap the
            // reported deadline around into a small number.
            timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        });
    };

    let duration = started.elapsed();
    // Safe to join now: the child has exited, so both pipes are at EOF. Joining
    // before reading would deadlock, which is why each stream got its own
    // thread at spawn time.
    let (stdout, stderr) = guard.collect()?;
    Ok(RunOutput {
        stdout,
        stderr,
        exit_code: status.code(),
        duration,
    })
}

/// Owns a spawned child plus the threads draining its pipes.
///
/// The invariant: whenever a [`ChildGuard`] is dropped the child is *dead and
/// reaped*, and the output readers are given a bounded chance to finish. The
/// kill matters most — a killed-but-unwaited child stays a zombie on Unix —
/// and the bound matters next, because a surviving grandchild holding the pipe
/// open must never turn a timeout into a hang (see [`join_reader`]).
struct ChildGuard {
    child: Child,
    /// The path we spawned, kept only so `try_wait` failures can name it.
    cli: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
}

impl ChildGuard {
    /// Spawn `cli` with its stdout/stderr piped and drained concurrently.
    ///
    /// stdin is `null`: llama.cpp falls back to an interactive prompt when it
    /// has a terminal, and a bench run must never block on one.
    fn spawn(cli: &Path, args: &[&str]) -> Result<Self, BenchError> {
        let mut child = Command::new(cli)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| BenchError::LlamaSpawn {
                cli: cli.display().to_string(),
                source,
            })?;

        // Take both handles *before* anything else so no output can sit in a
        // full pipe buffer: `read_to_end` on one thread while the other drains
        // the other stream is what makes this deadlock-free.
        let stdout = child.stdout.take().map(spawn_reader);
        let stderr = child.stderr.take().map(spawn_reader);
        Ok(Self {
            child,
            cli: cli.display().to_string(),
            stdout,
            stderr,
        })
    }

    /// Poll until the child exits or `timeout` elapses. `Ok(None)` = timed out
    /// (the child is still alive; dropping this guard cleans it up).
    fn wait_for(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, BenchError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Ok(Some(status)),
                Ok(None) if Instant::now() >= deadline => return Ok(None),
                Ok(None) => std::thread::sleep(POLL_INTERVAL),
                Err(source) => {
                    return Err(BenchError::LlamaSpawn {
                        cli: self.child_exe(),
                        source,
                    })
                }
            }
        }
    }

    /// Join both reader threads and decode their output.
    fn collect(&mut self) -> Result<(String, String), BenchError> {
        // One deadline for both streams: two independent budgets would double
        // the worst-case wait, and the streams drain concurrently anyway.
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        Ok((self.join_stdout(deadline), self.join_stderr(deadline)))
    }

    fn join_stdout(&mut self, deadline: Instant) -> String {
        match self.stdout.take() {
            // A panicking reader must not take the process down; whatever was
            // read (possibly nothing) is better than a poisoned run.
            Some(handle) => decode(join_reader(handle, deadline)),
            None => String::new(),
        }
    }

    fn join_stderr(&mut self, deadline: Instant) -> String {
        match self.stderr.take() {
            Some(handle) => decode(join_reader(handle, deadline)),
            None => String::new(),
        }
    }

    /// The path we spawned, for error messages only.
    fn child_exe(&self) -> String {
        self.cli.clone()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        // Kill first, then wait: `wait` is what reaps the zombie, and running it
        // on a still-running child would block the destructor indefinitely.
        // Errors are ignored on purpose — `Drop` cannot report them, and both
        // calls are best-effort cleanup.
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Give the readers their bounded chance to drain (they hit EOF once the
        // last writer is gone). Both joins are side-effect free.
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        let _ = self.join_stdout(deadline);
        let _ = self.join_stderr(deadline);
    }
}

/// Drain one piped stream on its own thread.
fn spawn_reader<R>(mut pipe: R) -> JoinHandle<Vec<u8>>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        // An I/O error here (broken pipe when the child dies first) leaves
        // whatever was read so far — a partial capture is still useful.
        let _ = pipe.read_to_end(&mut buf);
        buf
    })
}

/// Join a reader thread until `deadline`, degrading to an empty capture.
///
/// The bound matters: a reader only reaches EOF once *every* process holding
/// the write end has closed it, and the child we killed can have spawned
/// children of its own that inherited those handles (`cmd /C ping` is the
/// classic case — killing `cmd.exe` leaves `ping.exe` writing for another
/// minute). Unbounded `join` would turn a killed child into a hung bench, so
/// at the deadline we detach instead: dropping the handle leaves the thread to
/// finish and die on its own once the last writer goes away, and this call
/// returns whatever it has.
///
/// Callers pass a *shared* deadline for both streams so the worst case is one
/// budget, not two. In the normal (child exited on its own) case the readers
/// hit EOF immediately, so this budget is never actually spent.
fn join_reader(handle: JoinHandle<Vec<u8>>, deadline: Instant) -> Vec<u8> {
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return Vec::new();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    // `is_finished` guarantees this cannot block, and a panic in the reader
    // leaves nothing to collect.
    handle.join().unwrap_or_default()
}

/// Lossy UTF-8 decode: llama.cpp output is UTF-8, but a byte or two from a
/// model file must not fail a whole benchmark run.
fn decode(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Last [`STDERR_TAIL_CHARS`] characters of a stream, for error messages.
/// Cut on a character boundary (never mid-UTF-8) and trailing whitespace is
/// dropped first so the message never ends in a blank line.
pub fn stderr_tail(stderr: &str) -> String {
    let trimmed = stderr.trim_end();
    let len = trimmed.chars().count();
    if len <= STDERR_TAIL_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().skip(len - STDERR_TAIL_CHARS).collect()
}

// ---- parsing llama.cpp output ----------------------------------------------

/// Pure: scrape the build number out of `llama-cli --version` output.
///
/// Tolerant by design — the banner wording changed between `b`-versions and
/// build flags reorder the surrounding chatter (`ggml_vulkan`, `load_backend`,
/// `system_info` lines all come first on recent builds). Returns the bare
/// number (`"3505"`), or `None` when nothing looks like a version, which the
/// caller reports as [`BenchError::LlamaVersionUnknown`] instead of guessing.
pub fn parse_version(text: &str) -> Option<String> {
    for line in text.lines() {
        // `to_ascii_lowercase` is byte-length preserving, so a match found in
        // the lowered copy indexes the original line safely.
        let lowered = line.to_ascii_lowercase();
        for marker in VERSION_MARKERS {
            let Some(idx) = lowered.find(marker) else {
                continue;
            };
            let rest = &line[idx + marker.len()..];
            let Some(token) = rest.split_whitespace().next() else {
                continue;
            };
            let token = token.trim_matches(|c: char| !is_version_char(c));
            if looks_like_version(token) {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn is_version_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+')
}

/// A version is any non-empty token containing at least one digit — enough to
/// reject `build =` with nothing after it without pinning a format.
fn looks_like_version(token: &str) -> bool {
    !token.is_empty() && token.chars().any(|c| c.is_ascii_digit())
}

/// Pure: scrape the `llama_perf` context block out of a run's stderr.
///
/// Requires both halves (prefill *and* decode): a block with only one of them
/// is a truncated capture, not a measurement, and reporting half a number would
/// be worse than reporting none. The `total time` line is skipped explicitly —
/// it sits between the two halves we want and would otherwise be picked up as
/// a third reading, inflating throughput by the prefill time.
pub fn parse_perf(stderr: &str) -> Option<PerfSummary> {
    let mut prompt: Option<(f64, u64)> = None;
    let mut eval: Option<(f64, u64)> = None;

    for line in stderr.lines() {
        let lowered = line.to_ascii_lowercase();
        if lowered.contains("total time") {
            continue;
        }
        // "prompt eval time" / "prompt processing time" name the prefill half.
        let is_prompt = lowered.contains("prompt eval") || lowered.contains("prompt processing");
        let is_eval = lowered.contains("eval time") || lowered.contains("generation time");
        if !is_prompt && !is_eval {
            continue;
        }
        let Some((ms, tokens)) = parse_ms_and_count(line) else {
            continue;
        };
        // First occurrence wins: llama.cpp prints one context block, but a
        // repeated line must not silently overwrite the measurement we took.
        if is_prompt {
            prompt.get_or_insert((ms, tokens));
        } else {
            eval.get_or_insert((ms, tokens));
        }
    }

    match (prompt, eval) {
        (Some((prompt_eval_ms, prompt_tokens)), Some((eval_ms, eval_tokens))) => {
            Some(PerfSummary {
                prompt_eval_ms,
                prompt_tokens,
                eval_ms,
                eval_tokens,
            })
        }
        // Half a block (or none): no summary rather than a misleading one.
        _ => None,
    }
}

/// Pure: `... = 1234.56 ms / 24 tokens ( ... )` → `(1234.56, 24)`.
///
/// Anchored on the `ms` unit and the `/` separator rather than on fixed column
/// positions, which is what lets the same code read the b1xxx
/// (`prompt eval time = 55.29 ms / 7 tokens`) and b5xxx
/// (`prompt eval time = 142.09 ms / 7 tokens ( 20.30 ms per token, …)`) shapes.
fn parse_ms_and_count(line: &str) -> Option<(f64, u64)> {
    let fields: Vec<&str> = line.split_whitespace().collect();

    // The duration is the number attached to the `ms` unit. llama.cpp prints
    // them as separate fields (`364.41 ms`), so accept both that and a joined
    // `364.41ms` from any vendor build that reformats the line.
    let mut ms = None;
    let mut after_unit = 0;
    for (idx, field) in fields.iter().enumerate() {
        let field = field.trim_end_matches(',');
        if field.ends_with("ms") {
            if let Ok(value) = field.trim_end_matches("ms").parse::<f64>() {
                ms = Some(value);
                after_unit = idx + 1;
                break;
            }
        }
        // The unit sat in its own field: the number is the one before it.
        if field == "ms" && idx > 0 {
            if let Ok(value) = fields[idx - 1].trim_end_matches(',').parse::<f64>() {
                ms = Some(value);
                after_unit = idx + 1;
                break;
            }
        }
    }
    let ms = ms?;

    // Then the first integer after the `/` that follows it: tokens, or runs on
    // builds that word the decode line differently.
    let tail = &fields[after_unit..];
    let slash = tail
        .iter()
        .position(|field| field.trim_end_matches(',') == "/")?;
    let count = tail[slash + 1..]
        .iter()
        .find_map(|field| field.trim_end_matches(',').parse::<u64>().ok())?;

    Some((ms, count))
}

/// Command line for the minimal load test: tiny prompt, `LOAD_PREDICT` tokens,
/// a small context, and optional thread pinning.
///
/// Kept to flags every build since the b2xxx era understands, on purpose: the
/// point is a portable smoke test, so newer conveniences (`-ngl`, `-no-cnv`)
/// are deliberately absent — an unknown flag would make the probe fail on an
/// older checkout instead of telling us anything useful.
pub fn load_probe_args(model: &Path, threads: Option<u32>) -> Vec<String> {
    let mut args = vec![
        "-m".to_string(),
        model.display().to_string(),
        "-p".to_string(),
        LOAD_PROMPT.to_string(),
        "-n".to_string(),
        LOAD_PREDICT.to_string(),
        "-c".to_string(),
        LOAD_CTX.to_string(),
    ];
    if let Some(threads) = threads {
        args.push("-t".to_string());
        args.push(threads.to_string());
    }
    args
}

// ---- high-level probes ------------------------------------------------------

/// Ask `cli` for its build number (`<cli> --version`, bounded by
/// [`VERSION_TIMEOUT`]).
///
/// `--version` has historically printed to stdout *and*, on some builds, to
/// stderr — both are scanned. A non-zero exit with no version in either stream
/// means the binary could not even start (missing shared library, wrong
/// architecture); that surfaces as [`BenchError::LlamaFailed`] with its stderr
/// rather than as a silent "unknown version".
pub fn llama_version(cli: &Path) -> Result<String, BenchError> {
    let output = run_once(cli, &["--version"], VERSION_TIMEOUT)?;
    if let Some(version) = parse_version(&output.stdout).or_else(|| parse_version(&output.stderr)) {
        return Ok(version);
    }
    if !output.success() {
        return Err(BenchError::LlamaFailed {
            cli: cli.display().to_string(),
            args: "--version".to_string(),
            code: output.exit_code,
            stderr: stderr_tail(&output.stderr),
        });
    }
    Err(BenchError::LlamaVersionUnknown {
        cli: cli.display().to_string(),
    })
}

/// Full end-to-end probe: version, then a 1-token load test of `model`.
///
/// This is what `attune bench --model` runs. It answers one question — "is
/// this machine set up well enough for llama.cpp to load a model?" — and
/// nothing more; timings are reported when llama.cpp gives them, not measured.
pub fn probe_model(
    cli: &Path,
    model: &Path,
    timeout: Duration,
    threads: Option<u32>,
) -> Result<LlamaProbeReport, BenchError> {
    // Check the model up front so a typo fails in milliseconds rather than
    // after the version probe.
    if !is_runnable_file(model) {
        return Err(BenchError::InvalidConfig(format!(
            "model `{}` is not a readable file",
            model.display()
        )));
    }

    let version = llama_version(cli)?;
    let args = load_probe_args(model, threads);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = run_once(cli, &arg_refs, timeout)?;
    if !output.success() {
        return Err(BenchError::LlamaFailed {
            cli: cli.display().to_string(),
            args: args.join(" "),
            code: output.exit_code,
            stderr: stderr_tail(&output.stderr),
        });
    }

    Ok(LlamaProbeReport {
        cli: cli.display().to_string(),
        version,
        model: model.display().to_string(),
        load_ok: true,
        elapsed_ms: output.duration.as_secs_f64() * 1_000.0,
        perf: parse_perf(&output.stderr),
    })
}

/// Render a list of names as `` `a`, `b` `` for error messages.
fn quoted_list<'a, I: IntoIterator<Item = &'a str>>(names: I) -> String {
    names
        .into_iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- locating the binary (pure) -----------------------------------------

    fn hit(name: &str) -> (String, PathBuf) {
        (
            name.to_string(),
            PathBuf::from(format!("/opt/llama/{name}")),
        )
    }

    #[test]
    fn explicit_flag_beats_env_and_path() {
        let hits = vec![hit("llama-cli"), hit("main")];
        let (source, path) =
            select_llama_cli(Some("/flag/llama-cli"), Some("/env/llama-cli"), &hits)
                .expect("explicit flag should resolve");
        assert_eq!(source, LlamaSource::Flag);
        assert_eq!(path, PathBuf::from("/flag/llama-cli"));
    }

    #[test]
    fn env_beats_path_when_no_flag_is_given() {
        let hits = vec![hit("llama-cli"), hit("main")];
        let (source, path) =
            select_llama_cli(None, Some("/env/llama-cli"), &hits).expect("env should resolve");
        assert_eq!(source, LlamaSource::Env);
        assert_eq!(path, PathBuf::from("/env/llama-cli"));
    }

    #[test]
    fn path_is_the_last_resort_and_prefers_the_modern_binary_name() {
        let (source, path) = select_llama_cli(None, None, &[hit("llama-cli"), hit("main")])
            .expect("PATH should resolve");
        assert_eq!(
            source,
            LlamaSource::Path {
                name: "llama-cli".to_string()
            }
        );
        assert_eq!(path, PathBuf::from("/opt/llama/llama-cli"));

        // Only the legacy name present (old llama.cpp build tree).
        let (source, path) =
            select_llama_cli(None, None, &[hit("main")]).expect("legacy name should resolve");
        assert_eq!(
            source,
            LlamaSource::Path {
                name: "main".to_string()
            }
        );
        assert_eq!(path, PathBuf::from("/opt/llama/main"));
    }

    #[test]
    fn blank_values_are_treated_as_unset() {
        // Exporting an empty variable is a common accident and must not shadow
        // a perfectly good PATH entry.
        let hits = vec![hit("llama-cli")];
        let (source, _) = select_llama_cli(Some("   "), Some(""), &hits).expect("PATH should win");
        assert_eq!(
            source,
            LlamaSource::Path {
                name: "llama-cli".to_string()
            }
        );
    }

    #[test]
    fn nothing_found_is_a_clean_not_found_error() {
        let err = select_llama_cli(None, None, &[]).expect_err("nothing should be found");
        let message = err.to_string();
        assert!(matches!(err, BenchError::LlamaNotFound { .. }));
        assert!(
            message.contains("llama-cli") && message.contains("main"),
            "message should name the searched binaries: {message}"
        );
        assert!(
            message.contains(ENV_LLAMA_CLI),
            "message should name the env var: {message}"
        );
    }

    #[test]
    fn a_missing_explicit_path_is_reported_with_its_origin() {
        // The flag wins the precedence chain, then fails the existence check —
        // the error must name *which* level misfired, not just "not found".
        let cli = PathBuf::from("definitely-not-a-real-llama-cli-path");
        let (source, chosen) =
            select_llama_cli(cli.to_str(), None, &[]).expect("precedence picks the flag");
        assert_eq!(source, LlamaSource::Flag);
        assert!(!is_runnable_file(&chosen));
        assert!(!is_runnable_file(Path::new("/no/such/dir/llama-cli.exe")));
    }

    #[test]
    fn resolve_never_panics_even_with_a_bogus_explicit_path() {
        // End-to-end failure mode, and the one the CLI turns into a clean
        // message. Must be an error, never a panic.
        let err = resolve_llama_cli(Some("definitely-not-a-real-llama-cli-path"))
            .expect_err("bogus path should not resolve");
        let message = err.to_string();
        assert!(
            message.starts_with("llama-cli not found"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("--llama-cli") && message.contains(ENV_LLAMA_CLI),
            "message should tell the user how to fix it: {message}"
        );
    }

    #[test]
    fn source_descriptions_name_the_origin() {
        assert_eq!(LlamaSource::Flag.describe(), "`--llama-cli` path");
        assert_eq!(
            LlamaSource::Env.describe(),
            format!("`{ENV_LLAMA_CLI}` path")
        );
        assert_eq!(
            LlamaSource::Path {
                name: "main".to_string()
            }
            .describe(),
            "`main` on PATH"
        );
    }

    #[test]
    fn executable_variants_are_non_empty_on_every_platform() {
        for name in PATH_CANDIDATES {
            let variants = executable_variants(name);
            assert!(!variants.is_empty(), "{name} produced no candidates");
            assert!(
                variants.iter().any(|v| v.starts_with(name)),
                "{name} variants {variants:?} should all start with the name"
            );
        }
    }

    // ---- version parsing (pure) --------------------------------------------

    #[test]
    fn parse_version_reads_a_b3505_banner() {
        // b3505 (Nov 2023) and everything after it: `version: <build> (<sha>)`.
        assert_eq!(
            parse_version("version: 3505 (d09f13b4)\n"),
            Some("3505".to_string())
        );
        assert_eq!(
            parse_version("version: 5967 (a4ce208d)\n"),
            Some("5967".to_string())
        );
    }

    #[test]
    fn parse_version_reads_a_pre_b3000_banner() {
        // Older builds printed `main: build = <build> (<sha>)`.
        assert_eq!(
            parse_version("main: build = 4639 (a3cf7c6)\n"),
            Some("4639".to_string())
        );
        assert_eq!(
            parse_version("main: build = 1286 (4b0e6e1)\r\n"),
            Some("1286".to_string())
        );
    }

    #[test]
    fn parse_version_skips_chatter_from_recent_builds() {
        // b5xxx prints backend chatter before the version; the parser must
        // not latch onto one of those numbers.
        let stdout = "\
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 4070 Laptop GPU, ...
load_backend: loaded RPC backend from ./rpc-server-0:50056
system_info: n_threads = 8 / 8 | AVX = 1 | ...
version: 5967 (a4ce208d)
";
        assert_eq!(parse_version(stdout), Some("5967".to_string()));
    }

    #[test]
    fn parse_version_tolerates_unfamiliar_punctuation() {
        // Vendor builds do not always match upstream's shape.
        assert_eq!(
            parse_version("version: b3645-g7f9c2ab\n"),
            Some("b3645-g7f9c2ab".to_string())
        );
        assert_eq!(
            parse_version("llama-cli version = 4121 (deadbeef), built with GCC 13\n"),
            Some("4121".to_string())
        );
    }

    #[test]
    fn parse_version_returns_none_for_output_without_one() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("\n\n"), None);
        assert_eq!(
            parse_version("llama-cli: unrecognized option '--version'\n"),
            None
        );
        assert_eq!(parse_version("built with GCC 15.1.0 (GCC)\n"), None);
        // Marker present but the value is not a number: keep scanning, and end
        // up with no version rather than a bogus one.
        assert_eq!(parse_version("version:\nversion = none\n"), None);
    }

    // ---- perf parsing (pure) -----------------------------------------------

    #[test]
    fn parse_perf_reads_a_b4041_context_block() {
        let stderr = "\
llama_perf_context_print: prompt eval time =   364.41 ms /    7 tokens (    52.06 ms per token,    19.21 tokens per second)
llama_perf_context_print:        eval time =   517.44 ms /    6 runs   (    86.24 ms per token,    11.60 tokens per second)
llama_perf_context_print:       total time =   945.05 ms.
";
        let perf = parse_perf(stderr).expect("b4041 block should parse");
        assert!((perf.prompt_eval_ms - 364.41).abs() < 1e-9);
        assert_eq!(perf.prompt_tokens, 7);
        assert!((perf.eval_ms - 517.44).abs() < 1e-9);
        assert_eq!(perf.eval_tokens, 6);
        // Total time (945.05) must never be mistaken for either half.
        assert!(perf.eval_ms < 945.0);
    }

    #[test]
    fn parse_perf_reads_an_early_llama_perf_block() {
        // b1xxx-era wording, with no per-token breakdown at all.
        let stderr = "\
llama_print_timing: prompt eval time =    55.29 ms /    7 tokens
llama_print_timing:        eval time =    95.66 ms /    4 runs
llama_print_timing:      total time =   152.08 ms
";
        let perf = parse_perf(stderr).expect("early block should parse");
        assert!((perf.prompt_eval_ms - 55.29).abs() < 1e-9);
        assert_eq!(perf.prompt_tokens, 7);
        assert!((perf.eval_ms - 95.66).abs() < 1e-9);
        assert_eq!(perf.eval_tokens, 4);
    }

    #[test]
    fn parse_perf_reads_a_b5967_block_with_extra_chatter() {
        let stderr = "\
load_tensors: offloaded 21/21 layers to GPU
llama_perf_sampler_print:    sampling order = [ 0, 1, 2, 3 ] ...
llama_perf_context_print: prompt eval time =   142.09 ms /    7 tokens (    20.30 ms per token,    49.26 tokens per second)
llama_perf_context_print:        eval time =   352.15 ms /    6 runs   (    58.69 ms per token,    17.04 tokens per second)
llama_perf_context_print:       total time =   554.30 ms.
";
        let perf = parse_perf(stderr).expect("b5967 block should parse");
        assert_eq!(perf.prompt_tokens, 7);
        assert_eq!(perf.eval_tokens, 6);
        assert!((perf.prompt_tps() - 49.26).abs() < 0.01);
        assert!((perf.eval_tps() - 17.04).abs() < 0.01);
    }

    #[test]
    fn parse_perf_returns_none_for_a_missing_or_half_block() {
        assert_eq!(parse_perf(""), None);
        assert_eq!(parse_perf("main: loading model from 'x.gguf'...\n"), None);
        // Prefill only (run cut short): reporting half a measurement would be
        // worse than reporting none.
        let prompt_only = "llama_perf_context_print: prompt eval time = 12.00 ms / 3 tokens\n";
        assert_eq!(parse_perf(prompt_only), None);
        // Total-time line alone must not be read as decode.
        let total_only = "llama_perf_context_print:       total time =   945.05 ms.\n";
        assert_eq!(parse_perf(total_only), None);
    }

    #[test]
    fn parse_perf_keeps_the_first_block_when_one_is_repeated() {
        let stderr = "\
llama_perf_context_print: prompt eval time =   100.00 ms /    4 tokens ( ... )
llama_perf_context_print:        eval time =   200.00 ms /    2 runs   ( ... )
llama_perf_context_print:       total time =   300.00 ms.
llama_perf_context_print: prompt eval time =   999.00 ms /   99 tokens ( ... )
llama_perf_context_print:        eval time =   999.00 ms /   99 runs   ( ... )
";
        let perf = parse_perf(stderr).expect("first block should parse");
        assert!((perf.prompt_eval_ms - 100.00).abs() < 1e-9);
        assert_eq!(perf.prompt_tokens, 4);
        assert!((perf.eval_ms - 200.00).abs() < 1e-9);
        assert_eq!(perf.eval_tokens, 2);
    }

    #[test]
    fn perf_tps_guards_against_zero_division() {
        let perf = PerfSummary {
            prompt_eval_ms: 0.0,
            prompt_tokens: 0,
            eval_ms: 10.0,
            eval_tokens: 5,
        };
        assert_eq!(perf.prompt_tps(), 0.0);
        assert!((perf.eval_tps() - 500.0).abs() < 1e-9);
    }

    // ---- stderr tail --------------------------------------------------------

    #[test]
    fn stderr_tail_keeps_the_end_and_trims_trailing_newlines() {
        assert_eq!(stderr_tail("boom\n"), "boom");
        assert_eq!(stderr_tail("short"), "short");
        assert_eq!(stderr_tail("   \n\n"), "");
        let long = "x".repeat(STDERR_TAIL_CHARS * 2);
        let tail = stderr_tail(&long);
        assert_eq!(tail.chars().count(), STDERR_TAIL_CHARS);
        // Multi-byte input must be cut on a character boundary, never mid-UTF-8.
        let unicode = "\u{1F600}".repeat(STDERR_TAIL_CHARS * 2);
        let tail = stderr_tail(&unicode);
        assert_eq!(tail.chars().count(), STDERR_TAIL_CHARS);
        assert!(unicode.ends_with(&tail));
    }

    // ---- child processes ----------------------------------------------------

    /// Cross-platform child that does nothing for `seconds` and then exits 0.
    ///
    /// Both arms are gated (the PR #4 lesson): an ungated arm would be dead
    /// code on the other platform and trip `-D warnings` in CI.
    #[cfg(target_os = "windows")]
    fn sleeper(seconds: u32) -> (String, Vec<String>) {
        // There is no `/bin/sleep` on Windows and `timeout.exe` refuses to run
        // without a console, so ping the loopback interface instead: it waits
        // ~1 s per count and always exists.
        (
            "cmd".to_string(),
            vec![
                "/C".to_string(),
                format!("ping -n {} 127.0.0.1 >NUL", seconds + 1),
            ],
        )
    }

    /// POSIX twin of the Windows arm above.
    #[cfg(not(target_os = "windows"))]
    fn sleeper(seconds: u32) -> (String, Vec<String>) {
        ("/bin/sleep".to_string(), vec![seconds.to_string()])
    }

    /// Cross-platform child that prints `text` on stdout and exits 0.
    #[cfg(target_os = "windows")]
    fn echo(text: &str) -> (String, Vec<String>) {
        (
            "cmd".to_string(),
            vec!["/C".to_string(), format!("echo {text}")],
        )
    }

    /// POSIX twin of the Windows arm above.
    #[cfg(not(target_os = "windows"))]
    fn echo(text: &str) -> (String, Vec<String>) {
        ("/bin/echo".to_string(), vec![text.to_string()])
    }

    /// Run one of the helpers above through [`run_once`], borrowing its args.
    fn run_helper(
        helper: (String, Vec<String>),
        timeout: Duration,
    ) -> Result<RunOutput, BenchError> {
        let (exe, args) = helper;
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        run_once(Path::new(&exe), &arg_refs, timeout)
    }

    #[test]
    fn run_once_kills_a_child_that_overruns_its_timeout() {
        let started = Instant::now();
        let err = run_helper(sleeper(60), Duration::from_millis(250))
            .expect_err("a 60 s child must not finish inside a 250 ms timeout");
        let elapsed = started.elapsed();

        match err {
            BenchError::LlamaTimeout { timeout_ms, .. } => assert_eq!(timeout_ms, 250),
            other => panic!("expected LlamaTimeout, got {other:?}"),
        }
        // Generous headroom on purpose (the PR #7 lesson): a loaded CI runner
        // can stall a timer for a long time, but the child is *killed*, so this
        // can never grow towards the child's own 60 s.
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout path took {elapsed:?} — did the kill actually fire?"
        );
    }

    #[test]
    fn run_once_captures_stdout_and_the_exit_code() {
        // Generous timeout: this child finishes in milliseconds, and the only
        // thing under test is that we wait, decode and report it cleanly.
        let output = run_helper(echo("attune-ok"), Duration::from_secs(30))
            .expect("a quick echo should succeed");
        assert!(output.success(), "echo exited with {:?}", output.exit_code);
        assert_eq!(output.exit_code, Some(0));
        assert!(
            output.stdout.contains("attune-ok"),
            "stdout was {:?}",
            output.stdout
        );
    }

    #[test]
    fn run_once_reports_a_missing_binary_as_an_error_not_a_panic() {
        let err = run_once(
            Path::new("definitely-not-a-real-llama-cli-path"),
            &["--version"],
            Duration::from_secs(5),
        )
        .expect_err("spawning a missing binary must fail");
        assert!(matches!(err, BenchError::LlamaSpawn { .. }), "{err:?}");
    }

    #[test]
    fn run_output_treats_a_signalled_child_as_failure() {
        let signalled = RunOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            duration: Duration::ZERO,
        };
        assert!(!signalled.success());

        let failed = RunOutput {
            exit_code: Some(1),
            ..signalled
        };
        assert!(!failed.success());
    }

    // ---- arg building -------------------------------------------------------

    #[test]
    fn load_probe_args_ask_for_one_token_and_a_small_context() {
        let args = load_probe_args(Path::new("/models/tiny.gguf"), None);
        let joined = args.join(" ");
        assert!(joined.contains("-m /models/tiny.gguf"), "{joined}");
        assert!(joined.contains(&format!("-n {LOAD_PREDICT}")), "{joined}");
        assert!(joined.contains(&format!("-c {LOAD_CTX}")), "{joined}");
        assert!(joined.contains(&format!("-p {LOAD_PROMPT}")), "{joined}");
        assert!(!joined.contains("-t "), "no thread pin asked for: {joined}");

        let pinned = load_probe_args(Path::new("/models/tiny.gguf"), Some(6));
        assert!(pinned.join(" ").contains("-t 6"), "{:?}", pinned);
    }

    #[test]
    fn probe_model_rejects_a_missing_model_before_spawning_anything() {
        // No llama-cli needed: the model check happens first, so this is safe
        // to run in CI.
        let err = probe_model(
            Path::new("definitely-not-a-real-llama-cli-path"),
            Path::new("definitely-not-a-real-model.gguf"),
            Duration::from_millis(250),
            None,
        )
        .expect_err("a missing model must be rejected");
        assert!(matches!(err, BenchError::InvalidConfig(_)), "{err:?}");
    }

    // ---- JSON contract ------------------------------------------------------

    #[test]
    fn probe_report_serializes_and_deserializes() {
        let report = LlamaProbeReport {
            cli: "/usr/bin/llama-cli".to_string(),
            version: "5967".to_string(),
            model: "/models/tiny.gguf".to_string(),
            load_ok: true,
            elapsed_ms: 1234.5,
            perf: Some(PerfSummary {
                prompt_eval_ms: 142.09,
                prompt_tokens: 7,
                eval_ms: 352.15,
                eval_tokens: 6,
            }),
        };
        let json = serde_json::to_string(&report).expect("serialize probe report");
        for key in ["cli", "version", "model", "load_ok", "elapsed_ms", "perf"] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        let back: LlamaProbeReport = serde_json::from_str(&json).expect("deserialize probe report");
        assert_eq!(back, report);
    }

    #[test]
    fn probe_report_deserializes_without_the_optional_perf_block() {
        let json = r#"{
            "cli": "llama-cli",
            "version": "3505",
            "model": "tiny.gguf",
            "load_ok": true,
            "elapsed_ms": 12.0
        }"#;
        let report: LlamaProbeReport =
            serde_json::from_str(json).expect("deserialize probe report");
        assert_eq!(report.perf, None);
    }

    // ---- manual, real llama.cpp only ---------------------------------------
    //
    // The two tests below need a real llama.cpp build AND a GGUF model, so
    // they are `#[ignore]`d (never run by `cargo test` in CI) *and* env-gated
    // (never run on a dev box that has one but no model). Run them with:
    //
    //     ATTUNE_LLAMA_CLI=/path/to/llama-cli ATTUNE_TEST_MODEL=/path/to/tiny.gguf \
    //         cargo test -p attune-bench -- --ignored

    #[test]
    #[ignore = "manual: needs ATTUNE_LLAMA_CLI + ATTUNE_TEST_MODEL and a real llama.cpp build"]
    fn real_llama_cli_reports_a_version() {
        let cli = match resolve_llama_cli(std::env::var(ENV_LLAMA_CLI).ok().as_deref()) {
            Ok(cli) => cli,
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let version = llama_version(&cli).expect("real llama-cli should report a version");
        assert!(
            looks_like_version(&version),
            "implausible version {version:?}"
        );
    }

    #[test]
    #[ignore = "manual: needs ATTUNE_LLAMA_CLI + ATTUNE_TEST_MODEL and a real llama.cpp build"]
    fn real_llama_cli_loads_a_model() {
        let Some(model) = std::env::var_os(ENV_TEST_MODEL) else {
            eprintln!("skipping: {ENV_TEST_MODEL} not set");
            return;
        };
        let model = PathBuf::from(model);
        let cli = resolve_llama_cli(std::env::var(ENV_LLAMA_CLI).ok().as_deref())
            .expect("a real llama-cli must resolve");
        let report =
            probe_model(&cli, &model, LOAD_TIMEOUT, None).expect("model load should succeed");

        assert!(report.load_ok);
        assert_eq!(report.model, model.display().to_string());
        assert!(report.elapsed_ms > 0.0, "load took no measurable time");
        if let Some(perf) = report.perf {
            assert!(perf.eval_tokens > 0, "decode timed zero tokens");
            assert!(perf.eval_tps() > 0.0, "decode throughput was zero");
        }
        println!("{} ({}): {report:?}", report.cli, report.version);
    }
}
