//! llama.cpp `llama-server` subprocess adapter (M5 slice 1).
//!
//! This module deliberately duplicates the small process helpers from
//! `attune-bench::llama` instead of depending on that crate: importing the
//! benchmark stack into the serving path would pull in `attune-hal` and every
//! measurement kernel for the sake of ~100 lines (see the crate-level docs).
//! The duplication is kept close to the original (resolver precedence,
//! `PATHEXT`-aware variants, per-OS `is_runnable_file`, `ChildGuard`-style
//! `Drop`) so keeping them in sync is mechanical.
//!
//! Responsibilities:
//! - **Find it** - [`resolve_llama_server`] applies
//!   `--llama-server` > `ATTUNE_LLAMA_SERVER` > `PATH` search.
//! - **Build argv** - [`server_args`] reuses the profile mapper and adds
//!   `--host`/`--port` plus the CPU-affinity flag.
//! - **Run it** - [`ServerGuard`] owns the child and kills + reaps it on
//!   `Drop`; [`health_wait`] polls `GET /health` with a deadline.
//!
//! Test policy matches `attune-bench`: the resolver/argv/body parsers are pure
//! unit tests; the real-server integration test is `#[ignore]`d *and* gated on
//! `ATTUNE_LLAMA_SERVER` + `ATTUNE_TEST_MODEL`, so CI never runs it.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use attune_profile::{to_llama_args, Profile};

use crate::BackendError;

/// Environment variable holding an explicit `llama-server` path. Second in the
/// precedence chain, after the `--llama-server` flag.
pub const ENV_LLAMA_SERVER: &str = "ATTUNE_LLAMA_SERVER";

/// Environment variable pointing at a GGUF model. Read **only** by the manual,
/// `#[ignore]`d integration test - never by production code.
pub const ENV_TEST_MODEL: &str = "ATTUNE_TEST_MODEL";

/// Binary names looked up on `PATH`. `llama-server` is the only supported name
/// (there is no legacy alias for the server).
pub const PATH_CANDIDATES: &[&str] = &["llama-server"];

/// Polling period for both the process and the health wait. `Child` has no
/// portable blocking-wait-with-timeout; 10 ms is negligible next to a model
/// load and keeps a kill responsive.
pub const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long to wait for the child's output readers after it is gone. A
/// grandchild holding the pipe open must never hang shutdown.
const READER_JOIN_BUDGET: Duration = Duration::from_secs(2);

/// Health polling period. `llama-server` loads a model for seconds to minutes;
/// 200 ms gives a sub-second "ready" reaction without hammering the socket.
pub const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Per-request timeout for one health probe. Bounded so a wedged socket cannot
/// consume the whole caller deadline in a single call.
pub const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Number of trailing body characters kept in an error message.
const BODY_TAIL_CHARS: usize = 300;

/// `--numa` values accepted by llama-server. The M4 profile schema defaults to
/// `"local"`, which is *not* one of them; llama.cpp already runs on the local
/// node when `--numa` is omitted, so that sentinel is translated to omission.
const VALID_NUMA_POLICIES: &[&str] = &["distribute", "isolate", "numactl"];

/// Profile sentinel for "no NUMA tuning" (llama.cpp's default).
const NUMA_LOCAL: &str = "local";

/// Where a resolved `llama-server` came from. Reported in errors so the user
/// knows which precedence level misfired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaServerSource {
    /// The explicit `--llama-server` flag.
    Flag,
    /// The `ATTUNE_LLAMA_SERVER` environment variable.
    Env,
    /// Found on `PATH` under this binary name.
    Path {
        /// The name that matched, i.e. an entry of [`PATH_CANDIDATES`].
        name: String,
    },
}

impl LlamaServerSource {
    /// Human-readable origin, used inside error messages.
    pub fn describe(&self) -> String {
        match self {
            LlamaServerSource::Flag => "`--llama-server` path".to_string(),
            LlamaServerSource::Env => format!("`{ENV_LLAMA_SERVER}` path"),
            LlamaServerSource::Path { name } => format!("`{name}` on PATH"),
        }
    }
}

/// A built argv plus the warnings produced while building it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerArgs {
    /// Arguments after the executable name.
    pub args: Vec<String>,
    /// Non-fatal problems, e.g. an affinity value we could not map.
    pub warnings: Vec<String>,
}

// ---- locating the binary ----------------------------------------------------

/// Pure: apply the documented precedence over candidates that were *already*
/// gathered (no filesystem access here).
///
/// `path_hits` must be ordered the way [`search_path`] produced it. A blank
/// explicit/env path is treated as unset, because exporting an empty variable
/// is a common accident. Existence is checked by [`resolve_llama_server`] after
/// this function picks a winner.
pub fn select_llama_server(
    explicit: Option<&Path>,
    env_value: Option<&Path>,
    path_hits: &[(String, PathBuf)],
) -> Result<(LlamaServerSource, PathBuf), BackendError> {
    if let Some(path) = usable_path(explicit) {
        return Ok((LlamaServerSource::Flag, path.to_path_buf()));
    }
    if let Some(path) = usable_path(env_value) {
        return Ok((LlamaServerSource::Env, path.to_path_buf()));
    }
    if let Some((name, path)) = path_hits.first() {
        return Ok((LlamaServerSource::Path { name: name.clone() }, path.clone()));
    }
    Err(BackendError::NotFound {
        detail: format!(
            "searched PATH for {}",
            quoted_list(PATH_CANDIDATES.iter().copied())
        ),
    })
}

/// Locate `llama-server`, honouring the full precedence chain and verifying
/// that the winner is actually a runnable file.
pub fn resolve_llama_server(explicit: Option<&Path>) -> Result<PathBuf, BackendError> {
    let env_value = std::env::var_os(ENV_LLAMA_SERVER).map(PathBuf::from);
    let path_hits = search_path();
    let (source, chosen) = select_llama_server(explicit, env_value.as_deref(), &path_hits)?;
    if !is_runnable_file(&chosen) {
        return Err(BackendError::NotFound {
            detail: format!(
                "{} `{}` is not a runnable file",
                source.describe(),
                chosen.display()
            ),
        });
    }
    Ok(chosen)
}

/// A path is "usable" when it is present and not just whitespace.
fn usable_path(path: Option<&Path>) -> Option<&Path> {
    path.filter(|path| !path.as_os_str().is_empty() && !path.to_string_lossy().trim().is_empty())
}

/// Walk `PATH` looking for [`PATH_CANDIDATES`].
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
/// Duplicated from `attune_bench::llama::is_runnable_file` (see the crate
/// docs). Unix also checks the executable bit so a directory or a readable but
/// non-executable file in `PATH` never wins the search.
#[cfg(unix)]
pub fn is_runnable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

/// Windows has no executable bit - existence as a regular file is enough.
///
/// Duplicated from `attune_bench::llama::is_runnable_file` (see the crate docs).
#[cfg(not(unix))]
pub fn is_runnable_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.is_file())
        .unwrap_or(false)
}

// ---- building the argv ------------------------------------------------------

/// Map a profile to the full `llama-server` command line.
///
/// Starts from [`attune_profile::to_llama_args`] (model, context, batch,
/// threads, KV-cache types, NUMA, flash attention, mmap/mlock, GPU layers),
/// then appends the server-only `--host`/`--port` and the CPU-affinity flag.
/// The stored `cpu_affinity` is classified here because only the launcher knows
/// whether it is a mask or a range.
///
/// Two M4 values are translated for the current server before they reach the
/// child (see [`normalize_numa`] and [`normalize_flash_attn`]): the schema
/// default `numa: "local"` becomes "omit `--numa`", and a bare `-fa` becomes
/// `--flash-attn on` so its optional-value parser cannot swallow the next flag.
pub fn server_args(profile: &Profile, host: &str, port: u16) -> Vec<String> {
    server_args_with_warnings(profile, host, port).args
}

/// Same as [`server_args`], but also returns the non-fatal mapping warnings
/// (currently: an affinity value that is neither a hex mask nor a range).
pub fn server_args_with_warnings(profile: &Profile, host: &str, port: u16) -> ServerArgs {
    let mut args = to_llama_args(profile);
    let mut warnings = Vec::new();

    // The M4 mapper emits `--numa <policy>` unconditionally, but "local" (the
    // schema default) is not a valid llama-server value. Normalize before the
    // argv reaches the child.
    if let Some(warning) = normalize_numa(&mut args, &profile.load_config.numa) {
        warnings.push(warning);
    }

    // Current llama-server takes `--flash-attn [on|off|auto]`. A bare `-fa`
    // makes its optional-value parser swallow the next flag (e.g. `-ngl`) and
    // fail, so always spell the value out.
    normalize_flash_attn(&mut args);

    args.push("--host".to_string());
    args.push(host.to_string());
    args.push("--port".to_string());
    args.push(port.to_string());

    let (affinity_args, warning) =
        cpu_affinity_args(profile.compute_config.cpu_affinity.as_deref());
    args.extend(affinity_args);
    if let Some(warning) = warning {
        warnings.push(warning);
    }

    ServerArgs { args, warnings }
}

/// Rewrite the mapper's bare `-fa` into `--flash-attn on`.
///
/// llama.cpp historically accepted `-fa` as a boolean switch, but the current
/// server parses an *optional* value: `-fa -ngl 0` consumes `-ngl` as the
/// value and aborts. Spelling the value out keeps the rest of the argv intact.
fn normalize_flash_attn(args: &mut Vec<String>) {
    if let Some(index) = args.iter().position(|arg| arg == "-fa") {
        args[index] = "--flash-attn".to_string();
        args.insert(index + 1, "on".to_string());
    }
}

/// Pure: make a stored NUMA policy acceptable to `llama-server`.
///
/// * `local` (the M4 default) or blank -> omit `--numa` (llama.cpp already runs
///   on the local node by default).
/// * `distribute` / `isolate` / `numactl` -> keep, lowercased.
/// * anything else -> omit and return a warning.
///
/// Returns `Some(warning)` only for a genuinely unknown policy.
fn normalize_numa(args: &mut Vec<String>, policy: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == "--numa")?;
    if index + 1 >= args.len() {
        // Defensive: a dangling flag would make the child reject the whole
        // command line. Drop it and say so.
        args.remove(index);
        return Some("dropped a malformed `--numa` flag without a value".to_string());
    }

    let normalized = policy.trim().to_ascii_lowercase();
    if normalized == NUMA_LOCAL || normalized.is_empty() {
        args.drain(index..index + 2);
        return None;
    }
    if VALID_NUMA_POLICIES.contains(&normalized.as_str()) {
        args[index + 1] = normalized;
        return None;
    }

    args.drain(index..index + 2);
    Some(format!(
        "ignoring unsupported numa policy `{policy}`; omitting --numa (valid: {NUMA_LOCAL}, {})",
        VALID_NUMA_POLICIES.join(", ")
    ))
}

/// Pure: turn a stored `compute_config.cpu_affinity` into llama.cpp flags.
///
/// * `0xff` / `ff` / `0x0F` (hex digits) -> `--cpu-mask <value>`
/// * `0-7` / `2-5` (numeric range)        -> `--cpu-range <value>`
/// * anything else                        -> no flag + a warning string
///
/// A `None` or blank affinity is not a warning: the profile simply did not
/// record pinning.
pub fn cpu_affinity_args(affinity: Option<&str>) -> (Vec<String>, Option<String>) {
    let Some(raw) = affinity.map(str::trim).filter(|value| !value.is_empty()) else {
        return (Vec::new(), None);
    };

    match classify_cpu_affinity(raw) {
        AffinityArg::Mask(mask) => (vec!["--cpu-mask".to_string(), mask], None),
        AffinityArg::Range(range) => (vec!["--cpu-range".to_string(), range], None),
        AffinityArg::Skipped => (
            Vec::new(),
            Some(format!(
                "ignoring unsupported cpu_affinity `{raw}` (expected a hex mask like `0xff` or a \
                 range like `0-7`)"
            )),
        ),
    }
}

/// Classification of one affinity string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AffinityArg {
    Mask(String),
    Range(String),
    Skipped,
}

fn classify_cpu_affinity(raw: &str) -> AffinityArg {
    if is_cpu_range(raw) {
        return AffinityArg::Range(raw.to_string());
    }
    if is_hex_mask(raw) {
        return AffinityArg::Mask(raw.to_string());
    }
    AffinityArg::Skipped
}

/// `start-end`, both decimal, with `start <= end`.
fn is_cpu_range(raw: &str) -> bool {
    let Some((start, end)) = raw.split_once('-') else {
        return false;
    };
    if start.is_empty() || end.is_empty() || end.contains('-') {
        return false;
    }
    if !start.chars().all(|c| c.is_ascii_digit()) || !end.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    match (start.parse::<u32>(), end.parse::<u32>()) {
        (Ok(start), Ok(end)) => start <= end,
        _ => false,
    }
}

/// Optional `0x`/`0X` prefix followed by at least one hex digit.
fn is_hex_mask(raw: &str) -> bool {
    let digits = raw
        .strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .unwrap_or(raw);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_hexdigit())
}

/// Render a list of names as `` `a`, `b` `` for error messages.
fn quoted_list<'a, I: IntoIterator<Item = &'a str>>(names: I) -> String {
    names
        .into_iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- process guard ----------------------------------------------------------

/// Owns a spawned `llama-server` plus the threads draining its pipes.
///
/// The invariant: whenever a [`ServerGuard`] is dropped the child is *dead and
/// reaped*, and the output readers get a bounded chance to finish. This is what
/// makes `attune serve` killable without leaving a zombie.
pub struct ServerGuard {
    child: Child,
    /// The path we spawned, kept so errors can name it.
    binary: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    /// Non-fatal argv warnings, surfaced by the launcher.
    warnings: Vec<String>,
    reaped: bool,
}

impl ServerGuard {
    /// Spawn `binary` with `args`, inheriting the current directory.
    pub fn spawn(binary: &Path, args: &[String]) -> Result<Self, BackendError> {
        Self::spawn_in(binary, args, None, Vec::new())
    }

    /// Spawn with an optional working directory and argv warnings.
    pub(crate) fn spawn_in(
        binary: &Path,
        args: &[String],
        working_dir: Option<&Path>,
        warnings: Vec<String>,
    ) -> Result<Self, BackendError> {
        let mut command = Command::new(binary);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = working_dir {
            command.current_dir(dir);
        }

        let mut child = command.spawn().map_err(|source| BackendError::Spawn {
            binary: binary.display().to_string(),
            source,
        })?;

        // Take both handles before anything else so no output can sit in a
        // full pipe buffer while we wait for health.
        let stdout = child.stdout.take().map(spawn_reader);
        let stderr = child.stderr.take().map(spawn_reader);
        Ok(Self {
            child,
            binary: binary.display().to_string(),
            stdout,
            stderr,
            warnings,
            reaped: false,
        })
    }

    /// The OS-level child pid.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Non-fatal warnings produced while building this child's argv.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Non-blocking exit poll. `Ok(None)` means the child is still running.
    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>, BackendError> {
        self.child.try_wait().map_err(|source| BackendError::Spawn {
            binary: self.binary.clone(),
            source,
        })
    }

    /// Kill, reap and join the output readers. Idempotent; also called by
    /// `Drop`, so callers that never call it explicitly still clean up.
    pub fn shutdown(&mut self) {
        if self.reaped {
            return;
        }
        // Kill first, then wait: `wait` reaps the zombie, and running it on a
        // still-running child would block indefinitely. Both are best-effort:
        // `Drop`/shutdown cannot report an error.
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Bounded: a surviving grandchild holding the pipe open must never
        // turn shutdown into a hang.
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        let _ = join_reader(self.stdout.take(), deadline);
        let _ = join_reader(self.stderr.take(), deadline);
        self.reaped = true;
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Drain one piped stream on its own thread.
fn spawn_reader<R>(mut pipe: R) -> JoinHandle<Vec<u8>>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        // A broken pipe (child dies first) leaves whatever was read so far.
        let _ = pipe.read_to_end(&mut buf);
        buf
    })
}

/// Join a reader thread until `deadline`, detaching it afterwards.
///
/// A reader only reaches EOF once *every* process holding the write end has
/// closed it, and a killed child can leave grandchildren writing. Unbounded
/// `join` would hang shutdown, so at the deadline we drop the handle and let
/// the thread die on its own.
fn join_reader(handle: Option<JoinHandle<Vec<u8>>>, deadline: Instant) -> Vec<u8> {
    let Some(handle) = handle else {
        return Vec::new();
    };
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return Vec::new();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    // `is_finished` guarantees `join` cannot block.
    handle.join().unwrap_or_default()
}

// ---- health check -----------------------------------------------------------

/// Pure: decide whether a `/health` response means the model is loaded.
///
/// `llama-server` answers `200 {"status":"ok"}` when ready and
/// `503 {"error":...}` while loading. The body wording has drifted across
/// builds, so this accepts any 2xx, treats an empty body as healthy (some
/// builds answer `200` with no payload), honours an explicit `"status":"ok"`,
/// and rejects any body that carries an `error`.
pub fn health_is_ok(status_code: u16, body: &str) -> bool {
    if !(200..300).contains(&status_code) {
        return false;
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lowered = trimmed.to_ascii_lowercase();
    if lowered.contains("\"status\"") {
        return lowered.contains("\"ok\"");
    }
    !lowered.contains("\"error\"")
}

/// Poll `url` until it reports healthy or `deadline` elapses.
///
/// Returns how long the wait took on success. A non-2xx response or a transport
/// error is retried until the deadline (the server is usually still loading);
/// a 2xx whose body explicitly reports an error fails fast as
/// [`BackendError::HealthFailed`].
pub fn health_wait(url: &str, deadline: Duration) -> Result<Duration, BackendError> {
    let started = Instant::now();
    let deadline_at = started + deadline;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(HEALTH_REQUEST_TIMEOUT))
        .build()
        .into();

    let mut last_detail: String;
    loop {
        match agent.get(url).call() {
            Ok(mut response) => {
                let status = response.status().as_u16();
                match response.body_mut().read_to_string() {
                    Ok(body) if (200..300).contains(&status) => {
                        if health_is_ok(status, &body) {
                            return Ok(started.elapsed());
                        }
                        // 2xx but an explicit unhealthy body: the endpoint is
                        // up, so retrying the same answer cannot help.
                        return Err(BackendError::HealthFailed {
                            url: url.to_string(),
                            detail: tail(&body, BODY_TAIL_CHARS),
                        });
                    }
                    Ok(body) => {
                        last_detail = format!("HTTP {status}: {}", tail(&body, BODY_TAIL_CHARS));
                    }
                    Err(error) => {
                        last_detail = format!("HTTP {status}: body read failed: {error}");
                    }
                }
            }
            Err(error) => last_detail = error.to_string(),
        }

        if Instant::now() >= deadline_at {
            return Err(BackendError::Timeout {
                url: url.to_string(),
                timeout_ms: u64::try_from(deadline.as_millis()).unwrap_or(u64::MAX),
                detail: last_detail,
            });
        }
        std::thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

/// Last `max_chars` characters of `text`, trimmed, cut on a char boundary.
fn tail(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    let count = trimmed.chars().count();
    if count <= max_chars {
        return trimmed.to_string();
    }
    trimmed.chars().skip(count - max_chars).collect()
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use attune_profile::{ComputeConfig, LoadConfig, ModelConfig};

    fn hit(name: &str) -> (String, PathBuf) {
        (
            name.to_string(),
            PathBuf::from(format!("/opt/llama/{name}")),
        )
    }

    fn profile() -> Profile {
        Profile {
            model: ModelConfig {
                path: "models/tiny.gguf".to_string(),
                ..ModelConfig::default()
            },
            load_config: LoadConfig {
                n_ctx: 512,
                ..LoadConfig::default()
            },
            compute_config: ComputeConfig {
                n_threads: 2,
                n_threads_batch: 2,
                cpu_affinity: None,
            },
            ..Profile::default()
        }
    }

    /// True when `flag value` appears consecutively in `args`.
    fn has_pair(args: &[String], flag: &str, value: &str) -> bool {
        args.windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    // ---- resolver (pure) ----------------------------------------------------

    #[test]
    fn explicit_flag_beats_env_and_path() {
        let hits = vec![hit("llama-server")];
        let (source, path) = select_llama_server(
            Some(Path::new("/flag/llama-server")),
            Some(Path::new("/env/llama-server")),
            &hits,
        )
        .expect("explicit flag should resolve");
        assert_eq!(source, LlamaServerSource::Flag);
        assert_eq!(path, PathBuf::from("/flag/llama-server"));
    }

    #[test]
    fn env_beats_path_when_no_flag_is_given() {
        let hits = vec![hit("llama-server")];
        let (source, path) = select_llama_server(None, Some(Path::new("/env/llama-server")), &hits)
            .expect("env should resolve");
        assert_eq!(source, LlamaServerSource::Env);
        assert_eq!(path, PathBuf::from("/env/llama-server"));
    }

    #[test]
    fn path_is_the_last_resort() {
        let (source, path) =
            select_llama_server(None, None, &[hit("llama-server")]).expect("PATH should resolve");
        assert_eq!(
            source,
            LlamaServerSource::Path {
                name: "llama-server".to_string()
            }
        );
        assert_eq!(path, PathBuf::from("/opt/llama/llama-server"));
    }

    #[test]
    fn blank_values_are_treated_as_unset() {
        let hits = vec![hit("llama-server")];
        let (source, _) = select_llama_server(Some(Path::new("   ")), Some(Path::new("")), &hits)
            .expect("PATH should win");
        assert_eq!(
            source,
            LlamaServerSource::Path {
                name: "llama-server".to_string()
            }
        );
    }

    #[test]
    fn nothing_found_is_a_clean_not_found_error() {
        let error = select_llama_server(None, None, &[]).expect_err("nothing should be found");
        let message = error.to_string();
        assert!(matches!(error, BackendError::NotFound { .. }));
        assert!(
            message.contains("llama-server") && message.contains(ENV_LLAMA_SERVER),
            "message should name the binary and the env var: {message}"
        );
    }

    #[test]
    fn a_missing_explicit_path_is_reported_with_its_origin() {
        let chosen = PathBuf::from("definitely-not-a-real-llama-server-path");
        let (source, selected) =
            select_llama_server(Some(&chosen), None, &[]).expect("precedence picks the flag");
        assert_eq!(source, LlamaServerSource::Flag);
        assert!(!is_runnable_file(&selected));
        assert!(!is_runnable_file(Path::new("/no/such/dir/llama-server")));
    }

    #[test]
    fn resolve_never_panics_even_with_a_bogus_explicit_path() {
        let error =
            resolve_llama_server(Some(Path::new("definitely-not-a-real-llama-server-path")))
                .expect_err("bogus path should not resolve");
        let message = error.to_string();
        assert!(matches!(error, BackendError::NotFound { .. }));
        assert!(
            message.contains("--llama-server") && message.contains(ENV_LLAMA_SERVER),
            "message should tell the user how to fix it: {message}"
        );
    }

    #[test]
    fn source_descriptions_name_the_origin() {
        assert_eq!(LlamaServerSource::Flag.describe(), "`--llama-server` path");
        assert_eq!(
            LlamaServerSource::Env.describe(),
            format!("`{ENV_LLAMA_SERVER}` path")
        );
        assert_eq!(
            LlamaServerSource::Path {
                name: "llama-server".to_string()
            }
            .describe(),
            "`llama-server` on PATH"
        );
    }

    #[test]
    fn executable_variants_are_non_empty_and_start_with_the_name() {
        for name in PATH_CANDIDATES {
            let variants = executable_variants(name);
            assert!(!variants.is_empty(), "{name} produced no candidates");
            assert!(
                variants.iter().all(|variant| variant.starts_with(name)),
                "{name} variants {variants:?} should all start with the name"
            );
        }
    }

    #[test]
    fn is_runnable_file_accepts_a_file_and_rejects_a_directory() {
        let dir = std::env::temp_dir();
        let file = dir.join(format!(
            "attune-backend-runnable-test-{}",
            std::process::id()
        ));
        std::fs::write(&file, b"x").expect("write temp file");
        // Unix checks the executable bit, so make the temp file executable;
        // Windows only checks that it exists as a regular file.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = std::fs::metadata(&file)
                .expect("stat temp file")
                .permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&file, permissions).expect("chmod temp file");
        }
        assert!(is_runnable_file(&file));
        assert!(!is_runnable_file(&dir));
        let _ = std::fs::remove_file(&file);
    }

    // ---- argv ---------------------------------------------------------------

    #[test]
    fn server_args_extend_the_profile_mapper_with_host_and_port() {
        let args = server_args(&profile(), "127.0.0.1", 8081);
        assert!(has_pair(&args, "-m", "models/tiny.gguf"));
        assert!(has_pair(&args, "-c", "512"));
        assert!(has_pair(&args, "--host", "127.0.0.1"));
        assert!(has_pair(&args, "--port", "8081"));
        // The M4 default `numa: "local"` is not a llama-server value; it must
        // be translated to "no --numa flag" rather than passed through.
        assert!(!args.iter().any(|arg| arg == "--numa"));
        // Bare `-fa` would swallow the next flag; the value must be explicit.
        assert!(has_pair(&args, "--flash-attn", "on"));
        assert!(!args.iter().any(|arg| arg == "-fa"));
        // No affinity recorded -> neither flag appears.
        assert!(!args.iter().any(|arg| arg == "--cpu-mask"));
        assert!(!args.iter().any(|arg| arg == "--cpu-range"));
    }

    #[test]
    fn disabled_flash_attention_omits_the_flag() {
        let mut profile = profile();
        profile.load_config.flash_attn = false;
        let args = server_args(&profile, "127.0.0.1", 8081);
        assert!(!args.iter().any(|arg| arg == "-fa"));
        assert!(!args.iter().any(|arg| arg == "--flash-attn"));
    }

    #[test]
    fn numa_local_and_blank_are_omitted_silently() {
        let mut profile = profile();
        assert_eq!(profile.load_config.numa, "local");
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(!prepared.args.iter().any(|arg| arg == "--numa"));
        assert!(prepared.warnings.is_empty());

        profile.load_config.numa = "  ".to_string();
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(!prepared.args.iter().any(|arg| arg == "--numa"));
        assert!(prepared.warnings.is_empty());
    }

    #[test]
    fn valid_numa_policies_are_lowercased_and_kept() {
        let mut profile = profile();
        for policy in ["distribute", "isolate", "numactl"] {
            profile.load_config.numa = policy.to_string();
            let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
            assert!(has_pair(&prepared.args, "--numa", policy));
            assert!(prepared.warnings.is_empty());
        }
        profile.load_config.numa = " DISTRIBUTE ".to_string();
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(has_pair(&prepared.args, "--numa", "distribute"));
        assert!(prepared.warnings.is_empty());
    }

    #[test]
    fn unknown_numa_policy_is_omitted_with_a_warning() {
        let mut profile = profile();
        profile.load_config.numa = "bogus".to_string();
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(!prepared.args.iter().any(|arg| arg == "--numa"));
        assert_eq!(prepared.warnings.len(), 1);
        assert!(
            prepared.warnings[0].contains("bogus"),
            "{:?}",
            prepared.warnings
        );
    }

    #[test]
    fn hex_affinity_maps_to_cpu_mask() {
        let (args, warning) = cpu_affinity_args(Some("0xff"));
        assert_eq!(args, vec!["--cpu-mask", "0xff"]);
        assert_eq!(warning, None);

        let (args, warning) = cpu_affinity_args(Some("0x0F"));
        assert_eq!(args, vec!["--cpu-mask", "0x0F"]);
        assert_eq!(warning, None);

        // A bare hex string is a mask too.
        let (args, warning) = cpu_affinity_args(Some("ff"));
        assert_eq!(args, vec!["--cpu-mask", "ff"]);
        assert_eq!(warning, None);
    }

    #[test]
    fn range_affinity_maps_to_cpu_range() {
        let (args, warning) = cpu_affinity_args(Some("0-7"));
        assert_eq!(args, vec!["--cpu-range", "0-7"]);
        assert_eq!(warning, None);

        let (args, warning) = cpu_affinity_args(Some(" 2-5 "));
        assert_eq!(args, vec!["--cpu-range", "2-5"]);
        assert_eq!(warning, None);
    }

    #[test]
    fn unparsable_affinity_is_skipped_with_a_warning() {
        let (args, warning) = cpu_affinity_args(Some("all-the-cores"));
        assert!(args.is_empty());
        let warning = warning.expect("unparsable affinity must warn");
        assert!(
            warning.contains("all-the-cores"),
            "warning should quote the value: {warning}"
        );

        // An inverted range is not a valid pin.
        let (args, warning) = cpu_affinity_args(Some("9-2"));
        assert!(args.is_empty());
        assert!(warning.is_some());
    }

    #[test]
    fn absent_or_blank_affinity_is_silent() {
        assert_eq!(cpu_affinity_args(None), (Vec::new(), None));
        assert_eq!(cpu_affinity_args(Some("   ")), (Vec::new(), None));
    }

    #[test]
    fn server_args_carry_an_affinity_warning() {
        let mut profile = profile();
        profile.compute_config.cpu_affinity = Some("nonsense".to_string());
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(!prepared.args.iter().any(|arg| arg == "--cpu-mask"));
        assert_eq!(prepared.warnings.len(), 1);

        profile.compute_config.cpu_affinity = Some("0-3".to_string());
        let prepared = server_args_with_warnings(&profile, "127.0.0.1", 8081);
        assert!(has_pair(&prepared.args, "--cpu-range", "0-3"));
        assert!(prepared.warnings.is_empty());
    }

    // ---- health body parsing ------------------------------------------------

    #[test]
    fn health_body_parses_the_ready_shape() {
        assert!(health_is_ok(200, r#"{"status":"ok"}"#));
        assert!(health_is_ok(200, r#"{"status": "OK"}"#));
        assert!(health_is_ok(200, ""));
        assert!(health_is_ok(200, "ok"));
        assert!(health_is_ok(204, ""));
    }

    #[test]
    fn health_body_rejects_loading_and_error_shapes() {
        assert!(!health_is_ok(200, r#"{"status":"loading"}"#));
        assert!(!health_is_ok(
            200,
            r#"{"error":{"code":503,"message":"Loading model"}}"#
        ));
        assert!(!health_is_ok(
            503,
            r#"{"error":{"code":503,"message":"Loading model"}}"#
        ));
        assert!(!health_is_ok(404, ""));
    }

    #[test]
    fn tail_cuts_on_a_char_boundary() {
        let text = "\u{1F600}".repeat(10);
        let cut = tail(&text, 3);
        assert_eq!(cut.chars().count(), 3);
        assert!(text.ends_with(&cut));
    }

    // ---- process guard ------------------------------------------------------

    /// Cross-platform child that waits ~`seconds` and exits.
    #[cfg(target_os = "windows")]
    fn sleeper(seconds: u32) -> (String, Vec<String>) {
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

    fn spawn_sleeper(seconds: u32) -> ServerGuard {
        let (exe, args) = sleeper(seconds);
        ServerGuard::spawn(Path::new(&exe), &args).expect("spawn helper")
    }

    #[test]
    fn shutdown_kills_and_reaps_a_live_child() {
        let mut guard = spawn_sleeper(60);
        assert!(
            guard.try_wait().expect("poll child").is_none(),
            "the helper should still be running"
        );
        guard.shutdown();
        assert!(
            guard.try_wait().expect("poll child").is_some(),
            "shutdown must reap the child"
        );
        // Idempotent: a second call must not panic or hang.
        guard.shutdown();
    }

    #[test]
    fn drop_kills_the_child_promptly() {
        let started = Instant::now();
        {
            let _guard = spawn_sleeper(60);
        }
        // Generous headroom: a loaded CI runner can stall a timer, but the
        // child is killed, so this can never approach its own 60 s.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "Drop should kill the child instead of waiting for it"
        );
    }
}
