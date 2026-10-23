//! End-to-end llama.cpp decode benchmark (M2 slice 6).
//!
//! This module drives the small e2e block in `attune bench --model`: build a
//! portable llama.cpp command line, spawn the child, sample its resource
//! usage while it runs, parse the `llama_perf` block it printed, and fold
//! the numbers into an [`E2eResult`] shaped for the human report.
//!
//! Three principles, repeated from earlier slices:
//!
//! 1. **Old-build-safe flags only.** The CLI line is
//!    `-m -p -n -c [-st] [-v] [-t]`. Every flag here is one the b2xxx-b6xxx
//!    series understands — `-st` and `-v` both predate the chat rewrite — so
//!    the same `E2eConfig` works against any reasonably recent llama.cpp
//!    checkout the user might have built. New conveniences (`-ngl`, `-no-cnv`, …)
//!    would make the bench fail on an older build and report nothing useful.
//!    `-st` keeps the child from sitting in its interactive REPL and `-v` is
//!    what makes it print the per-slot timing block [`parse_perf`] reads (the
//!    non-verbose summary line has no `ms / tokens` pair to parse).
//! 2. **Generous deadline.** A real run loads a model (mmap + warm-up of
//!    every layer), processes the prefill, then decodes. On a slow box that
//!    is minutes; [`E2E_TIMEOUT`] is sized for that, so the user can ride out
//!    a cold cache. A wedged backend is still killed and reaped
//!    (`BenchError::LlamaTimeout`) — generous does not mean `forever`.
//! 3. **Pure math, then I/O.** The TTFT / TPOT / tok-s derivations from a
//!    [`PerfSummary`](crate::llama::PerfSummary) are *separate pure functions*
//!    unit-tested on canned input. `run_e2e` does the I/O; the maths stay
//!    testable without a real llama.cpp.
//!
//! **Monitoring choice.** The monitor points at the *child* pid
//! ([`SysinfoSampler::with_pid`](crate::monitor::SysinfoSampler::with_pid)),
//! because the parent (`attune`) is just waiting on the child and its own
//! CPU/RAM tells us nothing about the inference. We *own* the child on every
//! target platform (Linux: `/proc/<pid>/stat` is readable; Windows:
//! `Process::memory()` works for any child), so the sampler always has
//! something to read. When the pid vanishes (the child exited before the
//! first tick) the sampler degrades to zero readings rather than failing the
//! bench.
//!
//! No `.unwrap()` in non-test code (AGENTS.md §3.2) — spawn / wait / parse
//! failures all surface as [`BenchError`].

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::{sleep, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::llama::{self, parse_perf, stderr_tail, LlamaProbeReport, PerfSummary};
use crate::monitor::{Monitor, SysinfoSampler, DEFAULT_INTERVAL_MS};
use crate::BenchError;

// ---- config & defaults ----------------------------------------------------

/// Default prompt length in tokens. The prompt is built to that target, but
/// the *actual* prompt token count comes from llama.cpp's own `llama_perf`
/// block — this is only the size knob.
pub const DEFAULT_PROMPT_TOKENS: usize = 512;
/// Default decode length in tokens. A small-interaction figure.
pub const DEFAULT_GEN_TOKENS: u32 = 128;
/// Default context window. Big enough to hold the default prompt + decode
/// with a comfortable margin; smaller boxes should pass a lower value.
pub const DEFAULT_CTX_SIZE: u32 = 2048;
/// Generous deadline for an e2e run: model load + prefill + decode. Sized
/// for cold-cache loads; a wedged backend still times out and reaps cleanly.
pub const E2E_TIMEOUT: Duration = Duration::from_secs(600);
/// Polling interval for the wait-with-deadline loop. Cheap next to a
/// multi-second child; keeps the kill path responsive.
pub const POLL_INTERVAL: Duration = Duration::from_millis(10);
/// How long `collect` waits for both reader threads to drain after the
/// child exits. Comfortably longer than a dying process needs to flush its
/// pipe, short enough that a wedged grandchild cannot hang the bench.
const READER_JOIN_BUDGET: Duration = Duration::from_secs(2);
/// Approximate characters per English token, used only to seed the prompt
/// builder. The *real* token count comes from llama.cpp's own counter.
const APPROX_CHARS_PER_TOKEN: usize = 4;
/// Safety margin left in the context window for KV-cache bookkeeping
/// beyond the prompt + decode tokens billed to the user.
const CTX_SAFETY_MARGIN_TOKENS: u32 = 32;
/// Phrase the prompt is repeated from. Deliberately plain prose — domain
/// jargon would skew llama.cpp's tokeniser and make the prompt shorter or
/// longer than the target token count.
const PROMPT_PHRASE: &str = "The quick brown fox jumps over the lazy dog. ";

/// What `attune bench --model` measures. All fields have safe defaults; the
/// CLI exposes them as flags but never has to pass them all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct E2eConfig {
    /// GGUF model to load.
    pub model: PathBuf,
    /// Approximate number of prefill tokens. `run_e2e` clamps the prompt
    /// so this target fits inside `ctx_size - gen_tokens - safety`.
    pub prompt_tokens: usize,
    /// Number of tokens to generate (the `-n` value).
    pub gen_tokens: u32,
    /// Thread pin. `0` = let llama.cpp pick (no `-t` flag is emitted).
    pub threads: u32,
    /// Context window (`-c`).
    pub ctx_size: u32,
}

impl E2eConfig {
    /// Defaults for the four knobs, with the model picked from the CLI flag.
    pub fn new(model: PathBuf) -> Self {
        Self {
            model,
            prompt_tokens: DEFAULT_PROMPT_TOKENS,
            gen_tokens: DEFAULT_GEN_TOKENS,
            threads: 0,
            ctx_size: DEFAULT_CTX_SIZE,
        }
    }

    /// Clamped prompt tokens: 1 or above, never so large the prompt would
    /// crowd the context window. The cap is `(ctx - gen) - margin`, so the
    /// prompt + generated tokens always leave room for the KV cache plus
    /// llama.cpp's own bookkeeping.
    pub fn effective_prompt_tokens(&self) -> usize {
        let cap = self
            .ctx_size
            .saturating_sub(self.gen_tokens)
            .saturating_sub(CTX_SAFETY_MARGIN_TOKENS) as usize;
        self.prompt_tokens.max(1).min(cap.max(1))
    }

    /// Clamped context size: at least 32 (anything smaller would make
    /// llama.cpp itself refuse to start).
    pub fn ctx_size(&self) -> u32 {
        self.ctx_size.max(32)
    }

    /// Clamped generation length: at least 1 token, else `parse_perf`
    /// would report zero decode tokens and downstream tok-s would be 0.
    pub fn gen_tokens(&self) -> u32 {
        self.gen_tokens.max(1)
    }

    /// Prompt length actually fed to llama.cpp, in characters. Used by
    /// [`build_e2e_args`] to cap the repeating phrase.
    pub fn max_prompt_chars(&self) -> usize {
        self.effective_prompt_tokens()
            .saturating_mul(APPROX_CHARS_PER_TOKEN)
    }
}

// ---- result ---------------------------------------------------------------

/// End-to-end decode measurement. JSON contract for the `e2e` block of
/// `attune bench --model --json`. Every field is the report from llama.cpp
/// itself or derived from those numbers by [`ttft_ms_from`], [`tpot_ms_from`],
/// [`tok_s_from`]. `prompt_tokens` / `eval_tokens` reflect what llama.cpp
/// *actually* processed, not what was requested — they usually differ
/// slightly from the E2eConfig targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct E2eResult {
    /// Build number of the llama.cpp binary that ran (from `--version`).
    pub engine: String,
    /// Path of the model that ran.
    pub model: String,
    /// Number of prefill tokens llama.cpp *actually* processed.
    pub prompt_tokens: u64,
    /// Number of decode tokens llama.cpp *actually* generated. `0` when
    /// the perf block was missing or half-formed.
    pub eval_tokens: u64,
    /// Time-to-first-token, in milliseconds — approximated by the prefill
    /// wall clock. `0.0` when llama.cpp never printed a perf block.
    pub ttft_ms: f64,
    /// Time per output token, in milliseconds. `0.0` when no decode
    /// tokens were timed (run cut short / output missing).
    pub tpot_ms: f64,
    /// Decode throughput in tokens per second. `0.0` when tpot is zero or
    /// non-finite.
    pub tok_s: f64,
    /// Highest resident-set reading across the run, in bytes.
    pub ram_peak_bytes: u64,
    /// Mean of the per-tick CPU readings across the run.
    pub cpu_avg_pct: f64,
    /// Mean of the per-tick watt readings. `None` when the platform has
    /// no power counter or the run was too short to produce one delta.
    #[serde(default)]
    pub power_avg_watts: Option<f64>,
    /// Wall-clock time of the whole llama.cpp run (load + prefill + decode),
    /// in milliseconds. `0.0` in reports written before this field existed.
    #[serde(default)]
    pub elapsed_ms: f64,
}

impl E2eResult {
    /// True only when the perf block carried decode tokens — i.e. the run
    /// produced a meaningful decode measurement rather than just a load.
    pub fn is_complete(&self) -> bool {
        self.tpot_ms > 0.0 && self.tok_s > 0.0
    }

    /// The `llama` block for a full bench: the e2e run loaded the model, so
    /// it stands in for the separate load probe. `elapsed_ms` covers the
    /// whole run and `perf` is rebuilt from the e2e timings.
    pub fn load_report(&self, cli: &Path) -> LlamaProbeReport {
        let perf = (self.ttft_ms > 0.0 || self.eval_tokens > 0).then_some(PerfSummary {
            prompt_eval_ms: self.ttft_ms,
            prompt_tokens: self.prompt_tokens,
            eval_ms: self.tpot_ms * self.eval_tokens as f64,
            eval_tokens: self.eval_tokens,
        });
        LlamaProbeReport {
            cli: cli.display().to_string(),
            version: self.engine.clone(),
            model: self.model.clone(),
            load_ok: true,
            elapsed_ms: self.elapsed_ms,
            perf,
        }
    }
}

// ---- pure derivations from PerfSummary ------------------------------------

/// Time-to-first-token in milliseconds, approximated by the prefill wall
/// clock. Pure for testability.
pub fn ttft_ms_from(perf: &PerfSummary) -> f64 {
    if perf.prompt_eval_ms.is_finite() && perf.prompt_eval_ms > 0.0 {
        perf.prompt_eval_ms
    } else {
        0.0
    }
}

/// Time per output (decode) token in milliseconds. `0.0` when no decode
/// tokens were timed — a missing or half-formed perf block, never NaN.
pub fn tpot_ms_from(perf: &PerfSummary) -> f64 {
    if perf.eval_tokens == 0 || perf.eval_ms <= 0.0 || !perf.eval_ms.is_finite() {
        return 0.0;
    }
    let tpot = perf.eval_ms / perf.eval_tokens as f64;
    if tpot.is_finite() && tpot > 0.0 {
        tpot
    } else {
        0.0
    }
}

/// Decode throughput in tokens per second. `0.0` when tpot cannot be
/// computed — never divides by zero, never returns NaN/Inf.
pub fn tok_s_from(perf: &PerfSummary) -> f64 {
    let tpot = tpot_ms_from(perf);
    if tpot <= 0.0 {
        return 0.0;
    }
    let rate = 1_000.0 / tpot;
    if rate.is_finite() {
        rate
    } else {
        0.0
    }
}

// ---- arg / prompt builders -----------------------------------------------

/// Build a prompt string approximately `target_tokens` tokens long, capped
/// at `max_chars` characters. The actual token count comes from llama.cpp
/// itself, so this only needs to be in the right ballpark — we are seeding
/// it for that tokeniser, not measuring it ourselves.
pub fn build_prompt(target_tokens: usize, max_chars: usize) -> String {
    let target = target_tokens.saturating_mul(APPROX_CHARS_PER_TOKEN);
    let cap = max_chars.max(64); // always allow at least one phrase
    let target = target.min(cap);
    if target == 0 {
        return String::new();
    }
    let mut out = String::new();
    while out.len() < target {
        out.push_str(PROMPT_PHRASE);
    }
    if out.len() > cap {
        out.truncate(cap);
    }
    out
}

/// Build the llama.cpp command line for an e2e run. Only old-build-safe
/// flags (`-m`, `-p`, `-n`, `-c`, [`llama::COMMON_ARGS`], optional `-t`);
/// newer conveniences would make the bench fail on older llama.cpp checkouts
/// instead of telling us anything useful.
///
/// [`llama::COMMON_ARGS`] carries `-st` and `-v`, both of which predate the
/// chat rewrite: `-st` makes the child answer the prompt and exit instead of
/// opening its interactive REPL, `-v` makes it print the timing block
/// [`parse_perf`] reads instead of the compact `[ Prompt: … | Generation: … ]`
/// summary, which carries no parseable timings.
pub fn build_e2e_args(cfg: &E2eConfig) -> Vec<String> {
    let prompt = build_prompt(cfg.effective_prompt_tokens(), cfg.max_prompt_chars());
    let mut args = vec![
        "-m".to_string(),
        cfg.model.display().to_string(),
        "-p".to_string(),
        prompt,
        "-n".to_string(),
        cfg.gen_tokens().to_string(),
        "-c".to_string(),
        cfg.ctx_size().to_string(),
    ];
    args.extend(llama::COMMON_ARGS.iter().map(|flag| flag.to_string()));
    if cfg.threads > 0 {
        args.push("-t".to_string());
        args.push(cfg.threads.to_string());
    }
    args
}

// ---- child orchestration --------------------------------------------------

/// Millisecond-cheap checks an e2e run needs before anything slow starts:
/// the model is a readable file and `cli` reports a parseable version.
/// Returns the engine version. Callers that run other work first (the full
/// bench runs its kernels before the e2e run) call this up front so a bad
/// setup still fails fast.
pub fn preflight(cli: &Path, model: &Path) -> Result<String, BenchError> {
    if !llama::is_readable_file(model) {
        return Err(BenchError::InvalidConfig(format!(
            "model `{}` is not a readable file",
            model.display()
        )));
    }
    llama::llama_version(cli)
}

/// Run an end-to-end llama.cpp decode bench against `cli`. Spawns the
/// child, attaches the resource monitor to its pid, scrapes the perf block,
/// and folds the result with the resource readings into one [`E2eResult`].
pub fn run_e2e(cli: &Path, cfg: &E2eConfig) -> Result<E2eResult, BenchError> {
    // Failures here surface as `BenchError`, never as a panic.
    let engine = preflight(cli, &cfg.model)?;

    let args = build_e2e_args(cfg);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    // Spawn first so we have the child's pid before the monitor starts.
    let started = Instant::now();
    let mut guard = E2eGuard::spawn(cli, &arg_refs)?;
    let child_pid = guard.pid();

    let monitor = Monitor::start_with_interval(
        Box::new(SysinfoSampler::with_pid(sysinfo::Pid::from_u32(child_pid))),
        DEFAULT_INTERVAL_MS,
    )?;

    let deadline_result = guard.wait_for(E2E_TIMEOUT);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    // Always stop the monitor — dropping `monitor` would also stop it, but
    // we want the readings *now*, before printing the timeout error.
    let resources = monitor.stop();

    let status = match deadline_result {
        Ok(Some(status)) => status,
        Ok(None) => {
            return Err(BenchError::LlamaTimeout {
                cli: cli.display().to_string(),
                args: args.join(" "),
                timeout_ms: u64::try_from(E2E_TIMEOUT.as_millis()).unwrap_or(u64::MAX),
            });
        }
        Err(err) => return Err(err),
    };

    let (_stdout, stderr) = guard.collect()?;
    if !status.success() {
        return Err(BenchError::LlamaFailed {
            cli: cli.display().to_string(),
            args: args.join(" "),
            code: status.code(),
            stderr: stderr_tail(&stderr),
        });
    }

    let perf = parse_perf(&stderr);
    let (prompt_tokens, eval_tokens, ttft_ms, tpot_ms, tok_s) = match perf.as_ref() {
        Some(perf) => (
            perf.prompt_tokens,
            perf.eval_tokens,
            ttft_ms_from(perf),
            tpot_ms_from(perf),
            tok_s_from(perf),
        ),
        None => (cfg.effective_prompt_tokens() as u64, 0, 0.0, 0.0, 0.0),
    };

    Ok(E2eResult {
        engine,
        model: cfg.model.display().to_string(),
        prompt_tokens,
        eval_tokens,
        ttft_ms,
        tpot_ms,
        tok_s,
        ram_peak_bytes: resources.ram_peak_bytes,
        cpu_avg_pct: resources.cpu_avg_pct,
        power_avg_watts: resources.power_avg_watts,
        elapsed_ms,
    })
}

// ---- internal child guard (sibling of `llama::ChildGuard`) ---------------

/// RAII wrapper around a llama.cpp child plus its drained pipes.
///
/// Mirrors `llama::ChildGuard` so the e2e pipeline can have its own
/// `pid()` accessor without exposing `llama.rs`'s private type. Both
/// pieces of duplication are intentional: the dead-code-on-the-other-platform
/// hazard from PR #4 is easier to spot when each call site owns its guard.
struct E2eGuard {
    child: Child,
    cli: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
}

impl E2eGuard {
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
        let stdout = child.stdout.take().map(spawn_reader);
        let stderr = child.stderr.take().map(spawn_reader);
        Ok(Self {
            child,
            cli: cli.display().to_string(),
            stdout,
            stderr,
        })
    }

    /// The OS-level child pid — the very thing the resource monitor needs.
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Poll until the child exits or `timeout` elapses. `Ok(None)` = timed
    /// out (the child is still alive; dropping this guard cleans it up).
    fn wait_for(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, BenchError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Ok(Some(status)),
                Ok(None) if Instant::now() >= deadline => return Ok(None),
                Ok(None) => sleep(POLL_INTERVAL),
                Err(source) => {
                    return Err(BenchError::LlamaSpawn {
                        cli: self.cli.clone(),
                        source,
                    })
                }
            }
        }
    }

    /// Join the reader threads and decode whatever they captured.
    fn collect(&mut self) -> Result<(String, String), BenchError> {
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        let stdout = join_reader_or(self.stdout.take(), deadline);
        let stderr = join_reader_or(self.stderr.take(), deadline);
        Ok((decode(stdout), decode(stderr)))
    }
}

impl Drop for E2eGuard {
    fn drop(&mut self) {
        // Kill first, then wait, then read. `wait` is what reaps the
        // zombie; running it on a still-alive child would block the
        // destructor indefinitely. Errors are silently swallowed — `Drop`
        // can't report them and every step is best-effort cleanup.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        let _ = join_reader_or(self.stdout.take(), deadline);
        let _ = join_reader_or(self.stderr.take(), deadline);
    }
}

/// Drain one piped stream on its own thread.
fn spawn_reader<R>(mut pipe: R) -> JoinHandle<Vec<u8>>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        // A broken pipe (child dies first) leaves whatever was read so
        // far — a partial capture is still useful.
        let _ = pipe.read_to_end(&mut buf);
        buf
    })
}

/// Join a reader thread until `deadline`, returning whatever it produced
/// on the way (empty bytes if the deadline elapses first).
fn join_reader_or(handle: Option<JoinHandle<Vec<u8>>>, deadline: Instant) -> Vec<u8> {
    let Some(handle) = handle else {
        return Vec::new();
    };
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            // The reader is detached past this point. Detaching leaves
            // the thread to die on its own once the pipe finally closes,
            // and we return an empty capture rather than hang the bench.
            return Vec::new();
        }
        sleep(POLL_INTERVAL);
    }
    handle.join().unwrap_or_default()
}

/// Lossy UTF-8 decode. llama.cpp output is UTF-8 but one stray byte from a
/// model file must not fail a whole benchmark run.
fn decode(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::PerfSummary;

    // ---- pure math --------------------------------------------------------

    fn perf(
        prompt_eval_ms: f64,
        prompt_tokens: u64,
        eval_ms: f64,
        eval_tokens: u64,
    ) -> PerfSummary {
        PerfSummary {
            prompt_eval_ms,
            prompt_tokens,
            eval_ms,
            eval_tokens,
        }
    }

    #[test]
    fn ttft_is_the_prefill_wall_clock() {
        let p = perf(364.41, 7, 517.44, 6);
        assert!((ttft_ms_from(&p) - 364.41).abs() < 1e-9);
    }

    #[test]
    fn ttft_treats_non_finite_or_non_positive_prefill_as_zero() {
        // No meaningful measurement → 0.0, not NaN, not negative.
        assert_eq!(ttft_ms_from(&perf(0.0, 0, 10.0, 1)), 0.0);
        assert_eq!(ttft_ms_from(&perf(-1.0, 0, 10.0, 1)), 0.0);
        assert_eq!(ttft_ms_from(&perf(f64::NAN, 0, 10.0, 1)), 0.0);
    }

    #[test]
    fn tpot_is_eval_per_decode_token() {
        // 600 ms / 100 tokens = 6 ms/token.
        let p = perf(100.0, 7, 600.0, 100);
        assert!((tpot_ms_from(&p) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn tpot_returns_zero_for_empty_decode() {
        assert_eq!(tpot_ms_from(&perf(10.0, 1, 0.0, 0)), 0.0);
        assert_eq!(tpot_ms_from(&perf(10.0, 1, 100.0, 0)), 0.0);
        // A non-positive or non-finite window must read as zero.
        assert_eq!(tpot_ms_from(&perf(10.0, 1, -1.0, 1)), 0.0);
        assert_eq!(tpot_ms_from(&perf(10.0, 1, f64::NAN, 1)), 0.0);
        assert_eq!(tpot_ms_from(&perf(10.0, 1, f64::INFINITY, 1)), 0.0);
    }

    #[test]
    fn tok_s_is_inverse_of_tpot() {
        // 200 ms / 50 tokens = 4 ms/token → 250 tok/s.
        let p = perf(100.0, 7, 200.0, 50);
        assert!((tok_s_from(&p) - 250.0).abs() < 1e-9);

        // Realistic b4041 numbers (slice-5 test): 517.44 ms / 6 runs.
        let real = perf(364.41, 7, 517.44, 6);
        let expected = 6.0 / (517.44 / 1_000.0);
        assert!((tok_s_from(&real) - expected).abs() < 1e-9);
    }

    #[test]
    fn tok_s_is_zero_when_decode_is_unmeasurable() {
        assert_eq!(tok_s_from(&perf(10.0, 1, 0.0, 0)), 0.0);
        assert_eq!(tok_s_from(&perf(10.0, 1, 100.0, 0)), 0.0);
        // Tokenizer produced something infinite—must clamp to 0.
        assert_eq!(tok_s_from(&perf(10.0, 1, f64::INFINITY, 1)), 0.0);
    }

    // ---- E2eResult / E2eConfig ----------------------------------------------

    #[test]
    fn is_complete_only_when_decode_actually_ran() {
        let complete = E2eResult {
            engine: "3505".into(),
            model: "/m.gguf".into(),
            prompt_tokens: 7,
            eval_tokens: 6,
            ttft_ms: 364.41,
            tpot_ms: 86.24,
            tok_s: 11.6,
            ram_peak_bytes: 0,
            cpu_avg_pct: 0.0,
            power_avg_watts: None,
            elapsed_ms: 0.0,
        };
        assert!(complete.is_complete());

        let mut truncated = complete.clone();
        truncated.tpot_ms = 0.0;
        truncated.tok_s = 0.0;
        assert!(!truncated.is_complete());
    }

    #[test]
    fn effective_prompt_tokens_clamps_to_context_window() {
        let mut cfg = E2eConfig::new(PathBuf::from("/m.gguf"));
        // 512 tokens prompt, 128 gen, ctx 2048 → 512 fits with room to spare.
        assert_eq!(cfg.effective_prompt_tokens(), 512);

        // ctx too small for prompt + gen + safety margin → prompt shrinks.
        cfg.ctx_size = 200;
        assert_eq!(cfg.effective_prompt_tokens(), 200 - 128 - 32);

        // Floor of 1 (no zero-token prompts).
        cfg.prompt_tokens = 0;
        cfg.ctx_size = 4096;
        assert_eq!(cfg.effective_prompt_tokens(), 1);
    }

    #[test]
    fn clamps_have_a_real_floor() {
        let mut cfg = E2eConfig::new(PathBuf::from("/m.gguf"));
        cfg.ctx_size = 4; // below the 32 floor
        assert_eq!(cfg.ctx_size(), 32);
        cfg.gen_tokens = 0;
        assert_eq!(cfg.gen_tokens(), 1);
        cfg.prompt_tokens = 0;
        // Floor is 1, not 0.
        assert!(cfg.effective_prompt_tokens() >= 1);
    }

    // ---- arg / prompt builders --------------------------------------------

    #[test]
    fn build_e2e_args_uses_only_old_safe_flags() {
        let cfg = E2eConfig::new(PathBuf::from("/m.gguf"));
        let args = build_e2e_args(&cfg);
        let joined = args.join(" ");
        // Required flags:
        assert!(joined.contains("-m /m.gguf"), "{joined}");
        assert!(
            joined.contains(&format!("-n {}", DEFAULT_GEN_TOKENS)),
            "{joined}"
        );
        assert!(
            joined.contains(&format!("-c {}", DEFAULT_CTX_SIZE)),
            "{joined}"
        );
        // `-st` and `-v` predate the chat rewrite, so they are old-safe too:
        // `-st` stops the child from waiting in its interactive REPL and `-v`
        // makes it print the timing block `parse_perf` reads.
        for flag in llama::COMMON_ARGS {
            assert!(
                args.iter().any(|arg| arg == flag),
                "missing {flag}: {joined}"
            );
        }
        // The prompt must exist and be at least roughly the target length.
        let prompt_idx = args.iter().position(|s| s == "-p").expect("-p flag");
        let prompt_value = &args[prompt_idx + 1];
        assert!(
            prompt_value.len() >= 2048 - 32,
            "prompt was {}",
            prompt_value.len()
        );
        // No thread pin when threads == 0.
        assert!(!joined.contains("-t "), "no -t: {joined}");
    }

    #[test]
    fn build_e2e_args_pins_threads_only_when_requested() {
        let mut cfg = E2eConfig::new(PathBuf::from("/m.gguf"));
        let joined = build_e2e_args(&cfg).join(" ");
        assert!(!joined.contains("-t "), "no -t by default: {joined}");

        cfg.threads = 6;
        let pinned = build_e2e_args(&cfg).join(" ");
        assert!(pinned.contains("-t 6"), "{pinned}");
    }

    #[test]
    fn build_e2e_args_truncates_prompt_to_fit_context() {
        // ctx 64, gen 32 → prompt cap = 64 - 32 - 32 = 0 tokens → empty.
        let mut cfg = E2eConfig::new(PathBuf::from("/m.gguf"));
        cfg.ctx_size = 64;
        cfg.gen_tokens = 32;
        let args = build_e2e_args(&cfg);
        let prompt_idx = args.iter().position(|s| s == "-p").expect("-p flag");
        // At worst the prompt is the minimum 64 chars enforced by the
        // builder; what matters is that we did not blindly emit a 2048-char
        // prompt into a 64-token context.
        assert!(args[prompt_idx + 1].len() <= 64);
    }

    #[test]
    fn build_prompt_approximates_the_token_target() {
        // 128 tokens * 4 chars = 512 chars, exactly the typical phrase count.
        let prompt = build_prompt(128, 1024);
        assert!(prompt.len() >= 256 && prompt.len() <= 1024);
        assert!(!prompt.is_empty());
        // Empty input → empty output (degenerate corner).
        assert_eq!(build_prompt(0, 1024), "");
        // Cap (parameter) is honoured exactly.
        let tiny = build_prompt(10_000, 200);
        assert!(tiny.len() <= 200);
    }

    // ---- JSON contract ----------------------------------------------------

    #[test]
    fn e2e_result_roundtrip() {
        let result = E2eResult {
            engine: "3505".into(),
            model: "/m.gguf".into(),
            prompt_tokens: 7,
            eval_tokens: 6,
            ttft_ms: 364.41,
            tpot_ms: 86.24,
            tok_s: 11.6,
            ram_peak_bytes: 8_589_934_592,
            cpu_avg_pct: 75.0,
            power_avg_watts: Some(42.5),
            elapsed_ms: 12_345.0,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        for key in [
            "engine",
            "model",
            "prompt_tokens",
            "eval_tokens",
            "ttft_ms",
            "tpot_ms",
            "tok_s",
            "ram_peak_bytes",
            "cpu_avg_pct",
            "power_avg_watts",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        let back: E2eResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, result);
    }

    #[test]
    fn e2e_result_deserializes_without_power_when_unavailable() {
        let json = r#"{
            "engine": "3505",
            "model": "/m.gguf",
            "prompt_tokens": 7,
            "eval_tokens": 6,
            "ttft_ms": 364.41,
            "tpot_ms": 86.24,
            "tok_s": 11.6,
            "ram_peak_bytes": 1000,
            "cpu_avg_pct": 1.0
        }"#;
        let result: E2eResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.power_avg_watts, None);
    }

    // ---- pre-flight -------------------------------------------------------

    #[test]
    fn run_e2e_rejects_a_missing_model() {
        let cfg = E2eConfig::new(PathBuf::from("definitely-not-a-real-model.gguf"));
        let err = run_e2e(Path::new("definitely-not-a-real-llama-cli-path"), &cfg)
            .expect_err("a missing model must be rejected");
        assert!(err.to_string().contains("not a readable file"), "{err}");
    }

    #[test]
    fn run_e2e_accepts_a_non_executable_model() {
        // Regression for #27: a `0644` GGUF passes the model check, so the
        // failure comes from the missing llama-cli instead.
        let model =
            std::env::temp_dir().join(format!("attune-e2e-readable-{}.gguf", std::process::id()));
        std::fs::write(&model, b"GGUF").expect("write temp model");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&model, std::fs::Permissions::from_mode(0o644))
                .expect("chmod 0644");
        }
        let cfg = E2eConfig::new(model.clone());
        let err = run_e2e(Path::new("definitely-not-a-real-llama-cli-path"), &cfg)
            .expect_err("there is no llama-cli to run");
        let _ = std::fs::remove_file(&model);
        assert!(
            !err.to_string().contains("not a readable file"),
            "model check rejected a readable file: {err}"
        );
    }

    #[test]
    fn preflight_rejects_a_missing_model_before_the_version_probe() {
        // The model check comes first, so no llama-cli is needed to fail.
        let err = preflight(
            Path::new("definitely-not-a-real-llama-cli-path"),
            Path::new("definitely-not-a-real-model.gguf"),
        )
        .expect_err("a missing model must be rejected");
        assert!(matches!(err, BenchError::InvalidConfig(_)), "{err:?}");
    }

    // ---- e2e as the load report (#31) -------------------------------------

    fn e2e_sample() -> E2eResult {
        E2eResult {
            engine: "0.7.0-dev".to_string(),
            model: "/models/qwen3-4b.gguf".to_string(),
            prompt_tokens: 464,
            eval_tokens: 128,
            ttft_ms: 5_862.0,
            tpot_ms: 125.0,
            tok_s: 8.0,
            ram_peak_bytes: 4_699_271_168,
            cpu_avg_pct: 710.0,
            power_avg_watts: None,
            elapsed_ms: 26_000.0,
        }
    }

    #[test]
    fn load_report_is_built_from_the_e2e_run() {
        let report = e2e_sample().load_report(Path::new("/opt/llama/llama-cli"));
        assert_eq!(report.cli, "/opt/llama/llama-cli");
        assert_eq!(report.version, "0.7.0-dev");
        assert_eq!(report.model, "/models/qwen3-4b.gguf");
        assert!(report.load_ok);
        assert_eq!(report.elapsed_ms, 26_000.0);
        let perf = report.perf.expect("timed run carries perf");
        assert_eq!(perf.prompt_eval_ms, 5_862.0);
        assert_eq!(perf.prompt_tokens, 464);
        assert_eq!(perf.eval_ms, 16_000.0);
        assert_eq!(perf.eval_tokens, 128);
    }

    #[test]
    fn load_report_has_no_perf_when_the_run_printed_no_timings() {
        let mut untimed = e2e_sample();
        untimed.ttft_ms = 0.0;
        untimed.eval_tokens = 0;
        let report = untimed.load_report(Path::new("llama-cli"));
        assert!(report.load_ok);
        assert_eq!(report.perf, None);
    }

    #[test]
    fn e2e_result_without_elapsed_still_parses() {
        let json = r#"{
            "engine": "3505", "model": "/m.gguf", "prompt_tokens": 7,
            "eval_tokens": 6, "ttft_ms": 364.41, "tpot_ms": 86.24,
            "tok_s": 11.6, "ram_peak_bytes": 1000, "cpu_avg_pct": 1.0
        }"#;
        let result: E2eResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.elapsed_ms, 0.0);
    }

    // ---- manual, real llama.cpp only --------------------------------------

    /// Run a real llama.cpp e2e bench against an explicit model. CI has
    /// neither a llama.cpp build nor a model, so this test is
    /// `#[ignore]`d *and* env-gated (mirrors the policy in `llama.rs`).
    /// Run it manually with:
    ///
    ///     ATTUNE_LLAMA_CLI=/path/to/llama-cli ATTUNE_TEST_MODEL=/path/to/tiny.gguf \
    ///         cargo test -p attune-bench -- --ignored e2e::tests::real
    #[test]
    #[ignore = "manual: needs ATTUNE_LLAMA_CLI + ATTUNE_TEST_MODEL and a real llama.cpp build"]
    fn real_llama_cli_runs_an_e2e_bench() {
        let Some(model_os) = std::env::var_os(crate::llama::ENV_TEST_MODEL) else {
            eprintln!("skipping: {} not set", crate::llama::ENV_TEST_MODEL);
            return;
        };
        let cli = match llama::resolve_llama_cli(
            std::env::var(crate::llama::ENV_LLAMA_CLI).ok().as_deref(),
        ) {
            Ok(cli) => cli,
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let cfg = E2eConfig::new(PathBuf::from(model_os));
        let result = run_e2e(&cli, &cfg).expect("real e2e bench should run");

        assert_eq!(result.model, cfg.model.display().to_string());
        assert!(result.ram_peak_bytes > 0, "child RSS was never measured");
        assert!(
            result.is_complete(),
            "decode did not report any tokens: {result:?}"
        );
        assert!(result.tok_s > 0.0, "tok_s must be positive");
        println!("{result:?}");
    }
}
