//! Resource monitor — background RAM / CPU / power sampling (M2 slice 4).
//!
//! A benchmark tells you *how fast* the hardware went; the monitor tells you
//! *what that run cost*: peak resident memory, average / peak CPU, and (on
//! Linux) average package power. It runs on its own thread so sampling never
//! perturbs the timed kernels beyond one cheap `sysinfo` refresh per tick.
//!
//! Sampling is decoupled from timing through the [`Sampler`] trait, so the
//! aggregation math is unit-tested against canned readings instead of real
//! hardware, and [`Monitor`] itself is tested with a fake sampler.
//!
//! Lifecycle: [`Monitor::start`] spawns the worker, [`Monitor::stop`] signals
//! it, joins the thread and returns the [`ResourceStats`]. Dropping a
//! `Monitor` without calling `stop` does the same thing (minus handing you the
//! numbers), so an early `?` in a bench pipeline can never leak a thread.
//!
//! Power is Linux-only RAPL: energy counters under [`POWERCAP_ROOT`] in
//! micro-joules. The parser ([`parse_rapl_energy_uj`]) and the delta-to-watts
//! conversion ([`watts_from_energy_delta`]) are pure and platform-independent
//! so their unit tests run on every OS; only [`read_rapl_energy_uj`] — the
//! part that actually touches sysfs — is Linux-gated. Every other platform
//! returns `None`, and `ResourceStats::power_avg_watts` is then simply
//! `None`: monitoring never fails, it just reports less.
//!
//! `cpu_pct` is *total* CPU across all cores (so a saturated 8-thread kernel
//! reads several hundred percent). It is derived from CPU-millisecond deltas
//! by [`cpu_pct_from_ms`] rather than from `sysinfo`'s own `cpu_usage()`,
//! which is only recomputed every 200 ms on Windows and would therefore read a
//! flat 0 % at this module's 100 ms default.
//!
//! No `.unwrap()` in non-test code (AGENTS.md §3.2) — a thread that panics
//! degrades to an empty (zero-sample) report instead of poisoning the caller.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// Only the Linux RAPL reader builds `PathBuf` paths, so the import is gated
// with it — an ungated import would be dead code on Windows/macOS and trip
// `-D warnings` (the CI failure mode from PR #4).
#[cfg(target_os = "linux")]
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::BenchError;

/// Default sampling period: 100 ms. Fast enough to catch a bandwidth-kernel
/// RSS spike (a 256 MiB buffer touched in <1 s) without adding measurable
/// overhead to a multi-second bench run.
pub const DEFAULT_INTERVAL_MS: u64 = 100;

/// Floor for the sampling period. Below ~10 ms the `sysinfo` refresh itself
/// dominates the period and the readings stop meaning anything, so callers
/// asking for less get this instead.
pub const MIN_INTERVAL_MS: u64 = 10;

/// Linux RAPL root. Each package domain lives in a `intel-rapl:N` (or
/// `amd-rapl:N`) subdirectory holding an `energy_uj` file. Declared
/// unconditionally so the constant and the parser tests exist on every
/// platform; only [`read_rapl_energy_uj`] reads from it, and that is
/// Linux-gated.
pub const POWERCAP_ROOT: &str = "/sys/class/powercap";

/// Name of the RAPL energy file inside one powercap domain directory.
pub const RAPL_ENERGY_FILE: &str = "energy_uj";

/// One reading of process-level resource usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Resident set size of the monitored process, in bytes.
    pub ram_bytes: u64,
    /// CPU utilisation of the monitored process in percent. This is the
    /// *total across all cores*, so a fully saturated 16-thread kernel reads
    /// up to ~1600 %, not 100 %.
    pub cpu_pct: f64,
}

/// Anything that can produce [`Sample`]s on demand.
///
/// `sample_energy_uj` has a default `None` so a sampler only has to care about
/// what its platform can actually measure: a fake in tests returns nothing,
/// and [`SysinfoSampler`] returns RAPL energy on Linux and `None` elsewhere.
pub trait Sampler {
    /// Take one reading. Must be cheap and must not panic.
    fn sample(&mut self) -> Sample;

    /// Best-effort energy counter reading in micro-joules, or `None` when the
    /// platform has no power counter (or it could not be read). Only
    /// meaningful as the *difference* between two readings — see
    /// [`watts_from_energy_delta`].
    fn sample_energy_uj(&mut self) -> Option<u64> {
        None
    }
}

/// Aggregated resource usage over one monitored run. JSON contract for the
/// optional `resources` block of `attune bench --json`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceStats {
    /// Highest resident-set reading seen, in bytes.
    pub ram_peak_bytes: u64,
    /// Arithmetic mean of the CPU readings.
    pub cpu_avg_pct: f64,
    /// Highest CPU reading seen.
    pub cpu_peak_pct: f64,
    /// Mean of the per-tick watt readings. `None` when the platform exposes
    /// no power counter, or when the run was too short to produce a single
    /// energy *delta*. `serde(default)` keeps JSON written before slice 4
    /// parseable.
    #[serde(default)]
    pub power_avg_watts: Option<f64>,
    /// Number of readings folded into the numbers above.
    pub samples: usize,
    /// Effective sampling period after clamping to [`MIN_INTERVAL_MS`].
    pub interval_ms: u64,
}

impl ResourceStats {
    /// A zero-sample report carrying only the sampling period. Used when the
    /// monitor thread died before recording anything (or was stopped before
    /// its first tick) — monitoring degrades, it never fails the bench.
    pub fn empty(interval_ms: u64) -> Self {
        Self {
            ram_peak_bytes: 0,
            cpu_avg_pct: 0.0,
            cpu_peak_pct: 0.0,
            power_avg_watts: None,
            samples: 0,
            interval_ms,
        }
    }
}

/// Handle to a running sampler thread.
///
/// Drop stops and joins the thread, so `Monitor` must be owned by whoever
/// wants the numbers — [`stop`](Monitor::stop) consumes the handle and gives
/// back the [`ResourceStats`].
pub struct Monitor {
    interval_ms: u64,
    /// `None` once the worker has been signalled to stop; dropping the
    /// sender is what makes the worker's `recv_timeout` return.
    stop_tx: Option<Sender<()>>,
    /// `None` once the worker has been joined.
    worker: Option<JoinHandle<ResourceStats>>,
}

impl Monitor {
    /// Start sampling `sampler` every [`DEFAULT_INTERVAL_MS`].
    pub fn start(sampler: Box<dyn Sampler + Send>) -> Result<Self, BenchError> {
        Self::start_with_interval(sampler, DEFAULT_INTERVAL_MS)
    }

    /// Start sampling `sampler` every `interval_ms`, clamped to at least
    /// [`MIN_INTERVAL_MS`]. Fails only if the OS refuses to spawn the thread.
    pub fn start_with_interval(
        sampler: Box<dyn Sampler + Send>,
        interval_ms: u64,
    ) -> Result<Self, BenchError> {
        let interval_ms = clamp_interval_ms(interval_ms);
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let worker = std::thread::Builder::new()
            .name("attune-resource-monitor".to_string())
            .spawn(move || sampling_loop(sampler, stop_rx, interval_ms))
            .map_err(BenchError::MonitorSpawn)?;
        Ok(Self {
            interval_ms,
            stop_tx: Some(stop_tx),
            worker: Some(worker),
        })
    }

    /// Start monitoring *this* process with the real [`SysinfoSampler`] every
    /// `interval_ms`. This is the one-shot entry point used by the bench
    /// pipeline.
    pub fn start_sysinfo(interval_ms: u64) -> Result<Self, BenchError> {
        Self::start_with_interval(Box::new(SysinfoSampler::new()), interval_ms)
    }

    /// Effective sampling period in milliseconds (post-clamp).
    pub fn interval_ms(&self) -> u64 {
        self.interval_ms
    }

    /// Stop sampling, join the worker, and return what it collected.
    ///
    /// If the worker panicked (only reachable via a misbehaving [`Sampler`]),
    /// its samples are lost and an empty [`ResourceStats::empty`] report is
    /// returned instead of propagating the panic into the bench.
    pub fn stop(mut self) -> ResourceStats {
        self.shutdown()
            .unwrap_or_else(|| ResourceStats::empty(self.interval_ms))
    }

    /// Signal the worker, join it, and hand back its stats. Idempotent: the
    /// second call finds nothing to do and reports an empty run.
    fn shutdown(&mut self) -> Option<ResourceStats> {
        // Dropping the sender is the stop signal: the worker is blocked in
        // `recv_timeout`, and a disconnected channel wakes it immediately
        // instead of making `stop` wait out a full sampling period.
        self.stop_tx = None;
        self.worker.take().map(|worker| match worker.join() {
            Ok(stats) => stats,
            Err(_) => ResourceStats::empty(self.interval_ms),
        })
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        // A dropped-without-stopped monitor must not leave a thread behind.
        let _ = self.shutdown();
    }
}

/// Pure: clamp a requested sampling period to the supported floor.
pub fn clamp_interval_ms(interval_ms: u64) -> u64 {
    interval_ms.max(MIN_INTERVAL_MS)
}

// ---- worker ----------------------------------------------------------------

/// The sampling thread. Samples immediately (so even a very short run records
/// something), then sleeps one period between readings, waking early only if
/// the monitor is stopped.
fn sampling_loop(
    mut sampler: Box<dyn Sampler + Send>,
    stop_rx: Receiver<()>,
    interval_ms: u64,
) -> ResourceStats {
    let interval = Duration::from_millis(interval_ms);
    let mut acc = Accumulator::new(interval_ms);

    loop {
        let now = Instant::now();
        let sample = sampler.sample();
        // Energy is read immediately after the process reading so the watt
        // figure covers (approximately) the same window.
        let watts = sampler
            .sample_energy_uj()
            .and_then(|uj| acc.observe_energy(uj, now));
        acc.push(sample, watts);

        match stop_rx.recv_timeout(interval) {
            Err(RecvTimeoutError::Timeout) => continue,
            // Both arms mean "stop": an explicit message, or the sender being
            // dropped by `Monitor::stop` / `Monitor::drop`.
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    acc.finish()
}

/// Folds samples into [`ResourceStats`]. Kept separate from the thread so the
/// aggregation math (peak / mean / watt averaging) is testable without any
/// timing.
struct Accumulator {
    interval_ms: u64,
    samples: usize,
    ram_peak_bytes: u64,
    cpu_sum: f64,
    cpu_peak_pct: f64,
    watts_sum: f64,
    watts_samples: usize,
    /// Last `(energy_uj, when)` reading — a watt figure needs two of them.
    last_energy: Option<(u64, Instant)>,
}

impl Accumulator {
    fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            samples: 0,
            ram_peak_bytes: 0,
            cpu_sum: 0.0,
            cpu_peak_pct: 0.0,
            watts_sum: 0.0,
            watts_samples: 0,
            last_energy: None,
        }
    }

    /// Fold one reading into the running totals. A non-finite or negative
    /// `cpu_pct` is treated as 0 so one bad reading cannot poison the mean
    /// with `NaN`.
    fn push(&mut self, sample: Sample, watts: Option<f64>) {
        let cpu = if sample.cpu_pct.is_finite() && sample.cpu_pct > 0.0 {
            sample.cpu_pct
        } else {
            0.0
        };
        self.samples += 1;
        self.ram_peak_bytes = self.ram_peak_bytes.max(sample.ram_bytes);
        self.cpu_sum += cpu;
        self.cpu_peak_pct = self.cpu_peak_pct.max(cpu);
        if let Some(w) = watts {
            if w.is_finite() && w > 0.0 {
                self.watts_sum += w;
                self.watts_samples += 1;
            }
        }
    }

    /// Record an energy reading and, when a previous one exists, return the
    /// power drawn since it. The first reading only establishes a baseline.
    fn observe_energy(&mut self, energy_uj: u64, now: Instant) -> Option<f64> {
        let previous = self.last_energy.replace((energy_uj, now));
        let (prev_uj, prev_at) = previous?;
        // Saturating: a counter reset (RAPL wraps, and a driver can reset it
        // under us) must read as a decrease, not an underflow panic.
        let delta_uj = energy_uj.saturating_sub(prev_uj);
        watts_from_energy_delta(delta_uj, now.duration_since(prev_at))
    }

    fn finish(self) -> ResourceStats {
        let cpu_avg_pct = if self.samples == 0 {
            0.0
        } else {
            self.cpu_sum / self.samples as f64
        };
        let power_avg_watts = if self.watts_samples == 0 {
            None
        } else {
            Some(self.watts_sum / self.watts_samples as f64)
        };
        ResourceStats {
            ram_peak_bytes: self.ram_peak_bytes,
            cpu_avg_pct,
            cpu_peak_pct: self.cpu_peak_pct,
            power_avg_watts,
            samples: self.samples,
            interval_ms: self.interval_ms,
        }
    }
}

// ---- real sampler ----------------------------------------------------------

/// Cross-platform process sampler backed by `sysinfo`.
///
/// Justification for the dep: `sysinfo` is already the workspace-standard
/// source of RAM/CPU facts (`attune-hal` uses it for `detect`) and is the only
/// dependency that gives us per-process RSS + CPU on Linux *and* Windows with
/// the same code path. `ProcessRefreshKind::nothing().with_cpu().with_memory()`
/// keeps each tick down to a couple of syscalls instead of walking the whole
/// task list.
///
/// CPU is *not* taken from `Process::cpu_usage()`: on Windows that value is
/// only recomputed once every `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` (200 ms),
/// so a 100 ms monitor would report a flat 0 % forever. We read
/// `Process::accumulated_cpu_time()` — documented as total CPU-milliseconds on
/// every platform — and derive the percentage from its delta with
/// [`cpu_pct_from_ms`], which is rate-independent and unit-testable.
pub struct SysinfoSampler {
    system: sysinfo::System,
    pid: sysinfo::Pid,
    /// Last `(cpu_milliseconds, when)` reading — a percentage needs two.
    last_cpu_ms: Option<(u64, Instant)>,
}

impl SysinfoSampler {
    /// Sample the calling process (used by `attune bench`).
    pub fn new() -> Self {
        match sysinfo::get_current_pid() {
            Ok(pid) => Self::with_pid(pid),
            // `get_current_pid` only fails if sysinfo was compiled without
            // process support; `std::process::id` is the same number anyway.
            Err(_) => Self::with_pid(sysinfo::Pid::from_u32(std::process::id())),
        }
    }

    /// Sample an arbitrary process by pid (a later slice will point this at
    /// the backend's child process).
    pub fn with_pid(pid: sysinfo::Pid) -> Self {
        let mut system = sysinfo::System::new();
        refresh(&mut system, pid);
        Self {
            system,
            pid,
            last_cpu_ms: None,
        }
    }
}

impl Default for SysinfoSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SysinfoSampler {
    fn sample(&mut self) -> Sample {
        refresh(&mut self.system, self.pid);
        // Timestamped after the refresh so consecutive readings bracket the
        // window in which the process actually burned CPU.
        let now = Instant::now();
        let (ram_bytes, cpu_ms) = match self.system.process(self.pid) {
            Some(proc) => (proc.memory(), proc.accumulated_cpu_time()),
            // The pid vanished (process exited): report zero and keep the last
            // known CPU total so the next reading can still be a delta.
            None => (0, self.last_cpu_ms.map_or(0, |(ms, _)| ms)),
        };
        let cpu_pct = self
            .last_cpu_ms
            .and_then(|(previous_ms, at)| {
                cpu_pct_from_ms(cpu_ms.saturating_sub(previous_ms), now.duration_since(at))
            })
            .unwrap_or(0.0);
        self.last_cpu_ms = Some((cpu_ms, now));

        Sample { ram_bytes, cpu_pct }
    }

    fn sample_energy_uj(&mut self) -> Option<u64> {
        // RAPL is package-level, not per-process — it describes the socket
        // the process is running on, which is the only power figure Linux
        // offers without extra privileges.
        read_rapl_energy_uj()
    }
}

/// Pure: CPU-milliseconds burned over `elapsed` → percent of a single core.
///
/// 1000 ms of CPU in 1 s of wall clock is 100 % (one saturated core); the
/// value deliberately exceeds 100 % for a multi-threaded process. Returns
/// `None` for a non-positive or non-finite window, which is what the caller
/// sees when two readings land inside the same clock tick.
pub fn cpu_pct_from_ms(delta_ms: u64, elapsed: Duration) -> Option<f64> {
    let seconds = elapsed.as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let pct = delta_ms as f64 / 1_000.0 / seconds * 100.0;
    if pct.is_finite() {
        Some(pct.max(0.0))
    } else {
        None
    }
}

/// Refresh just the fields the monitor reads, for one pid.
fn refresh(system: &mut sysinfo::System, pid: sysinfo::Pid) {
    system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        sysinfo::ProcessRefreshKind::nothing()
            .with_cpu()
            .with_memory()
            .without_tasks(),
    );
}

// ---- RAPL power (Linux only, parser is portable) ----------------------------

/// Pure parser for the contents of a RAPL `energy_uj` file: one unsigned
/// decimal count of micro-joules, usually with a trailing newline. Returns
/// `None` for empty or non-numeric content so a mis-read never turns into a
/// bogus watt figure.
///
/// Platform-independent on purpose — the tests for it run on Windows and Linux
/// alike, and the Linux reader below is its only production caller.
pub fn parse_rapl_energy_uj(content: &str) -> Option<u64> {
    // `split_whitespace().next()` drops the trailing newline and rejects
    // leading junk in one step: if the first token is not the counter there is
    // nothing sensible to parse.
    let token = content.split_whitespace().next()?;
    token.parse::<u64>().ok()
}

/// Pure: micro-joules of energy consumed over `elapsed` → average watts.
///
/// Returns `None` for a non-positive or non-finite window, which is what the
/// caller sees when two readings land inside the same clock tick.
pub fn watts_from_energy_delta(delta_uj: u64, elapsed: Duration) -> Option<f64> {
    let seconds = elapsed.as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let joules = delta_uj as f64 / 1_000_000.0;
    let watts = joules / seconds;
    if watts.is_finite() {
        Some(watts)
    } else {
        None
    }
}

/// Read the lowest-index RAPL package domain's energy counter.
///
/// Linux only. Any failure — no `/sys/class/powercap`, permissions, a driver
/// that exposes no `energy_uj` — is a plain `None`; power monitoring is
/// strictly best-effort and must never break a bench run.
#[cfg(target_os = "linux")]
pub fn read_rapl_energy_uj() -> Option<u64> {
    let path = first_rapl_energy_file()?;
    let content = std::fs::read_to_string(path).ok()?;
    parse_rapl_energy_uj(&content)
}

/// The first domain directory under [`POWERCAP_ROOT`], sorted so the choice is
/// deterministic. `intel-rapl:0` (the package) sorts before its `intel-rapl:0:0`
/// sub-domains, which is the aggregate figure we want.
#[cfg(target_os = "linux")]
fn first_rapl_energy_file() -> Option<PathBuf> {
    let mut domains: Vec<PathBuf> = std::fs::read_dir(POWERCAP_ROOT)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join(RAPL_ENERGY_FILE).is_file())
        .collect();
    domains.sort();
    domains.into_iter().next()
}

/// Non-Linux platforms have no RAPL. Kept as a real (rather than absent)
/// function so [`SysinfoSampler::sample_energy_uj`] and its tests compile and
/// behave identically on every target — the field is just `None`.
#[cfg(not(target_os = "linux"))]
pub fn read_rapl_energy_uj() -> Option<u64> {
    None
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread::sleep;

    use super::*;

    /// Fake sampler replaying a canned list of readings. Records how many
    /// samples were actually taken so the test can compute the exact expected
    /// aggregates without depending on how many ticks fit in the wall clock.
    struct FakeSampler {
        samples: Vec<Sample>,
        energy: Option<Vec<u64>>,
        taken: Arc<AtomicUsize>,
    }

    impl FakeSampler {
        fn new(samples: Vec<Sample>) -> Self {
            Self {
                samples,
                energy: None,
                taken: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_energy(samples: Vec<Sample>, energy: Vec<u64>) -> Self {
            Self {
                samples,
                energy: Some(energy),
                taken: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Sampler for FakeSampler {
        fn sample(&mut self) -> Sample {
            let index = self.taken.fetch_add(1, Ordering::Relaxed);
            // Replay the last canned sample once the list runs out, so a test
            // never depends on exactly how many ticks elapsed.
            self.samples
                .get(index)
                .copied()
                .or_else(|| self.samples.last().copied())
                .unwrap_or(Sample {
                    ram_bytes: 0,
                    cpu_pct: 0.0,
                })
        }

        fn sample_energy_uj(&mut self) -> Option<u64> {
            let index = self.taken.load(Ordering::Relaxed).saturating_sub(1);
            self.energy
                .as_ref()
                .and_then(|e| e.get(index).copied().or_else(|| e.last().copied()))
        }
    }

    fn mib(n: u64) -> u64 {
        n * 1024 * 1024
    }

    // ---- interval clamp -----------------------------------------------------

    #[test]
    fn interval_is_clamped_to_the_supported_floor() {
        assert_eq!(clamp_interval_ms(0), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval_ms(1), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval_ms(MIN_INTERVAL_MS - 1), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval_ms(MIN_INTERVAL_MS), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval_ms(DEFAULT_INTERVAL_MS), DEFAULT_INTERVAL_MS);
        assert_eq!(clamp_interval_ms(5_000), 5_000);
    }

    #[test]
    fn monitor_reports_the_clamped_interval_not_the_requested_one() {
        let monitor = Monitor::start_with_interval(
            Box::new(FakeSampler::new(vec![Sample {
                ram_bytes: 1,
                cpu_pct: 1.0,
            }])),
            0,
        )
        .expect("monitor should start");
        assert_eq!(monitor.interval_ms(), MIN_INTERVAL_MS);
        let stats = monitor.stop();
        assert_eq!(stats.interval_ms, MIN_INTERVAL_MS);
    }

    #[test]
    fn default_interval_is_100ms() {
        assert_eq!(DEFAULT_INTERVAL_MS, 100);
        let monitor = Monitor::start(Box::new(FakeSampler::new(vec![Sample {
            ram_bytes: 1,
            cpu_pct: 1.0,
        }])))
        .expect("monitor should start");
        assert_eq!(monitor.interval_ms(), DEFAULT_INTERVAL_MS);
        assert_eq!(monitor.stop().interval_ms, DEFAULT_INTERVAL_MS);
    }

    // ---- aggregation math ---------------------------------------------------

    #[test]
    fn accumulator_aggregates_canned_samples_exactly() {
        let mut acc = Accumulator::new(DEFAULT_INTERVAL_MS);
        let canned = [
            Sample {
                ram_bytes: mib(100),
                cpu_pct: 10.0,
            },
            Sample {
                ram_bytes: mib(400),
                cpu_pct: 30.0,
            },
            Sample {
                ram_bytes: mib(250),
                cpu_pct: 20.0,
            },
            Sample {
                ram_bytes: mib(120),
                cpu_pct: 40.0,
            },
            Sample {
                ram_bytes: mib(310),
                cpu_pct: 50.0,
            },
        ];
        for sample in canned {
            acc.push(sample, None);
        }
        let stats = acc.finish();

        assert_eq!(stats.samples, canned.len());
        assert_eq!(stats.ram_peak_bytes, mib(400));
        assert_eq!(stats.cpu_peak_pct, 50.0);
        // mean(10, 30, 20, 40, 50) = 30
        assert!((stats.cpu_avg_pct - 30.0).abs() < 1e-9);
        assert_eq!(stats.interval_ms, DEFAULT_INTERVAL_MS);
        assert_eq!(stats.power_avg_watts, None);
    }

    #[test]
    fn accumulator_handles_single_sample_and_empty_run() {
        let mut acc = Accumulator::new(25);
        acc.push(
            Sample {
                ram_bytes: mib(64),
                cpu_pct: 12.5,
            },
            None,
        );
        let stats = acc.finish();
        assert_eq!(stats.samples, 1);
        assert_eq!(stats.ram_peak_bytes, mib(64));
        assert!((stats.cpu_avg_pct - 12.5).abs() < 1e-9);
        assert!((stats.cpu_peak_pct - 12.5).abs() < 1e-9);

        let empty = Accumulator::new(25).finish();
        assert_eq!(empty.samples, 0);
        assert_eq!(empty.ram_peak_bytes, 0);
        assert_eq!(empty.cpu_avg_pct, 0.0);
        assert_eq!(empty.cpu_peak_pct, 0.0);
        assert_eq!(empty.power_avg_watts, None);
    }

    #[test]
    fn accumulator_sanitises_non_finite_cpu_readings() {
        let mut acc = Accumulator::new(100);
        acc.push(
            Sample {
                ram_bytes: 10,
                cpu_pct: f64::NAN,
            },
            None,
        );
        acc.push(
            Sample {
                ram_bytes: 20,
                cpu_pct: -5.0,
            },
            None,
        );
        acc.push(
            Sample {
                ram_bytes: 5,
                cpu_pct: 30.0,
            },
            None,
        );
        let stats = acc.finish();
        // The two junk readings are floored at 0, so the mean is 30 / 3.
        assert!(stats.cpu_avg_pct.is_finite());
        assert!((stats.cpu_avg_pct - 10.0).abs() < 1e-9);
        assert!((stats.cpu_peak_pct - 30.0).abs() < 1e-9);
        assert_eq!(stats.ram_peak_bytes, 20);
    }

    #[test]
    fn accumulator_ignores_non_finite_watts() {
        let mut acc = Accumulator::new(100);
        acc.push(
            Sample {
                ram_bytes: 1,
                cpu_pct: 1.0,
            },
            Some(f64::NAN),
        );
        acc.push(
            Sample {
                ram_bytes: 1,
                cpu_pct: 1.0,
            },
            Some(10.0),
        );
        let stats = acc.finish();
        assert_eq!(stats.power_avg_watts, Some(10.0));
    }

    #[test]
    fn energy_delta_needs_two_readings() {
        let mut acc = Accumulator::new(100);
        // First reading is only a baseline.
        assert_eq!(acc.observe_energy(1_000_000, Instant::now()), None);
        // 5 J over 1 s == 5 W. The window is real wall-clock time, so the
        // result is approximate: `Instant` granularity differs per platform
        // (~15 ms on Windows), hence a tolerance instead of an exact match.
        let watts = acc
            .observe_energy(6_000_000, Instant::now() + Duration::from_secs(1))
            .expect("two readings should yield watts");
        assert!(
            (watts - 5.0).abs() < 0.5,
            "expected ~5 W over a 1 s window, got {watts}"
        );
    }

    // ---- RAPL parser (pure) -------------------------------------------------

    #[test]
    fn rapl_parser_reads_canned_content() {
        // Real-world shape: a bare decimal count plus a trailing newline.
        assert_eq!(parse_rapl_energy_uj("1234567\n"), Some(1_234_567));
        assert_eq!(parse_rapl_energy_uj("0\n"), Some(0));
        // Surrounding whitespace / CRLF is tolerated.
        assert_eq!(parse_rapl_energy_uj("  42  \r\n"), Some(42));
        // Largest plausible RAPL counter (u64::MAX worth of micro-joules).
        assert_eq!(parse_rapl_energy_uj(&u64::MAX.to_string()), Some(u64::MAX));
    }

    #[test]
    fn rapl_parser_rejects_garbage_without_panicking() {
        assert_eq!(parse_rapl_energy_uj(""), None);
        assert_eq!(parse_rapl_energy_uj("   \n"), None);
        assert_eq!(parse_rapl_energy_uj("n/a\n"), None);
        assert_eq!(parse_rapl_energy_uj("-1\n"), None);
        assert_eq!(parse_rapl_energy_uj("12.5\n"), None);
        assert_eq!(parse_rapl_energy_uj("Error 0\n"), None);
    }

    #[test]
    fn watts_from_energy_delta_converts_microjoules() {
        // 10 J over 2 s == 5 W.
        assert_eq!(
            watts_from_energy_delta(10_000_000, Duration::from_secs(2)),
            Some(5.0)
        );
        // 500 mJ over 100 ms == 5 W.
        assert_eq!(
            watts_from_energy_delta(500_000, Duration::from_millis(100)),
            Some(5.0)
        );
        // No energy burned == 0 W, which is a legitimate idle reading.
        assert_eq!(
            watts_from_energy_delta(0, Duration::from_millis(100)),
            Some(0.0)
        );
    }

    #[test]
    fn watts_from_energy_delta_rejects_degenerate_windows() {
        // Two readings inside the same clock tick: no elapsed time to divide by.
        assert_eq!(watts_from_energy_delta(1_000_000, Duration::ZERO), None);
        // Counter went backwards (reset/wrap): saturating delta == 0 W, not a panic.
        let mut acc = Accumulator::new(100);
        acc.observe_energy(5_000_000, Instant::now());
        assert_eq!(
            acc.observe_energy(1_000_000, Instant::now() + Duration::from_millis(10)),
            Some(0.0)
        );
    }

    #[test]
    fn read_rapl_energy_uj_is_optional_on_every_platform() {
        // The call must be safe everywhere — it must not panic on a machine
        // with no `/sys/class/powercap` (WSL, VMs, macOS, Windows).
        let reading = read_rapl_energy_uj();

        #[cfg(not(target_os = "linux"))]
        assert_eq!(reading, None);

        #[cfg(target_os = "linux")]
        {
            // If this box does expose a package counter, the reader must
            // actually return it — a real RAPL chip must never read as None.
            let has_counter = PathBuf::from(POWERCAP_ROOT)
                .join("intel-rapl:0")
                .join(RAPL_ENERGY_FILE)
                .is_file();
            if has_counter {
                assert!(
                    reading.is_some(),
                    "a readable intel-rapl:0 counter must parse"
                );
            }
        }
    }

    // ---- Monitor + fake sampler --------------------------------------------

    #[test]
    fn fake_sampler_reports_exact_aggregates() {
        let canned = vec![
            Sample {
                ram_bytes: mib(100),
                cpu_pct: 10.0,
            },
            Sample {
                ram_bytes: mib(400),
                cpu_pct: 30.0,
            },
            Sample {
                ram_bytes: mib(250),
                cpu_pct: 20.0,
            },
            Sample {
                ram_bytes: mib(120),
                cpu_pct: 40.0,
            },
            Sample {
                ram_bytes: mib(310),
                cpu_pct: 50.0,
            },
        ];
        let taken = Arc::new(AtomicUsize::new(0));
        let monitor = Monitor::start_with_interval(
            Box::new(FakeSampler {
                samples: canned.clone(),
                energy: None,
                taken: Arc::clone(&taken),
            }),
            20,
        )
        .expect("monitor should start");
        sleep(Duration::from_millis(120));
        let stats = monitor.stop();

        // The wall clock decides how many readings fit in 120 ms; the fake
        // records that count, so the expected aggregates are exact for
        // *whatever* was actually sampled. Past the end of the canned list the
        // fake keeps replaying its last reading, so the replayed sequence is
        // `canned` truncated to `n` (never padded).
        let n = taken.load(Ordering::Relaxed);
        assert!(n >= 1, "monitor should have sampled at least once");
        assert_eq!(stats.samples, n);
        assert_eq!(stats.interval_ms, 20);

        let replayed: Vec<Sample> = (0..n).map(|i| canned[i.min(canned.len() - 1)]).collect();
        assert_eq!(
            stats.ram_peak_bytes,
            replayed
                .iter()
                .map(|s| s.ram_bytes)
                .max()
                .expect("non-empty")
        );
        let mean = replayed.iter().map(|s| s.cpu_pct).sum::<f64>() / n as f64;
        assert!(
            (stats.cpu_avg_pct - mean).abs() < 1e-9,
            "avg {} != {mean}",
            stats.cpu_avg_pct
        );
        assert_eq!(
            stats.cpu_peak_pct,
            replayed.iter().map(|s| s.cpu_pct).fold(0.0, f64::max)
        );
        // A fake sampler with no energy source must never report power.
        assert_eq!(stats.power_avg_watts, None);
    }

    #[test]
    fn monitor_records_power_when_the_sampler_supplies_energy() {
        // Canned RAPL counter climbing by 1 J per tick. Wall-clock timing
        // makes the exact watt figure unknowable, but it must be present,
        // finite and positive.
        let energy: Vec<u64> = (0..200u64).map(|i| 1_000_000 * i).collect();
        let monitor = Monitor::start_with_interval(
            Box::new(FakeSampler::with_energy(
                vec![Sample {
                    ram_bytes: mib(1),
                    cpu_pct: 5.0,
                }],
                energy,
            )),
            20,
        )
        .expect("monitor should start");
        sleep(Duration::from_millis(120));
        let stats = monitor.stop();

        assert!(stats.samples >= 2, "power needs at least two readings");
        let watts = stats.power_avg_watts.expect("power should be reported");
        assert!(watts.is_finite(), "watts {watts} not finite");
        assert!(watts > 0.0, "watts {watts} should be positive");
    }

    #[test]
    fn dropping_a_monitor_stops_its_thread() {
        let taken = Arc::new(AtomicUsize::new(0));
        let monitor = Monitor::start_with_interval(
            Box::new(FakeSampler {
                samples: vec![Sample {
                    ram_bytes: 1,
                    cpu_pct: 1.0,
                }],
                energy: None,
                taken: Arc::clone(&taken),
            }),
            10,
        )
        .expect("monitor should start");
        sleep(Duration::from_millis(50));
        drop(monitor);

        // After the drop the thread is joined, so the count can never grow
        // again. Give it a couple of periods to prove it.
        let after_drop = taken.load(Ordering::Relaxed);
        sleep(Duration::from_millis(50));
        assert_eq!(taken.load(Ordering::Relaxed), after_drop);
        assert!(after_drop >= 1);
    }

    // ---- real sampler smoke -------------------------------------------------

    #[test]
    fn short_real_run_samples_this_process() {
        // Bound by LOGICAL cores, not physical: the test process runs other
        // test threads in parallel, so whole-process CPU% can legitimately
        // exceed physical_count * 100. The 1.5x headroom absorbs short-window
        // (25 ms) sampling noise on loaded CI machines. Exact accounting is
        // covered by the fake-sampler tests; this is only a smoke test.
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1) as f64;
        let bound = logical * 100.0 * 1.5;
        let started = Instant::now();
        let monitor =
            Monitor::start_sysinfo(25).expect("monitor should start against the real sampler");
        sleep(Duration::from_millis(150));
        let stats = monitor.stop();
        let elapsed = started.elapsed();

        assert_eq!(stats.interval_ms, 25);
        assert!(
            stats.samples >= 3,
            "expected >= 3 samples in 150 ms, got {}",
            stats.samples
        );
        assert!(stats.ram_peak_bytes > 0, "own RSS should never be 0 bytes");
        assert!(
            stats.cpu_avg_pct.is_finite() && stats.cpu_avg_pct >= 0.0,
            "cpu avg {} out of bounds",
            stats.cpu_avg_pct
        );
        // Total-CPU semantics: a multi-threaded process can exceed 100 %.
        assert!(
            stats.cpu_peak_pct <= bound,
            "cpu peak {} exceeds logical-core bound {} ({} logical cores)",
            stats.cpu_peak_pct,
            bound,
            logical
        );
        if let Some(watts) = stats.power_avg_watts {
            assert!(
                watts.is_finite() && watts >= 0.0,
                "watts {watts} out of bounds"
            );
        }
        assert!(
            elapsed < Duration::from_secs(2),
            "monitor run took {elapsed:?}"
        );
    }

    #[test]
    fn sysinfo_sampler_can_target_an_explicit_pid() {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        let mut sampler = SysinfoSampler::with_pid(pid);
        let first = sampler.sample();
        sleep(Duration::from_millis(60));
        let second = sampler.sample();
        assert!(first.ram_bytes > 0 && second.ram_bytes > 0);
        assert!(first.cpu_pct.is_finite() && second.cpu_pct.is_finite());
    }

    #[test]
    fn default_sysinfo_sampler_targets_this_process() {
        let mut sampler = SysinfoSampler::default();
        let sample = sampler.sample();
        assert!(sample.ram_bytes > 0);
    }

    /// Regression guard for the sampling rate: `Process::cpu_usage()` is only
    /// recomputed every `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` (200 ms on
    /// Windows), so reading it at the 100 ms default reports a flat 0 %.
    /// This spins real threads and requires a non-zero reading anyway.
    #[test]
    fn cpu_percentage_is_non_zero_while_the_process_is_busy() {
        let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let busy = Arc::clone(&busy);
                std::thread::spawn(move || {
                    let mut acc = 0u64;
                    while !busy.load(Ordering::Relaxed) {
                        for i in 0..50_000u64 {
                            acc = acc.wrapping_add(i ^ (acc >> 3));
                        }
                    }
                    std::hint::black_box(acc);
                })
            })
            .collect();

        let monitor = Monitor::start_sysinfo(MIN_INTERVAL_MS).expect("monitor should start");
        sleep(Duration::from_millis(300));
        let stats = monitor.stop();
        busy.store(true, Ordering::Relaxed);
        for worker in workers {
            let _ = worker.join();
        }

        assert!(
            stats.samples >= 3,
            "expected several samples, got {}",
            stats.samples
        );
        // Two spinning threads cannot read below ~1 % even on a heavily
        // oversubscribed CI runner, where the wall-clock window stretches and
        // only drags the percentage down.
        assert!(
            stats.cpu_peak_pct > 1.0,
            "busy process reported cpu peak {}",
            stats.cpu_peak_pct
        );
        assert!(stats.cpu_avg_pct > 0.0, "busy process reported cpu avg 0");
    }

    // ---- CPU conversion (pure) ----------------------------------------------

    #[test]
    fn cpu_pct_from_ms_converts_cpu_milliseconds() {
        // One saturated core for a second.
        assert_eq!(cpu_pct_from_ms(1_000, Duration::from_secs(1)), Some(100.0));
        // Half a core.
        assert_eq!(cpu_pct_from_ms(500, Duration::from_secs(1)), Some(50.0));
        // Multi-threaded: 4 cores' worth of CPU in 1 s is 400 %.
        assert_eq!(cpu_pct_from_ms(4_000, Duration::from_secs(1)), Some(400.0));
        // Sub-second window, same ratio: 100 ms in 50 ms of wall clock.
        assert_eq!(cpu_pct_from_ms(100, Duration::from_millis(50)), Some(200.0));
        // Idle process.
        assert_eq!(cpu_pct_from_ms(0, Duration::from_millis(100)), Some(0.0));
    }

    #[test]
    fn cpu_pct_from_ms_rejects_degenerate_windows() {
        assert_eq!(cpu_pct_from_ms(1_000, Duration::ZERO), None);
        // A counter reset must clamp to 0 %, never go negative.
        assert_eq!(cpu_pct_from_ms(0, Duration::from_millis(10)), Some(0.0));
    }

    // ---- JSON contract ------------------------------------------------------

    #[test]
    fn resource_stats_json_roundtrip() {
        let stats = ResourceStats {
            ram_peak_bytes: 1_073_741_824,
            cpu_avg_pct: 42.5,
            cpu_peak_pct: 87.25,
            power_avg_watts: Some(31.75),
            samples: 57,
            interval_ms: 100,
        };
        let json = serde_json::to_string(&stats).expect("serialize ResourceStats");
        for key in [
            "ram_peak_bytes",
            "cpu_avg_pct",
            "cpu_peak_pct",
            "power_avg_watts",
            "samples",
            "interval_ms",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }

        let back: ResourceStats = serde_json::from_str(&json).expect("deserialize ResourceStats");
        assert_eq!(back, stats);
    }

    #[test]
    fn resource_stats_deserializes_without_the_optional_power_field() {
        // JSON written by a platform with no RAPL, or by an older build, must
        // still parse with `power_avg_watts` defaulting to None.
        let json = r#"{
            "ram_peak_bytes": 268435456,
            "cpu_avg_pct": 15.0,
            "cpu_peak_pct": 22.0,
            "samples": 12,
            "interval_ms": 100
        }"#;
        let stats: ResourceStats = serde_json::from_str(json).expect("deserialize ResourceStats");
        assert_eq!(stats.ram_peak_bytes, 268_435_456);
        assert_eq!(stats.samples, 12);
        assert_eq!(stats.power_avg_watts, None);
    }

    #[test]
    fn empty_stats_serialize_and_carry_the_interval() {
        let stats = ResourceStats::empty(25);
        let json = serde_json::to_string(&stats).expect("serialize empty ResourceStats");
        let back: ResourceStats = serde_json::from_str(&json).expect("deserialize ResourceStats");
        assert_eq!(back, stats);
        assert_eq!(back.samples, 0);
        assert_eq!(back.interval_ms, 25);
    }
}
