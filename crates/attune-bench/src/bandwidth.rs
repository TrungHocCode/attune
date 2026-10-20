//! Memory bandwidth kernel — **read path** (M2 slice 1).
//!
//! Method (STREAM-like, deliberately simple):
//! 1. Allocate one contiguous buffer, pre-touch every byte with an
//!    index-derived pattern so page faults are NOT part of the measurement.
//! 2. Split the buffer into `threads` contiguous chunks — one per worker.
//! 3. One warm-up pass is run and thrown away (fills caches/TLB, warms the
//!    allocator) so rep #1 is not penalised.
//! 4. `reps` timed passes; we report the **best** (max) GB/s, which is the
//!    least noise-contaminated estimate of peak bandwidth.
//!
//! Units: GB/s is **decimal** (1 GB = 1e9 bytes) so the number is directly
//! comparable to vendor DRAM figures; buffer sizes are reported in MiB
//! (binary), matching how RAM is advertised on retail hardware.
//!
//! Per AGENTS.md §3.2 no `.unwrap()` appears in non-test code — allocation
//! failures surface as `BenchError::Alloc`, worker panics as
//! `BenchError::WorkerPanicked`.

use std::time::Instant;

use attune_hal::HardwareInfo;
use serde::{Deserialize, Serialize};

use crate::BenchError;

/// One mebibyte, in bytes.
pub const MIB: u64 = 1024 * 1024;
/// One (decimal) gigabyte, in bytes — the divisor behind `read_gbps`.
pub const GB: f64 = 1_000_000_000.0;

/// Default working-set size: big enough to blow past any desktop L3, small
/// enough to allocate and pre-touch in well under a second.
pub const DEFAULT_BUFFER_MIB: u64 = 256;
/// Floor for the working set. Below this the measurement is dominated by
/// cache/TLB noise rather than DRAM.
pub const MIN_BUFFER_MIB: u64 = 16;
/// Timed passes; the best one wins.
pub const DEFAULT_REPS: usize = 3;
/// Passes executed and discarded before timing starts.
pub const WARMUP_PASSES: usize = 1;
/// Working set must be at least this multiple of L3, otherwise we measure
/// cache bandwidth and call it DRAM bandwidth.
pub const L3_MULTIPLIER: u64 = 4;
/// Never take more than 1/`RAM_BUDGET_DIVISOR` of currently-available RAM —
/// allocating "all of RAM" would push the page cache out and skew the result.
pub const RAM_BUDGET_DIVISOR: u64 = 4;

/// Result of a read-bandwidth measurement. Field names are the JSON contract
/// consumed by `attune bench --quick --json`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReadResult {
    /// Best observed read bandwidth in decimal GB/s.
    pub read_gbps: f64,
    /// Working-set size actually allocated (MiB).
    pub buffer_mib: u64,
    /// Worker threads used (one contiguous chunk each).
    pub threads: usize,
    /// Timed passes executed (warm-up passes excluded).
    pub reps: usize,
}

/// Benchmark parameters. Build with [`BandwidthConfig::from_hardware`] for
/// sane defaults, or override any field for a custom sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandwidthConfig {
    pub buffer_bytes: u64,
    pub threads: usize,
    pub reps: usize,
}

impl BandwidthConfig {
    /// Default config for the detected machine: 256 MiB working set (raised
    /// to ≥4× L3, clamped to ≤¼ of available RAM) spread over the physical
    /// core count.
    pub fn from_hardware(hw: &HardwareInfo) -> Self {
        let buffer_bytes = choose_buffer_bytes(hw.memory.available_bytes, hw.cache.l3_bytes);
        let threads = usize::try_from(hw.cpu.physical_cores.max(1)).unwrap_or(1);
        Self {
            buffer_bytes,
            threads,
            reps: DEFAULT_REPS,
        }
    }
}

/// Pure conversion: bytes moved in `seconds` → decimal GB/s.
///
/// Kept separate from the timing loop so the byte accounting is unit-testable
/// without touching the allocator or the scheduler.
pub fn gbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 || !seconds.is_finite() {
        return 0.0;
    }
    bytes as f64 / seconds / GB
}

/// Working-set sizing rule (pure — no HAL calls, no allocation).
///
/// - start at [`DEFAULT_BUFFER_MIB`]
/// - raise to ≥4× L3 when L3 size is known
/// - clamp to ≤¼ of available RAM (fall back to the floor when RAM is
///   already nearly exhausted, so the kernel still runs)
/// - clamp to the [`MIN_BUFFER_MIB`] floor and round down to whole MiB
pub fn choose_buffer_bytes(available_bytes: u64, l3_bytes: u64) -> u64 {
    let mut buffer = DEFAULT_BUFFER_MIB * MIB;

    if l3_bytes > 0 {
        buffer = buffer.max(l3_bytes.saturating_mul(L3_MULTIPLIER));
    }

    if available_bytes > 0 {
        let budget = available_bytes / RAM_BUDGET_DIVISOR;
        buffer = buffer.min(budget.max(MIN_BUFFER_MIB * MIB));
    }

    buffer.max(MIN_BUFFER_MIB * MIB) / MIB * MIB
}

/// One-shot convenience entry point used by `attune bench --quick`.
pub fn quick_read() -> Result<ReadResult, BenchError> {
    let hw = HardwareInfo::detect()?;
    read_bandwidth(&BandwidthConfig::from_hardware(&hw))
}

/// Run the read kernel and return the best of `reps` passes.
pub fn read_bandwidth(cfg: &BandwidthConfig) -> Result<ReadResult, BenchError> {
    if cfg.threads == 0 {
        return Err(BenchError::InvalidConfig(
            "threads must be >= 1".to_string(),
        ));
    }
    if cfg.reps == 0 {
        return Err(BenchError::InvalidConfig("reps must be >= 1".to_string()));
    }
    let threads = cfg
        .threads
        .min(usize::try_from(cfg.buffer_bytes).unwrap_or(1));
    if threads == 0 {
        return Err(BenchError::InvalidConfig(
            "buffer must hold at least one byte per thread".to_string(),
        ));
    }

    let buffer = prefill(cfg.buffer_bytes)?;
    // `threads <= buffer_bytes`, so this division cannot yield a zero chunk
    // size (`slice::chunks(0)` panics).
    let thread_count = u64::try_from(threads).unwrap_or(u64::MAX).max(1);
    let per_thread = usize::try_from(cfg.buffer_bytes / thread_count)
        .unwrap_or(1)
        .max(1);
    let chunks: Vec<&[u8]> = buffer.chunks(per_thread).collect();

    for _ in 0..WARMUP_PASSES {
        // Bytes counted here are intentionally dropped: this pass only
        // warms caches/TLB and the freshly mapped pages.
        let _ = run_pass(&chunks)?;
    }

    let mut best_gbps = 0.0f64;
    for _ in 0..cfg.reps {
        let started = Instant::now();
        let bytes_read = run_pass(&chunks)?;
        let elapsed = started.elapsed().as_secs_f64();
        best_gbps = best_gbps.max(gbps(bytes_read, elapsed));
    }

    Ok(ReadResult {
        read_gbps: best_gbps,
        buffer_mib: cfg.buffer_bytes / MIB,
        threads: chunks.len(),
        reps: cfg.reps,
    })
}

/// Allocate `bytes` and touch every page with an index-derived pattern, so
/// the timed passes measure DRAM traffic and not first-touch page faults.
fn prefill(bytes: u64) -> Result<Vec<u8>, BenchError> {
    let len = usize::try_from(bytes).unwrap_or(usize::MAX);
    let mut buffer: Vec<u8> = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|e| BenchError::Alloc(bytes, e))?;
    buffer.resize(len, 0);
    for (i, byte) in buffer.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(0x9E);
    }
    Ok(buffer)
}

/// One pass over every chunk, each on its own scoped worker thread.
/// Returns the total number of bytes streamed.
fn run_pass(chunks: &[&[u8]]) -> Result<u64, BenchError> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let chunk: &[u8] = chunk;
                scope.spawn(move || sum_read(chunk))
            })
            .collect();

        for handle in handles {
            let checksum = handle.join().map_err(|_| BenchError::WorkerPanicked)?;
            // The checksum only exists to keep the loads alive. Bytes billed
            // are chunk lengths — never the checksum, which a periodic buffer
            // may legitimately fold down to 0.
            std::hint::black_box(checksum);
        }

        Ok(chunks
            .iter()
            .map(|chunk| u64::try_from(chunk.len()).unwrap_or(u64::MAX))
            .sum())
    })
}

/// Streaming read of `chunk` returning a checksum of every byte.
///
/// Four independent `u64` accumulators over 32-byte groups: a single
/// accumulator creates a loop-carried dependency of one add per cycle, which
/// caps a thread at ~1 byte/cycle and would make us report the ALU limit
/// instead of memory bandwidth. `black_box` keeps the optimiser from deleting
/// the loads (AGENTS.md §3.4: std only, no benchmark crate).
fn sum_read(chunk: &[u8]) -> u64 {
    let (mut a0, mut a1, mut a2, mut a3) = (0u64, 0u64, 0u64, 0u64);

    // `as_chunks` (stable since 1.88) gives fixed-size groups without any
    // per-byte bounds checks, which matter here: in an unoptimized build the
    // checks would dominate the loop and we would report the bounds checker,
    // not the memory subsystem.
    let (groups, tail) = chunk.as_chunks::<32>();
    for group in groups {
        let (words, _) = group.as_chunks::<8>();
        a0 = a0.wrapping_add(u64::from_ne_bytes(words[0]));
        a1 = a1.wrapping_add(u64::from_ne_bytes(words[1]));
        a2 = a2.wrapping_add(u64::from_ne_bytes(words[2]));
        a3 = a3.wrapping_add(u64::from_ne_bytes(words[3]));
    }

    // Trailing bytes (< 32) — tail of the buffer only, keep it simple.
    for &byte in tail {
        a0 = a0.wrapping_add(u64::from(byte));
    }

    // Wrapping *sum*, not XOR: a uniform buffer gives all four lanes the same
    // value, and XOR would cancel them to 0.
    let checksum = a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3);
    std::hint::black_box(checksum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(buffer_mib: u64, threads: usize, reps: usize) -> BandwidthConfig {
        BandwidthConfig {
            buffer_bytes: buffer_mib * MIB,
            threads,
            reps,
        }
    }

    // ---- pure byte accounting -------------------------------------------------

    #[test]
    fn gbps_converts_bytes_and_seconds() {
        // 1 GB in 1 s == 1 GB/s; 1 GB in 0.5 s == 2 GB/s.
        assert!((gbps(1_000_000_000, 1.0) - 1.0).abs() < 1e-9);
        assert!((gbps(1_000_000_000, 0.5) - 2.0).abs() < 1e-9);
        // 256 MiB in 4 s == 64 MiB/s == 0.067108864 GB/s.
        assert!((gbps(256 * MIB, 4.0) - 0.067_108_864).abs() < 1e-9);
    }

    #[test]
    fn gbps_is_zero_for_degenerate_durations() {
        assert_eq!(gbps(1_000_000, 0.0), 0.0);
        assert_eq!(gbps(1_000_000, -1.0), 0.0);
        assert_eq!(gbps(1_000_000, f64::NAN), 0.0);
        assert_eq!(gbps(1_000_000, f64::INFINITY), 0.0);
        assert_eq!(gbps(0, 1.0), 0.0);
    }

    #[test]
    fn sum_read_counts_every_byte() {
        assert_eq!(sum_read(&[0u8; 4096]), 0);
        assert_eq!(sum_read(&[0u8; 4096]), sum_read(&[0u8; 4096]));
        assert_ne!(sum_read(&[1u8; 4096]), 0);
        // Odd length exercises the <32 byte tail path.
        assert_ne!(sum_read(&[1u8; 31]), 0);
        assert_ne!(sum_read(&[1u8; 33]), 0);
    }

    #[test]
    fn pass_bills_chunk_bytes_not_checksums() {
        // Byte accounting must be the sum of chunk lengths — a buffer whose
        // checksum folds to zero must still report its full size.
        let buffer = vec![0xFFu8; 3000];
        let chunks: Vec<&[u8]> = vec![&buffer[..1000], &buffer[1000..3000]];
        assert_eq!(run_pass(&chunks).expect("pass should run"), 3000);
    }

    // ---- working-set sizing --------------------------------------------------

    #[test]
    fn buffer_defaults_to_256_mib_on_roomy_machines() {
        let picked = choose_buffer_bytes(64 * 1024 * MIB, 32 * MIB);
        assert_eq!(picked / MIB, DEFAULT_BUFFER_MIB);
    }

    #[test]
    fn buffer_clamped_by_available_ram() {
        // 512 MiB available => at most 128 MiB (1/4 of free RAM).
        let picked = choose_buffer_bytes(512 * MIB, 16 * MIB);
        assert_eq!(picked / MIB, 128);
        // 2 GiB available => budget 512 MiB, above the 256 MiB default, so the
        // default wins (clamping only ever lowers the working set).
        assert_eq!(
            choose_buffer_bytes(2 * 1024 * MIB, 16 * MIB) / MIB,
            DEFAULT_BUFFER_MIB
        );
    }

    #[test]
    fn buffer_raised_to_four_times_l3() {
        // 96 MiB L3 => 384 MiB working set, still MiB-aligned.
        let picked = choose_buffer_bytes(64 * 1024 * MIB, 96 * MIB);
        assert_eq!(picked / MIB, 4 * 96);
        assert_eq!(picked % MIB, 0);
        // L3 that fits the default must not inflate it.
        assert_eq!(
            choose_buffer_bytes(64 * 1024 * MIB, 8 * MIB) / MIB,
            DEFAULT_BUFFER_MIB
        );
    }

    #[test]
    fn buffer_never_below_floor() {
        let picked = choose_buffer_bytes(4 * MIB, 0);
        assert_eq!(picked / MIB, MIN_BUFFER_MIB);
    }

    #[test]
    fn buffer_handles_unknown_ram_and_l3() {
        // available == 0 (sysinfo failed) and l3 == 0 (no CPUID leaf).
        assert_eq!(choose_buffer_bytes(0, 0) / MIB, DEFAULT_BUFFER_MIB);
    }

    // ---- kernel smoke --------------------------------------------------------

    #[test]
    fn smoke_kernel_reports_plausible_bandwidth() {
        let cfg = test_cfg(16, 1, 1);
        let started = Instant::now();
        let result = read_bandwidth(&cfg).expect("read kernel should run");
        let elapsed = started.elapsed();

        assert_eq!(result.buffer_mib, 16);
        assert_eq!(result.threads, 1);
        assert_eq!(result.reps, 1);
        assert!(
            result.read_gbps > 0.5,
            "expected > 0.5 GB/s, got {}",
            result.read_gbps
        );
        assert!(
            result.read_gbps < 500.0,
            "implausible bandwidth {} GB/s",
            result.read_gbps
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "smoke kernel took {elapsed:?}"
        );
    }

    #[test]
    fn multi_thread_run_reports_requested_shape() {
        let cfg = test_cfg(16, 4, 1);
        let result = read_bandwidth(&cfg).expect("read kernel should run");
        assert_eq!(result.threads, 4);
        assert_eq!(result.buffer_mib, 16);
        assert!(result.read_gbps > 0.5);
    }

    #[test]
    fn rejects_invalid_config() {
        let threads = read_bandwidth(&BandwidthConfig {
            buffer_bytes: 16 * MIB,
            threads: 0,
            reps: 1,
        });
        assert!(matches!(threads, Err(BenchError::InvalidConfig(_))));

        let reps = read_bandwidth(&BandwidthConfig {
            buffer_bytes: 16 * MIB,
            threads: 1,
            reps: 0,
        });
        assert!(matches!(reps, Err(BenchError::InvalidConfig(_))));
    }

    #[test]
    fn more_threads_than_bytes_never_panics() {
        // Pathological config: 8 workers for a 4-byte buffer. Worker count is
        // clamped to the buffer so `slice::chunks(0)` can never be reached.
        let result = read_bandwidth(&BandwidthConfig {
            buffer_bytes: 4,
            threads: 8,
            reps: 1,
        })
        .expect("clamped config should run");
        assert_eq!(result.threads, 4);
        assert_eq!(result.buffer_mib, 0);
    }

    // ---- JSON contract -------------------------------------------------------

    #[test]
    fn read_result_json_roundtrip() {
        let result = ReadResult {
            read_gbps: 42.5,
            buffer_mib: 256,
            threads: 8,
            reps: 3,
        };
        let json = serde_json::to_string(&result).expect("serialize ReadResult");
        assert!(json.contains("\"read_gbps\""), "missing read_gbps: {json}");
        assert!(
            json.contains("\"buffer_mib\""),
            "missing buffer_mib: {json}"
        );
        assert!(json.contains("\"threads\""), "missing threads: {json}");
        assert!(json.contains("\"reps\""), "missing reps: {json}");

        let back: ReadResult = serde_json::from_str(&json).expect("deserialize ReadResult");
        assert_eq!(back, result);
    }
}
