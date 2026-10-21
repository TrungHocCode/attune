//! Memory bandwidth kernels — read / write / copy (M2 slice 2).
//!
//! STREAM-like: pre-touch one (or two) buffers so page faults are not in the
//! measurement, split into `threads` contiguous chunks, one warm-up pass
//! discarded, then `reps` timed passes. Quick mode reports best-of-reps;
//! full mode reports median + best so a noisy rep cannot drag the headline.
//!
//! GB/s is decimal (1e9 bytes) to match vendor DRAM figures; buffer sizes
//! are MiB (binary), matching retail RAM advertising.
//!
//! No `.unwrap()` in non-test code (AGENTS.md §3.2) — allocation failures
//! surface as `BenchError::Alloc`, worker panics as `BenchError::WorkerPanicked`.

use std::time::Instant;

use attune_hal::HardwareInfo;
use serde::{Deserialize, Serialize};

use crate::BenchError;

/// One mebibyte, in bytes.
pub const MIB: u64 = 1024 * 1024;
/// One (decimal) gigabyte, in bytes — the divisor behind GB/s numbers.
pub const GB: f64 = 1_000_000_000.0;

/// Default working-set size: big enough to blow past any desktop L3, small
/// enough to allocate and pre-touch in well under a second.
pub const DEFAULT_BUFFER_MIB: u64 = 256;
/// Floor for the working set. Below this the measurement is dominated by
/// cache/TLB noise rather than DRAM.
pub const MIN_BUFFER_MIB: u64 = 16;
/// Timed passes for `--quick` (best-of wins).
pub const DEFAULT_REPS: usize = 3;
/// Timed passes for full mode (median reported, best also captured).
pub const FULL_REPS: usize = 5;
/// Passes executed and discarded before timing starts.
pub const WARMUP_PASSES: usize = 1;
/// Working set must be at least this multiple of L3, otherwise we measure
/// cache bandwidth and call it DRAM bandwidth.
pub const L3_MULTIPLIER: u64 = 4;
/// Never take more than 1/`RAM_BUDGET_DIVISOR` of currently-available RAM —
/// allocating "all of RAM" would push the page cache out and skew the result.
pub const RAM_BUDGET_DIVISOR: u64 = 4;
/// Constant written by the write kernel. 0xA5 (1010 0101) is alternating bits
/// so DRAM DQ flipflops actually toggle on every transfer — no write-combining
/// short-circuits the measurement.
pub const FILL_BYTE: u8 = 0xA5;

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

/// Per-kernel statistics reported by the full sweep.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KernelStats {
    /// Median GB/s across `reps` timed passes (warm-up excluded).
    pub median_gbps: f64,
    /// Best GB/s across `reps` timed passes (warm-up excluded).
    pub best_gbps: f64,
}

/// Combined output of the read / write / copy sweep. JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FullBandwidthResult {
    pub read: KernelStats,
    pub write: KernelStats,
    pub copy: KernelStats,
    /// Working-set size for one buffer (MiB). The copy kernel allocates 2×.
    pub buffer_mib: u64,
    /// Worker threads used (one contiguous chunk each).
    pub threads: usize,
    /// Timed passes per kernel (warm-up excluded).
    pub reps: usize,
}

/// Benchmark parameters. Build with [`BandwidthConfig::from_hardware`] for
/// the quick path or [`BandwidthConfig::full_from_hardware`] for the full
/// read/write/copy sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandwidthConfig {
    pub buffer_bytes: u64,
    pub threads: usize,
    pub reps: usize,
}

impl BandwidthConfig {
    /// Quick-path default: 256 MiB working set (raised to ≥4× L3, clamped to
    /// ≤¼ of available RAM) spread over the physical core count, 3 reps.
    pub fn from_hardware(hw: &HardwareInfo) -> Self {
        let buffer_bytes = choose_buffer_bytes(hw.memory.available_bytes, hw.cache.l3_bytes);
        let threads = usize::try_from(hw.cpu.physical_cores.max(1)).unwrap_or(1);
        Self {
            buffer_bytes,
            threads,
            reps: DEFAULT_REPS,
        }
    }

    /// Full-sweep default: same sizing as quick, but `FULL_REPS` timed passes
    /// so the median is a real median (≥5 samples).
    pub fn full_from_hardware(hw: &HardwareInfo) -> Self {
        let mut cfg = Self::from_hardware(hw);
        cfg.reps = FULL_REPS;
        cfg
    }
}

/// Pure conversion: bytes moved in `seconds` → decimal GB/s.
/// Kept separate from the timing loop so the byte accounting is unit-testable
/// without touching the allocator or the scheduler.
pub fn gbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 || !seconds.is_finite() {
        return 0.0;
    }
    bytes as f64 / seconds / GB
}

/// Pure median: sorts in place, returns the middle value (or the mean of the
/// two middle values for even `n`). `0.0` for an empty input — never panics.
pub fn median_f64(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        let hi = samples[n / 2];
        let lo = samples[n / 2 - 1];
        (lo + hi) / 2.0
    }
}

/// Pure spread: `(max - min) / median` expressed as a percentage. Returns
/// `0.0` for any non-positive or non-finite median to keep callers safe
/// against a corrupt first rep.
pub fn spread_pct(min: f64, max: f64, median: f64) -> f64 {
    if !median.is_finite() || median <= 0.0 {
        return 0.0;
    }
    let span = max - min;
    if span <= 0.0 || !span.is_finite() {
        return 0.0;
    }
    span / median * 100.0
}

/// Working-set sizing rule (pure — no HAL calls, no allocation).
/// See module docs for the algorithm.
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

/// One-shot convenience entry point used by `attune bench` (full mode).
pub fn quick_full() -> Result<FullBandwidthResult, BenchError> {
    let hw = HardwareInfo::detect()?;
    full_bandwidth(&BandwidthConfig::full_from_hardware(&hw))
}

/// Run the read kernel and return the best of `reps` passes.
pub fn read_bandwidth(cfg: &BandwidthConfig) -> Result<ReadResult, BenchError> {
    if cfg.reps == 0 {
        return Err(BenchError::InvalidConfig("reps must be >= 1".to_string()));
    }
    let (_threads, per_thread, buffer) = setup_buffer(cfg.buffer_bytes, cfg.threads)?;
    let chunks: Vec<&[u8]> = buffer.chunks(per_thread).collect();

    for _ in 0..WARMUP_PASSES {
        // Bytes counted here are intentionally dropped: this pass only
        // warms caches/TLB and the freshly mapped pages.
        let _ = run_read_pass(&chunks)?;
    }

    let mut best_gbps = 0.0f64;
    for _ in 0..cfg.reps {
        let started = Instant::now();
        let bytes_read = run_read_pass(&chunks)?;
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

/// Run the full read + write + copy sweep. Each kernel runs its own warm-up
/// pass plus `reps` timed passes; we return median + best GB/s for every
/// kernel. Validates config once up front.
pub fn full_bandwidth(cfg: &BandwidthConfig) -> Result<FullBandwidthResult, BenchError> {
    if cfg.buffer_bytes == 0 {
        return Err(BenchError::InvalidConfig(
            "buffer_bytes must be > 0".to_string(),
        ));
    }
    if cfg.threads == 0 {
        return Err(BenchError::InvalidConfig(
            "threads must be >= 1".to_string(),
        ));
    }
    if cfg.reps == 0 {
        return Err(BenchError::InvalidConfig("reps must be >= 1".to_string()));
    }

    let read = read_bandwidth_full(cfg)?;
    let write = write_bandwidth_full(cfg)?;
    let copy = copy_bandwidth_full(cfg)?;

    Ok(FullBandwidthResult {
        read,
        write,
        copy,
        buffer_mib: cfg.buffer_bytes / MIB,
        threads: cfg.threads,
        reps: cfg.reps,
    })
}

// ---- full-mode per-kernel drivers -----------------------------------------

fn read_bandwidth_full(cfg: &BandwidthConfig) -> Result<KernelStats, BenchError> {
    let (_threads, per_thread, buffer) = setup_buffer(cfg.buffer_bytes, cfg.threads)?;
    let chunks: Vec<&[u8]> = buffer.chunks(per_thread).collect();

    // Warm-up pass is run and discarded (AGENTS.md §3 + STREAM convention).
    let _ = run_read_pass(&chunks)?;

    let mut samples: Vec<f64> = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let started = Instant::now();
        let bytes = run_read_pass(&chunks)?;
        let elapsed = started.elapsed().as_secs_f64();
        samples.push(gbps(bytes, elapsed));
    }
    Ok(summarize(&mut samples))
}

fn write_bandwidth_full(cfg: &BandwidthConfig) -> Result<KernelStats, BenchError> {
    let (_threads, per_thread, mut buffer) = setup_buffer(cfg.buffer_bytes, cfg.threads)?;
    let bytes_per_pass = cfg.buffer_bytes;

    run_write_pass(&mut buffer, per_thread)?; // warm-up pass

    let mut samples: Vec<f64> = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let started = Instant::now();
        run_write_pass(&mut buffer, per_thread)?;
        let elapsed = started.elapsed().as_secs_f64();
        samples.push(gbps(bytes_per_pass, elapsed));
    }
    Ok(summarize(&mut samples))
}

fn copy_bandwidth_full(cfg: &BandwidthConfig) -> Result<KernelStats, BenchError> {
    let (_threads, per_thread, src) = setup_buffer(cfg.buffer_bytes, cfg.threads)?;
    let len = usize::try_from(cfg.buffer_bytes).unwrap_or(usize::MAX);
    let mut dst: Vec<u8> = Vec::new();
    dst.try_reserve_exact(len)
        .map_err(|e| BenchError::Alloc(cfg.buffer_bytes, e))?;
    dst.resize(len, 0);

    // STREAM "copy" bills 2 × buffer_bytes per pass (one read of `src` plus
    // one write of `dst` for every element). Reporting that way means the
    // copy number is directly comparable to read+write in GB/s.
    let bytes_per_pass = cfg.buffer_bytes.saturating_mul(2);

    run_copy_pass(&src, &mut dst, per_thread)?; // warm-up pass

    let mut samples: Vec<f64> = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let started = Instant::now();
        run_copy_pass(&src, &mut dst, per_thread)?;
        let elapsed = started.elapsed().as_secs_f64();
        samples.push(gbps(bytes_per_pass, elapsed));
    }
    Ok(summarize(&mut samples))
}

// ---- internal helpers -----------------------------------------------------

/// Common setup: validate config, allocate + prefill one buffer, return the
/// per-thread chunk size in bytes.
fn setup_buffer(
    buffer_bytes: u64,
    requested_threads: usize,
) -> Result<(usize, usize, Vec<u8>), BenchError> {
    if buffer_bytes == 0 {
        return Err(BenchError::InvalidConfig(
            "buffer_bytes must be > 0".to_string(),
        ));
    }
    if requested_threads == 0 {
        return Err(BenchError::InvalidConfig(
            "threads must be >= 1".to_string(),
        ));
    }
    let buffer = prefill(buffer_bytes)?;
    let threads = requested_threads
        .min(usize::try_from(buffer_bytes).unwrap_or(1))
        .max(1);
    let thread_count = u64::try_from(threads).unwrap_or(u64::MAX).max(1);
    // `threads <= buffer_bytes`, so this division cannot yield a zero chunk
    // size (`slice::chunks(0)` panics).
    let per_thread = usize::try_from(buffer_bytes / thread_count)
        .unwrap_or(1)
        .max(1);
    Ok((threads, per_thread, buffer))
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

/// Summarise a sample vector of GB/s readings into (median, best).
fn summarize(samples: &mut [f64]) -> KernelStats {
    let best = samples.iter().copied().fold(0.0_f64, f64::max);
    let median = median_f64(samples);
    KernelStats {
        median_gbps: median,
        best_gbps: best,
    }
}

// ---- per-pass execution ---------------------------------------------------

/// One read pass over every chunk, each on its own scoped worker thread.
/// Returns the total number of bytes streamed.
fn run_read_pass(chunks: &[&[u8]]) -> Result<u64, BenchError> {
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

/// One write pass: each worker fills its chunk with [`FILL_BYTE`].
fn run_write_pass(buffer: &mut [u8], per_thread: usize) -> Result<(), BenchError> {
    let chunks: Vec<&mut [u8]> = buffer.chunks_mut(per_thread.max(1)).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| scope.spawn(move || fill_write(chunk)))
            .collect();
        let mut sum: u64 = 0;
        for handle in handles {
            sum = sum.wrapping_add(handle.join().map_err(|_| BenchError::WorkerPanicked)?);
        }
        std::hint::black_box(sum);
        Ok(())
    })
}

/// One copy pass: each worker copies its slice of `src` into its slice of
/// `dst`. `src` and `dst` MUST be the same length.
fn run_copy_pass(src: &[u8], dst: &mut [u8], per_thread: usize) -> Result<(), BenchError> {
    debug_assert_eq!(src.len(), dst.len());
    let s_chunks: Vec<&[u8]> = src.chunks(per_thread).collect();
    let d_chunks: Vec<&mut [u8]> = dst.chunks_mut(per_thread.max(1)).collect();
    if s_chunks.len() != d_chunks.len() {
        return Err(BenchError::InvalidConfig(
            "copy pass: src/dst chunk count mismatch".to_string(),
        ));
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = s_chunks
            .into_iter()
            .zip(d_chunks)
            .map(|(s, d)| scope.spawn(move || copy_bytes(s, d)))
            .collect();
        let mut sum: u64 = 0;
        for handle in handles {
            sum = sum.wrapping_add(handle.join().map_err(|_| BenchError::WorkerPanicked)?);
        }
        std::hint::black_box(sum);
        Ok(())
    })
}

// ---- per-chunk kernels ----------------------------------------------------

/// Streaming read of `chunk` returning a checksum of every byte.
/// Four independent `u64` accumulators so a single loop-carried add cannot
/// cap the thread at 1 byte/cycle.
fn sum_read(chunk: &[u8]) -> u64 {
    let (mut a0, mut a1, mut a2, mut a3) = (0u64, 0u64, 0u64, 0u64);
    // `as_chunks` (stable since 1.88) gives fixed-size groups without any
    // per-byte bounds checks.
    let (groups, tail) = chunk.as_chunks::<32>();
    for group in groups {
        let (words, _) = group.as_chunks::<8>();
        a0 = a0.wrapping_add(u64::from_ne_bytes(words[0]));
        a1 = a1.wrapping_add(u64::from_ne_bytes(words[1]));
        a2 = a2.wrapping_add(u64::from_ne_bytes(words[2]));
        a3 = a3.wrapping_add(u64::from_ne_bytes(words[3]));
    }
    for &byte in tail {
        a0 = a0.wrapping_add(u64::from(byte));
    }
    // Wrapping *sum*, not XOR: a uniform buffer gives all four lanes the same
    // value, and XOR would cancel them to 0.
    let checksum = a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3);
    std::hint::black_box(checksum)
}

/// Streaming write of `FILL_BYTE` into `chunk`. The accumulator is updated
/// per byte alongside the store so the writes are anchored — the compiler
/// can prove the stores are dead only by killing the sum, which is then
/// `black_box`'d at the end.
fn fill_write(chunk: &mut [u8]) -> u64 {
    let mut sum: u64 = 0;
    let (groups, tail) = chunk.as_chunks_mut::<32>();
    for group in groups {
        for byte in group {
            *byte = FILL_BYTE;
            sum = sum.wrapping_add(u64::from(FILL_BYTE));
        }
    }
    for byte in tail {
        *byte = FILL_BYTE;
        sum = sum.wrapping_add(u64::from(FILL_BYTE));
    }
    std::hint::black_box(sum)
}

/// Byte-by-byte copy of `src` into `dst` — deliberately NOT `copy_from_slice`.
/// `copy_from_slice` lowers to a memcpy intrinsic that beats the rest of the
/// suite and hides DRAM contention; we want the same stream-bandwidth
/// measurement as the read and write kernels.
fn copy_bytes(src: &[u8], dst: &mut [u8]) -> u64 {
    debug_assert_eq!(src.len(), dst.len());
    let mut sum: u64 = 0;
    let (sg, st) = src.as_chunks::<32>();
    let (dg, dt) = dst.as_chunks_mut::<32>();
    for (s_group, d_group) in sg.iter().zip(dg.iter_mut()) {
        for (s_byte, d_byte) in s_group.iter().zip(d_group.iter_mut()) {
            *d_byte = *s_byte;
            sum = sum.wrapping_add(u64::from(*s_byte));
        }
    }
    for (s_byte, d_byte) in st.iter().zip(dt.iter_mut()) {
        *d_byte = *s_byte;
        sum = sum.wrapping_add(u64::from(*s_byte));
    }
    std::hint::black_box(sum)
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
    fn fill_write_sets_every_byte_and_anchors() {
        let mut buf = vec![0u8; 1024];
        let sum = fill_write(&mut buf);
        // Every byte must be FILL_BYTE — proves the stores happened.
        assert!(buf.iter().all(|&b| b == FILL_BYTE));
        // Anchor sum must be deterministic: 1024 * FILL_BYTE.
        assert_eq!(sum, u64::from(FILL_BYTE) * 1024);
    }

    #[test]
    fn fill_write_handles_short_tail() {
        let mut buf = vec![0u8; 31]; // not a multiple of 32
        let sum = fill_write(&mut buf);
        assert!(buf.iter().all(|&b| b == FILL_BYTE));
        assert_eq!(sum, u64::from(FILL_BYTE) * 31);

        let mut buf = vec![0u8; 1];
        let sum = fill_write(&mut buf);
        assert_eq!(buf[0], FILL_BYTE);
        assert_eq!(sum, u64::from(FILL_BYTE));
    }

    #[test]
    fn copy_bytes_moves_every_byte() {
        let src: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();
        let mut dst = vec![0u8; 256];
        let sum = copy_bytes(&src, &mut dst);
        assert_eq!(src, dst);
        // Sum must equal sum of source bytes.
        let expected: u64 = src.iter().map(|&b| u64::from(b)).sum();
        assert_eq!(sum, expected);
    }

    #[test]
    fn copy_bytes_handles_short_tail() {
        // 31 bytes — exercises the <32 byte tail path.
        let src: Vec<u8> = (0..31u8).collect();
        let mut dst = vec![0u8; 31];
        copy_bytes(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn pass_bills_chunk_bytes_not_checksums() {
        // Byte accounting must be the sum of chunk lengths — a buffer whose
        // checksum folds to zero must still report its full size.
        let buffer = vec![0xFFu8; 3000];
        let chunks: Vec<&[u8]> = vec![&buffer[..1000], &buffer[1000..3000]];
        assert_eq!(run_read_pass(&chunks).expect("pass should run"), 3000);
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

    // ---- pure stats ----------------------------------------------------------

    #[test]
    fn median_f64_odd_count() {
        let mut v = [3.0, 1.0, 2.0];
        assert_eq!(median_f64(&mut v), 2.0);
    }

    #[test]
    fn median_f64_even_count() {
        let mut v = [4.0, 1.0, 3.0, 2.0];
        // sorted: [1, 2, 3, 4] -> midpoint between indices 1 and 2 -> 2.5.
        assert_eq!(median_f64(&mut v), 2.5);
    }

    #[test]
    fn median_f64_single_element() {
        let mut v = [42.0];
        assert_eq!(median_f64(&mut v), 42.0);
    }

    #[test]
    fn median_f64_empty_returns_zero() {
        let mut v: [f64; 0] = [];
        assert_eq!(median_f64(&mut v), 0.0);
    }

    #[test]
    fn median_f64_sorts_in_place() {
        let mut v = [5.0, 1.0, 3.0, 2.0, 4.0];
        let _ = median_f64(&mut v);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn median_f64_handles_nan_as_equal() {
        // NaN comparisons are partial; we collapse them to Equal so the call
        // never panics and still produces a finite number.
        let mut v = [f64::NAN, 1.0, 2.0];
        let m = median_f64(&mut v);
        assert!(m.is_finite());
    }

    #[test]
    fn spread_pct_normal_case() {
        // (120 - 80) / 100 = 40%
        assert!((spread_pct(80.0, 120.0, 100.0) - 40.0).abs() < 1e-9);
    }

    #[test]
    fn spread_pct_equal_min_max() {
        assert_eq!(spread_pct(50.0, 50.0, 50.0), 0.0);
    }

    #[test]
    fn spread_pct_zero_or_negative_median_is_zero() {
        assert_eq!(spread_pct(-10.0, 10.0, 0.0), 0.0);
        assert_eq!(spread_pct(-10.0, 10.0, -1.0), 0.0);
        assert_eq!(spread_pct(-10.0, 10.0, f64::NAN), 0.0);
        assert_eq!(spread_pct(-10.0, 10.0, f64::INFINITY), 0.0);
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

        let zero = full_bandwidth(&BandwidthConfig {
            buffer_bytes: 0,
            threads: 1,
            reps: 1,
        });
        assert!(matches!(zero, Err(BenchError::InvalidConfig(_))));
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

    #[test]
    fn smoke_write_kernel_8mib_1thread() {
        let cfg = test_cfg(8, 1, 1);
        let started = Instant::now();
        let stats = write_bandwidth_full(&cfg).expect("write kernel should run");
        let elapsed = started.elapsed();

        // Plausibility band: realistic DRAM tops out around 80 GB/s on a
        // single channel; we keep an order-of-magnitude lower bound and a
        // very wide upper bound so the test never flakes on slow VMs.
        assert!(
            stats.median_gbps > 0.5,
            "expected > 0.5 GB/s, got {}",
            stats.median_gbps
        );
        assert!(
            stats.median_gbps < 500.0,
            "implausible write bandwidth {} GB/s",
            stats.median_gbps
        );
        assert!(
            stats.best_gbps >= stats.median_gbps,
            "best {} < median {}",
            stats.best_gbps,
            stats.median_gbps
        );
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "write smoke took {elapsed:?}"
        );
    }

    #[test]
    fn smoke_copy_kernel_8mib_1thread() {
        let cfg = test_cfg(8, 1, 1);
        let started = Instant::now();
        let stats = copy_bandwidth_full(&cfg).expect("copy kernel should run");
        let elapsed = started.elapsed();

        assert!(
            stats.median_gbps > 0.5,
            "expected > 0.5 GB/s, got {}",
            stats.median_gbps
        );
        assert!(
            stats.median_gbps < 500.0,
            "implausible copy bandwidth {} GB/s",
            stats.median_gbps
        );
        assert!(
            stats.best_gbps >= stats.median_gbps,
            "best {} < median {}",
            stats.best_gbps,
            stats.median_gbps
        );
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "copy smoke took {elapsed:?}"
        );
    }

    #[test]
    fn full_sweep_reports_three_kernels() {
        let cfg = test_cfg(8, 1, 3);
        let res = full_bandwidth(&cfg).expect("full sweep should run");
        assert_eq!(res.buffer_mib, 8);
        assert_eq!(res.threads, 1);
        assert_eq!(res.reps, 3);
        assert!(res.read.median_gbps > 0.0);
        assert!(res.write.median_gbps > 0.0);
        assert!(res.copy.median_gbps > 0.0);
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

    #[test]
    fn full_result_json_roundtrip() {
        let res = FullBandwidthResult {
            read: KernelStats {
                median_gbps: 22.0,
                best_gbps: 25.0,
            },
            write: KernelStats {
                median_gbps: 14.0,
                best_gbps: 16.0,
            },
            copy: KernelStats {
                median_gbps: 20.0,
                best_gbps: 23.0,
            },
            buffer_mib: 256,
            threads: 8,
            reps: 5,
        };
        let json = serde_json::to_string(&res).expect("serialize FullBandwidthResult");
        assert!(json.contains("\"read\""), "missing read: {json}");
        assert!(json.contains("\"write\""), "missing write: {json}");
        assert!(json.contains("\"copy\""), "missing copy: {json}");
        assert!(
            json.contains("\"buffer_mib\""),
            "missing buffer_mib: {json}"
        );
        assert!(json.contains("\"threads\""), "missing threads: {json}");
        assert!(json.contains("\"reps\""), "missing reps: {json}");
        assert!(
            json.contains("\"median_gbps\""),
            "missing median_gbps: {json}"
        );
        assert!(json.contains("\"best_gbps\""), "missing best_gbps: {json}");

        let back: FullBandwidthResult =
            serde_json::from_str(&json).expect("deserialize FullBandwidthResult");
        assert_eq!(back, res);
    }
}
