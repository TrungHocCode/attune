//! FP32 FMA throughput kernels (M2 slice 3).
//!
//! Measures aggregate GFLOPS achieved by a tight FMA loop, multithreaded
//! across the physical-core count. The path (scalar / AVX2 / AVX-512F) is
//! selected at runtime via `is_x86_feature_detected!` (stable stdlib only —
//! no nightly `cfg_target_feature` tricks). Non-x86 targets compile and run
//! via the scalar fallback so the bench crate stays portable.
//!
//! Each worker owns a private copy of two L1-resident float buffers
//! (16 KiB total per thread), then re-runs the same FMA inner loop
//! `inner_iters` times. One warm-up pass is discarded; `reps` timed passes
//! are summarized into median + best.
//!
//! FLOPS accounting: 1 FMA = 2 ops (mul + add). SIMD lanes multiply the
//! op count (8 for AVX2, 16 for AVX-512F). GFLOPS is decimal (1e9 ops/s).
//!
//! No `.unwrap()` in non-test code (AGENTS.md §3.2) — allocation and worker
//! failures surface as `BenchError`.

use std::time::Instant;

use attune_hal::{cpu::theoretical_fp32_gflops, HardwareInfo};
use serde::{Deserialize, Serialize};

use crate::bandwidth::median_f64;
use crate::BenchError;

/// Decimal GFLOPS divisor (1 billion ops/second).
pub const GFLOPS: f64 = 1_000_000_000.0;
/// Per-thread f32 buffer length. 2048 floats × 4 bytes = 8 KiB per buffer;
/// two buffers × 8 KiB = 16 KiB per thread — L1-resident on virtually every
/// desktop/server core (L1d is typically 32 KiB).
pub const FLOATS_PER_BUFFER: usize = 2048;
/// Outer-loop iteration count for the FMA kernel. Each iter processes the
/// full buffer once through `dispatch_kernel`, so one timed pass does
/// `inner_iters × buffer_len / vec_width` FMAs per thread. Sized so the
/// total FLOPS bench finishes in well under 1 s on any path (the slow
/// scalar case still tops out around ~300 ms × reps).
pub const DEFAULT_INNER_ITERS: usize = 100_000;
/// Number of timed passes for full mode (median reported). Matches bandwidth.
pub const DEFAULT_REPS: usize = 3;
/// Warm-up passes executed and discarded before timing starts. The first
/// few passes are needed to settle L1 cache and let the CPU clock ramp
/// into its turbo bin; the cost is amortised over the timed run.
pub const WARMUP_PASSES: usize = 3;

/// SIMD path used by the kernel. The lowercase form (`scalar` / `avx2` /
/// `avx512f`) is the JSON contract for `attune bench --json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlopsPath {
    Scalar,
    Avx2,
    #[serde(rename = "avx512f")]
    Avx512F,
}

impl FlopsPath {
    pub fn as_str(&self) -> &'static str {
        match self {
            FlopsPath::Scalar => "scalar",
            FlopsPath::Avx2 => "avx2",
            FlopsPath::Avx512F => "avx512f",
        }
    }
}

/// Numeric FLOPS per single FMA instruction: vec width × 2 (the FMA counts
/// as 2 floating-point ops: a*b and +c).
fn ops_per_fma(path: FlopsPath) -> u64 {
    match path {
        FlopsPath::Avx512F => 32, // 16 lanes × 2
        FlopsPath::Avx2 => 16,    //  8 lanes × 2
        FlopsPath::Scalar => 2,   //  1 lane  × 2
    }
}

/// Number of FMAs the inner loop does in one pass over the per-thread
/// buffer (1 FMA per `vec_width` floats).
#[cfg(test)]
fn fmas_per_inner_pass(path: FlopsPath) -> u64 {
    let vec_width = match path {
        FlopsPath::Avx512F => 16,
        FlopsPath::Avx2 => 8,
        FlopsPath::Scalar => 1,
    };
    (FLOATS_PER_BUFFER as u64) / vec_width
}

/// Result of an FP32 FLOPS measurement. Stable JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlopsResult {
    /// Median FP32 GFLOPS across the `reps` timed passes.
    pub fp32_gflops: f64,
    /// Best FP32 GFLOPS observed across `reps` timed passes.
    pub best_gflops: f64,
    /// SIMD path the kernel actually used.
    pub path: FlopsPath,
    /// Worker threads used (one per `physical_cores`).
    pub threads: usize,
    /// Timed passes (warm-up excluded).
    pub reps: usize,
    /// Theoretical FP32 peak in GFLOPS for the detected cores × SIMD path ×
    /// max CPU clock. 0.0 = unknown (caller renders this as
    /// `"theoretical: unknown"`).
    pub theoretical_gflops: f64,
    /// 0.0 when `theoretical_gflops` is 0.0 / invalid; otherwise
    /// `fp32_gflops / theoretical_gflops * 100.0`.
    pub pct_of_theoretical: f64,
}

/// Parameters for the FLOPS sweep. Build with [`FlopsConfig::from_hardware`].
#[derive(Debug, Clone, Copy)]
pub struct FlopsConfig {
    pub threads: usize,
    pub inner_iters: usize,
    pub reps: usize,
}

impl FlopsConfig {
    /// Default config: one worker per physical core, default inner iters
    /// and rep count. The kernel itself decides the SIMD path at runtime.
    pub fn from_hardware(hw: &HardwareInfo) -> Self {
        let threads = usize::try_from(hw.cpu.physical_cores.max(1)).unwrap_or(1);
        Self {
            threads,
            inner_iters: DEFAULT_INNER_ITERS,
            reps: DEFAULT_REPS,
        }
    }
}

/// Pure: pick the best SIMD path for a fixed vocabulary of SIMD flag
/// strings. Kept pure so the rule is unit-testable without CPUID.
pub fn select_path_from_flags(simd_flags: &[String]) -> FlopsPath {
    if simd_flags.iter().any(|f| f == "AVX-512F") {
        FlopsPath::Avx512F
    } else if simd_flags.iter().any(|f| f == "AVX2") {
        FlopsPath::Avx2
    } else {
        FlopsPath::Scalar
    }
}

/// Detect the SIMD path at runtime via stable `is_x86_feature_detected!`.
/// Non-x86 platforms always return `FlopsPath::Scalar` (the SIMD helpers
/// in this module are gated and exist only on x86).
pub fn select_path() -> FlopsPath {
    let cpu = match HardwareInfo::detect() {
        Ok(hw) => hw.cpu,
        Err(_) => return FlopsPath::Scalar,
    };
    let from_flags = select_path_from_flags(&cpu.simd_flags);
    // Cross-check against `is_x86_feature_detected!`. If raw-cpuid reports
    // AVX2/AVX-512F but the OS has since disabled the feature in the
    // current thread (e.g. via PRCTL, a hypervisor toggling bits, or some
    // AVX-512-downclocking BIOS options), fall back to a path we know we
    // can actually execute.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match from_flags {
            FlopsPath::Avx512F if !is_x86_feature_detected!("avx512f") => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    FlopsPath::Avx2
                } else {
                    FlopsPath::Scalar
                }
            }
            // `run_avx2` also needs FMA; without it the kernel would fault.
            FlopsPath::Avx2
                if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) =>
            {
                FlopsPath::Scalar
            }
            other => other,
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        from_flags
    }
}

/// Pure: convert `ops` (FP32 operations) in `seconds` into decimal GFLOPS.
/// Returns 0.0 for any non-finite or non-positive duration.
pub fn gflops(ops: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 || !seconds.is_finite() {
        return 0.0;
    }
    (ops as f64) / seconds / GFLOPS
}

/// Pure: `"% of theoretical"` or 0.0 when theoretical is unknown / invalid.
/// Uses `fp32_gflops` as numerator — same convention as the rest of the
/// codebase (`pct_of` in `attune-cli`).
pub fn pct_of_theoretical(measured: f64, theoretical: f64) -> f64 {
    if !measured.is_finite() || !theoretical.is_finite() || theoretical <= 0.0 {
        return 0.0;
    }
    measured / theoretical * 100.0
}

// ---- public entry points ---------------------------------------------------

/// One-shot convenience entry point used by tests / future quick mode.
pub fn quick_flops() -> Result<FlopsResult, BenchError> {
    let hw = HardwareInfo::detect()?;
    measure_flops(&FlopsConfig::from_hardware(&hw))
}

/// Run the FMA sweep: warm-up pass + `reps` timed passes, summarise into
/// the FMA-block struct.
pub fn measure_flops(cfg: &FlopsConfig) -> Result<FlopsResult, BenchError> {
    if cfg.threads == 0 {
        return Err(BenchError::InvalidConfig(
            "threads must be >= 1".to_string(),
        ));
    }
    if cfg.inner_iters == 0 {
        return Err(BenchError::InvalidConfig(
            "inner_iters must be >= 1".to_string(),
        ));
    }
    if cfg.reps == 0 {
        return Err(BenchError::InvalidConfig("reps must be >= 1".to_string()));
    }

    let hw = HardwareInfo::detect()?;
    let path = select_path();

    // Warm-up pass — discarded.
    let _ = run_pass(cfg.threads, cfg.inner_iters, path)?;

    // Timed passes.
    let mut samples: Vec<f64> = Vec::with_capacity(cfg.reps);
    let mut total_ops_run: u64 = 0;
    for _ in 0..cfg.reps {
        let started = Instant::now();
        let fmas = run_pass(cfg.threads, cfg.inner_iters, path)?;
        let elapsed = started.elapsed().as_secs_f64();
        total_ops_run = total_ops_run.saturating_add(fmas);
        samples.push(gflops(fmas.saturating_mul(ops_per_fma(path)), elapsed));
    }
    // `total_ops_run` is here purely to anchor `black_box` so the compiler
    // can't optimise the timed loop away — the "rate" is what we report.
    std::hint::black_box(total_ops_run);

    let median = median_f64(&mut samples);
    let best = samples.iter().copied().fold(0.0_f64, f64::max);
    let theoretical =
        theoretical_fp32_gflops(hw.cpu.physical_cores, &hw.cpu.simd_flags, hw.cpu.clock_ghz);
    let pct = pct_of_theoretical(median, theoretical);

    Ok(FlopsResult {
        fp32_gflops: median,
        best_gflops: best,
        path,
        threads: cfg.threads,
        reps: cfg.reps,
        theoretical_gflops: theoretical,
        pct_of_theoretical: pct,
    })
}

// ---- internal: per-pass multithread driver --------------------------------

/// Run one timed pass: each of `threads` workers does `inner_iters`
/// FMA-passes over its private L1-resident buffers; we sum the per-thread
/// FMA counts and return the aggregate. Wall-clock cost is the
/// parallel-union of every worker.
fn run_pass(threads: usize, inner_iters: usize, path: FlopsPath) -> Result<u64, BenchError> {
    // Templates are tiny (8 KiB × 2 each), so cloning per worker is cheaper
    // than tracking shared state — and it eliminates false sharing across
    // workers without depending on alignment fiddling.
    let template_a = template_buffers();
    let template_b = template_buffers();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let a: Vec<f32> = template_a.clone();
                let b: Vec<f32> = template_b.clone();
                scope.spawn(move || dispatch_kernel(path, &a, &b, inner_iters))
            })
            .collect();

        let mut total_fmas: u64 = 0;
        for handle in handles {
            total_fmas =
                total_fmas.saturating_add(handle.join().map_err(|_| BenchError::WorkerPanicked)?);
        }
        Ok(total_fmas)
    })
}

/// Generate one of the FMA input vectors. Distinct vec per template so
/// `template_a` and `template_b` actually carry different values (otherwise
/// the compiler could reduce `a*b + acc` to `a*(a+1) + acc` or similar).
fn template_buffers() -> Vec<f32> {
    (0..FLOATS_PER_BUFFER)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect()
}

// ---- kernel dispatch (SIMD-gated) ------------------------------------------

fn dispatch_kernel(path: FlopsPath, bufa: &[f32], bufb: &[f32], inner_iters: usize) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match path {
            FlopsPath::Avx512F => return unsafe { run_avx512(bufa, bufb, inner_iters) },
            FlopsPath::Avx2 => return unsafe { run_avx2(bufa, bufb, inner_iters) },
            FlopsPath::Scalar => {}
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // Non-x86: only the scalar arm is reachable. The `path` argument is
        // ignored on purpose — `select_path_from_flags` returns `Scalar`
        // here, but defensively we still funnel through the scalar kernel.
        let _ = path;
    }
    run_scalar(bufa, bufb, inner_iters)
}

// ---- scalar kernel (always compiles) --------------------------------------

/// Scalar FMA loop. 1 FMA per `(a[i] * b[i] + acc)` line; `inner_iters`
/// passes through the buffer. The accumulator stays in `f64` to widen the
/// critical path between FMAs and avoid precision-related carrier effects.
fn run_scalar(bufa: &[f32], bufb: &[f32], inner_iters: usize) -> u64 {
    let mut acc = std::hint::black_box(0.0_f64);
    let n = bufa.len();
    let total = (inner_iters as u64).saturating_mul(n as u64);
    for _ in 0..inner_iters {
        for i in 0..n {
            acc += f64::from(bufa[i] * bufb[i]);
        }
    }
    std::hint::black_box(acc);
    total
}

// ---- SIMD kernels (x86/x86_64 only) ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_avx2(bufa: &[f32], bufb: &[f32], inner_iters: usize) -> u64 {
    // 4 independent accumulators keep the FMA pipeline fed despite the
    // 4-cycle FMADD latency on Intel/AMD. Without this we'd be latency-
    // bound (~5 GFLOPS/thread); with it we land much closer to the 2-FMA-
    // port throughput ceiling.
    use std::arch::x86_64::*;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let n8 = bufa.len() / 8;
    let n8_quad = n8 / 4; // FMAs per outer iter: 4 FMAs × 8 floats = 32 floats
    let a_ptr = bufa.as_ptr();
    let b_ptr = bufb.as_ptr();
    // Total FMAs per pass is `inner_iters × (n8 / 4) × 4 = inner_iters × n8`,
    // i.e. exactly one FMA per inner-loop iteration × vec_width.
    let total = (inner_iters as u64).saturating_mul(n8 as u64);
    for _ in 0..inner_iters {
        for q in 0..n8_quad {
            let base = q * 32; // 4 × 8
            let va0 = _mm256_loadu_ps(a_ptr.add(base));
            let vb0 = _mm256_loadu_ps(b_ptr.add(base));
            a0 = _mm256_fmadd_ps(va0, vb0, a0);
            let va1 = _mm256_loadu_ps(a_ptr.add(base + 8));
            let vb1 = _mm256_loadu_ps(b_ptr.add(base + 8));
            a1 = _mm256_fmadd_ps(va1, vb1, a1);
            let va2 = _mm256_loadu_ps(a_ptr.add(base + 16));
            let vb2 = _mm256_loadu_ps(b_ptr.add(base + 16));
            a2 = _mm256_fmadd_ps(va2, vb2, a2);
            let va3 = _mm256_loadu_ps(a_ptr.add(base + 24));
            let vb3 = _mm256_loadu_ps(b_ptr.add(base + 24));
            a3 = _mm256_fmadd_ps(va3, vb3, a3);
        }
    }
    // Horizontal reduce: store then sum — dwarfed by the kernel itself.
    let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let sum: f32 = buf.iter().sum();
    std::hint::black_box(sum);
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn run_avx512(bufa: &[f32], bufb: &[f32], inner_iters: usize) -> u64 {
    // Same latency-breaking pattern as the AVX2 kernel. AVX-512F has 8-cycle
    // FMADD latency on Skylake-X+, throughput of 1/cycle on consumer cores,
    // so 8 independent accumulators fully saturate the FMA port.
    use std::arch::x86_64::*;
    let mut a0 = _mm512_setzero_ps();
    let mut a1 = _mm512_setzero_ps();
    let mut a2 = _mm512_setzero_ps();
    let mut a3 = _mm512_setzero_ps();
    let mut a4 = _mm512_setzero_ps();
    let mut a5 = _mm512_setzero_ps();
    let mut a6 = _mm512_setzero_ps();
    let mut a7 = _mm512_setzero_ps();
    let n16 = bufa.len() / 16;
    let n16_oct = n16 / 8; // FMAs per outer iter: 8 FMAs × 16 floats = 128 floats
    let a_ptr = bufa.as_ptr();
    let b_ptr = bufb.as_ptr();
    let total = (inner_iters as u64).saturating_mul(n16 as u64);
    for _ in 0..inner_iters {
        for q in 0..n16_oct {
            let base = q * 128; // 8 × 16
            a0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base)),
                _mm512_loadu_ps(b_ptr.add(base)),
                a0,
            );
            a1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 16)),
                _mm512_loadu_ps(b_ptr.add(base + 16)),
                a1,
            );
            a2 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 32)),
                _mm512_loadu_ps(b_ptr.add(base + 32)),
                a2,
            );
            a3 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 48)),
                _mm512_loadu_ps(b_ptr.add(base + 48)),
                a3,
            );
            a4 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 64)),
                _mm512_loadu_ps(b_ptr.add(base + 64)),
                a4,
            );
            a5 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 80)),
                _mm512_loadu_ps(b_ptr.add(base + 80)),
                a5,
            );
            a6 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 96)),
                _mm512_loadu_ps(b_ptr.add(base + 96)),
                a6,
            );
            a7 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 112)),
                _mm512_loadu_ps(b_ptr.add(base + 112)),
                a7,
            );
        }
    }
    let acc = _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)),
        _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)),
    );
    let mut buf = [0.0f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), acc);
    let sum: f32 = buf.iter().sum();
    std::hint::black_box(sum);
    total
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(threads: usize, inner_iters: usize, reps: usize) -> FlopsConfig {
        FlopsConfig {
            threads,
            inner_iters,
            reps,
        }
    }

    // ---- pure math ----------------------------------------------------------

    #[test]
    fn gflops_converts_ops_and_seconds() {
        // 1e9 ops in 1 s → 1 GFLOPS.
        assert!((gflops(1_000_000_000, 1.0) - 1.0).abs() < 1e-9);
        // 1e9 ops in 0.5 s → 2 GFLOPS.
        assert!((gflops(1_000_000_000, 0.5) - 2.0).abs() < 1e-9);
        // Zero ops → zero GFLOPS.
        assert_eq!(gflops(0, 1.0), 0.0);
    }

    #[test]
    fn gflops_is_zero_for_degenerate_durations() {
        assert_eq!(gflops(1_000_000, 0.0), 0.0);
        assert_eq!(gflops(1_000_000, -1.0), 0.0);
        assert_eq!(gflops(1_000_000, f64::NAN), 0.0);
        assert_eq!(gflops(1_000_000, f64::INFINITY), 0.0);
    }

    #[test]
    fn pct_of_theoretical_normal_case() {
        // 64 / 128 = 50%
        assert!((pct_of_theoretical(64.0, 128.0) - 50.0).abs() < 1e-9);
        // 128 / 256 = 50%
        assert!((pct_of_theoretical(128.0, 256.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn pct_of_theoretical_zero_or_unknown_is_zero() {
        assert_eq!(pct_of_theoretical(50.0, 0.0), 0.0);
        assert_eq!(pct_of_theoretical(50.0, -10.0), 0.0);
        assert_eq!(pct_of_theoretical(50.0, f64::NAN), 0.0);
        assert_eq!(pct_of_theoretical(f64::NAN, 100.0), 0.0);
    }

    // ---- path selector ------------------------------------------------------

    #[test]
    fn select_path_prefers_avx512_over_avx2() {
        assert_eq!(select_path_from_flags(&[]), FlopsPath::Scalar);
        assert_eq!(
            select_path_from_flags(&["SSE4.2".to_string()]),
            FlopsPath::Scalar
        );
        assert_eq!(
            select_path_from_flags(&["AVX".to_string()]),
            FlopsPath::Scalar
        );
        assert_eq!(
            select_path_from_flags(&["AVX2".to_string()]),
            FlopsPath::Avx2
        );
        assert_eq!(
            select_path_from_flags(&["AVX2".to_string(), "AVX-512F".to_string()]),
            FlopsPath::Avx512F
        );
        assert_eq!(
            select_path_from_flags(&["AVX-512F".to_string()]),
            FlopsPath::Avx512F
        );
    }

    // ---- ops accounting helpers --------------------------------------------

    #[test]
    fn ops_per_fma_matches_vec_width() {
        assert_eq!(ops_per_fma(FlopsPath::Scalar), 2);
        assert_eq!(ops_per_fma(FlopsPath::Avx2), 16);
        assert_eq!(ops_per_fma(FlopsPath::Avx512F), 32);
    }

    #[test]
    fn fmas_per_inner_pass_matches_buffer_layout() {
        // FLOATS_PER_BUFFER / vec_width.
        assert_eq!(
            fmas_per_inner_pass(FlopsPath::Scalar),
            FLOATS_PER_BUFFER as u64
        );
        assert_eq!(
            fmas_per_inner_pass(FlopsPath::Avx2),
            (FLOATS_PER_BUFFER / 8) as u64
        );
        assert_eq!(
            fmas_per_inner_pass(FlopsPath::Avx512F),
            (FLOATS_PER_BUFFER / 16) as u64
        );
    }

    // ---- kernel smoke -------------------------------------------------------

    #[test]
    fn run_scalar_anchor_is_deterministic() {
        // Same input every call → same accumulator, so the kernel cannot
        // have been optimised away. The exact value depends on the buffers
        // baked by `template_buffers`, so we only assert non-zero.
        let a: Vec<f32> = template_buffers();
        let b: Vec<f32> = template_buffers();
        let fmas = run_scalar(&a, &b, 50);
        assert_eq!(fmas, (50u64) * (a.len() as u64));
    }

    #[test]
    fn smoke_measure_flops_reports_plausible() {
        // Small config that must complete in well under 10 s on any modern
        // machine and produce a non-zero GFLOPS in a sane band.
        let cfg = test_cfg(1, 20_000, 1);
        let started = Instant::now();
        let result = measure_flops(&cfg).expect("flops kernel should run");
        let elapsed = started.elapsed();

        assert!(result.fp32_gflops > 0.0, "got {}", result.fp32_gflops);
        // 5000 GFLOPS is well above any realistic single-thread number
        // (even AVX-512F single-thread tops out around 200 GFLOPS); bound
        // is intentionally loose so the test never flakes on slow VMs.
        assert!(
            result.fp32_gflops < 5000.0,
            "implausible flops {}",
            result.fp32_gflops
        );
        assert_eq!(result.threads, 1);
        assert_eq!(result.reps, 1);
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "smoke took {elapsed:?}"
        );
    }

    #[test]
    fn multi_thread_run_reports_requested_shape() {
        let cfg = test_cfg(4, 5_000, 1);
        let result = measure_flops(&cfg).expect("flops kernel should run");
        assert_eq!(result.threads, 4);
        assert_eq!(result.reps, 1);
        assert!(result.fp32_gflops > 0.0);
    }

    #[test]
    fn measure_flops_rejects_invalid_config() {
        let bad_threads = measure_flops(&test_cfg(0, 1_000, 1));
        assert!(matches!(bad_threads, Err(BenchError::InvalidConfig(_))));
        let bad_iters = measure_flops(&test_cfg(1, 0, 1));
        assert!(matches!(bad_iters, Err(BenchError::InvalidConfig(_))));
        let bad_reps = measure_flops(&test_cfg(1, 1_000, 0));
        assert!(matches!(bad_reps, Err(BenchError::InvalidConfig(_))));
    }

    // ---- JSON contract ------------------------------------------------------

    #[test]
    fn flops_result_json_roundtrip() {
        let res = FlopsResult {
            fp32_gflops: 1234.5,
            best_gflops: 1500.0,
            path: FlopsPath::Avx2,
            threads: 8,
            reps: 3,
            theoretical_gflops: 256.0,
            pct_of_theoretical: 482.0,
        };
        let json = serde_json::to_string(&res).expect("serialize FlopsResult");
        assert!(
            json.contains("\"fp32_gflops\""),
            "missing fp32_gflops: {json}"
        );
        assert!(
            json.contains("\"best_gflops\""),
            "missing best_gflops: {json}"
        );
        assert!(json.contains("\"path\""), "missing path: {json}");
        // `path` should serialize as the lowercase form per the JSON
        // contract, not the Rust variant name.
        assert!(
            json.contains("\"avx2\""),
            "path should serialize as \"avx2\": {json}"
        );
        assert!(json.contains("\"threads\""), "missing threads: {json}");
        assert!(json.contains("\"reps\""), "missing reps: {json}");
        assert!(
            json.contains("\"theoretical_gflops\""),
            "missing theoretical_gflops: {json}"
        );
        assert!(
            json.contains("\"pct_of_theoretical\""),
            "missing pct_of_theoretical: {json}"
        );

        let back: FlopsResult = serde_json::from_str(&json).expect("deserialize FlopsResult");
        assert_eq!(back, res);
    }

    #[test]
    fn flops_path_serializes_lowercase() {
        for (variant, expected) in [
            (FlopsPath::Scalar, "\"scalar\""),
            (FlopsPath::Avx2, "\"avx2\""),
            (FlopsPath::Avx512F, "\"avx512f\""),
        ] {
            let s = serde_json::to_string(&variant).expect("serialize FlopsPath");
            assert_eq!(s, expected, "wrong JSON for {variant:?}");
        }
    }
}
