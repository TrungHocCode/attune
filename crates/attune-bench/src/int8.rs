//! Int8 dot-product throughput kernel (issue #69).
//!
//! llama.cpp's quantized prefill runs on int8 dot products, not FP32 FMA, so
//! the FP32 peak is the wrong compute ceiling for q8_0/q4_* prefill. This
//! kernel measures what the CPU actually sustains on `u8 x i8 -> i32`
//! multiply-accumulate, multithreaded across physical cores like the FMA
//! kernel.
//!
//! Paths, chosen at runtime with `is_x86_feature_detected!`:
//! - `avxvnni`: VEX `vpdpbusd` via `_mm256_dpbusd_avx_epi32` (1 instruction).
//! - `avx2`: `vpmaddubsw` + `vpmaddwd` + `vpaddd` (3 instructions).
//! - `scalar`: portable fallback (also the only path off x86).
//!
//! Accounting: one multiply-add = 2 ops, matching the FP32 FLOPS convention,
//! so the unit is decimal GOPS (1e9 ops/s).

use std::time::Instant;

use attune_hal::HardwareInfo;
use serde::{Deserialize, Serialize};

use crate::bandwidth::median_f64;
use crate::flops::gflops as gops;
use crate::BenchError;

/// The kernel is a register-tiled micro-GEMM like real int8 matmul kernels:
/// each of `GROUPS` groups multiplies 2 activation chunks (32 B each) by 4
/// weight chunks, i.e. 8 dot-product instructions per 6 loads. Reusing loaded
/// registers matters: with one fresh load pair per instruction the kernel is
/// load-port bound and under-reports what the multiply units sustain.
pub const GROUPS: usize = 8;
/// Activation (u8) buffer length: `GROUPS` x 2 chunks x 32 B.
pub const A_LEN: usize = GROUPS * 64;
/// Weight (i8) buffer length: `GROUPS` x 4 chunks x 32 B.
pub const B_LEN: usize = GROUPS * 128;
/// Multiply-adds per outer iteration (`GROUPS` x 8 chunk pairs x 32 lanes).
const MACS_PER_ITER: u64 = (GROUPS * 8 * 32) as u64;
/// Outer iterations per timed pass for the fastest path (about 70 ms on an
/// AVX-VNNI core); slower paths run proportionally fewer, see `measure_int8`.
pub const DEFAULT_INNER_ITERS: usize = 1_000_000;
/// Timed passes; the median is reported.
pub const DEFAULT_REPS: usize = 3;

/// Kernel path actually used. Lowercase names are the JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Int8Path {
    Scalar,
    Avx2,
    AvxVnni,
}

impl Int8Path {
    pub fn as_str(&self) -> &'static str {
        match self {
            Int8Path::Scalar => "scalar",
            Int8Path::Avx2 => "avx2",
            Int8Path::AvxVnni => "avxvnni",
        }
    }
}

/// Result of an int8 throughput measurement. Stable JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Int8Result {
    /// Median aggregate GOPS (multiply-add = 2 ops) across timed passes.
    pub int8_gops: f64,
    /// Best GOPS across timed passes.
    pub best_gops: f64,
    pub path: Int8Path,
    pub threads: usize,
    pub reps: usize,
}

/// Best path this CPU can execute right now.
pub fn select_path() -> Int8Path {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if path_supported(Int8Path::AvxVnni) {
            return Int8Path::AvxVnni;
        }
        if path_supported(Int8Path::Avx2) {
            return Int8Path::Avx2;
        }
    }
    Int8Path::Scalar
}

/// Whether `path` can run on this CPU (also gates the per-path tests).
pub fn path_supported(path: Int8Path) -> bool {
    match path {
        Int8Path::Scalar => true,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Int8Path::Avx2 => is_x86_feature_detected!("avx2"),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Int8Path::AvxVnni => {
            is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        _ => false,
    }
}

/// Measure int8 throughput with the quick budget (about 0.4 s).
pub fn quick_int8() -> Result<Int8Result, BenchError> {
    let hw = HardwareInfo::detect()?;
    let threads = usize::try_from(hw.cpu.physical_cores.max(1)).unwrap_or(1);
    measure_int8(threads, DEFAULT_INNER_ITERS, DEFAULT_REPS)
}

/// One discarded warm-up pass plus `reps` timed passes.
pub fn measure_int8(
    threads: usize,
    inner_iters: usize,
    reps: usize,
) -> Result<Int8Result, BenchError> {
    if threads == 0 || inner_iters == 0 || reps == 0 {
        return Err(BenchError::InvalidConfig(
            "threads, inner_iters and reps must be >= 1".to_string(),
        ));
    }
    let path = select_path();
    // Keep every path near the same wall-clock budget.
    let inner_iters = match path {
        Int8Path::AvxVnni => inner_iters,
        Int8Path::Avx2 => (inner_iters / 2).max(1),
        Int8Path::Scalar => (inner_iters / 20).max(1),
    };
    run_pass(threads, inner_iters, path)?;

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        let ops = run_pass(threads, inner_iters, path)?;
        samples.push(gops(ops, started.elapsed().as_secs_f64()));
    }
    let median = median_f64(&mut samples);
    let best = samples.iter().copied().fold(0.0_f64, f64::max);
    Ok(Int8Result {
        int8_gops: median,
        best_gops: best,
        path,
        threads,
        reps,
    })
}

/// Total ops (2 per multiply-add) done by all workers in one pass.
fn run_pass(threads: usize, inner_iters: usize, path: Int8Path) -> Result<u64, BenchError> {
    let (a, b) = template_buffers();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (a, b) = (a.clone(), b.clone());
                scope.spawn(move || std::hint::black_box(kernel(path, &a, &b, inner_iters)))
            })
            .collect();
        for handle in handles {
            handle.join().map_err(|_| BenchError::WorkerPanicked)?;
        }
        Ok((threads as u64) * (inner_iters as u64) * MACS_PER_ITER * 2)
    })
}

/// Small values (a in 0..16, b in -8..8) so the AVX2 `vpmaddubsw` i16
/// saturation never triggers and every path computes the identical sum.
fn template_buffers() -> (Vec<u8>, Vec<i8>) {
    let a = (0..A_LEN).map(|i| (i % 16) as u8).collect();
    let b = (0..B_LEN).map(|i| ((i * 7) % 16) as i8 - 8).collect();
    (a, b)
}

/// Run `inner_iters` passes over the buffers and return the wrapping sum of
/// all products. Buffers must be `A_LEN` / `B_LEN` long. Falls back to the
/// scalar path if `path` is unsupported here, so it can never fault.
pub fn kernel(path: Int8Path, a: &[u8], b: &[i8], inner_iters: usize) -> i32 {
    if a.len() != A_LEN || b.len() != B_LEN || !path_supported(path) {
        return scalar(a, b, inner_iters);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match path {
        // SAFETY: feature support was checked by `path_supported` above.
        Int8Path::AvxVnni => return unsafe { vnni(a, b, inner_iters) },
        Int8Path::Avx2 => return unsafe { avx2(a, b, inner_iters) },
        Int8Path::Scalar => {}
    }
    scalar(a, b, inner_iters)
}

fn scalar(a: &[u8], b: &[i8], inner_iters: usize) -> i32 {
    let (a, b) = (std::hint::black_box(a), std::hint::black_box(b));
    let mut sum = 0_i32;
    for _ in 0..inner_iters {
        for g in 0..GROUPS {
            for i in 0..2 {
                for j in 0..4 {
                    for l in 0..32 {
                        let x = i32::from(a[g * 64 + i * 32 + l]);
                        let y = i32::from(b[g * 128 + j * 32 + l]);
                        sum = sum.wrapping_add(x * y);
                    }
                }
            }
        }
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! x86_kernel {
    ($name:ident, $features:literal, |$acc:ident, $va:ident, $vb:ident| $step:expr) => {
        #[target_feature(enable = $features)]
        unsafe fn $name(a: &[u8], b: &[i8], inner_iters: usize) -> i32 {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            // 2x4 register tile: 8 independent accumulators hide the latency.
            let mut accs = [[_mm256_setzero_si256(); 4]; 2];
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            for _ in 0..inner_iters {
                for g in 0..GROUPS {
                    let ld_a =
                        |i: usize| _mm256_loadu_si256(pa.add(g * 64 + i * 32) as *const __m256i);
                    let ld_b =
                        |j: usize| _mm256_loadu_si256(pb.add(g * 128 + j * 32) as *const __m256i);
                    let vas = [ld_a(0), ld_a(1)];
                    let vbs = [ld_b(0), ld_b(1), ld_b(2), ld_b(3)];
                    for (row, $va) in accs.iter_mut().zip(vas) {
                        for ($acc, $vb) in row.iter_mut().zip(vbs) {
                            *$acc = $step;
                        }
                    }
                }
            }
            let mut total = _mm256_setzero_si256();
            for acc in accs.iter().flatten() {
                total = _mm256_add_epi32(total, *acc);
            }
            let mut lanes = [0_i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, total);
            lanes.iter().fold(0_i32, |s, v| s.wrapping_add(*v))
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
x86_kernel!(vnni, "avx2,avxvnni", |acc, va, vb| _mm256_dpbusd_avx_epi32(
    *acc, va, vb
));
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
x86_kernel!(avx2, "avx2", |acc, va, vb| _mm256_add_epi32(
    *acc,
    _mm256_madd_epi16(_mm256_maddubs_epi16(va, vb), _mm256_set1_epi16(1))
));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_path_matches_the_scalar_reference() {
        let (a, b) = template_buffers();
        // Reference computed independently of the kernels' loops.
        let mut one_pass = 0_i32;
        for g in 0..GROUPS {
            for ac in a[g * 64..(g + 1) * 64].chunks(32) {
                for bc in b[g * 128..(g + 1) * 128].chunks(32) {
                    one_pass += ac
                        .iter()
                        .zip(bc)
                        .map(|(x, y)| i32::from(*x) * i32::from(*y))
                        .sum::<i32>();
                }
            }
        }
        for path in [Int8Path::Scalar, Int8Path::Avx2, Int8Path::AvxVnni] {
            if !path_supported(path) {
                continue;
            }
            assert_eq!(kernel(path, &a, &b, 1), one_pass, "{path:?}");
            assert_eq!(kernel(path, &a, &b, 3), one_pass * 3, "{path:?}");
        }
    }

    #[test]
    fn known_tiny_input() {
        let mut a = vec![0_u8; A_LEN];
        let mut b = vec![0_i8; B_LEN];
        a[0] = 3;
        b[0] = -2;
        // a[100] is group 1, chunk 1, lane 4; b[196] is group 1, chunk 2, lane 4.
        a[100] = 5;
        b[196] = 4;
        for path in [Int8Path::Scalar, Int8Path::Avx2, Int8Path::AvxVnni] {
            if path_supported(path) {
                assert_eq!(kernel(path, &a, &b, 1), 14, "{path:?}");
            }
        }
    }

    #[test]
    fn quick_measurement_reports_positive_gops() {
        // Headroom only: no upper bound, tiny workload.
        let result = measure_int8(1, 200, 2).expect("measure");
        assert!(result.int8_gops > 0.0 && result.best_gops >= result.int8_gops);
    }

    #[test]
    fn rejects_zero_config() {
        assert!(measure_int8(0, 1, 1).is_err());
    }
}
