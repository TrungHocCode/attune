//! Pure profile → llama.cpp argv mapping (M4 slice 3).
//!
//! The order is stable so a saved profile always launches with the same
//! argument vector, and the mapper is pure (no `std::process`, no file I/O) so
//! it can be unit-tested without llama.cpp installed.

use crate::Profile;

/// Map a profile to llama.cpp load/runtime flags.
///
/// Emits `-m`, `-c`, `-b`, `--ubatch-size`, `-t`, `-tb`, `--cache-type-k`,
/// `--cache-type-v`, `--numa`, `-fa` (when enabled), `--no-mmap` (when mmap is
/// off), `--mlock` (when enabled) and `-ngl`.
///
/// `compute_config.cpu_affinity` is intentionally not mapped here: the stored
/// value may be either a mask or a range, and choosing between `--cpu-mask`
/// and `--cpu-range` belongs to the M5 launcher that knows the semantics.
pub fn to_llama_args(profile: &Profile) -> Vec<String> {
    let load = &profile.load_config;
    let compute = &profile.compute_config;
    let mut args = Vec::with_capacity(24);

    push_pair(&mut args, "-m", &profile.model.path);
    push_pair(&mut args, "-c", load.n_ctx);
    push_pair(&mut args, "-b", load.n_batch);
    push_pair(&mut args, "--ubatch-size", load.n_ubatch);
    push_pair(&mut args, "-t", compute.n_threads);
    push_pair(&mut args, "-tb", compute.n_threads_batch);
    push_pair(&mut args, "--cache-type-k", &load.cache_type_k);
    push_pair(&mut args, "--cache-type-v", &load.cache_type_v);
    push_pair(&mut args, "--numa", &load.numa);
    if load.flash_attn {
        args.push("-fa".to_string());
    }
    if !load.mmap {
        args.push("--no-mmap".to_string());
    }
    if load.mlock {
        args.push("--mlock".to_string());
    }
    push_pair(&mut args, "-ngl", load.n_gpu_layers);

    args
}

fn push_pair(args: &mut Vec<String>, flag: &str, value: impl ToString) {
    args.push(flag.to_string());
    args.push(value.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ComputeConfig, LoadConfig, ModelConfig};

    fn best_config_profile() -> Profile {
        Profile {
            model: ModelConfig {
                path: "models/tiny.gguf".to_string(),
                quant: "q4_k_m".to_string(),
                size_bytes: 82_600_000,
            },
            load_config: LoadConfig {
                n_ctx: 4_096,
                n_batch: 512,
                n_ubatch: 128,
                n_gpu_layers: 8,
                mmap: false,
                mlock: true,
                flash_attn: true,
                cache_type_k: "q8_0".to_string(),
                cache_type_v: "q8_0".to_string(),
                numa: "local".to_string(),
            },
            compute_config: ComputeConfig {
                n_threads: 8,
                n_threads_batch: 8,
                cpu_affinity: Some("0-7".to_string()),
            },
            ..Profile::default()
        }
    }

    #[test]
    fn maps_the_best_config_to_the_expected_flags() {
        let args = to_llama_args(&best_config_profile());
        assert_eq!(
            args,
            vec![
                "-m",
                "models/tiny.gguf",
                "-c",
                "4096",
                "-b",
                "512",
                "--ubatch-size",
                "128",
                "-t",
                "8",
                "-tb",
                "8",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--numa",
                "local",
                "-fa",
                "--no-mmap",
                "--mlock",
                "-ngl",
                "8",
            ]
        );
    }

    #[test]
    fn defaults_omit_the_optional_switches() {
        let args = to_llama_args(&Profile::default());
        assert!(!args.iter().any(|arg| arg == "--no-mmap"));
        assert!(!args.iter().any(|arg| arg == "--mlock"));
        assert!(args.iter().any(|arg| arg == "-fa"));
        assert_eq!(args[args.len() - 1], "0");
        assert_eq!(args[args.len() - 2], "-ngl");
    }
}
