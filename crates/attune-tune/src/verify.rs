//! Two-tier end-to-end verification of the top-K estimate ranking (M4 slice 4).
//!
//! Slice 1 ranks every surviving config with the roofline estimate. This
//! module lets the CLI measure the best few for real: it maps a
//! [`TuneConfig`] onto an [`E2eConfig`], runs the existing
//! [`attune_bench::e2e`] pipeline, and re-scores the measured configs with the
//! same [`Objective`] the estimate used.
//!
//! Two layers, deliberately separated:
//!
//! * **Pure** — [`e2e_config_for`] and [`rerank_with_e2e`] do no I/O, so the
//!   mapping and the re-score are unit-tested on canned numbers.
//! * **Orchestration** — [`run_verify_topk`] is the one function in this crate
//!   that spawns a process. It preflights the backend *once* (model readable,
//!   `llama-cli` resolvable, version parseable) before any slow run, then
//!   measures the top-K configs one at a time. A single config failing is a
//!   warning, never a failed sweep: that config keeps its estimate score.
//!
//! The whole path is opt-in (`attune tune --verify`); the default run stays
//! estimate-only, spawns nothing and keeps CI fast.

use std::cmp::Ordering;
use std::path::Path;

use attune_bench::e2e::{preflight, run_e2e, E2eConfig, E2eResult};
use attune_bench::llama::{is_readable_file, resolve_llama_cli};
use attune_bench::BenchError;

use crate::sweep::RankedConfig;
use crate::{Objective, TuneConfig, TuneError};

/// Smallest context llama.cpp accepts (same floor as
/// [`E2eConfig::ctx_size`]).
const MIN_E2E_CTX: u32 = 32;

/// Result of [`run_verify_topk`]: the re-scored configs plus the non-fatal
/// notices from any config whose real run failed.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOutcome {
    /// All ranked configs, best first. Measured entries carry
    /// [`RankedConfig::verified`] and [`RankedConfig::measured`]; the rest
    /// keep their estimate numbers.
    pub configs: Vec<RankedConfig>,
    /// One notice per config whose e2e run failed or produced no usable
    /// decode timing. Empty when every attempted run succeeded.
    pub warnings: Vec<String>,
}

/// Map a tune grid point onto the llama.cpp knobs [`run_e2e`] understands.
///
/// Only the fields the current e2e command line supports are mapped:
/// `n_threads` → `threads` (`0` means "let llama.cpp decide", so no `-t` flag
/// is emitted) and `n_ctx` → `ctx_size`, clamped to `MIN_E2E_CTX`. The
/// `-st` / `-v` flags live in [`attune_bench::llama::COMMON_ARGS`] and are
/// added by [`attune_bench::e2e::build_e2e_args`].
pub fn e2e_config_for(
    config: &TuneConfig,
    model_path: &Path,
    prompt_tokens: usize,
    gen_tokens: u32,
) -> E2eConfig {
    E2eConfig {
        model: model_path.to_path_buf(),
        prompt_tokens,
        gen_tokens,
        threads: config.n_threads,
        ctx_size: config.n_ctx.max(MIN_E2E_CTX),
    }
}

/// Replace estimate numbers with measured ones and re-sort with the same
/// [`Objective`].
///
/// `measured` is keyed by the exact [`TuneConfig`] that ran. Entries without a
/// complete measurement are copied through untouched, so a failed or truncated
/// e2e run never erases the estimate. Sorting is stable, so unmeasured configs
/// keep their relative estimate order among themselves.
///
/// The original score may include the near-ceiling RAM penalty applied by
/// [`crate::rank_by_estimate`], but [`RankedConfig`] does not store the ratio.
/// The penalty is therefore recovered as `estimate_raw − score` and carried
/// onto the measured score, so verification does not silently drop that nudge.
pub fn rerank_with_e2e(
    ranked: &[RankedConfig],
    measured: &[(TuneConfig, E2eResult)],
    objective: Objective,
) -> Vec<RankedConfig> {
    let mut updated: Vec<RankedConfig> = ranked
        .iter()
        .map(|entry| {
            let Some((_, result)) = measured.iter().find(|(config, _)| config == &entry.config)
            else {
                return entry.clone();
            };
            if !result.is_complete() {
                return entry.clone();
            }

            let score = rescore(entry, result.tok_s, result.ttft_ms, objective);
            let mut updated = entry.clone();
            updated.decode_tps = result.tok_s;
            updated.ttft_ms = result.ttft_ms;
            updated.score = score;
            updated.verified = true;
            updated.measured = Some(result.clone());
            updated
        })
        .collect();

    updated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    updated
}

/// Run the best `top_k` configs through a real llama.cpp e2e and fold the
/// measurements back into the ranking.
///
/// Setup failures (unreadable model, no `llama-cli`, unparseable version) are
/// returned as [`TuneError`] before a single model load starts. Individual run
/// failures after setup are warnings: the affected config keeps its estimate
/// score and the sweep continues. `ctx_size` overrides each config's `n_ctx`
/// when supplied (the CLI's `--ctx`, mirroring `attune bench --model`).
#[allow(clippy::too_many_arguments)]
pub fn run_verify_topk(
    ranked: &[RankedConfig],
    objective: Objective,
    top_k: usize,
    model_path: &Path,
    llama_cli: Option<&str>,
    prompt_tokens: usize,
    gen_tokens: u32,
    ctx_size: Option<u32>,
) -> Result<VerifyOutcome, TuneError> {
    if top_k == 0 {
        return Err(TuneError::InvalidConfig("top_k must be > 0".to_string()));
    }
    if ranked.is_empty() {
        return Ok(VerifyOutcome {
            configs: Vec::new(),
            warnings: Vec::new(),
        });
    }

    // Check the model before resolving the binary so a typo fails fast even on
    // a machine with no llama.cpp installed. `preflight` repeats this check
    // together with the version probe.
    if !is_readable_file(model_path) {
        return Err(TuneError::InvalidConfig(format!(
            "model `{}` is not a readable file",
            model_path.display()
        )));
    }
    let cli = resolve_llama_cli(llama_cli).map_err(|err| TuneError::Verify(err.to_string()))?;
    preflight(&cli, model_path).map_err(map_preflight)?;

    let engine_version = cli.display().to_string();
    let limit = top_k.min(ranked.len());
    let mut measured = Vec::with_capacity(limit);
    let mut warnings = Vec::new();

    for entry in ranked.iter().take(limit) {
        let mut config = e2e_config_for(&entry.config, model_path, prompt_tokens, gen_tokens);
        if let Some(ctx) = ctx_size {
            config.ctx_size = ctx;
        }

        match run_e2e(&cli, &config) {
            Ok(result) if result.is_complete() => {
                measured.push((entry.config.clone(), result));
            }
            Ok(result) => warnings.push(format!(
                "e2e verification of {} with `{engine_version}` produced no decode timing \
                 ({} prompt / {} gen tokens); keeping the estimate",
                config_label(&entry.config),
                result.prompt_tokens,
                result.eval_tokens,
            )),
            Err(err) => warnings.push(format!(
                "e2e verification of {} failed ({err}); keeping the estimate",
                config_label(&entry.config),
            )),
        }
    }

    Ok(VerifyOutcome {
        configs: rerank_with_e2e(ranked, &measured, objective),
        warnings,
    })
}

/// Turn a bench preflight error into a tune error. A bad model stays
/// [`TuneError::InvalidConfig`] so the CLI's message matches the estimate
/// path; everything else is a verification-setup failure.
fn map_preflight(err: BenchError) -> TuneError {
    match err {
        BenchError::InvalidConfig(message) => TuneError::InvalidConfig(message),
        other => TuneError::Verify(other.to_string()),
    }
}

/// Re-score one config from new decode/TTFT numbers, preserving the estimate's
/// RAM penalty.
///
/// Shared by [`rerank_with_e2e`] (measured numbers) and
/// [`crate::sweep::calibrate_ranked`] (calibrated estimates): both replace the
/// decode/TTFT pair but must not silently drop the near-ceiling nudge.
pub(crate) fn rescore(
    entry: &RankedConfig,
    decode_tps: f64,
    ttft_ms: f64,
    objective: Objective,
) -> f64 {
    let estimate_raw = objective.score(entry.decode_tps, entry.ttft_ms, entry.fits, 0.0);
    let penalty = if estimate_raw.is_finite() && entry.score.is_finite() {
        (estimate_raw - entry.score).max(0.0)
    } else {
        0.0
    };
    objective.score(decode_tps, ttft_ms, entry.fits, 0.0) - penalty
}

/// Compact config identity used in warning messages.
fn config_label(config: &TuneConfig) -> String {
    format!(
        "{} ctx={} threads={} batch={} kv={}",
        config.quant.as_str(),
        config.n_ctx,
        config.n_threads,
        config.n_batch,
        config.cache_kv.as_str(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use attune_estimate::Quant;
    use std::path::PathBuf;

    fn ranked(config: TuneConfig, decode_tps: f64, ttft_ms: f64, score: f64) -> RankedConfig {
        RankedConfig {
            config,
            decode_tps,
            ttft_ms,
            fits: true,
            score,
            verified: false,
            measured: None,
            calibration: None,
        }
    }

    fn e2e(tok_s: f64, ttft_ms: f64) -> E2eResult {
        E2eResult {
            engine: "test".into(),
            model: "/m.gguf".into(),
            prompt_tokens: 32,
            eval_tokens: 8,
            ttft_ms,
            tpot_ms: 1_000.0 / tok_s,
            tok_s,
            ram_peak_bytes: 0,
            cpu_avg_pct: 0.0,
            power_avg_watts: None,
            elapsed_ms: 1.0,
        }
    }

    #[test]
    fn e2e_config_for_maps_the_tune_fields() {
        let config = TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16);
        let mapped = e2e_config_for(&config, Path::new("/models/tiny.gguf"), 64, 16);

        assert_eq!(mapped.model, PathBuf::from("/models/tiny.gguf"));
        assert_eq!(mapped.prompt_tokens, 64);
        assert_eq!(mapped.gen_tokens, 16);
        assert_eq!(mapped.threads, 8);
        assert_eq!(mapped.ctx_size, 2_048);
    }

    #[test]
    fn e2e_config_for_clamps_ctx_and_keeps_zero_threads_meaningful() {
        let config = TuneConfig {
            n_ctx: 4,
            n_threads: 0,
            ..TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16)
        };
        let mapped = e2e_config_for(&config, Path::new("/m.gguf"), 8, 4);

        // 0 threads means "let llama.cpp decide": the e2e builder emits no -t.
        assert_eq!(mapped.threads, 0);
        // Context is clamped to the llama.cpp floor, raw and via the accessor.
        assert_eq!(mapped.ctx_size, MIN_E2E_CTX);
        assert_eq!(mapped.ctx_size(), MIN_E2E_CTX);
    }

    #[test]
    fn rerank_promotes_a_measured_config_over_the_estimate_winner() {
        let estimate_winner = TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16);
        let estimate_runner_up = TuneConfig::new(Quant::Q4_K_M, 2_048, 8, 512, Quant::Fp16);
        let ranked = [
            ranked(estimate_winner.clone(), 300.0, 100.0, 300.0),
            ranked(estimate_runner_up.clone(), 200.0, 200.0, 200.0),
        ];

        // The config that was second by estimate measured much better.
        let measured = [(estimate_runner_up.clone(), e2e(400.0, 40.0))];
        let reranked = rerank_with_e2e(&ranked, &measured, Objective::Throughput);

        assert_eq!(reranked[0].config, estimate_runner_up);
        assert!(reranked[0].verified);
        assert_eq!(reranked[0].decode_tps, 400.0);
        assert_eq!(reranked[0].ttft_ms, 40.0);
        assert!(reranked[0].score > reranked[1].score);
        assert!(reranked[0].measured.is_some());

        // The unmeasured config keeps its estimate score untouched.
        assert!(!reranked[1].verified);
        assert_eq!(reranked[1].config, estimate_winner);
        assert_eq!(reranked[1].decode_tps, 300.0);
        assert_eq!(reranked[1].score, 300.0);
    }

    #[test]
    fn rerank_ignores_an_incomplete_measurement() {
        let config = TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16);
        let ranked = [ranked(config.clone(), 300.0, 100.0, 300.0)];
        let measured = [(config, e2e(0.0, 0.0))];

        let reranked = rerank_with_e2e(&ranked, &measured, Objective::Throughput);
        assert!(!reranked[0].verified);
        assert!(reranked[0].measured.is_none());
        assert_eq!(reranked[0].decode_tps, 300.0);
    }

    #[test]
    fn run_verify_topk_rejects_a_missing_model_without_spawning() {
        let config = TuneConfig::new(Quant::Q4_K_M, 2_048, 4, 512, Quant::Fp16);
        let ranked = [ranked(config, 300.0, 100.0, 300.0)];

        let err = run_verify_topk(
            &ranked,
            Objective::Throughput,
            1,
            Path::new("definitely-not-a-real-model.gguf"),
            None,
            32,
            8,
            Some(64),
        )
        .expect_err("a missing model must be rejected");

        assert!(
            matches!(err, TuneError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );
        assert!(
            err.to_string().contains("not a readable file"),
            "message should name the model: {err}"
        );
    }

    #[test]
    fn run_verify_topk_rejects_zero_top_k() {
        let err = run_verify_topk(
            &[],
            Objective::Throughput,
            0,
            Path::new("/m.gguf"),
            None,
            32,
            8,
            None,
        )
        .expect_err("top_k = 0 must be rejected");
        assert!(matches!(err, TuneError::InvalidConfig(_)), "got {err:?}");
    }

    // ---- manual, real llama.cpp only --------------------------------------

    /// Mirrors `attune-bench`'s real-test policy: `#[ignore]` plus an env gate
    /// so CI (no llama.cpp, no model) can never run it. Run manually with:
    ///
    ///     ATTUNE_LLAMA_CLI=/path/to/llama-cli ATTUNE_TEST_MODEL=/path/to/tiny.gguf \
    ///         cargo test -p attune-tune -- --ignored verify::tests::real
    #[test]
    #[ignore = "manual: needs ATTUNE_LLAMA_CLI + ATTUNE_TEST_MODEL and a real llama.cpp build"]
    fn real_llama_cli_verifies_the_top_config() {
        let Some(model) = std::env::var_os(attune_bench::llama::ENV_TEST_MODEL) else {
            eprintln!("skipping: {} not set", attune_bench::llama::ENV_TEST_MODEL);
            return;
        };
        let model = PathBuf::from(model);
        let config = TuneConfig::new(Quant::Q4_K_M, 128, 2, 32, Quant::Fp16);
        let ranked = [ranked(config, 100.0, 500.0, 100.0)];

        let outcome = run_verify_topk(
            &ranked,
            Objective::Throughput,
            1,
            &model,
            None,
            16,
            4,
            Some(128),
        )
        .expect("real e2e verification should run");

        assert_eq!(outcome.configs.len(), 1);
        let verified = &outcome.configs[0];
        assert!(
            verified.verified,
            "top config was not measured: {outcome:?}"
        );
        assert!(verified.decode_tps > 0.0, "measured tok/s must be positive");
        let measured = verified.measured.as_ref().expect("measured block recorded");
        assert!(measured.is_complete());
        println!("{verified:?}");
    }
}
