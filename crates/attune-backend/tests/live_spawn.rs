//! Manual end-to-end test: spawn a **real** `llama-server` with a tiny model,
//! wait for `/health`, then prove the guard reaps it.
//!
//! This test is `#[ignore]`d *and* env-gated on `ATTUNE_LLAMA_SERVER` +
//! `ATTUNE_TEST_MODEL`, so it never runs in CI (AGENTS.md §4 / §6). Run it by
//! hand:
//!
//! ```text
//! $env:ATTUNE_LLAMA_SERVER = "C:\\path\\to\\llama-server.exe"
//! $env:ATTUNE_TEST_MODEL   = "C:\\path\\to\\tiny.gguf"
//! cargo test -p attune-backend --test live_spawn -- --ignored --nocapture
//! ```

use std::path::Path;
use std::time::Duration;

use attune_backend::llama_subprocess::{ENV_LLAMA_SERVER, ENV_TEST_MODEL};
use attune_backend::{health_wait, resolve_llama_server, server_args, ServerGuard};
use attune_profile::{ComputeConfig, LoadConfig, ModelConfig, Profile};

/// Bind an ephemeral port, then release it for the child.
///
/// There is a tiny race between the drop and the child's bind; for a manual
/// single-process test that is acceptable.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
    listener.local_addr().expect("read local addr").port()
}

fn test_profile(model: &str) -> Profile {
    Profile {
        name: "live-spawn".to_string(),
        model: ModelConfig {
            path: model.to_string(),
            quant: "q4_k_m".to_string(),
            size_bytes: 0,
        },
        load_config: LoadConfig {
            n_ctx: 256,
            n_batch: 64,
            n_ubatch: 32,
            n_gpu_layers: 0,
            mmap: true,
            mlock: false,
            // f16 KV keeps this independent of quantized-KV support. Flash
            // attention stays on (the profile default) so the live run proves
            // the backend's `-fa` -> `--flash-attn on` translation.
            flash_attn: true,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            // The M4 profile default. It is normalized to "omit --numa"
            // because `--numa local` is not a llama-server value.
            numa: "local".to_string(),
        },
        compute_config: ComputeConfig {
            n_threads: 2,
            n_threads_batch: 2,
            cpu_affinity: None,
        },
        ..Profile::default()
    }
}

#[test]
#[ignore = "requires ATTUNE_LLAMA_SERVER and ATTUNE_TEST_MODEL (real llama-server + GGUF)"]
fn spawns_a_real_server_waits_for_health_and_reaps_it() {
    let (Ok(server), Ok(model)) = (
        std::env::var(ENV_LLAMA_SERVER),
        std::env::var(ENV_TEST_MODEL),
    ) else {
        eprintln!("skipping: set {ENV_LLAMA_SERVER} and {ENV_TEST_MODEL}");
        return;
    };

    let binary = resolve_llama_server(Some(Path::new(&server)))
        .expect("explicit llama-server path should resolve");
    let profile = test_profile(&model);
    let port = free_port();
    let host = "127.0.0.1";
    let args = server_args(&profile, host, port);
    eprintln!("spawning {} {}", binary.display(), args.join(" "));

    let mut guard = ServerGuard::spawn(&binary, &args).expect("spawn llama-server");
    eprintln!("child pid {}", guard.pid());

    let url = format!("http://{host}:{port}/health");
    let waited = health_wait(&url, Duration::from_secs(60))
        .unwrap_or_else(|error| panic!("health wait failed for {url}: {error}"));
    eprintln!("health OK after {:.2}s", waited.as_secs_f64());

    assert!(
        guard.try_wait().expect("poll live child").is_none(),
        "the server must still be running after a successful health check"
    );

    guard.shutdown();
    assert!(
        guard.try_wait().expect("poll reaped child").is_some(),
        "shutdown must reap the child (no zombie)"
    );
}
