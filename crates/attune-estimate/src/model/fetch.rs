//! Online `config.json` fetch with an offline disk cache.
//!
//! `estimate --model <org>/<name>` resolves models outside the embedded
//! catalog by reading Hugging Face's `config.json` (about 2 KiB). The network
//! is an enhancement, never a requirement:
//!
//! 1. a fresh (< 30 days) cache is used without touching the network,
//! 2. otherwise the config is fetched and the cache rewritten,
//! 3. a failed fetch degrades to the stale cache entry,
//! 4. then to the embedded catalog when the repository name matches exactly,
//! 5. otherwise the caller gets a clean error naming `--params` / `--arch`.
//!
//! The cache lives at `.attune/catalog-cache.json` relative to the current
//! working directory. That keeps the project free of a home-directory crate
//! while still surviving across runs; the file is gitignored and holds one
//! entry, `{name, fetched_at_unix, entry}`.
//!
//! Only [`fetch_config`] performs network I/O. [`parse_config_json`], the
//! cache helpers and [`resolve_hf_model_at`] are deterministic, so tests never
//! need network access (the one transport test talks to a closed localhost
//! port).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::catalog::{ModelEntry, MODEL_CATALOG};
use super::Arch;
use crate::EstimateError;

/// Timeout for the whole ~2 KiB config request.
pub const FETCH_TIMEOUT_SECS: u64 = 5;

/// How long a cached config stays fresh: 30 days.
pub const CACHE_MAX_AGE_SECS: u64 = 30 * 24 * 60 * 60;

/// Cache directory, relative to the current working directory.
pub const CACHE_DIR: &str = ".attune";

/// Cache file name inside [`CACHE_DIR`].
pub const CACHE_FILE: &str = "catalog-cache.json";

const HF_BASE_URL: &str = "https://huggingface.co";
const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// Model facts recovered from a Hugging Face `config.json`.
///
/// This is the owned, runtime counterpart of [`ModelEntry`]: it cannot borrow
/// the `'static` catalog strings. `params` is derived from the published
/// architecture fields (see [`parse_config_json`]) because `config.json` does
/// not carry a parameter count.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FetchedConfig {
    /// Derived parameter count used for the weight footprint.
    pub params: u64,
    /// `num_hidden_layers`.
    pub n_layer: u32,
    /// `num_key_value_heads`, falling back to `num_attention_heads`.
    pub n_kv_heads: u32,
    /// `head_dim` when published, otherwise `hidden_size / num_attention_heads`.
    pub head_dim: u32,
    /// `max_position_embeddings`, or 0 when the config omits it.
    pub ctx_default: u32,
}

impl FetchedConfig {
    /// KV-cache shape for the existing roofline math.
    ///
    /// `Arch::name` is display-only on this path, so the runtime-fetched
    /// model keeps the static label `"hf-config"`.
    pub fn arch(&self) -> Arch {
        Arch {
            name: "hf-config",
            n_layer: self.n_layer,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

/// One cached config: the repository id, when it was fetched, and its facts.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CachedConfig {
    /// Hugging Face repository id (`org/name`).
    pub name: String,
    /// Unix timestamp (seconds) of the successful fetch.
    pub fetched_at_unix: u64,
    /// Facts parsed from that `config.json`.
    pub entry: FetchedConfig,
}

/// Where a resolved model's facts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// A fresh on-disk cache entry, no network access.
    Cache,
    /// A successful fetch from Hugging Face.
    Fetch,
    /// A stale cache entry served because the refetch failed.
    StaleCache,
    /// The embedded catalog, matched by the repository name after a failed
    /// fetch (e.g. `HuggingFaceTB/SmolLM2-135M`).
    EmbeddedCatalog,
}

/// The model facts plus provenance and non-fatal notices for `estimate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedHfModel {
    /// Report label: the repository id (or catalog name on the embedded path).
    pub label: String,
    /// Parameter count used for the weight footprint.
    pub params: u64,
    /// KV-cache shape.
    pub arch: Arch,
    /// Which source supplied the facts.
    pub source: ConfigSource,
    /// Notices to attach to every report (`offline, using …`, cache failures).
    pub warnings: Vec<String>,
}

/// The cache path for the current working directory.
pub fn cache_path() -> PathBuf {
    Path::new(CACHE_DIR).join(CACHE_FILE)
}

/// True when `input` looks like a Hugging Face repository id (`org/name`).
///
/// The GGUF-file check runs first in the CLI, so an existing `.gguf` path is
/// never mistaken for an id. Absolute paths, `./…`, `../…`, backslashes and
/// ids with more than one slash are rejected: short catalog names stay a
/// catalog-only lookup and are never guessed into an organization.
pub fn is_hf_id(input: &str) -> bool {
    hf_id(input).is_some()
}

/// Fetch `config.json` for `hf_id` from Hugging Face.
///
/// One bounded HTTPS request with [`FETCH_TIMEOUT_SECS`].
pub fn fetch_config(hf_id: &str) -> Result<String, EstimateError> {
    fetch_text(&config_url(hf_id), Duration::from_secs(FETCH_TIMEOUT_SECS))
}

/// Parse a `config.json` body into the facts the roofline needs.
///
/// The parameter count is derived from the standard transformer architecture:
/// per layer the Q/K/V/O projections, the three SwiGLU MLP matrices and the
/// two norms, plus the token embedding (and the LM head when embeddings are
/// not tied). This is an estimate — it intentionally ignores biases that only
/// some architectures have — and lands within ~1% of the published card for
/// the tested Llama/Qwen families.
///
/// `num_key_value_heads` is optional (MHA falls back to
/// `num_attention_heads`). `head_dim` is preferred when the config publishes
/// it (Qwen3 does; `hidden_size / num_attention_heads` would be wrong there).
pub fn parse_config_json(text: &str) -> Result<FetchedConfig, EstimateError> {
    let raw: RawConfig = serde_json::from_str(text)
        .map_err(|err| EstimateError::InvalidConfig(format!("config.json is not valid: {err}")))?;

    if raw.hidden_size == 0 {
        return Err(EstimateError::InvalidConfig(
            "hidden_size must be positive".to_string(),
        ));
    }
    if raw.num_hidden_layers == 0 {
        return Err(EstimateError::InvalidConfig(
            "num_hidden_layers must be positive".to_string(),
        ));
    }
    if raw.num_attention_heads == 0 {
        return Err(EstimateError::InvalidConfig(
            "num_attention_heads must be positive".to_string(),
        ));
    }

    let n_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
    if n_kv_heads == 0 {
        return Err(EstimateError::InvalidConfig(
            "num_key_value_heads must be positive".to_string(),
        ));
    }

    let head_dim = match raw.head_dim {
        Some(0) => {
            return Err(EstimateError::InvalidConfig(
                "head_dim must be positive".to_string(),
            ));
        }
        Some(head_dim) => head_dim,
        None => {
            if !raw.hidden_size.is_multiple_of(raw.num_attention_heads) {
                return Err(EstimateError::InvalidConfig(format!(
                    "hidden_size {} is not divisible by num_attention_heads {}",
                    raw.hidden_size, raw.num_attention_heads
                )));
            }
            raw.hidden_size / raw.num_attention_heads
        }
    };

    let intermediate_size = raw
        .intermediate_size
        .ok_or_else(|| EstimateError::InvalidConfig("missing intermediate_size".to_string()))?;
    let vocab_size = raw
        .vocab_size
        .ok_or_else(|| EstimateError::InvalidConfig("missing vocab_size".to_string()))?;

    let hidden = u128::from(raw.hidden_size);
    let layers = u128::from(raw.num_hidden_layers);
    let heads = u128::from(raw.num_attention_heads);
    let kv_heads = u128::from(n_kv_heads);
    let dim = u128::from(head_dim);
    let intermediate = u128::from(intermediate_size);
    let vocab = u128::from(vocab_size);

    let attention = hidden * heads * dim + 2 * hidden * kv_heads * dim + hidden * hidden;
    let mlp = 3 * hidden * intermediate;
    let norm = 2 * hidden;
    let embeddings = vocab * hidden;
    let mut params = (attention + mlp + norm) * layers + embeddings + hidden;
    if !raw.tie_word_embeddings.unwrap_or(false) {
        params += embeddings;
    }

    let params = u64::try_from(params).map_err(|_| {
        EstimateError::InvalidConfig("derived parameter count does not fit in u64".to_string())
    })?;

    Ok(FetchedConfig {
        params,
        n_layer: raw.num_hidden_layers,
        n_kv_heads,
        head_dim,
        ctx_default: raw.max_position_embeddings.unwrap_or(0),
    })
}

/// Read the cache, returning `None` for a missing, unreadable or corrupt file.
///
/// A corrupt cache is never an error: it is equivalent to no cache, and the
/// caller simply refetches.
pub fn read_cache(path: &Path) -> Option<CachedConfig> {
    let text = fs::read_to_string(path).ok()?;
    // Tolerate a UTF-8 BOM written by Windows editors/tools; our own writer
    // never emits one, but a hand-edited cache should not be discarded.
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    serde_json::from_str(text).ok()
}

/// Write the cache, creating [`CACHE_DIR`] when needed.
pub fn write_cache(path: &Path, cached: &CachedConfig) -> Result<(), EstimateError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|err| EstimateError::Cache(format!("create {}: {err}", parent.display())))?;
    }
    let text = serde_json::to_string_pretty(cached)
        .map_err(|err| EstimateError::Cache(err.to_string()))?;
    fs::write(path, text)
        .map_err(|err| EstimateError::Cache(format!("write {}: {err}", path.display())))
}

/// True when a cache entry fetched at `fetched_at_unix` is younger than
/// [`CACHE_MAX_AGE_SECS`] at `now_unix`.
///
/// A timestamp in the future is treated as stale so a clock jump triggers a
/// refetch instead of pinning the entry forever.
pub fn cache_is_fresh(fetched_at_unix: u64, now_unix: u64) -> bool {
    now_unix >= fetched_at_unix && now_unix - fetched_at_unix < CACHE_MAX_AGE_SECS
}

/// Resolve an `org/model` id for `estimate`, using the cache and network.
///
/// The cache path is `.attune/catalog-cache.json` under the current working
/// directory. `refresh` skips a fresh cache and forces a refetch; a failed
/// refetch still serves the stale entry.
pub fn resolve_hf_model(input: &str, refresh: bool) -> Result<ResolvedHfModel, EstimateError> {
    resolve_hf_model_at(input, refresh, &cache_path(), unix_now(), fetch_config)
}

/// Cache/fetch/fallback policy with injectable clock, path and fetcher.
///
/// Split from [`resolve_hf_model`] so tests can run the exact production
/// policy against canned JSON without network access.
fn resolve_hf_model_at<F>(
    input: &str,
    refresh: bool,
    cache_path: &Path,
    now_unix: u64,
    fetch: F,
) -> Result<ResolvedHfModel, EstimateError>
where
    F: FnOnce(&str) -> Result<String, EstimateError>,
{
    let hf_id = hf_id(input).ok_or_else(|| {
        EstimateError::InvalidModelName(format!(
            "{input} is not a Hugging Face repository id (expected org/name)"
        ))
    })?;

    let cached = read_cache(cache_path).filter(|cached| cached.name.eq_ignore_ascii_case(hf_id));
    if !refresh {
        if let Some(cached) = cached.as_ref() {
            if cache_is_fresh(cached.fetched_at_unix, now_unix) {
                return Ok(resolved_from(
                    &cached.name,
                    &cached.entry,
                    ConfigSource::Cache,
                    vec![],
                ));
            }
        }
    }

    match fetch(hf_id) {
        Ok(text) => {
            let entry = parse_config_json(&text)?;
            let cached = CachedConfig {
                name: hf_id.to_string(),
                fetched_at_unix: now_unix,
                entry: entry.clone(),
            };
            let mut warnings = Vec::new();
            if let Err(err) = write_cache(cache_path, &cached) {
                warnings.push(format!("catalog cache not written: {err}"));
            }
            Ok(resolved_from(hf_id, &entry, ConfigSource::Fetch, warnings))
        }
        Err(fetch_err) => {
            if let Some(cached) = cached {
                let age_days = now_unix.saturating_sub(cached.fetched_at_unix) / SECS_PER_DAY;
                return Ok(resolved_from(
                    &cached.name,
                    &cached.entry,
                    ConfigSource::StaleCache,
                    vec![format!(
                        "offline, using cached config ({age_days} days old)"
                    )],
                ));
            }
            if let Some(entry) = catalog_for_hf_id(hf_id) {
                return Ok(ResolvedHfModel {
                    label: entry.name.to_string(),
                    params: entry.params,
                    arch: entry.arch,
                    source: ConfigSource::EmbeddedCatalog,
                    warnings: vec!["offline, using embedded catalog".to_string()],
                });
            }
            let detail = match fetch_err {
                EstimateError::FetchConfig(detail) => detail,
                other => other.to_string(),
            };
            Err(EstimateError::FetchConfig(format!(
                "{detail}; no cache or embedded entry for {hf_id}, pass --params <SIZE> --arch <NAME>"
            )))
        }
    }
}

/// Build a resolved model from fetched/cached facts.
fn resolved_from(
    label: &str,
    config: &FetchedConfig,
    source: ConfigSource,
    warnings: Vec<String>,
) -> ResolvedHfModel {
    ResolvedHfModel {
        label: label.to_string(),
        params: config.params,
        arch: config.arch(),
        source,
        warnings,
    }
}

/// Match the repository segment of an id against a catalog name.
///
/// Only an exact (case-insensitive) name match is used; a suffix such as
/// `Qwen2.5-7B-Instruct` is not guessed down to `qwen2.5-7b`.
fn catalog_for_hf_id(hf_id: &str) -> Option<&'static ModelEntry> {
    let short = hf_id.rsplit('/').next()?;
    MODEL_CATALOG
        .iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(short))
}

/// Split and validate an `org/name` repository id.
fn hf_id(input: &str) -> Option<&str> {
    let value = input.trim();
    if value.is_empty() || value.contains('\\') || value.starts_with('/') || value.contains("://") {
        return None;
    }

    let mut parts = value.split('/');
    let org = parts.next()?;
    let name = parts.next()?;
    if parts.next().is_some() || org.is_empty() || name.is_empty() {
        return None;
    }
    if is_hf_segment(org) && is_hf_segment(name) {
        Some(value)
    } else {
        None
    }
}

/// Allowed characters in a repository-owner/name segment, plus the `./..`
/// path escape guard.
fn is_hf_segment(segment: &str) -> bool {
    segment != "."
        && segment != ".."
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// The `config.json` endpoint for a repository id.
fn config_url(hf_id: &str) -> String {
    format!("{HF_BASE_URL}/{hf_id}/resolve/main/config.json")
}

/// One bounded GET, translating every transport/status failure.
fn fetch_text(url: &str, timeout: Duration) -> Result<String, EstimateError> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|err| EstimateError::FetchConfig(format!("{url}: {err}")))?;
    response
        .body_mut()
        .read_to_string()
        .map_err(|err| EstimateError::FetchConfig(format!("{url}: {err}")))
}

/// Current Unix time, falling back to 0 if the clock is before the epoch.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// The subset of `config.json` the roofline needs.
#[derive(Debug, serde::Deserialize)]
struct RawConfig {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from HuggingFaceTB/SmolLM2-135M config.json (tied embeddings).
    const SMOLLM2_135M: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 576,
        "intermediate_size": 1536,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 9,
        "num_hidden_layers": 30,
        "num_key_value_heads": 3,
        "tie_word_embeddings": true,
        "vocab_size": 49152
    }"#;

    /// Trimmed from Qwen/Qwen3-0.6B config.json: explicit head_dim (128)
    /// differs from hidden_size / num_attention_heads (64).
    const QWEN3_06B: &str = r#"{
        "head_dim": 128,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "tie_word_embeddings": true,
        "vocab_size": 151936
    }"#;

    fn temp_cache(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-fetch-{}-{tag}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir.join(CACHE_FILE)
    }

    fn cleanup(path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = fs::remove_dir_all(dir);
        }
    }

    fn cached(name: &str, fetched_at_unix: u64) -> CachedConfig {
        CachedConfig {
            name: name.to_string(),
            fetched_at_unix,
            entry: parse_config_json(SMOLLM2_135M).expect("canned config"),
        }
    }

    #[test]
    fn parses_the_smollm2_config_shape() {
        let config = parse_config_json(SMOLLM2_135M).expect("SmolLM2 config");
        assert_eq!(config.n_layer, 30);
        assert_eq!(config.n_kv_heads, 3);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.ctx_default, 8_192);
        // 30 layers × (attention 884_736 + MLP 2_654_208 + norms 1_152)
        // + tied embedding 28_311_552 + final norm 576.
        assert_eq!(config.params, 134_515_008);
        assert_eq!(config.arch().name, "hf-config");
        assert_eq!(config.arch().n_layer, 30);
    }

    #[test]
    fn prefers_the_published_head_dim() {
        let config = parse_config_json(QWEN3_06B).expect("Qwen3 config");
        // hidden / heads = 64, but the config publishes 128.
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.params, 566_682_624);
    }

    #[test]
    fn kv_heads_fall_back_to_attention_heads() {
        let text = r#"{
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 128256
        }"#;
        let config = parse_config_json(text).expect("MHA config");
        assert_eq!(config.n_kv_heads, 32);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn non_tied_embeddings_add_the_lm_head() {
        let tied = parse_config_json(SMOLLM2_135M).expect("tied");
        let untied = parse_config_json(&SMOLLM2_135M.replace("true", "false")).expect("untied");
        // One extra vocab × hidden matrix: 49_152 × 576.
        assert_eq!(untied.params - tied.params, 28_311_552);
    }

    #[test]
    fn rejects_configs_with_unusable_shapes() {
        let cases = [
            "{}",
            r#"{"hidden_size":0,"num_hidden_layers":30,"num_attention_heads":9,"intermediate_size":1536,"vocab_size":49152}"#,
            r#"{"hidden_size":576,"num_hidden_layers":0,"num_attention_heads":9,"intermediate_size":1536,"vocab_size":49152}"#,
            r#"{"hidden_size":576,"num_hidden_layers":30,"num_attention_heads":0,"intermediate_size":1536,"vocab_size":49152}"#,
            r#"{"hidden_size":100,"num_hidden_layers":30,"num_attention_heads":3,"intermediate_size":1536,"vocab_size":49152}"#,
            r#"{"hidden_size":576,"num_hidden_layers":30,"num_attention_heads":9,"head_dim":0,"intermediate_size":1536,"vocab_size":49152}"#,
            r#"{"hidden_size":576,"num_hidden_layers":30,"num_attention_heads":9,"vocab_size":49152}"#,
            r#"{"hidden_size":576,"num_hidden_layers":30,"num_attention_heads":9,"intermediate_size":1536}"#,
        ];
        for text in cases {
            let err = parse_config_json(text).expect_err("config should be rejected");
            assert!(
                matches!(err, EstimateError::InvalidConfig(_)),
                "{text}: got {err:?}"
            );
        }
    }

    #[test]
    fn recognizes_repository_ids_but_not_paths_or_short_names() {
        for id in ["Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-135M"] {
            assert!(is_hf_id(id), "{id} should be an id");
            assert_eq!(hf_id(id), Some(id));
        }
        for value in [
            "llama-3.1-8b",
            "org/",
            "/org/name",
            "./model",
            "../model",
            "org/model/extra",
            "org\\model",
            "https://huggingface.co/org/model",
            "  ",
        ] {
            assert!(!is_hf_id(value), "{value:?} should not be an id");
        }
        assert_eq!(hf_id("  Qwen/Qwen3-0.6B  "), Some("Qwen/Qwen3-0.6B"));
    }

    #[test]
    fn config_url_points_at_the_hugging_face_resolve_endpoint() {
        assert_eq!(
            config_url("Qwen/Qwen2.5-0.5B"),
            "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/config.json"
        );
    }

    #[test]
    fn cache_round_trips_and_freshness_is_thirty_days() {
        let path = temp_cache("roundtrip");
        let entry = cached("Qwen/Qwen2.5-0.5B", 1_000);
        write_cache(&path, &entry).expect("write cache");
        assert_eq!(read_cache(&path), Some(entry));

        assert!(cache_is_fresh(1_000, 1_000));
        assert!(cache_is_fresh(1_000, 1_000 + CACHE_MAX_AGE_SECS - 1));
        assert!(!cache_is_fresh(1_000, 1_000 + CACHE_MAX_AGE_SECS));
        // A future timestamp must not pin the entry.
        assert!(!cache_is_fresh(2_000, 1_000));
        cleanup(&path);
    }

    #[test]
    fn corrupt_cache_is_treated_as_missing() {
        let path = temp_cache("corrupt");
        fs::write(&path, "{ not json").expect("write junk");
        assert_eq!(read_cache(&path), None);
        cleanup(&path);
    }

    #[test]
    fn cache_read_tolerates_a_utf8_bom() {
        let path = temp_cache("bom");
        let entry = cached("Qwen/Qwen2.5-0.5B", 42);
        let mut text = String::from("\u{feff}");
        text.push_str(&serde_json::to_string(&entry).expect("serialize cache"));
        fs::write(&path, text).expect("write BOM cache");
        assert_eq!(read_cache(&path), Some(entry));
        cleanup(&path);
    }

    #[test]
    fn fresh_cache_is_used_without_fetching() {
        let path = temp_cache("fresh");
        let now = 1_700_000_000;
        write_cache(&path, &cached("Qwen/Qwen2.5-0.5B", now - 60)).expect("seed cache");
        let fetch = |_: &str| -> Result<String, EstimateError> {
            panic!("a fresh cache must not touch the network")
        };

        let resolved =
            resolve_hf_model_at("Qwen/Qwen2.5-0.5B", false, &path, now, fetch).expect("cache hit");
        assert_eq!(resolved.source, ConfigSource::Cache);
        assert_eq!(resolved.params, 134_515_008);
        assert!(resolved.warnings.is_empty());
        cleanup(&path);
    }

    #[test]
    fn refresh_ignores_a_fresh_cache() {
        let path = temp_cache("refresh");
        let now = 1_700_000_000;
        write_cache(&path, &cached("Qwen/Qwen3-0.6B", now - 60)).expect("seed cache");
        let fetch = |id: &str| {
            assert_eq!(id, "Qwen/Qwen3-0.6B");
            Ok(QWEN3_06B.to_string())
        };

        let resolved = resolve_hf_model_at("Qwen/Qwen3-0.6B", true, &path, now, fetch)
            .expect("forced refetch");
        assert_eq!(resolved.source, ConfigSource::Fetch);
        assert_eq!(resolved.arch.head_dim, 128);
        // The cache was rewritten with the fresh timestamp/facts.
        let cached = read_cache(&path).expect("rewritten cache");
        assert_eq!(cached.fetched_at_unix, now);
        assert_eq!(cached.entry.head_dim, 128);
        cleanup(&path);
    }

    #[test]
    fn stale_cache_is_refetched() {
        let path = temp_cache("stale-refetch");
        let now = 1_700_000_000;
        write_cache(
            &path,
            &cached("Qwen/Qwen3-0.6B", now - CACHE_MAX_AGE_SECS - 1),
        )
        .expect("seed stale cache");
        let fetch = |_: &str| Ok(QWEN3_06B.to_string());

        let resolved = resolve_hf_model_at("Qwen/Qwen3-0.6B", false, &path, now, fetch)
            .expect("refetch stale cache");
        assert_eq!(resolved.source, ConfigSource::Fetch);
        assert_eq!(read_cache(&path).expect("cache").fetched_at_unix, now);
        cleanup(&path);
    }

    #[test]
    fn stale_cache_serves_offline_when_fetch_fails() {
        let path = temp_cache("stale-offline");
        let now = 1_700_000_000;
        write_cache(&path, &cached("Qwen/Qwen3-0.6B", now - 31 * SECS_PER_DAY))
            .expect("seed stale cache");
        let fetch = |_: &str| Err(EstimateError::FetchConfig("network down".to_string()));

        let resolved = resolve_hf_model_at("Qwen/Qwen3-0.6B", false, &path, now, fetch)
            .expect("offline fallback");
        assert_eq!(resolved.source, ConfigSource::StaleCache);
        assert_eq!(resolved.params, 134_515_008);
        assert_eq!(
            resolved.warnings,
            vec!["offline, using cached config (31 days old)".to_string()]
        );
        cleanup(&path);
    }

    #[test]
    fn embedded_catalog_is_the_last_resort() {
        let path = temp_cache("embedded");
        let fetch = |_: &str| Err(EstimateError::FetchConfig("network down".to_string()));

        let resolved = resolve_hf_model_at(
            "HuggingFaceTB/SmolLM2-135M",
            false,
            &path,
            1_700_000_000,
            fetch,
        )
        .expect("catalog fallback");
        assert_eq!(resolved.source, ConfigSource::EmbeddedCatalog);
        assert_eq!(resolved.label, "smollm2-135m");
        assert_eq!(resolved.params, 135_000_000);
        assert_eq!(resolved.arch.n_layer, 30);
        assert_eq!(
            resolved.warnings,
            vec!["offline, using embedded catalog".to_string()]
        );
        // The failed fetch must not create a cache file.
        assert!(!path.exists());
        cleanup(&path);
    }

    #[test]
    fn no_cache_and_no_catalog_match_is_a_clean_error() {
        let path = temp_cache("no-fallback");
        let fetch = |_: &str| Err(EstimateError::FetchConfig("network down".to_string()));

        let err = resolve_hf_model_at("Qwen/Unknown-Model", false, &path, 1_700_000_000, fetch)
            .expect_err("no fallback must fail");
        assert!(matches!(err, EstimateError::FetchConfig(_)), "got {err:?}");
        let message = err.to_string();
        assert!(message.contains("network down"), "message: {message}");
        assert!(message.contains("--params"), "message: {message}");
        // The low-level prefix must not be repeated by the wrapper.
        assert_eq!(message.matches("config fetch failed").count(), 1);
        cleanup(&path);
    }

    #[test]
    fn non_repository_input_is_rejected_before_fetching() {
        let path = temp_cache("not-an-id");
        let fetch = |_: &str| -> Result<String, EstimateError> {
            panic!("a short name must never be fetched")
        };
        let err = resolve_hf_model_at("smollm2-135m", false, &path, 1_700_000_000, fetch)
            .expect_err("short names are catalog-only");
        assert!(
            matches!(err, EstimateError::InvalidModelName(_)),
            "got {err:?}"
        );
        cleanup(&path);
    }

    #[test]
    fn connection_refusal_maps_to_a_fetch_error() {
        // Closed localhost port: no external network, no DNS, fast refusal.
        let err = fetch_text("http://127.0.0.1:9/config.json", Duration::from_millis(500))
            .expect_err("closed port must fail");
        assert!(matches!(err, EstimateError::FetchConfig(_)), "got {err:?}");
        assert!(err.to_string().contains("config fetch failed"));
    }
}
