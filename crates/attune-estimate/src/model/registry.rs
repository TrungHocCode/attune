//! The downloadable model catalog (`attune-catalog/1`, M6).
//!
//! Unlike the compiled-in [`super::MODEL_CATALOG`] (estimate-only names), every
//! entry here pins real files: repo, revision, size and SHA-256, read from the
//! Hugging Face API and the GGUF header by the daily generator.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Arch, EstimateError, Quant};

/// The only schema this build understands.
pub const SCHEMA: &str = "attune-catalog/1";

/// Repo owners the catalog generator is allowed to pull files from. Kept in
/// sync with `catalog/publishers.json` (pinned by a test).
pub const TRUSTED_PUBLISHERS: &[&str] = &[
    "Qwen",
    "microsoft",
    "mistralai",
    "HuggingFaceTB",
    "ibm-granite",
    "ggml-org",
    "bartowski",
    "unsloth",
];

/// Only llama.cpp builds from the official release feed may be installed.
const LLAMA_ASSET_PREFIX: &str = "https://github.com/ggml-org/llama.cpp/releases/download/";

/// The whole catalog document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: String,
    pub generated_at: String,
    pub llama_cpp: LlamaPin,
    pub models: Vec<CatalogModel>,
}

/// The llama.cpp release attune is tested with, per platform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaPin {
    pub tag: String,
    pub assets: Vec<LlamaAsset>,
}

/// One downloadable llama.cpp build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaAsset {
    /// `std::env::consts::OS` value (`windows`, `linux`).
    pub os: String,
    /// `std::env::consts::ARCH` value (`x86_64`).
    pub arch: String,
    /// `cpu` in M6 (GPU builds arrive with M7).
    pub variant: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

/// One model (merged across publisher repos).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub display: String,
    pub publisher: String,
    pub license: String,
    /// Original publisher's base repo creation date (`YYYY-MM-DD`), if known.
    #[serde(default)]
    pub released: Option<String>,
    #[serde(default)]
    pub gated: bool,
    pub params: u64,
    pub arch: CatalogArch,
    pub ctx_default: u32,
    pub files: Vec<CatalogFile>,
}

/// KV-cache shape from the GGUF header. Separate from [`Arch`] because that
/// type names a compiled-in preset (`&'static str`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogArch {
    pub n_layer: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl CatalogArch {
    /// The estimator's KV shape for this model.
    pub fn to_arch(self) -> Arch {
        Arch {
            name: "catalog",
            n_layer: self.n_layer,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

/// One pinned GGUF file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogFile {
    pub quant: Quant,
    pub repo: String,
    pub file: String,
    /// Repo commit the file is pinned to.
    pub revision: String,
    pub size: u64,
    pub sha256: String,
}

impl CatalogFile {
    /// Download URL pinned to the revision.
    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.file
        )
    }
}

impl CatalogModel {
    /// The file for `quant`, if this model ships one.
    pub fn file_for(&self, quant: Quant) -> Option<&CatalogFile> {
        self.files.iter().find(|file| file.quant == quant)
    }
}

impl Catalog {
    /// Exact id (case-insensitive), else a unique id prefix.
    pub fn find(&self, query: &str) -> Result<&CatalogModel, EstimateError> {
        let query = query.trim().to_ascii_lowercase();
        if let Some(model) = self
            .models
            .iter()
            .find(|m| m.id.eq_ignore_ascii_case(&query))
        {
            return Ok(model);
        }
        let matches: Vec<&CatalogModel> = self
            .models
            .iter()
            .filter(|m| m.id.to_ascii_lowercase().starts_with(&query))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one),
            [] => Err(EstimateError::InvalidModelName(format!(
                "`{query}` is not in the model catalog"
            ))),
            many => Err(EstimateError::InvalidModelName(format!(
                "`{query}` matches several catalog models: {}",
                many.iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

/// Parse and validate a catalog document.
pub fn parse_catalog(text: &str) -> Result<Catalog, EstimateError> {
    let catalog: Catalog =
        serde_json::from_str(text).map_err(|err| EstimateError::InvalidCatalog(err.to_string()))?;
    let bad = |msg: String| Err(EstimateError::InvalidCatalog(msg));
    if catalog.schema != SCHEMA {
        return bad(format!(
            "schema `{}` is not supported (this attune reads {SCHEMA})",
            catalog.schema
        ));
    }
    let mut ids = HashSet::new();
    for model in &catalog.models {
        if !ids.insert(model.id.to_ascii_lowercase()) {
            return bad(format!("duplicate model id `{}`", model.id));
        }
        if model.files.is_empty() {
            return bad(format!("model `{}` has no files", model.id));
        }
        for file in &model.files {
            if !is_sha256(&file.sha256) {
                return bad(format!("`{}` has an invalid sha256", file.file));
            }
            let owner = file.repo.split('/').next().unwrap_or("");
            if !TRUSTED_PUBLISHERS.contains(&owner) {
                return bad(format!(
                    "`{}` comes from untrusted owner `{owner}`",
                    file.file
                ));
            }
        }
    }
    for asset in &catalog.llama_cpp.assets {
        if !asset.url.starts_with(LLAMA_ASSET_PREFIX) {
            return bad(format!(
                "llama.cpp asset `{}` is not from the official release feed",
                asset.url
            ));
        }
        if !is_sha256(&asset.sha256) {
            return bad(format!(
                "llama.cpp asset `{}` has an invalid sha256",
                asset.url
            ));
        }
    }
    Ok(catalog)
}

/// Where the daily GitHub Action publishes the catalog.
pub const CATALOG_URL: &str =
    "https://raw.githubusercontent.com/TrungHocCode/attune/catalog/catalog.json";
/// Cached catalog inside the attune home dir.
pub const CACHE_FILE: &str = "catalog.json";
/// When the cached catalog was fetched (`{"fetched_at": <unix secs>}`).
pub const CACHE_META_FILE: &str = "catalog.meta.json";
/// A cached catalog younger than this is used without a network call.
pub const FRESH_SECS: u64 = 24 * 60 * 60;
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Built-in catalog: the floor when there is no network and no usable cache.
pub const SNAPSHOT: &str = include_str!("../../../../catalog/snapshot.json");

/// Which source a loaded catalog came from (shown to the user).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogOrigin {
    FreshCache,
    Fetched,
    StaleCache,
    Snapshot,
}

impl CatalogOrigin {
    /// Short human label for report footers.
    pub fn label(self) -> &'static str {
        match self {
            Self::FreshCache => "cached",
            Self::Fetched => "updated",
            Self::StaleCache => "cached, offline",
            Self::Snapshot => "built-in, offline",
        }
    }
}

/// A catalog plus where it came from.
#[derive(Debug, Clone)]
pub struct LoadedCatalog {
    pub catalog: Catalog,
    pub origin: CatalogOrigin,
    /// `generated_at` date (`YYYY-MM-DD`).
    pub date: String,
    pub warnings: Vec<String>,
}

/// Load the catalog: fresh cache → network → stale cache → built-in snapshot.
/// Never fails; the snapshot is always there.
pub fn load_catalog(refresh: bool) -> LoadedCatalog {
    let fetch = || {
        attune_fetch::fetch_text(
            CATALOG_URL,
            &attune_fetch::HostPolicy::production(),
            FETCH_TIMEOUT,
        )
        .map_err(|err| err.to_string())
    };
    match attune_fetch::home_dir() {
        Ok(home) => load_catalog_at(&home, refresh, unix_now(), &fetch),
        Err(err) => {
            let mut loaded = snapshot();
            loaded
                .warnings
                .push(format!("{err}; using the built-in catalog"));
            loaded
        }
    }
}

/// [`load_catalog`] with an explicit home dir, clock and fetcher (tests).
pub fn load_catalog_at(
    home: &std::path::Path,
    refresh: bool,
    now_unix: u64,
    fetch: &dyn Fn() -> Result<String, String>,
) -> LoadedCatalog {
    let cache_path = home.join(CACHE_FILE);
    let mut warnings = Vec::new();
    let cached = match std::fs::read_to_string(&cache_path) {
        Ok(text) => match parse_catalog(&text) {
            Ok(catalog) => Some(catalog),
            Err(err) => {
                warnings.push(format!("ignoring {}: {err}", cache_path.display()));
                None
            }
        },
        Err(_) => None,
    };
    let fetched_at = std::fs::read_to_string(home.join(CACHE_META_FILE))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|meta| meta.get("fetched_at")?.as_u64());

    if let (Some(catalog), Some(at), false) = (&cached, fetched_at, refresh) {
        if now_unix.saturating_sub(at) < FRESH_SECS {
            return loaded(catalog.clone(), CatalogOrigin::FreshCache, warnings);
        }
    }
    match fetch().and_then(|text| {
        parse_catalog(&text)
            .map(|c| (c, text))
            .map_err(|e| e.to_string())
    }) {
        Ok((catalog, text)) => {
            let meta = format!(r#"{{"fetched_at": {now_unix}}}"#);
            if let Err(err) = std::fs::write(&cache_path, text)
                .and_then(|()| std::fs::write(home.join(CACHE_META_FILE), meta))
            {
                warnings.push(format!("could not cache the catalog: {err}"));
            }
            return loaded(catalog, CatalogOrigin::Fetched, warnings);
        }
        Err(err) => warnings.push(format!("catalog update failed ({err})")),
    }
    match cached {
        Some(catalog) => loaded(catalog, CatalogOrigin::StaleCache, warnings),
        None => {
            let mut loaded = snapshot();
            loaded.warnings.splice(0..0, warnings);
            loaded
        }
    }
}

fn snapshot() -> LoadedCatalog {
    match parse_catalog(SNAPSHOT) {
        Ok(catalog) => loaded(catalog, CatalogOrigin::Snapshot, Vec::new()),
        // A broken snapshot is a build bug; the unit test pins it.
        Err(err) => LoadedCatalog {
            catalog: Catalog {
                schema: SCHEMA.to_string(),
                generated_at: String::new(),
                llama_cpp: LlamaPin {
                    tag: String::new(),
                    assets: Vec::new(),
                },
                models: Vec::new(),
            },
            origin: CatalogOrigin::Snapshot,
            date: String::new(),
            warnings: vec![format!("built-in catalog is invalid: {err}")],
        },
    }
}

fn loaded(catalog: Catalog, origin: CatalogOrigin, warnings: Vec<String>) -> LoadedCatalog {
    let date = catalog.generated_at.chars().take(10).collect();
    LoadedCatalog {
        catalog,
        origin,
        date,
        warnings,
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn model(id: &str, sha: &str) -> String {
        format!(
            r#"{{
              "id": "{id}", "display": "Qwen3 4B Instruct 2507", "publisher": "Qwen",
              "license": "apache-2.0", "released": "2025-07-30", "gated": false,
              "params": 4022468096,
              "arch": {{ "n_layer": 36, "n_kv_heads": 8, "head_dim": 128 }},
              "ctx_default": 262144,
              "files": [
                {{ "quant": "q4_k_m", "repo": "Qwen/Qwen3-4B-Instruct-2507-GGUF",
                  "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "revision": "0123abcd",
                  "size": 2497280480, "sha256": "{sha}" }}
              ]
            }}"#
        )
    }

    fn catalog(schema: &str, models: &[String]) -> String {
        format!(
            r#"{{
              "schema": "{schema}", "generated_at": "2026-11-26T03:00:00Z",
              "llama_cpp": {{ "tag": "b11804", "assets": [
                {{ "os": "windows", "arch": "x86_64", "variant": "cpu",
                  "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-b11804-bin-win-cpu-x64.zip",
                  "size": 12345678, "sha256": "{SHA_B}" }} ] }},
              "models": [{}]
            }}"#,
            models.join(",")
        )
    }

    #[test]
    fn parses_the_spec_example() {
        let text = catalog(SCHEMA, &[model("qwen3-4b-instruct-2507", SHA_A)]);
        let parsed = parse_catalog(&text).expect("parse");
        assert_eq!(parsed.llama_cpp.tag, "b11804");
        let file = &parsed.models[0].files[0];
        assert_eq!(file.quant, Quant::Q4_K_M);
        assert_eq!(
            file.url(),
            "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF/resolve/0123abcd/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        );
        assert_eq!(parsed.models[0].arch.to_arch().n_layer, 36);
        assert!(parsed.models[0].file_for(Quant::Q4_K_M).is_some());
        assert!(parsed.models[0].file_for(Quant::Q8_0).is_none());
    }

    #[test]
    fn rejects_unknown_schema() {
        let text = catalog("attune-catalog/2", &[model("a", SHA_A)]);
        assert!(parse_catalog(&text).is_err());
    }

    #[test]
    fn rejects_duplicate_ids_and_bad_sha() {
        let dup = catalog(SCHEMA, &[model("a", SHA_A), model("a", SHA_A)]);
        assert!(parse_catalog(&dup).is_err());
        let bad = catalog(SCHEMA, &[model("a", "not-a-sha")]);
        assert!(parse_catalog(&bad).is_err());
        // Moves the file list under an ignored key, leaving `files: []`.
        let empty_files = catalog(SCHEMA, &[model("a", SHA_A)])
            .replace(r#""files": ["#, r#""files": [], "unused": ["#);
        assert!(parse_catalog(&empty_files).is_err(), "{empty_files}");
    }

    #[test]
    fn rejects_untrusted_repo_owner() {
        let text = catalog(SCHEMA, &[model("a", SHA_A)]).replace(
            "Qwen/Qwen3-4B-Instruct-2507-GGUF",
            "evilcorp/Qwen3-4B-Instruct-2507-GGUF",
        );
        let err = parse_catalog(&text).expect_err("untrusted owner must be rejected");
        assert!(err.to_string().contains("evilcorp"), "{err}");
    }

    #[test]
    fn rejects_foreign_llama_asset_url() {
        let text = catalog(SCHEMA, &[model("a", SHA_A)]).replace(
            "https://github.com/ggml-org/llama.cpp/releases/download/",
            "https://evil.example/llama.cpp/",
        );
        let err = parse_catalog(&text).expect_err("foreign asset must be rejected");
        assert!(err.to_string().contains("official release feed"), "{err}");
    }

    #[test]
    fn trusted_list_covers_the_generator_publishers() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../../../../catalog/publishers.json"))
                .expect("publishers.json parses");
        let publishers = json["publishers"]
            .as_array()
            .expect("publishers is an array");
        assert!(!publishers.is_empty());
        for publisher in publishers {
            let name = publisher.as_str().expect("publisher is a string");
            assert!(
                TRUSTED_PUBLISHERS.contains(&name),
                "TRUSTED_PUBLISHERS is missing `{name}`"
            );
        }
    }

    fn temp_home(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-cat-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    const NOW: u64 = 1_800_000_000;
    const HOUR: u64 = 3600;

    fn seed_cache(home: &std::path::Path, text: &str, fetched_at: u64) {
        std::fs::write(home.join(CACHE_FILE), text).expect("cache");
        std::fs::write(
            home.join(CACHE_META_FILE),
            format!(r#"{{"fetched_at": {fetched_at}}}"#),
        )
        .expect("meta");
    }

    fn remote() -> String {
        catalog(SCHEMA, &[model("remote-model", SHA_A)])
    }

    #[test]
    fn snapshot_parses_and_has_every_tier() {
        let snapshot = parse_catalog(SNAPSHOT).expect("snapshot parses");
        let tiers = [
            (0, 2_000_000_000),
            (2_000_000_000, 5_000_000_000),
            (5_000_000_000, 10_000_000_000),
            (10_000_000_000, u64::MAX),
        ];
        for (lo, hi) in tiers {
            assert!(
                snapshot
                    .models
                    .iter()
                    .any(|m| m.params >= lo && m.params < hi),
                "no model in tier {lo}..{hi}"
            );
        }
        assert!(snapshot.models.iter().all(|m| !m.gated));
    }

    #[test]
    fn fresh_cache_is_used_without_fetching() {
        let home = temp_home("fresh");
        seed_cache(&home, &remote(), NOW - HOUR);
        let loaded = load_catalog_at(&home, false, NOW, &|| panic!("must not fetch"));
        assert_eq!(loaded.origin, CatalogOrigin::FreshCache);
        assert_eq!(loaded.catalog.models[0].id, "remote-model");
    }

    #[test]
    fn stale_cache_triggers_fetch_and_rewrites_cache() {
        let home = temp_home("stale");
        seed_cache(
            &home,
            &catalog(SCHEMA, &[model("old", SHA_A)]),
            NOW - 30 * HOUR,
        );
        let loaded = load_catalog_at(&home, false, NOW, &|| Ok(remote()));
        assert_eq!(loaded.origin, CatalogOrigin::Fetched);
        let cached = std::fs::read_to_string(home.join(CACHE_FILE)).expect("cache");
        assert!(cached.contains("remote-model"));
        let again = load_catalog_at(&home, false, NOW + HOUR, &|| panic!("fresh now"));
        assert_eq!(again.origin, CatalogOrigin::FreshCache);
    }

    #[test]
    fn offline_falls_back_to_stale_cache_then_snapshot() {
        let home = temp_home("offline");
        seed_cache(&home, &remote(), NOW - 30 * HOUR);
        let loaded = load_catalog_at(&home, false, NOW, &|| Err("offline".to_string()));
        assert_eq!(loaded.origin, CatalogOrigin::StaleCache);
        assert!(!loaded.warnings.is_empty());

        let empty = temp_home("offline-empty");
        let loaded = load_catalog_at(&empty, false, NOW, &|| Err("offline".to_string()));
        assert_eq!(loaded.origin, CatalogOrigin::Snapshot);
    }

    #[test]
    fn corrupt_or_newer_schema_cache_falls_back() {
        for (tag, text) in [
            ("corrupt", "{oops".to_string()),
            ("newer", catalog("attune-catalog/2", &[model("x", SHA_A)])),
        ] {
            let home = temp_home(tag);
            seed_cache(&home, &text, NOW - HOUR);
            let loaded = load_catalog_at(&home, false, NOW, &|| Err("offline".to_string()));
            assert_eq!(loaded.origin, CatalogOrigin::Snapshot, "{tag}");
            assert!(
                loaded.warnings.iter().any(|w| w.contains(CACHE_FILE)),
                "{tag}: {:?}",
                loaded.warnings
            );
        }
    }

    #[test]
    fn refresh_forces_fetch_even_when_fresh() {
        let home = temp_home("refresh");
        seed_cache(&home, &catalog(SCHEMA, &[model("old", SHA_A)]), NOW - HOUR);
        let loaded = load_catalog_at(&home, true, NOW, &|| Ok(remote()));
        assert_eq!(loaded.origin, CatalogOrigin::Fetched);
        assert_eq!(loaded.catalog.models[0].id, "remote-model");
    }

    #[test]
    fn find_exact_then_unique_prefix() {
        let one = parse_catalog(&catalog(SCHEMA, &[model("qwen3-4b-instruct-2507", SHA_A)]))
            .expect("parse");
        assert_eq!(
            one.find("qwen3-4b").expect("prefix").id,
            "qwen3-4b-instruct-2507"
        );
        assert_eq!(
            one.find("QWEN3-4B-Instruct-2507").expect("exact").id,
            "qwen3-4b-instruct-2507"
        );
        assert!(one.find("llama").is_err());

        let two = parse_catalog(&catalog(
            SCHEMA,
            &[
                model("qwen3-4b-instruct-2507", SHA_A),
                model("qwen3-4b-thinking-2507", SHA_A),
            ],
        ))
        .expect("parse");
        let err = two.find("qwen3-4b").expect_err("ambiguous").to_string();
        assert!(err.contains("qwen3-4b-instruct-2507"), "{err}");
        assert!(err.contains("qwen3-4b-thinking-2507"), "{err}");
    }
}
