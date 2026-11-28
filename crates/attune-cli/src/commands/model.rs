//! Shared `--model` resolution for `attune estimate` and `attune tune`.
//!
//! Both commands accept the same sources, in this priority order:
//!
//! 1. an existing `.gguf` file, probed from its header;
//! 2. an `org/model` Hugging Face id, resolved from the 30-day cache or a
//!    fetch (with stale-cache / embedded-catalog fallbacks);
//! 3. a live catalog id (exact or unique prefix);
//! 4. the built-in estimate-only registry, so old names such as
//!    `llama-3.1-8b` and `smollm2-135m` keep working;
//! 5. manual `--params` plus an optional `--arch`.
//!
//! Only wiring lives here: probing, fetching, catalog access and the KV math
//! are in `attune-estimate`. `tune` passes the quant it will rank so a catalog
//! entry pins its file (size + download source); `estimate` passes `None` and
//! keeps the nominal bytes-per-param footprint, because it predicts formats
//! that may not be published as a file.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
use attune_estimate::model::registry::{load_catalog, Catalog, CatalogModel, LoadedCatalog};
use attune_estimate::{
    is_gguf_file, is_hf_id, kv_bytes_per_token, parse_arch, parse_model, parse_params,
    probe_gguf_file, resolve_hf_model, Arch, ConfigSource, GgufProbe, Quant,
};
use attune_profile::ModelSource;

/// Upper bound on the live-catalog ids named in an unknown-model error. Enough
/// to make a typo obvious without flooding the terminal.
const SUGGESTION_LIMIT: usize = 8;

/// Notice attached when a probed GGUF has no usable attention head count, so
/// the KV-cache shape cannot be derived from the header.
pub const GGUF_KV_EXCLUDED_WARNING: &str = "KV excluded, GGUF has no usable attention head count";

/// Notice attached when manual `--params` has no `--arch`, so the decode
/// prediction omits the KV-cache read.
pub const KV_EXCLUDED_WARNING: &str = "KV excluded, unknown arch (pass --arch)";

/// Where the resolved model facts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProvenance {
    /// Facts read from a `.gguf` header.
    Gguf,
    /// Facts fetched from Hugging Face (fresh fetch or disk cache).
    Fetched,
    /// Facts from a catalog (live or the built-in registry).
    Catalog,
    /// Facts from `--params` / `--arch`.
    Manual,
}

/// Model inputs resolved from GGUF, fetch/cache, catalog or manual flags.
///
/// One resolver feeds both commands, so the KV shape, context, weights and
/// warnings cannot drift between `estimate` and `tune`.
#[derive(Debug)]
pub struct ModelResolution {
    /// Report label: catalog id, GGUF path or the literal `--params` text.
    pub label: String,
    /// Human-readable source label for the pretty `Model:` line
    /// (`provider/name`, an `org/model` id, a path or the `--params` text).
    pub display: String,
    /// Parameter count used for the weight footprint.
    pub params: u64,
    /// Resolved KV-cache shape, when the source had one.
    pub arch: Option<Arch>,
    /// KV-cache traffic per generated token; 0 when `arch` is unknown.
    pub kv_per_token: f64,
    /// Context window documented by the source (catalog entry, fetched
    /// `config.json` or GGUF `context_length`). An explicit `--ctx` always
    /// wins over this.
    pub ctx_default: Option<u32>,
    /// Which source supplied the facts.
    pub provenance: ModelProvenance,
    /// Probe result when `--model` was a `.gguf` file path.
    pub gguf: Option<GgufProbe>,
    /// Real weight footprint when the source pins a file (GGUF or a catalog
    /// file). `None` keeps the nominal bytes-per-param table.
    pub weight_bytes: Option<u64>,
    /// Every non-fatal notice from resolution (fetch fallbacks, catalog/file
    /// disagreement, omitted KV).
    pub warnings: Vec<String>,
    /// `model.path` for a saved profile when it differs from `label` (a
    /// catalog model's portable `~/.attune/models/...` path).
    pub model_path: Option<String>,
    /// Download source for a catalog model (profile v2).
    pub source: Option<ModelSource>,
}

/// The path when `--model` is an existing GGUF file (`.gguf` extension or
/// `GGUF` magic), which is probed instead of being looked up as a catalog name.
pub fn as_gguf_path(value: &str) -> Option<&Path> {
    let path = Path::new(value.trim());
    is_gguf_file(path).then_some(path)
}

/// Resolve model facts from GGUF, fetch/cache, catalog or manual flags.
///
/// The catalog is loaded only when the input reaches the catalog branch, so a
/// `.gguf` path or an `org/model` id never pays for a catalog fetch.
pub fn resolve_model(
    model: Option<&str>,
    params: Option<&str>,
    arch: Option<&str>,
    refresh_catalog: bool,
    quant: Option<Quant>,
) -> Result<ModelResolution> {
    let Some(name) = model else {
        return resolve_manual(params, arch);
    };
    if let Some(path) = as_gguf_path(name) {
        return resolve_gguf(name, path);
    }
    if is_hf_id(name) {
        return resolve_hf(name, refresh_catalog);
    }
    let loaded = load_catalog(refresh_catalog);
    resolve_catalog_name(&loaded, name, quant)
}

/// Resolve a catalog short name against an already loaded catalog.
///
/// Split from [`resolve_model`] so tests can inject `parse_catalog(SNAPSHOT)`
/// (and its warnings) without touching the network.
pub fn resolve_catalog_name(
    loaded: &LoadedCatalog,
    name: &str,
    quant: Option<Quant>,
) -> Result<ModelResolution> {
    match loaded.catalog.find(name) {
        Ok(model) => {
            let mut resolution = resolve_catalog_model(model, quant)?;
            resolution.warnings.extend(loaded.warnings.iter().cloned());
            Ok(resolution)
        }
        Err(catalog_err) => {
            // The built-in registry is the fallback for estimate-only names
            // (`llama-3.1-8b`, `gemma-2-9b`, ...) that the live catalog drops.
            if let Ok(entry) = parse_model(name) {
                return Ok(ModelResolution {
                    label: entry.name.to_string(),
                    display: format!("{}/{}", entry.provider, entry.name),
                    params: entry.params,
                    arch: Some(entry.arch),
                    kv_per_token: kv_bytes_per_token(&entry.arch),
                    ctx_default: Some(entry.ctx_default),
                    provenance: ModelProvenance::Catalog,
                    gguf: None,
                    weight_bytes: None,
                    warnings: loaded.warnings.clone(),
                    model_path: None,
                    source: None,
                });
            }
            Err(anyhow!(
                "{catalog_err}; {}",
                unknown_model_hint(&loaded.catalog, name)
            ))
        }
    }
}

/// Facts and download source for one catalog model at `quant`.
///
/// The file does not need to exist: the catalog pins its exact size and shape.
/// `quant: None` (estimate) skips the file lookup so every format can still be
/// predicted; `Some` (tune) fails when the catalog ships no such file.
pub fn resolve_catalog_model(
    model: &CatalogModel,
    quant: Option<Quant>,
) -> Result<ModelResolution> {
    let arch = model.arch.to_arch();
    let mut resolution = ModelResolution {
        label: model.id.clone(),
        display: format!("{}/{}", model.publisher, model.display),
        params: model.params,
        arch: Some(arch),
        kv_per_token: kv_bytes_per_token(&arch),
        ctx_default: Some(model.ctx_default),
        provenance: ModelProvenance::Catalog,
        gguf: None,
        weight_bytes: None,
        warnings: Vec::new(),
        model_path: None,
        source: None,
    };
    let Some(quant) = quant else {
        return Ok(resolution);
    };
    let file = model.file_for(quant).ok_or_else(|| {
        let have: Vec<&str> = model.files.iter().map(|file| file.quant.as_str()).collect();
        anyhow!(
            "{} has no {} file in the catalog; available: {}",
            model.id,
            quant.as_str(),
            have.join(", ")
        )
    })?;
    let source = ModelSource {
        repo: file.repo.clone(),
        file: file.file.clone(),
        revision: file.revision.clone(),
        sha256: file.sha256.clone(),
        size: file.size,
    };
    resolution.weight_bytes = Some(file.size);
    resolution.model_path = Some(source.home_path());
    resolution.source = Some(source);
    Ok(resolution)
}

/// The actionable tail of an unknown-model error: up to [`SUGGESTION_LIMIT`]
/// live-catalog ids that look like the input, plus the `org/model` and
/// `--params` escapes.
fn unknown_model_hint(catalog: &Catalog, name: &str) -> String {
    let suggestions = catalog.suggest_ids(name, SUGGESTION_LIMIT);
    let close = if suggestions.is_empty() {
        "no close ids in the catalog".to_string()
    } else {
        format!("close ids: {}", suggestions.join(", "))
    };
    format!("{close}; or pass a Hugging Face id like `org/model`, or --params <SIZE> --arch <NAME>")
}

/// Probe an existing `.gguf` file: facts plus its real on-disk size.
fn resolve_gguf(name: &str, path: &Path) -> Result<ModelResolution> {
    let probe = probe_gguf_file(path)
        .map_err(|err| anyhow!("failed to probe {}: {err}", path.display()))?;
    let params = probe.params().ok_or_else(|| {
        anyhow!(
            "{} has no general.parameter_count and no catalog match; use --params <SIZE> --arch <NAME> instead of --model",
            path.display()
        )
    })?;
    let weight_bytes = fs::metadata(path)
        .map_err(|err| anyhow!("failed to stat {}: {err}", path.display()))?
        .len();
    let arch = probe.arch;
    let mut warnings = Vec::new();
    if let Some(warning) = probe.warning.clone() {
        warnings.push(warning);
    }
    if arch.is_none() {
        warnings.push(GGUF_KV_EXCLUDED_WARNING.to_string());
    }
    Ok(ModelResolution {
        label: name.to_string(),
        display: name.to_string(),
        params,
        arch,
        kv_per_token: arch.map_or(0.0, |arch| kv_bytes_per_token(&arch)),
        ctx_default: probe.facts.context_length,
        provenance: ModelProvenance::Gguf,
        gguf: Some(probe),
        weight_bytes: Some(weight_bytes),
        warnings,
        model_path: None,
        source: None,
    })
}

/// Resolve an `org/model` id through the Hugging Face config cache.
fn resolve_hf(name: &str, refresh_catalog: bool) -> Result<ModelResolution> {
    let resolved = resolve_hf_model(name, refresh_catalog).map_err(|err| anyhow!("{err}"))?;
    let provenance = match resolved.source {
        // The offline fallback returns embedded catalog facts even though it
        // was reached through the fetch path.
        ConfigSource::EmbeddedCatalog => ModelProvenance::Catalog,
        ConfigSource::Cache | ConfigSource::Fetch | ConfigSource::StaleCache => {
            ModelProvenance::Fetched
        }
    };
    let arch = resolved.arch;
    Ok(ModelResolution {
        label: resolved.label.clone(),
        display: resolved.label,
        params: resolved.params,
        arch: Some(arch),
        kv_per_token: kv_bytes_per_token(&arch),
        ctx_default: resolved.ctx_default,
        provenance,
        gguf: None,
        weight_bytes: None,
        warnings: resolved.warnings,
        model_path: None,
        source: None,
    })
}

/// Manual `--params` (+ optional `--arch`): no catalog, no file.
fn resolve_manual(params: Option<&str>, arch: Option<&str>) -> Result<ModelResolution> {
    let params = params.ok_or_else(|| anyhow!("pass --model <NAME_OR_PATH> or --params <SIZE>"))?;
    let arch = match arch {
        Some(name) => Some(*parse_arch(name)?),
        None => None,
    };
    Ok(ModelResolution {
        label: params.trim().to_string(),
        display: params.trim().to_string(),
        params: parse_params(params)?,
        arch,
        kv_per_token: arch.map_or(0.0, |arch| kv_bytes_per_token(&arch)),
        ctx_default: None,
        provenance: ModelProvenance::Manual,
        gguf: None,
        weight_bytes: None,
        warnings: if arch.is_none() {
            vec![KV_EXCLUDED_WARNING.to_string()]
        } else {
            Vec::new()
        },
        model_path: None,
        source: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use attune_estimate::model::registry::{parse_catalog, CatalogOrigin, SNAPSHOT};

    fn loaded_snapshot() -> LoadedCatalog {
        LoadedCatalog {
            catalog: parse_catalog(SNAPSHOT).expect("snapshot parses"),
            origin: CatalogOrigin::Snapshot,
            date: String::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn live_catalog_id_resolves_with_its_documented_context() {
        let loaded = loaded_snapshot();
        let resolved =
            resolve_catalog_name(&loaded, "qwen3-4b-instruct-2507", None).expect("catalog id");

        assert_eq!(resolved.label, "qwen3-4b-instruct-2507");
        assert_eq!(resolved.params, 4_022_468_096);
        assert_eq!(resolved.arch.map(|arch| arch.name), Some("catalog"));
        assert_eq!(resolved.ctx_default, Some(262_144));
        assert_eq!(resolved.provenance, ModelProvenance::Catalog);
        // estimate does not pin a file, so the nominal footprint stays.
        assert_eq!(resolved.weight_bytes, None);
        assert!(resolved.warnings.is_empty());
    }

    #[test]
    fn live_catalog_prefix_matches_like_tune() {
        let loaded = loaded_snapshot();
        let resolved = resolve_catalog_name(&loaded, "QWEN3-4B-INSTRUCT", None).expect("prefix");
        assert_eq!(resolved.label, "qwen3-4b-instruct-2507");
    }

    #[test]
    fn old_registry_name_still_resolves_when_the_live_catalog_lacks_it() {
        let loaded = loaded_snapshot();
        assert!(
            loaded.catalog.find("llama-3.1-8b").is_err(),
            "fixture moved"
        );

        let resolved = resolve_catalog_name(&loaded, "Llama-3.1-8B", None).expect("built-in name");
        assert_eq!(resolved.label, "llama-3.1-8b");
        assert_eq!(resolved.display, "meta/llama-3.1-8b");
        assert_eq!(resolved.params, 8_030_000_000);
        assert_eq!(resolved.ctx_default, Some(131_072));
        assert_eq!(resolved.provenance, ModelProvenance::Catalog);
    }

    #[test]
    fn unknown_name_lists_close_live_ids_and_the_hf_escape() {
        let loaded = loaded_snapshot();
        let message = resolve_catalog_name(&loaded, "4b-instruct", None)
            .expect_err("unknown name must list close ids")
            .to_string();

        assert!(
            message.contains("qwen3-4b-instruct-2507"),
            "message: {message}"
        );
        assert!(message.contains("Hugging Face id"), "message: {message}");
        assert!(message.contains("`org/model`"), "message: {message}");
        assert!(message.contains("--params"), "message: {message}");
    }

    #[test]
    fn unknown_name_without_a_close_id_still_suggests_an_hf_id() {
        let loaded = loaded_snapshot();
        let message = resolve_catalog_name(&loaded, "not-a-model", None)
            .expect_err("unknown model must fail")
            .to_string();
        assert!(message.contains("not-a-model"), "message: {message}");
        assert!(message.contains("no close ids"), "message: {message}");
        assert!(message.contains("`org/model`"), "message: {message}");
    }

    #[test]
    fn catalog_file_is_pinned_only_when_a_quant_is_requested() {
        let loaded = loaded_snapshot();
        let model = loaded
            .catalog
            .find("qwen3-4b-instruct-2507")
            .expect("id")
            .clone();

        let no_quant = resolve_catalog_model(&model, None).expect("facts only");
        assert_eq!(no_quant.weight_bytes, None);
        assert!(no_quant.source.is_none());

        let file = model.file_for(Quant::Q4_K_M).expect("q4_k_m file");
        let pinned = resolve_catalog_model(&model, Some(Quant::Q4_K_M)).expect("pinned file");
        assert_eq!(pinned.weight_bytes, Some(file.size));
        assert_eq!(
            pinned.model_path.as_deref(),
            Some(format!("~/.attune/models/{}/{}", file.repo, file.file).as_str())
        );
        assert_eq!(pinned.source.expect("source").sha256, file.sha256);

        let err = resolve_catalog_model(&model, Some(Quant::Fp16))
            .expect_err("missing fp16 file")
            .to_string();
        assert!(err.contains("q4_k_m"), "{err}");
    }
}
