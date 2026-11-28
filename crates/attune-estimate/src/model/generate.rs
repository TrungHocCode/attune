//! Pure core of the daily catalog generator (`attune catalog build`, M6).
//!
//! The CLI does the HTTP calls (HF model list, file tree, GGUF header range,
//! base repo info); everything that decides *what* enters the catalog lives
//! here so it is testable without the network.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::model::registry::{
    parse_catalog, Catalog, CatalogArch, CatalogFile, CatalogModel, LlamaPin, SCHEMA,
};
use crate::{EstimateError, Quant};

/// `catalog/publishers.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GenConfig {
    /// The llama.cpp pin, copied verbatim (bumping it is a human edit).
    pub llama_cpp: LlamaPin,
    /// Larger models are left out (they do not run on a typical machine).
    pub max_params: u64,
    /// Quants kept per model.
    pub quants: Vec<Quant>,
    /// GGUF repo owners to list, in preference order per quant.
    pub publishers: Vec<String>,
    /// Orgs whose *base* models may enter (the original model makers).
    pub base_orgs: Vec<String>,
    /// Most-downloaded repos listed per publisher.
    pub per_publisher_limit: usize,
}

/// One entry of the HF `/api/models` list.
#[derive(Debug, Clone, PartialEq)]
pub struct ListedRepo {
    pub id: String,
    pub sha: String,
    pub gated: bool,
    pub pipeline_tag: Option<String>,
    pub base_model: Option<String>,
    pub files: Vec<String>,
}

/// One chosen file before its size and hash are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedFile {
    pub quant: Quant,
    pub repo: String,
    pub revision: String,
    pub file: String,
}

/// One model with its preferred file per quant.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedModel {
    pub id: String,
    pub display: String,
    pub publisher: String,
    pub base_model: String,
    pub files: Vec<SelectedFile>,
}

/// Shape facts read from the first file's GGUF header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeaderFacts {
    pub params: u64,
    pub arch: CatalogArch,
    pub ctx: u32,
}

/// What the base (original) repo says.
#[derive(Debug, Clone, PartialEq)]
pub struct BaseInfo {
    pub created_at: Option<String>,
    pub gated: bool,
    pub license: String,
}

/// A selected model with everything the network told us.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    pub selected: SelectedModel,
    /// Each file with `(size, sha256)` from the repo tree, when found.
    pub files: Vec<(SelectedFile, Option<(u64, String)>)>,
    pub facts: Option<HeaderFacts>,
    pub base: BaseInfo,
}

/// Name tokens that mark a model llama-server cannot chat with as-is.
const EXCLUDED_TOKENS: &[&str] = &[
    "base",
    "vl",
    "vision",
    "video",
    "embed",
    "embedding",
    "reranker",
    "audio",
    "speech",
    "omni",
    "mmproj",
    "guard",
    "guardian",
];

/// Parse an HF `/api/models?...&expand[]=...` list.
pub fn parse_model_list(json: &str) -> Result<Vec<ListedRepo>, EstimateError> {
    let value: Value =
        serde_json::from_str(json).map_err(|err| EstimateError::InvalidCatalog(err.to_string()))?;
    let items = value
        .as_array()
        .ok_or_else(|| EstimateError::InvalidCatalog("model list is not an array".to_string()))?;
    Ok(items
        .iter()
        .filter_map(|item| {
            let card = item.get("cardData");
            let base_model = card
                .and_then(|c| c.get("base_model"))
                .and_then(|b| match b {
                    Value::String(s) => Some(s.clone()),
                    Value::Array(list) => list.first()?.as_str().map(str::to_string),
                    _ => None,
                });
            Some(ListedRepo {
                id: item.get("id")?.as_str()?.to_string(),
                sha: item
                    .get("sha")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                // `false`, or a mode string ("auto", "manual") when gated.
                gated: item.get("gated").is_some_and(|g| g != &Value::Bool(false)),
                pipeline_tag: card
                    .and_then(|c| c.get("pipeline_tag"))
                    .or_else(|| item.get("pipeline_tag"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                base_model,
                files: item
                    .get("siblings")
                    .and_then(Value::as_array)
                    .map(|s| {
                        s.iter()
                            .filter_map(|f| f.get("rfilename")?.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        })
        .collect())
}

/// LFS files of an HF `/api/models/<repo>/tree/<rev>` listing:
/// path -> (size, sha256).
pub fn parse_tree(json: &str) -> Result<BTreeMap<String, (u64, String)>, EstimateError> {
    let value: Value =
        serde_json::from_str(json).map_err(|err| EstimateError::InvalidCatalog(err.to_string()))?;
    let items = value
        .as_array()
        .ok_or_else(|| EstimateError::InvalidCatalog("tree is not an array".to_string()))?;
    Ok(items
        .iter()
        .filter_map(|item| {
            let lfs = item.get("lfs")?;
            Some((
                item.get("path")?.as_str()?.to_string(),
                (
                    lfs.get("size")?.as_u64()?,
                    lfs.get("oid")?.as_str()?.to_string(),
                ),
            ))
        })
        .collect())
}

/// Date, gate and license of a base repo (`/api/models/<base>`).
pub fn parse_base_info(json: &str) -> Result<BaseInfo, EstimateError> {
    let value: Value =
        serde_json::from_str(json).map_err(|err| EstimateError::InvalidCatalog(err.to_string()))?;
    let license = value
        .get("cardData")
        .and_then(|c| c.get("license"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            value
                .get("tags")?
                .as_array()?
                .iter()
                .find_map(|tag| tag.as_str()?.strip_prefix("license:").map(str::to_string))
        })
        .unwrap_or_else(|| "unknown".to_string());
    Ok(BaseInfo {
        created_at: value
            .get("createdAt")
            .and_then(Value::as_str)
            .map(str::to_string),
        gated: value.get("gated").is_some_and(|g| g != &Value::Bool(false)),
        license,
    })
}

/// The file in `files` for `quant` (single-file GGUF only).
fn file_for_quant(files: &[String], quant: Quant) -> Option<&String> {
    let tail = format!("{}.gguf", quant.as_str());
    files.iter().find(|name| {
        let lower = name.to_ascii_lowercase();
        !lower.contains("-of-")
            && lower.ends_with(&tail)
            && lower[..lower.len() - tail.len()].ends_with(['-', '_', '.'])
    })
}

/// Group repos by base model and pick each quant's file from the most
/// preferred publisher. Gated, foreign-org, non-text and base/vision models
/// are dropped.
pub fn select(config: &GenConfig, repos: &[ListedRepo]) -> Vec<SelectedModel> {
    // id -> (model, per-quant publisher rank)
    let mut models: BTreeMap<String, (SelectedModel, BTreeMap<String, usize>)> = BTreeMap::new();
    for repo in repos {
        let owner = repo.id.split('/').next().unwrap_or("");
        let Some(rank) = config.publishers.iter().position(|p| p == owner) else {
            continue;
        };
        let Some(base) = repo.base_model.as_deref() else {
            continue;
        };
        let Some((base_org, name)) = base.split_once('/') else {
            continue;
        };
        let text_only = repo
            .pipeline_tag
            .as_deref()
            .is_none_or(|tag| tag == "text-generation");
        let lower = name.to_ascii_lowercase();
        let excluded = lower
            .split(['-', '_', '.'])
            .any(|token| EXCLUDED_TOKENS.contains(&token) || token.contains("vlm"));
        if repo.gated || !text_only || excluded || !config.base_orgs.iter().any(|o| o == base_org) {
            continue;
        }
        let (model, ranks) = models.entry(lower.clone()).or_insert_with(|| {
            (
                SelectedModel {
                    id: lower.clone(),
                    display: name.replace(['-', '_'], " "),
                    publisher: base_org.to_string(),
                    base_model: base.to_string(),
                    files: Vec::new(),
                },
                BTreeMap::new(),
            )
        });
        for &quant in &config.quants {
            let Some(file) = file_for_quant(&repo.files, quant) else {
                continue;
            };
            let key = quant.as_str().to_string();
            if ranks.get(&key).is_some_and(|&best| best <= rank) {
                continue;
            }
            ranks.insert(key, rank);
            model.files.retain(|f| f.quant != quant);
            model.files.push(SelectedFile {
                quant,
                repo: repo.id.clone(),
                revision: repo.sha.clone(),
                file: file.clone(),
            });
        }
    }
    models
        .into_values()
        .map(|(mut model, _)| {
            model
                .files
                .sort_by_key(|f| config.quants.iter().position(|&q| q == f.quant));
            model
        })
        .filter(|model| !model.files.is_empty())
        .collect()
}

/// Build the catalog from resolved models; unusable ones are dropped.
pub fn assemble(config: &GenConfig, models: Vec<ResolvedModel>, generated_at: &str) -> Catalog {
    let mut out: Vec<CatalogModel> = models
        .into_iter()
        .filter_map(|model| {
            let facts = model.facts?;
            if facts.params == 0 || facts.params > config.max_params || model.base.gated {
                return None;
            }
            let files: Vec<CatalogFile> = model
                .files
                .into_iter()
                .filter_map(|(file, meta)| {
                    let (size, sha256) = meta?;
                    Some(CatalogFile {
                        quant: file.quant,
                        repo: file.repo,
                        file: file.file,
                        revision: file.revision,
                        size,
                        sha256,
                    })
                })
                .collect();
            (!files.is_empty()).then(|| CatalogModel {
                id: model.selected.id,
                display: model.selected.display,
                publisher: model.selected.publisher,
                license: model.base.license,
                released: model
                    .base
                    .created_at
                    .map(|date| date.chars().take(10).collect()),
                gated: false,
                params: facts.params,
                arch: facts.arch,
                ctx_default: facts.ctx,
                files,
            })
        })
        .collect();
    out.sort_by_key(|m| m.params);
    Catalog {
        schema: SCHEMA.to_string(),
        generated_at: generated_at.to_string(),
        llama_cpp: config.llama_cpp.clone(),
        models: out,
    }
}

/// Refuse to publish an empty or sharply shrunk catalog (a broken run must
/// not wipe yesterday's good one), and re-validate the schema.
pub fn guard(new: &Catalog, previous: Option<&Catalog>) -> Result<(), EstimateError> {
    if new.models.is_empty() {
        return Err(EstimateError::InvalidCatalog(
            "generated catalog has no models".to_string(),
        ));
    }
    if let Some(previous) = previous {
        if new.models.len() * 10 < previous.models.len() * 7 {
            return Err(EstimateError::InvalidCatalog(format!(
                "generated catalog shrank from {} to {} models (> 30%)",
                previous.models.len(),
                new.models.len()
            )));
        }
    }
    let text =
        serde_json::to_string(new).map_err(|err| EstimateError::InvalidCatalog(err.to_string()))?;
    parse_catalog(&text).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::registry::{parse_catalog, CatalogArch, LlamaPin};

    fn config() -> GenConfig {
        GenConfig {
            llama_cpp: LlamaPin {
                tag: "b11804".to_string(),
                assets: Vec::new(),
            },
            max_params: 70_000_000_000,
            quants: vec![Quant::Q4_K_M, Quant::Q8_0],
            publishers: vec!["Qwen".into(), "bartowski".into(), "unsloth".into()],
            base_orgs: vec!["Qwen".into(), "microsoft".into()],
            per_publisher_limit: 100,
        }
    }

    fn listed(id: &str, base: &str, files: &[&str]) -> ListedRepo {
        ListedRepo {
            id: id.to_string(),
            sha: format!("sha-{id}"),
            gated: false,
            pipeline_tag: Some("text-generation".to_string()),
            base_model: Some(base.to_string()),
            files: files.iter().map(|f| f.to_string()).collect(),
        }
    }

    #[test]
    fn parses_the_hf_model_list_shape() {
        let json = r#"[{"id":"bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
          "cardData":{"pipeline_tag":"text-generation","base_model":"Qwen/Qwen3-4B-Instruct-2507"},
          "gated":false,"sha":"abc",
          "siblings":[{"rfilename":".gitattributes"},{"rfilename":"Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"}]},
          {"id":"x/gated-GGUF","gated":"auto","sha":"def","siblings":[]}]"#;
        let repos = parse_model_list(json).expect("parse");
        assert_eq!(repos.len(), 2);
        assert_eq!(
            repos[0].base_model.as_deref(),
            Some("Qwen/Qwen3-4B-Instruct-2507")
        );
        assert_eq!(repos[0].files.len(), 2);
        assert!(repos[1].gated, "a string `gated` value means gated");
    }

    #[test]
    fn groups_repos_into_one_model_preferring_publisher_order() {
        let base = "Qwen/Qwen3-4B-Instruct-2507";
        let models = select(
            &config(),
            &[
                listed(
                    "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
                    base,
                    &[
                        "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                        "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf",
                    ],
                ),
                listed(
                    "Qwen/Qwen3-4B-Instruct-2507-GGUF",
                    base,
                    &["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"],
                ),
            ],
        );
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.id, "qwen3-4b-instruct-2507");
        assert_eq!(model.display, "Qwen3 4B Instruct 2507");
        assert_eq!(model.publisher, "Qwen");
        let q4 = model
            .files
            .iter()
            .find(|f| f.quant == Quant::Q4_K_M)
            .expect("q4");
        assert_eq!(q4.repo, "Qwen/Qwen3-4B-Instruct-2507-GGUF");
        let q8 = model
            .files
            .iter()
            .find(|f| f.quant == Quant::Q8_0)
            .expect("q8");
        assert_eq!(q8.repo, "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(
            q8.revision,
            "sha-bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
        );
    }

    #[test]
    fn skips_gated_foreign_vision_base_and_split_repos() {
        let mut gated = listed(
            "Qwen/Qwen3-8B-GGUF",
            "Qwen/Qwen3-8B",
            &["Qwen3-8B-Q4_K_M.gguf"],
        );
        gated.gated = true;
        let foreign = listed(
            "bartowski/Foo-GGUF",
            "SomeOrg/Foo-7B",
            &["Foo-7B-Q4_K_M.gguf"],
        );
        let mut vision = listed(
            "bartowski/Qwen_VL-GGUF",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            &["VL-Q4_K_M.gguf"],
        );
        vision.pipeline_tag = Some("image-text-to-text".to_string());
        let base_model = listed(
            "bartowski/Qwen_Qwen3-8B-Base-GGUF",
            "Qwen/Qwen3-8B-Base",
            &["Qwen3-8B-Base-Q4_K_M.gguf"],
        );
        let split = listed(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "Qwen/Qwen2.5-14B-Instruct",
            &["qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"],
        );
        let named = |id: &str, base: &str| listed(id, base, &["x-Q4_K_M.gguf"]);
        let models = select(
            &config(),
            &[
                gated,
                foreign,
                vision,
                base_model,
                split,
                named("Qwen/SmolVLM2-GGUF", "Qwen/SmolVLM2-2.2B-Instruct"),
                named("Qwen/speech-GGUF", "Qwen/granite-speech-4.1-2b"),
                named("Qwen/guardian-GGUF", "Qwen/granite-guardian-3.3-8b"),
                named("Qwen/omni-GGUF", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            ],
        );
        assert!(models.is_empty(), "{models:?}");
    }

    fn resolved(params: u64, gated_base: bool, created: Option<&str>) -> ResolvedModel {
        let selected = select(
            &config(),
            &[listed(
                "Qwen/Qwen3-8B-GGUF",
                "Qwen/Qwen3-8B",
                &["Qwen3-8B-Q4_K_M.gguf"],
            )],
        )
        .remove(0);
        ResolvedModel {
            files: selected
                .files
                .iter()
                .map(|f| (f.clone(), Some((5_000_000_000, "d".repeat(64)))))
                .collect(),
            selected,
            facts: Some(HeaderFacts {
                params,
                arch: CatalogArch {
                    n_layer: 36,
                    n_kv_heads: 8,
                    head_dim: 128,
                },
                ctx: 40_960,
            }),
            base: BaseInfo {
                created_at: created.map(str::to_string),
                gated: gated_base,
                license: "apache-2.0".to_string(),
            },
        }
    }

    #[test]
    fn release_date_comes_from_the_base_repo() {
        let catalog = assemble(
            &config(),
            vec![resolved(8_190_735_360, false, Some("2025-04-27"))],
            "2026-11-27T03:00:00Z",
        );
        assert_eq!(catalog.models.len(), 1);
        let model = &catalog.models[0];
        assert_eq!(model.released.as_deref(), Some("2025-04-27"));
        assert_eq!(model.files[0].size, 5_000_000_000);
        assert_eq!(model.license, "apache-2.0");
        let none = assemble(&config(), vec![resolved(8_190_735_360, false, None)], "x");
        assert_eq!(none.models[0].released, None);
    }

    #[test]
    fn assemble_drops_too_large_gated_base_and_unreadable_headers() {
        let mut no_header = resolved(8_000_000_000, false, None);
        no_header.facts = None;
        let catalog = assemble(
            &config(),
            vec![
                resolved(72_000_000_000, false, None),
                resolved(8_000_000_000, true, None),
                no_header,
            ],
            "x",
        );
        assert!(catalog.models.is_empty());
    }

    #[test]
    fn guard_refuses_empty_or_shrunk_catalog() {
        let one = assemble(&config(), vec![resolved(8_000_000_000, false, None)], "x");
        let empty = assemble(&config(), Vec::new(), "x");
        assert!(guard(&empty, None).is_err());
        assert!(guard(&one, None).is_ok());
        let mut big = one.clone();
        for i in 0..3 {
            let mut m = big.models[0].clone();
            m.id = format!("m{i}");
            big.models.push(m);
        }
        // 1 of 4 models left: shrank by 75%.
        assert!(guard(&one, Some(&big)).is_err());
        assert!(guard(&big, Some(&one)).is_ok());
    }

    #[test]
    fn llama_pin_variants_and_min_driver_survive_the_round_trip() {
        let sha = "d".repeat(64);
        let text = format!(
            r#"{{
              "llama_cpp": {{ "tag": "b11804", "assets": [
                {{ "os": "windows", "arch": "x86_64", "variant": "cuda12",
                  "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-win-cuda.zip",
                  "size": 10, "sha256": "{sha}", "min_driver": "551.61" }},
                {{ "os": "linux", "arch": "x86_64", "variant": "vulkan",
                  "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-linux-vulkan.tar.gz",
                  "size": 20, "sha256": "{sha}" }}
              ] }},
              "max_params": 1, "quants": ["q4_k_m"],
              "publishers": ["Qwen"], "base_orgs": ["Qwen"], "per_publisher_limit": 1
            }}"#
        );
        let config: GenConfig = serde_json::from_str(&text).expect("config");
        let catalog = assemble(&config, Vec::new(), "x");
        let round_trip = serde_json::to_string(&catalog).expect("serialize");
        let parsed = parse_catalog(&round_trip).expect("reparse");
        let assets = &parsed.llama_cpp.assets;
        assert_eq!(assets[0].variant, "cuda12");
        assert_eq!(assets[0].min_driver.as_deref(), Some("551.61"));
        assert_eq!(assets[1].variant, "vulkan");
        assert_eq!(assets[1].min_driver, None);
    }

    #[test]
    fn tree_maps_lfs_files_to_size_and_sha() {
        let json = r#"[{"type":"file","path":".gitattributes","size":1570},
          {"type":"file","path":"Qwen3-8B-Q4_K_M.gguf","size":5027783488,
           "lfs":{"oid":"d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785","size":5027783488}}]"#;
        let tree = parse_tree(json).expect("tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(
            tree.get("Qwen3-8B-Q4_K_M.gguf"),
            Some(&(
                5_027_783_488,
                "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785".to_string()
            ))
        );
    }

    #[test]
    fn base_info_reads_date_gate_and_license() {
        let open = r#"{"id":"Qwen/Qwen3-8B","createdAt":"2025-04-27T03:42:00.000Z","gated":false,
          "cardData":{"license":"apache-2.0"},"tags":["license:apache-2.0"]}"#;
        let info = parse_base_info(open).expect("info");
        assert_eq!(info.created_at.as_deref(), Some("2025-04-27T03:42:00.000Z"));
        assert!(!info.gated);
        assert_eq!(info.license, "apache-2.0");
        let gated = r#"{"id":"meta-llama/x","gated":"manual","tags":["license:llama3.1"]}"#;
        let info = parse_base_info(gated).expect("info");
        assert!(info.gated);
        assert_eq!(info.license, "llama3.1");
    }
}
