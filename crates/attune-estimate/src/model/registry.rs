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
        }
    }
    for asset in &catalog.llama_cpp.assets {
        if !is_sha256(&asset.sha256) {
            return bad(format!(
                "llama.cpp asset `{}` has an invalid sha256",
                asset.url
            ));
        }
    }
    Ok(catalog)
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
