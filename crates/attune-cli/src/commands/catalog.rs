//! `attune catalog build` (hidden): regenerate the model catalog from the
//! Hugging Face API. Run daily by `.github/workflows/catalog.yml`; the
//! selection rules live in `attune_estimate::model::generate`.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use attune_estimate::model::generate::{
    assemble, guard, parse_base_info, parse_model_list, parse_tree, select, BaseInfo, GenConfig,
    HeaderFacts, ResolvedModel,
};
use attune_estimate::model::gguf::probe_gguf_bytes;
use attune_estimate::model::registry::{parse_catalog, Catalog, CatalogArch};
use attune_fetch::{fetch_range, fetch_text, HostPolicy};

const HF: &str = "https://huggingface.co";
const TIMEOUT: Duration = Duration::from_secs(60);
/// Enough of a GGUF file to cover its metadata (tokenizer arrays included).
const HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// `YYYY-MM-DDTHH:MM:SSZ` for unix seconds (no date crate needed).
fn iso8601(unix: u64) -> String {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    // Civil-from-days (Howard Hinnant), proleptic Gregorian.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

/// Header facts already known for this (repo, revision) in the previous
/// catalog, so unchanged files are not range-downloaded again.
fn known_facts(previous: Option<&Catalog>, repo: &str, revision: &str) -> Option<HeaderFacts> {
    previous?.models.iter().find_map(|model| {
        model
            .files
            .iter()
            .any(|f| f.repo == repo && f.revision == revision)
            .then_some(HeaderFacts {
                params: model.params,
                arch: model.arch,
                ctx: model.ctx_default,
            })
    })
}

fn header_facts(url: &str, policy: &HostPolicy) -> Result<HeaderFacts> {
    let bytes = fetch_range(url, policy, HEADER_BYTES, TIMEOUT)?;
    let (params, arch, ctx) = probe_gguf_bytes(&bytes).map_err(|err| anyhow!("{err}"))?;
    let arch = arch.ok_or_else(|| anyhow!("no KV shape in the header"))?;
    Ok(HeaderFacts {
        params,
        arch: CatalogArch {
            n_layer: arch.n_layer,
            n_kv_heads: arch.n_kv_heads,
            head_dim: arch.head_dim,
        },
        ctx: ctx.unwrap_or(4096),
    })
}

/// Regenerate the catalog into `out`; refuses an empty or shrunk result.
pub fn build(publishers: &Path, previous: Option<&Path>, out: &Path) -> Result<()> {
    let config: GenConfig = serde_json::from_str(
        &std::fs::read_to_string(publishers)
            .with_context(|| format!("failed to read {}", publishers.display()))?,
    )
    .with_context(|| format!("invalid {}", publishers.display()))?;
    let previous: Option<Catalog> = previous
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| parse_catalog(&text).ok());
    let policy = HostPolicy::production();

    let mut listed = Vec::new();
    for publisher in &config.publishers {
        let url = format!(
            "{HF}/api/models?author={publisher}&filter=gguf&sort=downloads&direction=-1&limit={}&expand%5B%5D=gated&expand%5B%5D=sha&expand%5B%5D=cardData&expand%5B%5D=siblings&expand%5B%5D=pipeline_tag",
            config.per_publisher_limit
        );
        let text = fetch_text(&url, &policy, TIMEOUT)?;
        listed.extend(parse_model_list(&text).map_err(|err| anyhow!("{publisher}: {err}"))?);
    }
    let selected = select(&config, &listed);
    eprintln!(
        "[catalog] {} repos listed, {} models selected",
        listed.len(),
        selected.len()
    );

    let mut trees: BTreeMap<(String, String), BTreeMap<String, (u64, String)>> = BTreeMap::new();
    let mut resolved = Vec::new();
    for model in selected {
        let mut files = Vec::new();
        for file in &model.files {
            let key = (file.repo.clone(), file.revision.clone());
            if !trees.contains_key(&key) {
                let url = format!("{HF}/api/models/{}/tree/{}", file.repo, file.revision);
                let tree = fetch_text(&url, &policy, TIMEOUT)
                    .map_err(|err| anyhow!("{err}"))
                    .and_then(|text| parse_tree(&text).map_err(|err| anyhow!("{err}")))
                    .unwrap_or_else(|err| {
                        eprintln!("[catalog] warning: {}: {err}", file.repo);
                        BTreeMap::new()
                    });
                trees.insert(key.clone(), tree);
            }
            files.push((file.clone(), trees[&key].get(&file.file).cloned()));
        }
        let first = &model.files[0];
        let facts = match known_facts(previous.as_ref(), &first.repo, &first.revision) {
            Some(facts) => Some(facts),
            None => {
                let url = format!(
                    "{HF}/{}/resolve/{}/{}",
                    first.repo, first.revision, first.file
                );
                header_facts(&url, &policy)
                    .map_err(|err| eprintln!("[catalog] warning: {}: {err}", model.id))
                    .ok()
            }
        };
        let base = fetch_text(
            &format!("{HF}/api/models/{}", model.base_model),
            &policy,
            TIMEOUT,
        )
        .map_err(|err| anyhow!("{err}"))
        .and_then(|text| parse_base_info(&text).map_err(|err| anyhow!("{err}")))
        .unwrap_or_else(|err| {
            eprintln!("[catalog] warning: {}: {err}", model.base_model);
            // Unknown base: treat as gated so it is left out.
            BaseInfo {
                created_at: None,
                gated: true,
                license: "unknown".to_string(),
            }
        });
        resolved.push(ResolvedModel {
            selected: model,
            files,
            facts,
            base,
        });
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let catalog = assemble(&config, resolved, &iso8601(now));
    guard(&catalog, previous.as_ref()).map_err(|err| anyhow!("{err}"))?;
    std::fs::write(out, serde_json::to_string_pretty(&catalog)? + "\n")
        .with_context(|| format!("failed to write {}", out.display()))?;
    eprintln!(
        "[catalog] wrote {} models to {}",
        catalog.models.len(),
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_formats_unix_seconds_in_utc() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(1_795_522_719), "2026-11-24T12:18:39Z");
        assert_eq!(iso8601(951_782_400), "2000-02-29T00:00:00Z");
    }
}
