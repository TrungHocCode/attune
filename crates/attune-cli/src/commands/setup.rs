//! `attune setup`: install the catalog's pinned llama.cpp builds (M6/M7).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use attune_backend::{choose_variants, GpuRequest};
use attune_estimate::model::registry::{load_catalog, LlamaAsset, LlamaPin};
use attune_fetch::{
    download, install_archives, installed_variant_binary, variant_dir, Download, HostPolicy,
};
use attune_serve::DownloadConsent;

/// One archive of a variant's download.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetDownload {
    pub asset: LlamaAsset,
    /// Where the archive is downloaded to.
    pub archive: PathBuf,
}

/// What `setup` will do for one build variant on this machine.
#[derive(Debug, Clone, PartialEq)]
pub enum VariantInstallPlan {
    AlreadyInstalled {
        variant: String,
        server: PathBuf,
    },
    Download {
        variant: String,
        /// Every archive of this build (a CUDA build also ships a cudart
        /// companion), in catalog order.
        downloads: Vec<AssetDownload>,
        /// Where the archives are unpacked.
        dest: PathBuf,
    },
}

/// Where `asset` is downloaded to under `<home>/downloads`.
fn archive_path(home: &Path, asset: &LlamaAsset) -> Result<PathBuf> {
    let file_name = asset
        .url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow!("llama.cpp asset URL has no file name: {}", asset.url))?;
    Ok(home.join("downloads").join(file_name))
}

/// Plan one build `variant` for `os`/`arch`, or report the existing install.
pub fn plan_variant(
    home: &Path,
    pin: &LlamaPin,
    os: &str,
    arch: &str,
    variant: &str,
) -> Result<VariantInstallPlan> {
    if let Some(server) = installed_variant_binary(home, &pin.tag, variant, "llama-server") {
        return Ok(VariantInstallPlan::AlreadyInstalled {
            variant: variant.to_string(),
            server,
        });
    }
    let assets = pin.assets_for(os, arch, variant);
    if assets.is_empty() {
        let have: Vec<String> = pin
            .assets
            .iter()
            .map(|a| format!("{}/{}", a.os, a.arch))
            .collect();
        return Err(anyhow!(
            "no llama.cpp {} {variant} build for {os}/{arch} (catalog has {}); install llama.cpp yourself and pass --llama-server",
            pin.tag,
            have.join(", ")
        ));
    }
    let downloads = assets
        .into_iter()
        .map(|asset| {
            Ok(AssetDownload {
                asset: asset.clone(),
                archive: archive_path(home, asset)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(VariantInstallPlan::Download {
        variant: variant.to_string(),
        downloads,
        dest: variant_dir(home, &pin.tag, variant),
    })
}

/// Ask once for a whole download plan (`--yes` / non-TTY rules as before).
fn confirm(yes: bool, total: u64, what: &str) -> Result<()> {
    let allowed = match super::download::consent_for_stdin(yes) {
        DownloadConsent::Yes => true,
        DownloadConsent::Ask(ask) => ask(total, what),
        DownloadConsent::No => return Err(anyhow!("pass --yes to download {what}")),
    };
    if !allowed {
        return Err(anyhow!("download declined"));
    }
    Ok(())
}

/// Download every archive and unpack them into `dest`.
fn fetch_all(downloads: &[AssetDownload], dest: &Path) -> Result<()> {
    let mut archives = Vec::with_capacity(downloads.len());
    for item in downloads {
        let request = Download {
            url: item.asset.url.clone(),
            dest: item.archive.clone(),
            size: item.asset.size,
            sha256: item.asset.sha256.clone(),
        };
        download(
            &request,
            &HostPolicy::production(),
            &mut super::download::progress,
        )
        .map_err(|err| anyhow!("{err}"))?;
        archives.push(item.archive.clone());
    }
    install_archives(&archives, dest).map_err(|err| anyhow!("{err}"))?;
    for archive in &archives {
        let _ = std::fs::remove_file(archive);
    }
    Ok(())
}

/// `attune setup [--yes] [--refresh-catalog] [--gpu BUILD]`.
pub fn run(yes: bool, refresh_catalog: bool, gpu: GpuRequest) -> Result<()> {
    install_selected(yes, refresh_catalog, gpu).map(|_| ())
}

/// The `--gpu` plan: keep the CPU build, add a GPU build when chosen.
fn install_selected(yes: bool, refresh_catalog: bool, gpu: GpuRequest) -> Result<()> {
    let loaded = load_catalog(refresh_catalog);
    for warning in &loaded.warnings {
        eprintln!("warning: {warning}");
    }
    let home = attune_fetch::home_dir().map_err(|err| anyhow!("{err}"))?;
    let pin = &loaded.catalog.llama_cpp;
    let gpus = attune_hal::gpu::detect().devices;
    let selection = choose_variants(
        gpu,
        &gpus,
        pin,
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
    .map_err(|err| anyhow!("{err}"))?;
    if let Some(note) = &selection.note {
        println!("{note}");
    }
    let plans = selection
        .variants
        .iter()
        .map(|variant| {
            plan_variant(
                &home,
                pin,
                std::env::consts::OS,
                std::env::consts::ARCH,
                variant,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let mut pending = Vec::new();
    for plan in &plans {
        match plan {
            VariantInstallPlan::AlreadyInstalled { variant, server } => {
                println!(
                    "llama.cpp {} ({variant}) already installed: {}",
                    pin.tag,
                    server.display()
                );
            }
            VariantInstallPlan::Download { .. } => pending.push(plan),
        }
    }
    if pending.is_empty() {
        return Ok(());
    }

    let total: u64 = pending
        .iter()
        .map(|plan| match plan {
            VariantInstallPlan::Download { downloads, .. } => {
                downloads.iter().map(|item| item.asset.size).sum()
            }
            VariantInstallPlan::AlreadyInstalled { .. } => 0,
        })
        .sum();
    let each: Vec<String> = pending
        .iter()
        .filter_map(|plan| match plan {
            VariantInstallPlan::Download {
                variant, downloads, ..
            } => {
                let size: u64 = downloads.iter().map(|item| item.asset.size).sum();
                Some(format!("{variant} ({:.0} MB)", size as f64 / 1e6))
            }
            VariantInstallPlan::AlreadyInstalled { .. } => None,
        })
        .collect();
    let what = format!(
        "llama.cpp {} builds: {} - {:.0} MB total",
        pin.tag,
        each.join(", "),
        total as f64 / 1e6
    );
    confirm(yes, total, &what)?;

    for plan in pending {
        if let VariantInstallPlan::Download {
            variant,
            downloads,
            dest,
        } = plan
        {
            fetch_all(downloads, dest)?;
            let server = installed_variant_binary(&home, &pin.tag, variant, "llama-server")
                .ok_or_else(|| {
                    anyhow!(
                        "installed {} but found no llama-server in it",
                        dest.display()
                    )
                })?;
            println!(
                "llama.cpp {} ({variant}) installed: {}",
                pin.tag,
                server.display()
            );
        }
    }
    Ok(())
}

/// Install the CPU pin (or report the existing one) and return the
/// `llama-server` path. Shares its output and consent rules with `setup`;
/// `attune run` keeps using this CPU-only path.
pub fn install(yes: bool, refresh_catalog: bool) -> Result<PathBuf> {
    let loaded = load_catalog(refresh_catalog);
    for warning in &loaded.warnings {
        eprintln!("warning: {warning}");
    }
    let home = attune_fetch::home_dir().map_err(|err| anyhow!("{err}"))?;
    let pin = &loaded.catalog.llama_cpp;
    let plan = plan_variant(
        &home,
        pin,
        std::env::consts::OS,
        std::env::consts::ARCH,
        "cpu",
    )?;
    let (downloads, dest) = match plan {
        VariantInstallPlan::AlreadyInstalled { server, .. } => {
            println!(
                "llama.cpp {} already installed: {}",
                pin.tag,
                server.display()
            );
            return Ok(server);
        }
        VariantInstallPlan::Download {
            downloads, dest, ..
        } => (downloads, dest),
    };
    let asset = downloads
        .first()
        .ok_or_else(|| anyhow!("no CPU asset for llama.cpp {}", pin.tag))?;
    let what = format!(
        "llama.cpp {} ({:.0} MB)",
        pin.tag,
        asset.asset.size as f64 / 1e6
    );
    confirm(yes, asset.asset.size, &what)?;
    fetch_all(&downloads, &dest)?;
    let server =
        installed_variant_binary(&home, &pin.tag, "cpu", "llama-server").ok_or_else(|| {
            anyhow!(
                "installed {} but found no llama-server in it",
                dest.display()
            )
        })?;
    println!("Installed llama.cpp {}: {}", pin.tag, server.display());
    Ok(server)
}

/// Make sure a usable `llama-server` exists, installing the pinned build with
/// the same consent rules as `setup` when it does not. An explicit
/// `--llama-server` path is returned untouched.
pub fn ensure_llama_server(explicit: Option<&Path>, yes: bool) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = attune_serve::resolve_llama_server(None) {
        return Ok(path);
    }
    install(yes, false)
}

/// The explicit `--llama-cli` path, else the `attune setup` install when
/// the env var and PATH have none (bench resolves those itself).
pub fn llama_cli_or_installed(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    if attune_bench::llama::resolve_llama_cli(None).is_ok() {
        return None;
    }
    let home = attune_fetch::home_dir().ok()?;
    attune_fetch::find_installed(&home, "llama-cli")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin() -> LlamaPin {
        LlamaPin {
            tag: "b11804".to_string(),
            assets: vec![LlamaAsset {
                os: "windows".to_string(),
                arch: "x86_64".to_string(),
                variant: "cpu".to_string(),
                url: "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-b11804-bin-win-cpu-x64.zip".to_string(),
                size: 24_067_910,
                sha256: "3".repeat(64),
                min_driver: None,
            }],
        }
    }

    fn cuda_pin() -> LlamaPin {
        LlamaPin {
            tag: "b11804".to_string(),
            assets: vec![
                LlamaAsset {
                    os: "windows".to_string(),
                    arch: "x86_64".to_string(),
                    variant: "cuda12".to_string(),
                    url: "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-b11804-bin-win-cuda-12.4-x64.zip".to_string(),
                    size: 2,
                    sha256: "3".repeat(64),
                    min_driver: Some("551.61".to_string()),
                },
                LlamaAsset {
                    os: "windows".to_string(),
                    arch: "x86_64".to_string(),
                    variant: "cuda12".to_string(),
                    url: "https://github.com/ggml-org/llama.cpp/releases/download/b11804/cudart-llama-bin-win-cuda-12.4-x64.zip".to_string(),
                    size: 3,
                    sha256: "4".repeat(64),
                    min_driver: Some("551.61".to_string()),
                },
            ],
        }
    }

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-setup-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn setup_skips_when_already_installed() {
        let home = temp_home("installed");
        let bin = variant_dir(&home, "b11804", "cpu");
        std::fs::create_dir_all(&bin).expect("mkdir");
        std::fs::write(bin.join("llama-server.exe"), b"MZ").expect("write");
        match plan_variant(&home, &pin(), "windows", "x86_64", "cpu").expect("plan") {
            VariantInstallPlan::AlreadyInstalled { variant, server } => {
                assert_eq!(variant, "cpu");
                assert_eq!(server, bin.join("llama-server.exe"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn setup_plans_the_matching_asset_download() {
        let home = temp_home("fresh");
        match plan_variant(&home, &pin(), "windows", "x86_64", "cpu").expect("plan") {
            VariantInstallPlan::Download {
                variant,
                downloads,
                dest,
            } => {
                assert_eq!(variant, "cpu");
                assert_eq!(downloads.len(), 1);
                assert_eq!(downloads[0].asset.size, 24_067_910);
                assert_eq!(
                    downloads[0].archive,
                    home.join("downloads")
                        .join("llama-b11804-bin-win-cpu-x64.zip")
                );
                assert_eq!(dest, variant_dir(&home, "b11804", "cpu"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cuda_plan_downloads_both_archives_into_one_dir() {
        let home = temp_home("cuda");
        match plan_variant(&home, &cuda_pin(), "windows", "x86_64", "cuda12").expect("plan") {
            VariantInstallPlan::Download {
                variant,
                downloads,
                dest,
            } => {
                assert_eq!(variant, "cuda12");
                assert_eq!(downloads.len(), 2);
                assert_eq!(
                    downloads[0].archive,
                    home.join("downloads")
                        .join("llama-b11804-bin-win-cuda-12.4-x64.zip")
                );
                assert_eq!(
                    downloads[1].archive,
                    home.join("downloads")
                        .join("cudart-llama-bin-win-cuda-12.4-x64.zip")
                );
                assert_eq!(dest, variant_dir(&home, "b11804", "cuda12"));
                assert_eq!(dest, home.join("llama.cpp").join("b11804-cuda12"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unsupported_platform_names_what_exists() {
        let home = temp_home("mac");
        let err = plan_variant(&home, &pin(), "macos", "aarch64", "cpu")
            .expect_err("no asset")
            .to_string();
        assert!(err.contains("windows/x86_64"), "{err}");
        assert!(err.contains("--llama-server"), "{err}");
    }
}
