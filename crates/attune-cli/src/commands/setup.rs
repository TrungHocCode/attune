//! `attune setup`: install the catalog's pinned llama.cpp build (M6).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use attune_estimate::model::registry::{load_catalog, LlamaAsset, LlamaPin};
use attune_fetch::{
    download, install_archive, install_dir, installed_binary, Download, HostPolicy,
};
use attune_serve::DownloadConsent;

/// What `setup` will do on this machine.
#[derive(Debug, Clone, PartialEq)]
pub enum SetupPlan {
    AlreadyInstalled(PathBuf),
    Download {
        asset: LlamaAsset,
        /// Where the archive is downloaded to.
        archive: PathBuf,
        /// Where it is unpacked.
        dest: PathBuf,
    },
}

/// Pick the CPU asset for `os`/`arch`, or report the existing install.
pub fn plan_setup(home: &Path, pin: &LlamaPin, os: &str, arch: &str) -> Result<SetupPlan> {
    if let Some(path) = installed_binary(home, &pin.tag, "llama-server") {
        return Ok(SetupPlan::AlreadyInstalled(path));
    }
    let asset = pin
        .assets
        .iter()
        .find(|a| a.os == os && a.arch == arch && a.variant == "cpu")
        .ok_or_else(|| {
            let have: Vec<String> = pin
                .assets
                .iter()
                .map(|a| format!("{}/{}", a.os, a.arch))
                .collect();
            anyhow!(
                "no llama.cpp {} build for {os}/{arch} (catalog has {}); install llama.cpp yourself and pass --llama-server",
                pin.tag,
                have.join(", ")
            )
        })?;
    let file_name = asset
        .url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow!("llama.cpp asset URL has no file name: {}", asset.url))?;
    Ok(SetupPlan::Download {
        asset: asset.clone(),
        archive: home.join("downloads").join(file_name),
        dest: install_dir(home, &pin.tag),
    })
}

/// `attune setup [--yes] [--refresh-catalog]`.
pub fn run(yes: bool, refresh_catalog: bool) -> Result<()> {
    let loaded = load_catalog(refresh_catalog);
    for warning in &loaded.warnings {
        eprintln!("warning: {warning}");
    }
    let home = attune_fetch::home_dir().map_err(|err| anyhow!("{err}"))?;
    let pin = &loaded.catalog.llama_cpp;
    let (asset, archive, dest) =
        match plan_setup(&home, pin, std::env::consts::OS, std::env::consts::ARCH)? {
            SetupPlan::AlreadyInstalled(path) => {
                println!(
                    "llama.cpp {} already installed: {}",
                    pin.tag,
                    path.display()
                );
                return Ok(());
            }
            SetupPlan::Download {
                asset,
                archive,
                dest,
            } => (asset, archive, dest),
        };
    let what = format!("llama.cpp {} ({:.0} MB)", pin.tag, asset.size as f64 / 1e6);
    let allowed = match super::download::consent_for_stdin(yes) {
        DownloadConsent::Yes => true,
        DownloadConsent::Ask(ask) => ask(asset.size, &what),
        DownloadConsent::No => {
            return Err(anyhow!("pass --yes to download {what}"));
        }
    };
    if !allowed {
        return Err(anyhow!("download declined"));
    }
    let request = Download {
        url: asset.url.clone(),
        dest: archive.clone(),
        size: asset.size,
        sha256: asset.sha256.clone(),
    };
    download(
        &request,
        &HostPolicy::production(),
        &mut super::download::progress,
    )
    .map_err(|err| anyhow!("{err}"))?;
    install_archive(&archive, &dest).map_err(|err| anyhow!("{err}"))?;
    let _ = std::fs::remove_file(&archive);
    let server = installed_binary(&home, &pin.tag, "llama-server").ok_or_else(|| {
        anyhow!(
            "installed {} but found no llama-server in it",
            dest.display()
        )
    })?;
    println!("Installed llama.cpp {}: {}", pin.tag, server.display());
    Ok(())
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
            }],
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
        let bin = install_dir(&home, "b11804");
        std::fs::create_dir_all(&bin).expect("mkdir");
        std::fs::write(bin.join("llama-server.exe"), b"MZ").expect("write");
        match plan_setup(&home, &pin(), "windows", "x86_64").expect("plan") {
            SetupPlan::AlreadyInstalled(path) => assert_eq!(path, bin.join("llama-server.exe")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn setup_plans_the_matching_asset_download() {
        let home = temp_home("fresh");
        match plan_setup(&home, &pin(), "windows", "x86_64").expect("plan") {
            SetupPlan::Download {
                asset,
                archive,
                dest,
            } => {
                assert_eq!(asset.size, 24_067_910);
                assert_eq!(
                    archive,
                    home.join("downloads")
                        .join("llama-b11804-bin-win-cpu-x64.zip")
                );
                assert_eq!(dest, install_dir(&home, "b11804"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unsupported_platform_names_what_exists() {
        let home = temp_home("mac");
        let err = plan_setup(&home, &pin(), "macos", "aarch64")
            .expect_err("no asset")
            .to_string();
        assert!(err.contains("windows/x86_64"), "{err}");
        assert!(err.contains("--llama-server"), "{err}");
    }
}
