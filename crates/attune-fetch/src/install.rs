//! Unpack and find llama.cpp installs under `<home>/llama.cpp/<tag>/`.

use std::fs::File;
use std::path::{Component, Path, PathBuf};

use crate::FetchError;

/// Where build `tag` is installed.
pub fn install_dir(home: &Path, tag: &str) -> PathBuf {
    home.join("llama.cpp").join(tag)
}

/// `name` (or `name.exe`) anywhere inside build `tag`'s install, shallowest
/// first (Windows zips are flat, Linux tarballs nest under `build/bin/`).
pub fn installed_binary(home: &Path, tag: &str, name: &str) -> Option<PathBuf> {
    let exe = format!("{name}.exe");
    let mut queue = vec![install_dir(home, tag)];
    while !queue.is_empty() {
        let mut next = Vec::new();
        for dir in queue {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    next.push(path);
                } else if path
                    .file_name()
                    .is_some_and(|f| f == name || f == exe.as_str())
                {
                    return Some(path);
                }
            }
        }
        queue = next;
    }
    None
}

/// `name` from the newest installed build (`b11804` beats `b900`), for
/// callers that do not know the catalog's pinned tag.
pub fn find_installed(home: &Path, name: &str) -> Option<PathBuf> {
    let root = home.join("llama.cpp");
    let mut tags: Vec<String> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|tag| !tag.ends_with(".tmp"))
        .collect();
    let build = |tag: &str| tag.trim_start_matches('b').parse::<u64>().unwrap_or(0);
    tags.sort_by_key(|tag| std::cmp::Reverse(build(tag)));
    tags.iter()
        .find_map(|tag| installed_binary(home, tag, name))
}

/// Unpack `archive` (`.zip` or `.tar.gz`) into `dest` atomically: into
/// `<dest>.tmp` first, renamed only when every entry unpacked. Entries that
/// would escape the directory (`..`, absolute paths) fail the install.
pub fn install_archive(archive: &Path, dest: &Path) -> Result<(), FetchError> {
    let tmp = PathBuf::from(format!("{}.tmp", dest.display()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|err| FetchError::io(&tmp, err))?;
    let name = archive.to_string_lossy().to_ascii_lowercase();
    let result = if name.ends_with(".zip") {
        unpack_zip(archive, &tmp)
    } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        unpack_tar_gz(archive, &tmp)
    } else {
        Err(archive_error(archive, "unknown archive type"))
    };
    if let Err(err) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(err);
    }
    let _ = std::fs::remove_dir_all(dest);
    std::fs::rename(&tmp, dest).map_err(|err| FetchError::io(dest, err))
}

fn archive_error(archive: &Path, detail: impl std::fmt::Display) -> FetchError {
    FetchError::Archive {
        path: archive.to_path_buf(),
        detail: detail.to_string(),
    }
}

/// A relative path made only of normal components.
fn is_contained(path: &Path) -> bool {
    path.components().all(|c| matches!(c, Component::Normal(_)))
        && path.components().next().is_some()
}

fn unpack_zip(archive: &Path, into: &Path) -> Result<(), FetchError> {
    let file = File::open(archive).map_err(|err| FetchError::io(archive, err))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|err| archive_error(archive, err))?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|err| archive_error(archive, err))?;
        let Some(relative) = entry.enclosed_name().filter(|p| is_contained(p)) else {
            return Err(archive_error(
                archive,
                format!("entry `{}` escapes the install dir", entry.name()),
            ));
        };
        let out = into.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|err| FetchError::io(&out, err))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|err| FetchError::io(parent, err))?;
        }
        let mut target = File::create(&out).map_err(|err| FetchError::io(&out, err))?;
        std::io::copy(&mut entry, &mut target).map_err(|err| FetchError::io(&out, err))?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode));
        }
    }
    Ok(())
}

fn unpack_tar_gz(archive: &Path, into: &Path) -> Result<(), FetchError> {
    let file = File::open(archive).map_err(|err| FetchError::io(archive, err))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let entries = tar.entries().map_err(|err| archive_error(archive, err))?;
    for entry in entries {
        let mut entry = entry.map_err(|err| archive_error(archive, err))?;
        let relative = entry
            .path()
            .map_err(|err| archive_error(archive, err))?
            .into_owned();
        if !is_contained(&relative) {
            return Err(archive_error(
                archive,
                format!("entry `{}` escapes the install dir", relative.display()),
            ));
        }
        // A symlink may only point inside the install (shared libs in the
        // Linux build are `libx.so -> libx.so.0`).
        if let Some(link) = entry
            .link_name()
            .map_err(|err| archive_error(archive, err))?
        {
            if link.is_absolute() || link.components().any(|c| c == Component::ParentDir) {
                return Err(archive_error(
                    archive,
                    format!("link `{}` escapes the install dir", relative.display()),
                ));
            }
        }
        let out = into.join(&relative);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|err| FetchError::io(parent, err))?;
        }
        entry
            .unpack(&out)
            .map_err(|err| archive_error(archive, err))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-install-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn zip_with(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create");
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, bytes) in entries {
            zip.start_file(*name, options).expect("entry");
            zip.write_all(bytes).expect("write");
        }
        zip.finish().expect("finish");
    }

    fn tar_gz_with(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create");
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for (name, bytes) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o755);
            // `set_path` refuses `..`; write the raw name to build a hostile archive.
            let raw = &mut header.as_old_mut().name;
            raw[..name.len()].copy_from_slice(name.as_bytes());
            header.set_cksum();
            tar.append(&header, *bytes).expect("append");
        }
        tar.into_inner().expect("tar").finish().expect("gz");
    }

    #[test]
    fn install_unpacks_zip_atomically() {
        let dir = temp("zip");
        let archive = dir.join("llama.zip");
        zip_with(
            &archive,
            &[("llama-server.exe", b"MZ"), ("ggml.dll", b"dll")],
        );
        let dest = install_dir(&dir, "b1");
        install_archive(&archive, &dest).expect("install");
        assert_eq!(
            installed_binary(&dir, "b1", "llama-server"),
            Some(dest.join("llama-server.exe"))
        );
        assert!(!dir.join("llama.cpp").join("b1.tmp").exists());
    }

    #[test]
    fn corrupt_archive_leaves_no_install_dir() {
        let dir = temp("corrupt");
        let archive = dir.join("bad.zip");
        std::fs::write(&archive, b"not a zip").expect("write");
        let dest = install_dir(&dir, "b1");
        assert!(install_archive(&archive, &dest).is_err());
        assert!(!dest.exists());
    }

    #[test]
    fn zip_slip_entries_are_rejected() {
        let dir = temp("zipslip");
        let archive = dir.join("evil.zip");
        zip_with(&archive, &[("../evil.txt", b"x")]);
        let dest = install_dir(&dir, "b1");
        assert!(install_archive(&archive, &dest).is_err());
        assert!(!dir.join("llama.cpp").join("evil.txt").exists());
        assert!(!dest.exists());
    }

    #[test]
    fn tar_gz_installs_nested_binary_and_rejects_slip() {
        let dir = temp("tar");
        let archive = dir.join("llama.tar.gz");
        tar_gz_with(&archive, &[("build/bin/llama-server", b"\x7fELF")]);
        let dest = install_dir(&dir, "b2");
        install_archive(&archive, &dest).expect("install");
        assert_eq!(
            installed_binary(&dir, "b2", "llama-server"),
            Some(dest.join("build").join("bin").join("llama-server"))
        );

        let evil = dir.join("evil.tar.gz");
        tar_gz_with(&evil, &[("../../evil", b"x")]);
        let dest = install_dir(&dir, "b3");
        assert!(install_archive(&evil, &dest).is_err());
        assert!(!dest.exists());
    }

    #[test]
    fn newest_installed_build_wins() {
        let dir = temp("newest");
        for tag in ["b900", "b11804"] {
            let bin = install_dir(&dir, tag);
            std::fs::create_dir_all(&bin).expect("mkdir");
            std::fs::write(bin.join("llama-server"), b"x").expect("write");
        }
        assert_eq!(
            find_installed(&dir, "llama-server"),
            Some(install_dir(&dir, "b11804").join("llama-server"))
        );
    }
}
