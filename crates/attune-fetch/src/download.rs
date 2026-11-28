use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::{FetchError, HostPolicy};

/// Redirect hops followed by hand (each one checked against the policy).
const MAX_REDIRECTS: usize = 5;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// A read that makes no progress for this long fails the download. The
/// resumable `.part` file is kept, so a retry continues where it stopped.
const STALL_TIMEOUT: Duration = Duration::from_secs(60);
const CHUNK: usize = 64 * 1024;

/// One pinned file: where it comes from, where it goes, and what it must be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Download {
    pub url: String,
    pub dest: PathBuf,
    pub size: u64,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: String,
}

/// Download `d` to `d.dest`: resumable through `<dest>.part`, verified by size
/// and SHA-256, renamed into place only when both match. `progress` receives
/// `(bytes_done, bytes_total)`.
pub fn download(
    d: &Download,
    policy: &HostPolicy,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), FetchError> {
    download_with(d, policy, &free_disk_bytes, progress)
}

/// [`download`] with an injectable free-space probe (tests).
pub fn download_with(
    d: &Download,
    policy: &HostPolicy,
    free: &dyn Fn(&Path) -> Option<u64>,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), FetchError> {
    download_with_stall(d, policy, free, STALL_TIMEOUT, progress)
}

/// [`download_with`] with an injectable stall timeout (tests).
fn download_with_stall(
    d: &Download,
    policy: &HostPolicy,
    free: &dyn Fn(&Path) -> Option<u64>,
    stall: Duration,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), FetchError> {
    policy.check(&d.url)?;
    let dir = d.dest.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir).map_err(|err| FetchError::io(dir, err))?;
    let part = part_path(&d.dest);

    let mut have = std::fs::metadata(&part).map_or(0, |m| m.len());
    if have > d.size {
        remove(&part)?;
        have = 0;
    }
    let need = (d.size - have) + d.size / 10;
    if let Some(free) = free(dir) {
        if free < need {
            return Err(FetchError::DiskSpace { need, free });
        }
    }

    let mut hasher = Sha256::new();
    if have > 0 {
        hash_file(&part, &mut hasher)?;
    }
    if have < d.size {
        let response = get(&d.url, policy, (have > 0).then_some(have), None, stall)?;
        let status = response.status().as_u16();
        let append = match status {
            206 => true,
            200 => false,
            other => {
                return Err(FetchError::Http {
                    url: d.url.clone(),
                    detail: format!("HTTP {other}"),
                })
            }
        };
        if !append && have > 0 {
            // The server ignored Range and sent the whole file: start over.
            hasher = Sha256::new();
            have = 0;
        }
        let mut out = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .map_err(|err| FetchError::io(&part, err))?;
        let mut reader = response.into_body().into_reader();
        let mut buf = vec![0_u8; CHUNK];
        loop {
            let n = reader.read(&mut buf).map_err(|err| FetchError::Http {
                url: d.url.clone(),
                detail: err.to_string(),
            })?;
            if n == 0 {
                break;
            }
            if have + n as u64 > d.size {
                drop(out);
                remove(&part)?;
                return Err(FetchError::SizeMismatch {
                    expected: d.size,
                    actual: have + n as u64,
                });
            }
            out.write_all(&buf[..n])
                .map_err(|err| FetchError::io(&part, err))?;
            hasher.update(&buf[..n]);
            have += n as u64;
            progress(have, d.size);
        }
        out.flush().map_err(|err| FetchError::io(&part, err))?;
    }

    if have != d.size {
        // Short body: keep the part so the next run resumes.
        return Err(FetchError::SizeMismatch {
            expected: d.size,
            actual: have,
        });
    }
    let actual = hex(&hasher.finalize());
    if !actual.eq_ignore_ascii_case(&d.sha256) {
        remove(&part)?;
        return Err(FetchError::ChecksumMismatch {
            expected: d.sha256.clone(),
            actual,
        });
    }
    progress(have, d.size);
    std::fs::rename(&part, &d.dest).map_err(|err| FetchError::io(&d.dest, err))
}

/// GET a small text document (catalog JSON), following checked redirects.
pub fn fetch_text(url: &str, policy: &HostPolicy, timeout: Duration) -> Result<String, FetchError> {
    let response = get(url, policy, None, Some(timeout), STALL_TIMEOUT)?;
    let status = response.status().as_u16();
    if status != 200 {
        return Err(FetchError::Http {
            url: url.to_string(),
            detail: format!("HTTP {status}"),
        });
    }
    response
        .into_body()
        .read_to_string()
        .map_err(|err| FetchError::Http {
            url: url.to_string(),
            detail: err.to_string(),
        })
}

/// The first `len` bytes of `url` (HTTP Range; a server that ignores Range
/// is cut off after `len` bytes). Used to read GGUF headers.
pub fn fetch_range(
    url: &str,
    policy: &HostPolicy,
    len: u64,
    timeout: Duration,
) -> Result<Vec<u8>, FetchError> {
    let response = get_range(url, policy, 0, Some(len), Some(timeout), STALL_TIMEOUT)?;
    let status = response.status().as_u16();
    if status != 200 && status != 206 {
        return Err(FetchError::Http {
            url: url.to_string(),
            detail: format!("HTTP {status}"),
        });
    }
    let mut out = Vec::new();
    response
        .into_body()
        .into_reader()
        .take(len)
        .read_to_end(&mut out)
        .map_err(|err| FetchError::Http {
            url: url.to_string(),
            detail: err.to_string(),
        })?;
    Ok(out)
}

/// Free bytes on the disk holding `path` (longest matching mount point).
pub fn free_disk_bytes(path: &Path) -> Option<u64> {
    let path = std::path::absolute(path).ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|disk| path.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
}

/// One request with manual redirects, so every hop meets the policy.
fn get(
    url: &str,
    policy: &HostPolicy,
    from: Option<u64>,
    timeout: Option<Duration>,
    stall: Duration,
) -> Result<ureq::http::Response<ureq::Body>, FetchError> {
    match from {
        Some(from) => get_range(url, policy, from, None, timeout, stall),
        None => get_inner(url, policy, None, timeout, stall),
    }
}

/// `Range: bytes=from-` (or `from-(from+len-1)`).
fn get_range(
    url: &str,
    policy: &HostPolicy,
    from: u64,
    len: Option<u64>,
    timeout: Option<Duration>,
    stall: Duration,
) -> Result<ureq::http::Response<ureq::Body>, FetchError> {
    let range = match len {
        Some(len) => format!("bytes={from}-{}", from + len.saturating_sub(1)),
        None => format!("bytes={from}-"),
    };
    get_inner(url, policy, Some(range), timeout, stall)
}

fn get_inner(
    url: &str,
    policy: &HostPolicy,
    range: Option<String>,
    timeout: Option<Duration>,
    stall: Duration,
) -> Result<ureq::http::Response<ureq::Body>, FetchError> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_global(timeout)
        // Limits each transport read, not the total exchange, so a multi-GB
        // download may take an hour while a stalled socket fails in `stall`.
        .timeout_per_read(Some(stall))
        .build()
        .into();
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        policy.check(&current)?;
        let mut request = agent.get(&current);
        if let Some(range) = &range {
            request = request.header("Range", range);
        }
        let response = request.call().map_err(|err| FetchError::Http {
            url: current.clone(),
            detail: err.to_string(),
        })?;
        let status = response.status().as_u16();
        if !matches!(status, 301 | 302 | 303 | 307 | 308) {
            return Ok(response);
        }
        let location = response
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| FetchError::Http {
                url: current.clone(),
                detail: format!("HTTP {status} without a Location header"),
            })?;
        current = absolute_location(&current, location);
    }
    Err(FetchError::Http {
        url: url.to_string(),
        detail: format!("more than {MAX_REDIRECTS} redirects"),
    })
}

/// Resolve a `Location` that may be origin-relative (`/path`).
fn absolute_location(base: &str, location: &str) -> String {
    if !location.starts_with('/') {
        return location.to_string();
    }
    let origin_end = base
        .find("://")
        .and_then(|i| base[i + 3..].find('/').map(|j| i + 3 + j))
        .unwrap_or(base.len());
    format!("{}{location}", &base[..origin_end])
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

fn hash_file(path: &Path, hasher: &mut Sha256) -> Result<(), FetchError> {
    let mut file = File::open(path).map_err(|err| FetchError::io(path, err))?;
    let mut buf = vec![0_u8; CHUNK];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|err| FetchError::io(path, err))?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

fn remove(path: &Path) -> Result<(), FetchError> {
    match std::fs::remove_file(path) {
        Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(FetchError::io(path, err)),
        _ => Ok(()),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn stalled_body_times_out_and_keeps_the_part_file() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stall server");
        let url = format!("http://{}/model.gguf", listener.local_addr().expect("addr"));
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut head = [0_u8; 1024];
            let _ = stream.read(&mut head);
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n");
            let _ = stream.flush();
            // Keep the socket open without sending the body.
            std::thread::sleep(Duration::from_secs(5));
        });

        let dir = std::env::temp_dir().join(format!("attune-stall-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let dest = dir.join("model.gguf");
        let request = Download {
            url,
            dest: dest.clone(),
            size: 1024,
            sha256: "0".repeat(64),
        };
        let err = download_with_stall(
            &request,
            &HostPolicy::allow_local_for_tests(),
            &|_| Some(u64::MAX),
            Duration::from_millis(300),
            &mut |_, _| {},
        )
        .expect_err("a stalled body must fail");
        assert!(matches!(err, FetchError::Http { .. }), "{err}");
        assert!(
            part_path(&dest).exists(),
            "the resumable .part file must survive a stall"
        );
    }

    #[test]
    fn relative_location_keeps_the_origin() {
        assert_eq!(
            absolute_location("https://huggingface.co/a/b?x=1", "/c/d"),
            "https://huggingface.co/c/d"
        );
        assert_eq!(
            absolute_location(
                "https://huggingface.co/a",
                "https://cdn-lfs.huggingface.co/z"
            ),
            "https://cdn-lfs.huggingface.co/z"
        );
    }
}
