//! Downloader against a local stub HTTP server (no real network).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use attune_fetch::{download_with, Download, FetchError, HostPolicy};
use sha2::{Digest, Sha256};

const BODY_LEN: usize = 1024 * 1024;

fn body() -> Vec<u8> {
    (0..BODY_LEN).map(|i| (i * 31 % 251) as u8).collect()
}

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Serves each connection with `respond(request_head)`; records the heads.
struct Stub {
    url: String,
    seen: Arc<Mutex<Vec<String>>>,
}

fn stub<F>(respond: F) -> Stub
where
    F: Fn(&str) -> Vec<u8> + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/model.gguf", listener.local_addr().expect("addr"));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut head = Vec::new();
            let mut byte = [0_u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap_or(0) == 0 {
                    break;
                }
                head.push(byte[0]);
            }
            let head = String::from_utf8_lossy(&head).to_string();
            log.lock().expect("lock").push(head.clone());
            let _ = stream.write_all(&respond(&head));
        }
    });
    Stub { url, seen }
}

fn response(status: &str, extra: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn temp_dest(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("attune-dl-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("model.gguf")
}

fn part(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

fn plenty(_: &Path) -> Option<u64> {
    Some(u64::MAX)
}

fn req(url: &str, dest: &Path, sha: &str) -> Download {
    Download {
        url: url.to_string(),
        dest: dest.to_path_buf(),
        size: BODY_LEN as u64,
        sha256: sha.to_string(),
    }
}

#[test]
fn fresh_download_verifies_and_renames() {
    let data = body();
    let served = data.clone();
    let server = stub(move |_| response("200 OK", "", &served));
    let dest = temp_dest("fresh");
    let mut last = 0;
    download_with(
        &req(&server.url, &dest, &sha_hex(&data)),
        &HostPolicy::allow_local_for_tests(),
        &plenty,
        &mut |done, _| last = done,
    )
    .expect("download");
    assert_eq!(std::fs::read(&dest).expect("read"), data);
    assert!(!part(&dest).exists());
    assert_eq!(last, BODY_LEN as u64);
}

#[test]
fn wrong_checksum_deletes_part_and_errors() {
    let data = body();
    let served = data.clone();
    let server = stub(move |_| response("200 OK", "", &served));
    let dest = temp_dest("badsha");
    let err = download_with(
        &req(&server.url, &dest, &"0".repeat(64)),
        &HostPolicy::allow_local_for_tests(),
        &plenty,
        &mut |_, _| {},
    )
    .expect_err("must fail");
    assert!(matches!(err, FetchError::ChecksumMismatch { .. }), "{err}");
    assert!(!dest.exists());
    assert!(!part(&dest).exists());
}

#[test]
fn resume_sends_range_and_completes() {
    let data = body();
    let served = data.clone();
    let server = stub(move |head| {
        assert!(
            head.to_ascii_lowercase().contains("range: bytes=307200-"),
            "{head}"
        );
        response(
            "206 Partial Content",
            &format!(
                "Content-Range: bytes 307200-{}/{}\r\n",
                BODY_LEN - 1,
                BODY_LEN
            ),
            &served[307_200..],
        )
    });
    let dest = temp_dest("resume");
    std::fs::write(part(&dest), &data[..307_200]).expect("seed part");
    download_with(
        &req(&server.url, &dest, &sha_hex(&data)),
        &HostPolicy::allow_local_for_tests(),
        &plenty,
        &mut |_, _| {},
    )
    .expect("download");
    assert_eq!(std::fs::read(&dest).expect("read"), data);
    assert_eq!(server.seen.lock().expect("lock").len(), 1);
}

#[test]
fn server_ignoring_range_restarts_from_zero() {
    let data = body();
    let served = data.clone();
    let server = stub(move |_| response("200 OK", "", &served));
    let dest = temp_dest("norange");
    std::fs::write(part(&dest), &data[..307_200]).expect("seed part");
    download_with(
        &req(&server.url, &dest, &sha_hex(&data)),
        &HostPolicy::allow_local_for_tests(),
        &plenty,
        &mut |_, _| {},
    )
    .expect("download");
    let got = std::fs::read(&dest).expect("read");
    assert_eq!(got.len(), BODY_LEN);
    assert_eq!(got, data);
}

#[test]
fn redirect_to_unlisted_host_is_refused() {
    let server = stub(|_| response("302 Found", "Location: http://10.0.0.1/x\r\n", b""));
    let dest = temp_dest("redirect");
    let err = download_with(
        &req(&server.url, &dest, &"0".repeat(64)),
        &HostPolicy::allow_local_for_tests(),
        &plenty,
        &mut |_, _| {},
    )
    .expect_err("must refuse");
    assert!(matches!(err, FetchError::HostNotAllowed { .. }), "{err}");
}

#[test]
fn not_enough_disk_space_errors_before_downloading() {
    let server = stub(|_| response("200 OK", "", b"x"));
    let dest = temp_dest("disk");
    let err = download_with(
        &req(&server.url, &dest, &"0".repeat(64)),
        &HostPolicy::allow_local_for_tests(),
        &|_| Some(10),
        &mut |_, _| {},
    )
    .expect_err("must fail");
    assert!(matches!(err, FetchError::DiskSpace { .. }), "{err}");
    assert!(server.seen.lock().expect("lock").is_empty());
}

#[test]
fn fetch_range_asks_for_a_prefix_and_caps_the_body() {
    let data = body();
    let served = data.clone();
    let server = stub(move |head| {
        assert!(
            head.to_ascii_lowercase().contains("range: bytes=0-1023"),
            "{head}"
        );
        // A server that ignores Range still yields only the first `len` bytes.
        response("200 OK", "", &served)
    });
    let got = attune_fetch::fetch_range(
        &server.url,
        &HostPolicy::allow_local_for_tests(),
        1024,
        std::time::Duration::from_secs(10),
    )
    .expect("range");
    assert_eq!(got, data[..1024]);
}
