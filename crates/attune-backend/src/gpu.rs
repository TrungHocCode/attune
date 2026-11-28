//! llama.cpp GPU device discovery for `serve` (M7 slice 5A).
//!
//! The bench crate already parses `--list-devices` (`attune_bench::gpu`), but
//! `attune-bench` may only depend on `attune-hal` (AGENTS.md §1), so this
//! crate cannot re-export that helper without inverting the dependency. The
//! small pure parsers and the bounded runner are therefore duplicated here,
//! the same precedent as the resolver/guard in [`crate::llama_subprocess`].
//! Keep them in sync with `attune_bench::gpu` when llama.cpp's listing format
//! changes.
//!
//! The device id (`Vulkan1`, `CUDA0`, …) is what `-dev` expects, so
//! [`pick_device`] chooses deliberately instead of letting llama.cpp default
//! to `Vulkan0`, which on a laptop is often the integrated GPU listed first.

use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::BackendError;

/// Deadline for `--list-devices`. The probe only enumerates adapters, so a
/// healthy build answers in well under a second.
pub const LIST_DEVICES_TIMEOUT: Duration = Duration::from_secs(10);

/// Polling period while waiting for the probe to exit.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long the output readers get after the probe is gone.
const READER_JOIN_BUDGET: Duration = Duration::from_secs(2);

/// Characters of stderr kept in a probe error.
const STDERR_TAIL_CHARS: usize = 300;

/// One device as listed by llama.cpp: `(id, name, total MiB)`.
///
/// `id` is the `-dev` value (`Vulkan0`, `CUDA1`, …). `total` is the total
/// memory in MiB when the build reports it; `None` when the parenthesis is
/// absent or unparseable.
pub type DeviceEntry = (String, String, Option<u64>);

/// Parse `llama.cpp --list-devices` output into `(id, name, total MiB)`.
///
/// Unrecognised lines (headers, empty lines, chatter) are ignored rather than
/// failing the parse. The total is the *first* `N MiB` value inside the
/// trailing parenthesis; the free value that follows it is ignored.
pub fn parse_list_devices(text: &str) -> Vec<DeviceEntry> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((id, rest)) = line.split_once(':') else {
            continue;
        };
        let id = id.trim();
        if !is_device_id(id) {
            continue;
        }
        let rest = rest.trim();
        if rest.is_empty() {
            continue;
        }
        let (name, params) = match rest.rfind('(') {
            Some(idx) => (rest[..idx].trim(), Some(&rest[idx + 1..])),
            None => (rest, None),
        };
        if name.is_empty() {
            continue;
        }
        out.push((
            id.to_string(),
            name.to_string(),
            params.and_then(parse_total_mib),
        ));
    }
    out
}

/// A device id is one token of ASCII alphanumerics that starts with a letter:
/// `Vulkan1`, `CUDA0`, `Metal`, `BLAS`. That rejects `Available devices:` and
/// log prefixes like `ggml_vulkan:` without a hard-coded list of backends.
fn is_device_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric())
}

/// First `N MiB` value in a parenthesis body, e.g. `5921 MiB, 5153 MiB free`.
fn parse_total_mib(params: &str) -> Option<u64> {
    let fields: Vec<&str> = params.split_whitespace().collect();
    for pair in fields.windows(2) {
        if pair[1].trim_end_matches(',') != "MiB" {
            continue;
        }
        if let Ok(mib) = pair[0].trim_end_matches(',').parse::<u64>() {
            return Some(mib);
        }
    }
    None
}

/// True for adapters we do not want to prefer when a discrete GPU is present:
/// Intel graphics and anything labelled integrated.
fn is_integrated(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("intel")
        || lower.contains("integrated")
        || lower.contains("uhd graphics")
        || lower.contains("iris")
        || lower.contains("radeon graphics")
}

/// Pick the device id to pass to `-dev`.
///
/// A `preferred_name` (the GPU recorded in the profile) wins when some listed
/// name contains it (case-insensitive). Otherwise the discrete adapter with the
/// most reported memory wins, skipping Intel/integrated devices; if every
/// listed device is integrated, the largest is still used rather than returning
/// nothing.
pub fn pick_device(devices: &[DeviceEntry], preferred_name: Option<&str>) -> Option<String> {
    if let Some(preferred) = preferred_name.map(str::trim).filter(|s| !s.is_empty()) {
        let needle = preferred.to_ascii_lowercase();
        if let Some((id, _, _)) = devices
            .iter()
            .find(|(_, name, _)| name.to_ascii_lowercase().contains(&needle))
        {
            return Some(id.clone());
        }
    }

    let discrete: Vec<&DeviceEntry> = devices
        .iter()
        .filter(|(_, name, _)| !is_integrated(name))
        .collect();
    let pool: Vec<&DeviceEntry> = if discrete.is_empty() {
        devices.iter().collect()
    } else {
        discrete
    };
    pool.into_iter()
        .max_by_key(|(_, _, total)| total.unwrap_or(0))
        .map(|(id, _, _)| id.clone())
}

/// Run `<llama_bin> --list-devices` and parse the result.
///
/// A non-zero exit with no parseable device names is reported as a
/// [`BackendError::GpuProbe`] carrying the stderr tail, so a wrong binary or a
/// missing GPU runtime surfaces clearly instead of as an empty list.
pub fn list_devices(llama_bin: &Path) -> Result<Vec<DeviceEntry>, BackendError> {
    let mut child = ProbeChild::spawn(llama_bin)?;
    let status = child.wait(LIST_DEVICES_TIMEOUT)?;
    let Some(status) = status else {
        return Err(BackendError::GpuProbe {
            binary: llama_bin.display().to_string(),
            detail: format!(
                "`--list-devices` did not finish within {} s",
                LIST_DEVICES_TIMEOUT.as_secs()
            ),
        });
    };
    let (stdout, stderr) = child.collect();
    let mut devices = parse_list_devices(&stdout);
    if devices.is_empty() {
        devices = parse_list_devices(&stderr);
    }
    if devices.is_empty() && !status.success() {
        let code = status.code().map_or_else(
            || "terminated by a signal".to_string(),
            |code| code.to_string(),
        );
        return Err(BackendError::GpuProbe {
            binary: llama_bin.display().to_string(),
            detail: format!("exited with status {code}: {}", stderr_tail(&stderr)),
        });
    }
    Ok(devices)
}

/// Last [`STDERR_TAIL_CHARS`] characters of a stream, trimmed.
fn stderr_tail(stderr: &str) -> String {
    let trimmed = stderr.trim_end();
    let len = trimmed.chars().count();
    if len <= STDERR_TAIL_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().skip(len - STDERR_TAIL_CHARS).collect()
}

// ---- bounded probe runner ---------------------------------------------------

/// Owns the probe child plus the threads draining its pipes.
///
/// Dropping it kills and reaps the child, so a timeout never leaks a process.
struct ProbeChild {
    child: Child,
    binary: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
}

impl ProbeChild {
    fn spawn(binary: &Path) -> Result<Self, BackendError> {
        let mut child = Command::new(binary)
            .arg("--list-devices")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| BackendError::Spawn {
                binary: binary.display().to_string(),
                source,
            })?;
        let stdout = child.stdout.take().map(spawn_reader);
        let stderr = child.stderr.take().map(spawn_reader);
        Ok(Self {
            child,
            binary: binary.display().to_string(),
            stdout,
            stderr,
        })
    }

    /// Poll until the child exits or `timeout` elapses.
    fn wait(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, BackendError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Ok(Some(status)),
                Ok(None) if Instant::now() >= deadline => return Ok(None),
                Ok(None) => std::thread::sleep(POLL_INTERVAL),
                Err(source) => {
                    return Err(BackendError::Spawn {
                        binary: self.binary.clone(),
                        source,
                    })
                }
            }
        }
    }

    /// Join both readers, sharing one deadline.
    fn collect(&mut self) -> (String, String) {
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        (
            decode(join_reader(self.stdout.take(), deadline)),
            decode(join_reader(self.stderr.take(), deadline)),
        )
    }
}

impl Drop for ProbeChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let deadline = Instant::now() + READER_JOIN_BUDGET;
        let _ = join_reader(self.stdout.take(), deadline);
        let _ = join_reader(self.stderr.take(), deadline);
    }
}

fn spawn_reader<R>(mut pipe: R) -> JoinHandle<Vec<u8>>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        buf
    })
}

fn join_reader(handle: Option<JoinHandle<Vec<u8>>>, deadline: Instant) -> Vec<u8> {
    let Some(handle) = handle else {
        return Vec::new();
    };
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return Vec::new();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    handle.join().unwrap_or_default()
}

fn decode(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vulkan_with_integrated_and_discrete_gpus() {
        let text = "\
Available devices:
  Vulkan0: Intel(R) Graphics (RPL-P) (5934 MiB, 5150 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (5921 MiB, 5153 MiB free)
";
        let devices = parse_list_devices(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].0, "Vulkan0");
        assert_eq!(devices[0].1, "Intel(R) Graphics (RPL-P)");
        assert_eq!(devices[0].2, Some(5934));
        assert_eq!(devices[1].0, "Vulkan1");
        assert_eq!(devices[1].1, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(devices[1].2, Some(5921));
    }

    #[test]
    fn parses_a_single_cuda_device() {
        let text = "  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU (6140 MiB, 5075 MiB free)\n";
        assert_eq!(
            parse_list_devices(text),
            vec![(
                "CUDA0".to_string(),
                "NVIDIA GeForce RTX 4050 Laptop GPU".to_string(),
                Some(6140)
            )]
        );
    }

    #[test]
    fn parse_ignores_garbage_lines() {
        let text = "\
Available devices:
not a device line
ggml_vulkan: backend loaded
  : missing id
  Vulkan1: GPU without memory
  Vulkan2: GPU with bad memory (not a number MiB)
";
        let devices = parse_list_devices(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(
            devices[0],
            ("Vulkan1".into(), "GPU without memory".into(), None)
        );
        assert_eq!(
            devices[1],
            ("Vulkan2".into(), "GPU with bad memory".into(), None)
        );
    }

    #[test]
    fn parse_returns_empty_for_unrelated_output() {
        assert!(parse_list_devices("").is_empty());
        assert!(parse_list_devices("llama-cli: unrecognized option\n").is_empty());
    }

    fn vulkan_pair() -> Vec<DeviceEntry> {
        vec![
            (
                "Vulkan0".into(),
                "Intel(R) Graphics (RPL-P)".into(),
                Some(5934),
            ),
            (
                "Vulkan1".into(),
                "NVIDIA GeForce RTX 4050 Laptop GPU".into(),
                Some(5921),
            ),
        ]
    }

    #[test]
    fn pick_prefers_the_named_discrete_gpu() {
        let picked = pick_device(&vulkan_pair(), Some("NVIDIA GeForce RTX 4050 Laptop GPU"));
        assert_eq!(picked.as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_matches_the_preferred_name_case_insensitively() {
        assert_eq!(
            pick_device(&vulkan_pair(), Some("nvidia geforce rtx 4050")).as_deref(),
            Some("Vulkan1")
        );
    }

    #[test]
    fn pick_skips_intel_and_chooses_the_largest_discrete_adapter() {
        let devices = vec![
            ("Vulkan0".into(), "Intel(R) Graphics".into(), Some(8192)),
            (
                "Vulkan1".into(),
                "AMD Radeon RX 7800 XT".into(),
                Some(16384),
            ),
            (
                "Vulkan2".into(),
                "NVIDIA GeForce RTX 4060".into(),
                Some(8192),
            ),
        ];
        assert_eq!(pick_device(&devices, None).as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_falls_back_to_the_largest_when_everything_is_integrated() {
        let devices = vec![
            ("Vulkan0".into(), "Intel UHD Graphics".into(), Some(4096)),
            (
                "Vulkan1".into(),
                "AMD Radeon(TM) Graphics".into(),
                Some(8192),
            ),
        ];
        assert_eq!(pick_device(&devices, None).as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn pick_returns_none_for_no_devices() {
        assert_eq!(pick_device(&[], None), None);
        assert_eq!(pick_device(&[], Some("NVIDIA")), None);
    }
}
