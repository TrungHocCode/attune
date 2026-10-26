//! GGUF header probing backing `estimate --model <path.gguf>`.
//!
//! Only the metadata key/value section from the first
//! [`MAX_HEADER_BYTES`] is read; tensor data and the weight payload are never
//! touched (no mmap, no full file read). The parser extracts the facts the
//! roofline needs:
//!
//! - `general.architecture` (required),
//! - `<arch>.block_count`,
//! - `<arch>.embedding_length`,
//! - `<arch>.attention.head_count_kv`, falling back to
//!   `<arch>.attention.head_count` when the KV variant is absent (the
//!   documented MHA fallback: without GQA every head is a KV head),
//! - `general.parameter_count` (optional; `0` when the writer did not store
//!   it — this module deliberately does not stat the file size).
//!
//! The internal parse also keeps `<arch>.attention.head_count` so
//! [`GgufProbe`] can derive the per-head dimension used by the existing KV
//! math: `head_dim = embedding_length / head_count`.
//!
//! Metadata is written in arbitrary order. Keys are therefore collected while
//! walking the header and resolved against the architecture once it is known.
//! Arrays are skipped without loading their contents: fixed-width elements by
//! their width, strings by their length prefix, and nested arrays recursively.
//! The walk normally stops at the 64 KiB cap inside one of the tokenizer
//! arrays; that is expected, so a truncated tail is tolerated as long as the
//! requested facts were already seen.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::catalog::{ModelEntry, MODEL_CATALOG};
use super::Arch;
use crate::EstimateError;

/// Hard cap on how many header bytes [`probe_gguf`] reads.
///
/// 64 KiB covers the metadata prefix used by llama.cpp and the common
/// converters before the (often multi-megabyte) tokenizer arrays.
pub const MAX_HEADER_BYTES: u64 = 64 * 1024;

/// Current GGUF magic.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF versions whose metadata layout this parser understands.
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

// GGUF metadata value types (`gguf_meta_value_type`).
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

/// Maximum nesting depth accepted for arrays of arrays.
const MAX_ARRAY_DEPTH: u32 = 16;

/// Facts read from one GGUF header.
///
/// All numeric fields use `0` as the "writer did not store it" sentinel; the
/// exception is [`GgufFacts::architecture`], which must be present for the
/// arch-specific keys to mean anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFacts {
    /// `general.architecture` (e.g. `llama`, `qwen2`).
    pub architecture: String,
    /// `<arch>.block_count`; 0 when absent.
    pub block_count: u32,
    /// `<arch>.embedding_length`; 0 when absent.
    pub embedding_length: u32,
    /// `<arch>.attention.head_count_kv`, falling back to
    /// `<arch>.attention.head_count`; 0 when both are absent.
    pub n_kv_heads: u32,
    /// `general.parameter_count`; 0 when absent (the file is *not* stat'ed
    /// here, so a file without this key cannot supply a parameter count).
    pub param_estimate: u64,
}

/// A probed file plus the catalog cross-check and the KV shape it implies.
///
/// The user-facing entry point for file resolution: it carries the requested
/// facts alongside an optional catalog warning. A mismatch is a notice, never
/// an error — the file always wins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufProbe {
    /// Facts read from the header.
    pub facts: GgufFacts,
    /// KV-cache shape derived from the header, when both `block_count`,
    /// `n_kv_heads` and a usable `head_dim` are available.
    pub arch: Option<Arch>,
    /// Catalog entry whose name appears in the file stem, if any.
    pub catalog: Option<&'static ModelEntry>,
    /// Catalog/file disagreement, e.g.
    /// `catalog says 32 layers, file has 36`.
    pub warning: Option<String>,
}

impl GgufProbe {
    /// Parameter count to feed the roofline: the file's
    /// `general.parameter_count` first, then a matching catalog entry.
    ///
    /// `None` means neither source had a value; the caller should surface a
    /// clean error or fall back to explicit `--params` rather than predicting
    /// with zero parameters.
    pub fn params(&self) -> Option<u64> {
        if self.facts.param_estimate > 0 {
            Some(self.facts.param_estimate)
        } else {
            self.catalog.map(|entry| entry.params)
        }
    }
}

/// Read the header-only facts from `path`.
///
/// See the module docs for the fields and limits. I/O failures, a bad magic,
/// an unsupported version and a missing `general.architecture` are clean
/// [`EstimateError::InvalidGguf`] errors.
pub fn probe_gguf(path: &Path) -> Result<GgufFacts, EstimateError> {
    Ok(parse_header(path)?.facts)
}

/// Probe `path` and cross-check the result against a catalog entry inferred
/// from the file name.
///
/// The file is authoritative: a catalog disagreement only produces
/// [`GgufProbe::warning`]. Catalog names are matched case-insensitively as a
/// substring of the stem, preferring the longest match (so
/// `smollm2-135m-q4km.gguf` resolves to `smollm2-135m`).
pub fn probe_gguf_file(path: &Path) -> Result<GgufProbe, EstimateError> {
    let parsed = parse_header(path)?;
    let catalog = catalog_for_path(path);
    let warning = catalog.and_then(|entry| catalog_warning(&parsed.facts, entry));
    Ok(GgufProbe {
        arch: parsed.arch(),
        facts: parsed.facts,
        catalog,
        warning,
    })
}

/// Header parse result with the extra head count needed to size KV cache.
struct ParsedHeader {
    facts: GgufFacts,
    head_count: u32,
}

impl ParsedHeader {
    /// Derive the KV shape, requiring a non-zero layer/KV-head count and a
    /// head dimension that divides `embedding_length` evenly.
    fn arch(&self) -> Option<Arch> {
        let head_dim = self.head_dim()?;
        let facts = &self.facts;
        (facts.block_count > 0 && facts.n_kv_heads > 0).then_some(Arch {
            // `Arch::name` is display-only for this path; the file has no
            // static catalog name.
            name: "gguf",
            n_layer: facts.block_count,
            n_kv_heads: facts.n_kv_heads,
            head_dim,
        })
    }

    fn head_dim(&self) -> Option<u32> {
        if self.head_count == 0 || self.facts.embedding_length == 0 {
            return None;
        }
        let head_dim = self.facts.embedding_length / self.head_count;
        (head_dim > 0 && self.facts.embedding_length.is_multiple_of(self.head_count))
            .then_some(head_dim)
    }
}

/// Read up to [`MAX_HEADER_BYTES`] and parse the metadata prefix.
fn parse_header(path: &Path) -> Result<ParsedHeader, EstimateError> {
    let file = File::open(path).map_err(|err| {
        EstimateError::InvalidGguf(format!("cannot open {}: {err}", path.display()))
    })?;

    // `take` bounds the read, and `BufReader` keeps it buffered: no mmap, no
    // weight payload, no more than 64 KiB of I/O.
    let mut reader = BufReader::new(file).take(MAX_HEADER_BYTES);
    let mut header = Vec::with_capacity(MAX_HEADER_BYTES as usize);
    reader.read_to_end(&mut header).map_err(|err| {
        EstimateError::InvalidGguf(format!("cannot read {}: {err}", path.display()))
    })?;

    parse_metadata(&header)
}

/// Parse the GGUF prefix held in `data`.
fn parse_metadata(data: &[u8]) -> Result<ParsedHeader, EstimateError> {
    let mut reader = ByteReader::new(data);

    let magic = reader.take(4).map_err(header_error)?;
    if magic != GGUF_MAGIC {
        return Err(EstimateError::InvalidGguf(format!(
            "bad magic {:?}, expected \"GGUF\"",
            String::from_utf8_lossy(magic)
        )));
    }

    let version = reader.u32().map_err(header_error)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(EstimateError::InvalidGguf(format!(
            "unsupported GGUF version {version} (expected 2 or 3)"
        )));
    }

    let _tensor_count = reader.u64().map_err(header_error)?;
    let kv_count = reader.u64().map_err(header_error)?;

    let mut architecture: Option<String> = None;
    let mut integers: Vec<(String, u64)> = Vec::new();
    let mut truncated = false;

    for _ in 0..kv_count {
        let key = match reader.gguf_str() {
            Ok(key) => key,
            Err(HeaderError::Truncated) => {
                truncated = true;
                break;
            }
            Err(err) => return Err(header_error(err)),
        };
        let value_type = match reader.u32() {
            Ok(value_type) => value_type,
            Err(HeaderError::Truncated) => {
                truncated = true;
                break;
            }
            Err(err) => return Err(header_error(err)),
        };
        let value = match reader.read_value(value_type) {
            Ok(value) => value,
            Err(HeaderError::Truncated) => {
                truncated = true;
                break;
            }
            Err(err) => return Err(header_error(err)),
        };

        match value {
            MetaValue::String(value) if key == "general.architecture" => {
                architecture = Some(value);
            }
            MetaValue::Unsigned(value) if is_integer_fact(key) => {
                integers.push((key.to_string(), value));
            }
            // Non-negative signed values are accepted as a courtesy for
            // writers that encode an unsigned field as INT32/INT64.
            MetaValue::Signed(value) if value >= 0 && is_integer_fact(key) => {
                integers.push((key.to_string(), value as u64));
            }
            _ => {}
        }
    }

    let architecture = architecture.ok_or_else(|| {
        let reason = if truncated {
            format!(
                "missing general.architecture (header truncated after {} bytes)",
                data.len()
            )
        } else {
            "missing general.architecture".to_string()
        };
        EstimateError::InvalidGguf(reason)
    })?;

    let lookup = |suffix: &str| -> Option<u64> {
        let key = format!("{architecture}.{suffix}");
        integers
            .iter()
            .rev()
            .find_map(|(candidate, value)| (candidate == &key).then_some(*value))
    };
    let lookup_global = |key: &str| -> Option<u64> {
        integers
            .iter()
            .rev()
            .find_map(|(candidate, value)| (candidate == key).then_some(*value))
    };

    let block_count = narrow("block_count", lookup("block_count"))?.unwrap_or(0);
    let embedding_length = narrow("embedding_length", lookup("embedding_length"))?.unwrap_or(0);
    let head_count = narrow("attention.head_count", lookup("attention.head_count"))?;
    let head_count_kv = narrow("attention.head_count_kv", lookup("attention.head_count_kv"))?;

    Ok(ParsedHeader {
        facts: GgufFacts {
            architecture,
            block_count,
            embedding_length,
            // MQA/GQA fallback: with no KV-specific key every head is a KV
            // head.
            n_kv_heads: head_count_kv.or(head_count).unwrap_or(0),
            param_estimate: lookup_global("general.parameter_count").unwrap_or(0),
        },
        head_count: head_count.unwrap_or(0),
    })
}

/// Convert a metadata integer to the `u32` the architecture keys use.
fn narrow(field: &str, value: Option<u64>) -> Result<Option<u32>, EstimateError> {
    value
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                EstimateError::InvalidGguf(format!("{field} value {value} does not fit in u32"))
            })
        })
        .transpose()
}

/// Keys whose integer value the parser keeps.
fn is_integer_fact(key: &str) -> bool {
    key == "general.parameter_count"
        || key.ends_with(".block_count")
        || key.ends_with(".embedding_length")
        || key.ends_with(".attention.head_count")
        || key.ends_with(".attention.head_count_kv")
}

/// Infer the catalog entry named in a GGUF file stem.
fn catalog_for_path(path: &Path) -> Option<&'static ModelEntry> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    let stem = stem.replace('_', "-");
    MODEL_CATALOG
        .iter()
        .filter(|entry| stem.contains(entry.name))
        .max_by_key(|entry| entry.name.len())
}

/// One-line catalog/file disagreement, if the file's shape differs.
fn catalog_warning(facts: &GgufFacts, entry: &ModelEntry) -> Option<String> {
    let mut parts = Vec::new();
    if facts.block_count > 0 && facts.block_count != entry.arch.n_layer {
        parts.push(format!(
            "catalog says {} layers, file has {}",
            entry.arch.n_layer, facts.block_count
        ));
    }
    if facts.n_kv_heads > 0 && facts.n_kv_heads != entry.arch.n_kv_heads {
        parts.push(format!(
            "catalog says {} KV heads, file has {}",
            entry.arch.n_kv_heads, facts.n_kv_heads
        ));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

/// A metadata value this parser can walk past.
///
/// Only the payloads needed for the facts are kept; floats/bools/arrays are
/// consumed and collapsed to [`MetaValue::Other`].
enum MetaValue {
    Unsigned(u64),
    Signed(i64),
    String(String),
    Other,
}

/// Bounds-checked little-endian reader for the header slice.
struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], HeaderError> {
        let end = self.pos.checked_add(len).ok_or(HeaderError::Truncated)?;
        if end > self.data.len() {
            return Err(HeaderError::Truncated);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<(), HeaderError> {
        self.take(len).map(|_| ())
    }

    fn array<T, const N: usize>(&mut self, convert: fn([u8; N]) -> T) -> Result<T, HeaderError> {
        let bytes = self.take(N)?;
        let array = bytes.try_into().map_err(|_| HeaderError::Truncated)?;
        Ok(convert(array))
    }

    fn u8(&mut self) -> Result<u8, HeaderError> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, HeaderError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, HeaderError> {
        self.array(u16::from_le_bytes)
    }

    fn i16(&mut self) -> Result<i16, HeaderError> {
        self.array(i16::from_le_bytes)
    }

    fn u32(&mut self) -> Result<u32, HeaderError> {
        self.array(u32::from_le_bytes)
    }

    fn i32(&mut self) -> Result<i32, HeaderError> {
        self.array(i32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, HeaderError> {
        self.array(u64::from_le_bytes)
    }

    fn i64(&mut self) -> Result<i64, HeaderError> {
        self.array(i64::from_le_bytes)
    }

    fn f32(&mut self) -> Result<f32, HeaderError> {
        self.array(f32::from_le_bytes)
    }

    fn f64(&mut self) -> Result<f64, HeaderError> {
        self.array(f64::from_le_bytes)
    }

    /// GGUF string: `u64` length followed by `len` UTF-8 bytes (no NUL).
    fn gguf_str(&mut self) -> Result<&'a str, HeaderError> {
        let len = self.u64()?;
        let len = usize::try_from(len)
            .map_err(|_| HeaderError::Invalid("string length exceeds usize".to_string()))?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map_err(|_| HeaderError::Invalid("metadata string is not UTF-8".to_string()))
    }

    fn read_value(&mut self, value_type: u32) -> Result<MetaValue, HeaderError> {
        Ok(match value_type {
            TYPE_UINT8 => MetaValue::Unsigned(self.u8()? as u64),
            TYPE_INT8 => MetaValue::Signed(self.i8()? as i64),
            TYPE_UINT16 => MetaValue::Unsigned(self.u16()? as u64),
            TYPE_INT16 => MetaValue::Signed(self.i16()? as i64),
            TYPE_UINT32 => MetaValue::Unsigned(self.u32()? as u64),
            TYPE_INT32 => MetaValue::Signed(self.i32()? as i64),
            TYPE_UINT64 => MetaValue::Unsigned(self.u64()?),
            TYPE_INT64 => MetaValue::Signed(self.i64()?),
            TYPE_FLOAT32 => {
                self.f32()?;
                MetaValue::Other
            }
            TYPE_FLOAT64 => {
                self.f64()?;
                MetaValue::Other
            }
            TYPE_BOOL => {
                self.u8()?;
                MetaValue::Other
            }
            TYPE_STRING => MetaValue::String(self.gguf_str()?.to_string()),
            TYPE_ARRAY => {
                self.skip_array(0)?;
                MetaValue::Other
            }
            other => return Err(HeaderError::UnsupportedType(other)),
        })
    }

    /// Walk an array without materialising it.
    fn skip_array(&mut self, depth: u32) -> Result<(), HeaderError> {
        if depth > MAX_ARRAY_DEPTH {
            return Err(HeaderError::Invalid(
                "metadata array nesting is too deep".to_string(),
            ));
        }
        let element_type = self.u32()?;
        let count = self.u64()?;
        match element_type {
            TYPE_STRING => {
                for _ in 0..count {
                    let len = self.u64()?;
                    let len = usize::try_from(len).map_err(|_| {
                        HeaderError::Invalid("array string length exceeds usize".to_string())
                    })?;
                    self.skip(len)?;
                }
            }
            TYPE_ARRAY => {
                for _ in 0..count {
                    self.skip_array(depth + 1)?;
                }
            }
            fixed => {
                let size = element_size(fixed).ok_or(HeaderError::UnsupportedType(fixed))?;
                let total = count
                    .checked_mul(size)
                    .ok_or_else(|| HeaderError::Invalid("array is too large".to_string()))?;
                let total = usize::try_from(total)
                    .map_err(|_| HeaderError::Invalid("array is too large".to_string()))?;
                self.skip(total)?;
            }
        }
        Ok(())
    }
}

/// Fixed byte width of a scalar metadata type.
fn element_size(value_type: u32) -> Option<u64> {
    match value_type {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

/// Internal parse failure before it is mapped to [`EstimateError`].
enum HeaderError {
    Truncated,
    Invalid(String),
    UnsupportedType(u32),
}

fn header_error(err: HeaderError) -> EstimateError {
    match err {
        HeaderError::Truncated => {
            EstimateError::InvalidGguf("header is truncated before the metadata".to_string())
        }
        HeaderError::Invalid(message) => EstimateError::InvalidGguf(message),
        HeaderError::UnsupportedType(value_type) => {
            EstimateError::InvalidGguf(format!("unsupported metadata value type {value_type}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Build a fake GGUF header without any tensor payload.
    struct FakeGguf {
        bytes: Vec<u8>,
    }

    impl FakeGguf {
        fn new(version: u32, kv_count: u64) -> Self {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(GGUF_MAGIC);
            push_u32(&mut bytes, version);
            push_u64(&mut bytes, 0); // tensor_count
            push_u64(&mut bytes, kv_count);
            Self { bytes }
        }

        fn key(&mut self, key: &str, value_type: u32) {
            push_string(&mut self.bytes, key);
            push_u32(&mut self.bytes, value_type);
        }

        fn string(&mut self, key: &str, value: &str) {
            self.key(key, TYPE_STRING);
            push_string(&mut self.bytes, value);
        }

        fn u32(&mut self, key: &str, value: u32) {
            self.key(key, TYPE_UINT32);
            push_u32(&mut self.bytes, value);
        }

        fn u64(&mut self, key: &str, value: u64) {
            self.key(key, TYPE_UINT64);
            push_u64(&mut self.bytes, value);
        }

        fn f32(&mut self, key: &str, value: f32) {
            self.key(key, TYPE_FLOAT32);
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn boolean(&mut self, key: &str, value: bool) {
            self.key(key, TYPE_BOOL);
            self.bytes.push(u8::from(value));
        }

        fn string_array(&mut self, key: &str, values: &[&str]) {
            self.key(key, TYPE_ARRAY);
            push_u32(&mut self.bytes, TYPE_STRING);
            push_u64(&mut self.bytes, values.len() as u64);
            for value in values {
                push_string(&mut self.bytes, value);
            }
        }

        fn u32_array(&mut self, key: &str, values: &[u32]) {
            self.key(key, TYPE_ARRAY);
            push_u32(&mut self.bytes, TYPE_UINT32);
            push_u64(&mut self.bytes, values.len() as u64);
            for value in values {
                push_u32(&mut self.bytes, *value);
            }
        }

        fn bytes(&self) -> &[u8] {
            &self.bytes
        }
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    /// Write `bytes` to a unique temp directory and return the file path.
    fn write_temp(tag: &str, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("attune-gguf-{}-{tag}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        fs::write(&path, bytes).expect("write fake GGUF");
        path
    }

    fn cleanup(path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn probe_reads_architecture_shape_and_param_count() {
        let mut file = FakeGguf::new(3, 9);
        file.string("general.architecture", "llama");
        // A string array before the architecture keys mirrors real writers.
        file.string_array("general.languages", &["en"]);
        file.u32("llama.block_count", 30);
        file.u32("llama.embedding_length", 576);
        file.u32("llama.attention.head_count", 9);
        file.u32("llama.attention.head_count_kv", 3);
        file.f32("llama.rope.freq_base", 10_000.0);
        file.boolean("tokenizer.ggml.add_bos_token", true);
        file.u64("general.parameter_count", 135_000_000);

        let path = write_temp("valid", "smollm2-135m-q4km.gguf", file.bytes());
        let facts = probe_gguf(&path).expect("valid GGUF header");

        assert_eq!(facts.architecture, "llama");
        assert_eq!(facts.block_count, 30);
        assert_eq!(facts.embedding_length, 576);
        assert_eq!(facts.n_kv_heads, 3);
        assert_eq!(facts.param_estimate, 135_000_000);

        let probe = probe_gguf_file(&path).expect("probe with catalog check");
        assert_eq!(probe.facts, facts);
        assert_eq!(
            probe.arch,
            Some(Arch {
                name: "gguf",
                n_layer: 30,
                n_kv_heads: 3,
                // 576 / 9 = 64.
                head_dim: 64,
            })
        );
        assert_eq!(probe.catalog.map(|entry| entry.name), Some("smollm2-135m"));
        assert_eq!(probe.params(), Some(135_000_000));
        assert_eq!(probe.warning, None);
        cleanup(&path);
    }

    #[test]
    fn probe_rejects_corrupt_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NOPE");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        let path = write_temp("corrupt", "bad.gguf", &bytes);
        let err = probe_gguf(&path).expect_err("bad magic must fail");
        assert!(matches!(err, EstimateError::InvalidGguf(_)), "got {err:?}");
        assert!(err.to_string().contains("magic"), "message: {err}");
        cleanup(&path);
    }

    #[test]
    fn kv_heads_fall_back_to_the_total_head_count() {
        let mut file = FakeGguf::new(3, 5);
        file.string("general.architecture", "llama");
        file.u32("llama.block_count", 12);
        file.u32("llama.embedding_length", 768);
        file.u32("llama.attention.head_count", 12);
        file.u32_array("llama.rope.dimension_sections", &[0, 0, 0, 0]);

        let path = write_temp("fallback", "mha.gguf", file.bytes());
        let facts = probe_gguf(&path).expect("MHA metadata");
        assert_eq!(facts.n_kv_heads, 12);
        assert_eq!(facts.embedding_length / 12, 64);
        cleanup(&path);
    }

    #[test]
    fn file_probe_warns_on_catalog_mismatch_and_file_wins() {
        let mut file = FakeGguf::new(3, 5);
        file.string("general.architecture", "llama");
        file.u32("llama.block_count", 36);
        file.u32("llama.embedding_length", 4096);
        file.u32("llama.attention.head_count", 32);
        file.u32("llama.attention.head_count_kv", 8);

        let path = write_temp("catalog", "llama-3.1-8b.gguf", file.bytes());
        let probe = probe_gguf_file(&path).expect("valid GGUF header");

        assert_eq!(probe.catalog.map(|entry| entry.name), Some("llama-3.1-8b"));
        assert_eq!(
            probe.warning.as_deref(),
            Some("catalog says 32 layers, file has 36")
        );
        // No `general.parameter_count`: the catalog supplies the params.
        assert_eq!(probe.params(), Some(8_030_000_000));
        // The file shape wins over the catalog shape.
        let arch = probe.arch.expect("file arch");
        assert_eq!((arch.n_layer, arch.n_kv_heads, arch.head_dim), (36, 8, 128));
        cleanup(&path);
    }

    #[test]
    fn minimal_header_without_head_or_param_keys_still_parses() {
        // The smallest useful header: architecture + layer/embedding shape.
        // Missing optional keys become 0 and disable the KV shape instead of
        // failing the probe.
        let mut file = FakeGguf::new(3, 3);
        file.string("general.architecture", "llama");
        file.u32("llama.block_count", 12);
        file.u32("llama.embedding_length", 768);

        let path = write_temp("minimal", "minimal.gguf", file.bytes());
        let facts = probe_gguf(&path).expect("minimal header should parse");
        assert_eq!(facts.block_count, 12);
        assert_eq!(facts.embedding_length, 768);
        assert_eq!(facts.n_kv_heads, 0);
        assert_eq!(facts.param_estimate, 0);

        let probe = probe_gguf_file(&path).expect("probe with catalog check");
        assert_eq!(probe.arch, None);
        assert_eq!(probe.params(), None);
        cleanup(&path);
    }

    #[test]
    fn file_param_count_wins_over_the_catalog() {
        let mut file = FakeGguf::new(3, 6);
        file.string("general.architecture", "llama");
        file.u32("llama.block_count", 32);
        file.u32("llama.embedding_length", 4096);
        file.u32("llama.attention.head_count", 32);
        file.u32("llama.attention.head_count_kv", 8);
        file.u64("general.parameter_count", 8_000_000_000);

        let path = write_temp("params", "llama-3.1-8b.gguf", file.bytes());
        let probe = probe_gguf_file(&path).expect("valid GGUF header");
        // The file's own count wins over the catalog's 8.03B, and the matching
        // shape produces no warning.
        assert_eq!(probe.params(), Some(8_000_000_000));
        assert_eq!(probe.warning, None);
        cleanup(&path);
    }

    #[test]
    fn probe_rejects_unsupported_versions_and_missing_architecture() {
        let mut old = FakeGguf::new(1, 0);
        old.string("general.architecture", "llama");
        let path = write_temp("version", "old.gguf", old.bytes());
        let err = probe_gguf(&path).expect_err("version 1 is not supported");
        assert!(err.to_string().contains("version"), "message: {err}");
        cleanup(&path);

        let mut no_arch = FakeGguf::new(3, 1);
        no_arch.u32("llama.block_count", 30);
        let path = write_temp("no-arch", "no-arch.gguf", no_arch.bytes());
        let err = probe_gguf(&path).expect_err("architecture is required");
        assert!(
            err.to_string().contains("general.architecture"),
            "message: {err}"
        );
        cleanup(&path);
    }

    #[test]
    fn truncated_metadata_still_returns_the_facts_already_seen() {
        // Claim more entries than the file contains: the walk hits the end of
        // the buffer after the facts, which is the normal tokenizer-array case.
        let mut file = FakeGguf::new(3, 10);
        file.string("general.architecture", "llama");
        file.u32("llama.block_count", 30);
        file.u32("llama.embedding_length", 576);
        file.u32("llama.attention.head_count", 9);
        file.u32("llama.attention.head_count_kv", 3);

        let path = write_temp("truncated", "truncated.gguf", file.bytes());
        let facts = probe_gguf(&path).expect("facts before the truncation");
        assert_eq!(facts.block_count, 30);
        assert_eq!(facts.n_kv_heads, 3);
        cleanup(&path);
    }
}
