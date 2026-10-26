//! attune-estimate — performance prediction (M3).
//!
//! Slice 1 ships the roofline **decode** core: parse a model size + quant,
//! combine it with measured bandwidth, and report a predicted decode rate.
//! Slice 2 adds **prefill / TTFT**, the memory/compute **ridge** (`AI* =
//! FLOPS / BW`) and **bottleneck classification**. Slice 3 adds **calibration**
//! from a measured bench. Slice 4 closes M3 with **confidence intervals**
//! derived from the measured bandwidth spread plus the **multi-config** report
//! contract.
//!
//! The JSON contract is [`report::EstimateReport`]; evolve it with
//! `serde(default)` like `attune-hal` / `attune-bench`, never by silently
//! retyping fields.

pub mod bottleneck;
pub mod calibrate;
pub mod error;
pub mod model;
pub mod prefill;
pub mod report;
pub mod roofline;

pub use bottleneck::{classify, decode_ai, prefill_ai, ridge_ai};
pub use calibrate::{
    apply_decode, apply_prefill, clamped, fit_eta, CalibratedInputs, MAX_ETA, MIN_ETA,
};
pub use error::EstimateError;
pub use model::{
    kv_bytes_per_token, kv_total_bytes, parse_arch, parse_model, parse_params, parse_quants,
    probe_gguf, probe_gguf_file, Arch, GgufFacts, GgufProbe, ModelEntry, ModelSpec, Quant,
    ARCH_PRESETS, BITS_PER_WEIGHT, MAX_HEADER_BYTES, MODEL_CATALOG,
};
pub use prefill::{predict_prefill_ms, predict_prefill_ms_quant};
pub use report::{
    interval, spread_pct_from_stats, Bottleneck, CalibrationReport, Confidence, EstimateInputs,
    EstimateReport, MultiEstimateReport, RamFit,
};
pub use roofline::{predict_decode_tps, predict_decode_tps_with_kv, HwInputs};

/// Schema stamped on every [`EstimateReport`] so consumers can tell a
/// slice-1 (M3) document from anything older.
pub const SCHEMA: &str = "attune-estimate/1";
