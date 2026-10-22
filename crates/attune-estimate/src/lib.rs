//! attune-estimate — performance prediction (M3).
//!
//! Slice 1 ships the roofline **decode** core: parse a model size + quant,
//! combine it with measured bandwidth, and report a predicted decode rate.
//! Slice 2 adds **prefill / TTFT**, the memory/compute **ridge** (`AI* =
//! FLOPS / BW`) and **bottleneck classification**. Later slices add
//! calibration, confidence intervals and multi-config sweeps.
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
pub use calibrate::{apply_decode, apply_prefill, fit_eta, CalibratedInputs, MAX_ETA, MIN_ETA};
pub use error::EstimateError;
pub use model::{parse_params, ModelSpec, Quant, BITS_PER_WEIGHT};
pub use prefill::predict_prefill_ms;
pub use report::{Bottleneck, CalibrationReport, EstimateInputs, EstimateReport};
pub use roofline::{predict_decode_tps, HwInputs};

/// Schema stamped on every [`EstimateReport`] so consumers can tell a
/// slice-1 (M3) document from anything older.
pub const SCHEMA: &str = "attune-estimate/1";
