//! attune-estimate — performance prediction (M3).
//!
//! Slice 1 ships the roofline **decode** core: parse a model size + quant,
//! combine it with measured bandwidth, and report a predicted decode rate
//! plus the current bottleneck classification. Later slices add prefill,
//! calibration and confidence intervals, then compute the memory/compute
//! ridge properly.
//!
//! The JSON contract is [`report::EstimateReport`]; evolve it with
//! `serde(default)` like `attune-hal` / `attune-bench`, never by silently
//! retyping fields.

pub mod error;
pub mod model;
pub mod report;
pub mod roofline;

pub use error::EstimateError;
pub use model::{parse_params, ModelSpec, Quant};
pub use report::{Bottleneck, EstimateReport};
pub use roofline::{predict_decode_tps, HwInputs};

/// Schema stamped on every [`EstimateReport`] so consumers can tell a
/// slice-1 (M3) document from anything older.
pub const SCHEMA: &str = "attune-estimate/1";
