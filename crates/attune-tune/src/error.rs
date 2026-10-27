//! Errors raised while expanding or ranking a tune sweep.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TuneError {
    /// A sweep point has a zero or otherwise unusable field.
    #[error("invalid tune config: {0}")]
    InvalidConfig(String),
    /// The caller asked to rank an empty grid. Ranking nothing is a caller
    /// bug, not a valid "no configs fit" result.
    #[error("tune grid is empty")]
    EmptyGrid,
    /// The objective string is not one of the documented names.
    #[error("invalid objective: {0} (expected throughput, latency or balanced)")]
    InvalidObjective(String),
    /// The pure roofline math rejected an input (zero parameters, bad
    /// bandwidth/FLOPS, ...). Propagated unchanged so the CLI can print the
    /// actionable `attune-estimate` message.
    #[error(transparent)]
    Estimate(#[from] attune_estimate::EstimateError),
}
