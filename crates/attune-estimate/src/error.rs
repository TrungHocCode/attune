//! Errors raised while parsing a model spec or predicting performance.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EstimateError {
    /// The model size string could not be parsed, or was zero/out of range.
    #[error("invalid model size: {0}")]
    InvalidModel(String),
    /// The quantization label is not one of the supported names.
    #[error("invalid quantization: {0}")]
    InvalidQuant(String),
    /// The architecture preset name is not one of the documented names.
    #[error("invalid architecture: {0}")]
    InvalidArch(String),
    /// The model name is not one of the verified catalog entries.
    #[error("invalid catalog model: {0}")]
    InvalidModelName(String),
    /// A GGUF file could not be opened or its header is malformed/unsupported.
    #[error("invalid GGUF: {0}")]
    InvalidGguf(String),
    /// An online `config.json` fetch failed (timeout, DNS, TLS, non-2xx).
    #[error("config fetch failed: {0}")]
    FetchConfig(String),
    /// A fetched or cached `config.json` lacks usable architecture fields.
    #[error("invalid model config: {0}")]
    InvalidConfig(String),
    /// The on-disk catalog cache could not be written.
    #[error("catalog cache error: {0}")]
    Cache(String),
    /// A hardware input is missing, non-finite or non-positive.
    #[error("invalid hardware inputs: {0}")]
    BadHw(String),
    /// A calibration ratio could not be computed from the supplied values.
    #[error("invalid calibration input: {0}")]
    InvalidCalibration(String),
    /// Calibration was requested but the bench report has no usable measured
    /// data (missing/partial `e2e` block). Expected condition, not a bug.
    #[error("calibration unavailable: {0}")]
    CalibrationUnavailable(String),
}
