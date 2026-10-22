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
    /// A hardware input is missing, non-finite or non-positive.
    #[error("invalid hardware inputs: {0}")]
    BadHw(String),
}
