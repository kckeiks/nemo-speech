//! Owned configuration. No caller-owned pointers escape a native creation call.

use std::path::PathBuf;

use crate::{Error, Result};

/// Native backend selection. A GPU index uses the installed SDK's GPU backend
/// (CUDA when using a CUDA SDK). This does not certify per-op GPU coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Explicit CPU execution, useful for validation without a GPU.
    #[default]
    Cpu,
    /// GPU index. Creation fails if the native runtime cannot select that GPU.
    Gpu(u32),
}

/// Streaming geometry. Defaults match the pinned native SDK. This wrapper
/// bounds the CTC window to 60 seconds, chunk size to at least 1 ms and RNNT
/// lookahead to 1024 frames to avoid unchecked native conversions/zero steps.
#[derive(Debug, Clone, Copy)]
pub struct StreamingConfig {
    pub chunk_size_seconds: f32,
    pub ctc_left_padding_seconds: f32,
    pub ctc_right_padding_seconds: f32,
    /// RNNT lookahead in encoder frames. `6` selects the Nemotron 560 ms
    /// operating point; `-1` preserves the value stored in the model. Native
    /// model support determines which nonnegative values are valid.
    pub rnnt_right_context: i32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size_seconds: 0.16,
            ctc_left_padding_seconds: 1.92,
            ctc_right_padding_seconds: 1.92,
            rnnt_right_context: 1,
        }
    }
}

impl StreamingConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.chunk_size_seconds.is_finite() || self.chunk_size_seconds < 0.001 {
            return Err(Error::invalid(
                "chunk_size_seconds",
                "must be finite and at least 0.001 seconds",
            ));
        }
        for (field, value) in [
            ("ctc_left_padding_seconds", self.ctc_left_padding_seconds),
            ("ctc_right_padding_seconds", self.ctc_right_padding_seconds),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(field, "must be finite and nonnegative"));
            }
        }
        let window = f64::from(self.chunk_size_seconds)
            + f64::from(self.ctc_left_padding_seconds)
            + f64::from(self.ctc_right_padding_seconds);
        if window > 60.0 {
            return Err(Error::invalid(
                "streaming_window",
                "chunk plus padding must not exceed 60 seconds",
            ));
        }
        if !(-1..=1024).contains(&self.rnnt_right_context) {
            return Err(Error::invalid(
                "rnnt_right_context",
                "must be between -1 and 1024 frames",
            ));
        }
        Ok(())
    }
}

/// Opt-in neural batching across independent concurrent calls on one recognizer.
/// This bounds native stage queues; the application must also bound ingress.
#[derive(Debug, Clone, Copy)]
pub struct BatchingConfig {
    pub max_batch_size: u32,
    pub max_queue_delay_us: u32,
    pub max_queue_depth: u32,
    /// Provision for the maximum concurrent stateful streams, independently of
    /// `max_batch_size`. Exhaustion can wait up to 60 seconds before a native
    /// error; keep application admission below this capacity.
    pub state_arena_slots: u32,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_queue_delay_us: 5_000,
            max_queue_depth: 2_048,
            state_arena_slots: 32,
        }
    }
}

/// Startup configuration for one shared model. Optional configurations preserve
/// native defaults when absent. Batching is disabled when `batching` is `None`.
#[derive(Debug, Clone)]
pub struct RecognizerConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub streaming: Option<StreamingConfig>,
    pub batching: Option<BatchingConfig>,
}

impl RecognizerConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            device: Device::Cpu,
            streaming: None,
            batching: None,
        }
    }
}

/// Recognition options supported by this focused safe wrapper. For additional
/// native capabilities use the sys crate. Words are populated only when the
/// native model supports offsets; greedy models currently return one alternative.
#[derive(Debug, Clone)]
pub struct RecognitionOptions {
    /// Native language/prompt selection. `None` preserves native behavior;
    /// `Some("auto")` requests detection only on models with an auto prompt.
    pub language: Option<String>,
    pub word_time_offsets: bool,
    pub verbatim_transcripts: bool,
    pub max_alternatives: u32,
}

impl Default for RecognitionOptions {
    fn default() -> Self {
        Self {
            language: None,
            word_time_offsets: false,
            verbatim_transcripts: false,
            max_alternatives: 1,
        }
    }
}

/// Supported input sample rate. Explicit rates avoid ambiguity about model rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRate(i32);

impl SampleRate {
    pub const HZ_16000: Self = Self(16_000);

    pub fn new(hz: u32) -> Result<Self> {
        if !(8_000..=96_000).contains(&hz) {
            return Err(Error::invalid(
                "sample_rate",
                "must be between 8000 and 96000 Hz",
            ));
        }
        Ok(Self(hz as i32))
    }

    pub fn hz(self) -> u32 {
        self.0 as u32
    }

    pub(crate) fn raw(self) -> i32 {
        self.0
    }
}

pub(crate) fn native_int(field: &'static str, value: u32, positive: bool) -> Result<i32> {
    if positive && value == 0 {
        return Err(Error::invalid(field, "must be positive"));
    }
    i32::try_from(value)
        .map_err(|_| Error::invalid(field, "exceeds the native signed 32-bit limit"))
}
