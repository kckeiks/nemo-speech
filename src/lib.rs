//! In-process ASR using NVIDIA's official NeMo-Speech.cpp C ABI.
//!
//! Load a trusted SDK once, share a [`Recognizer`] across threads, and create
//! each [`Stream`] on the blocking thread that will drive it. Streams are
//! deliberately neither `Send` nor `Sync`, matching the native SDK contract.
//! Native calls may block: do not run them on an async executor's worker threads.
//!
//! No SDK is needed to compile this crate. Loading native code is the single
//! unsafe entry point; model, stream, result, and library lifetimes are owned.
#![deny(unsafe_op_in_unsafe_fn)]
#![doc = include_str!("../README.md")]

mod config;
mod result;
mod stream;

pub use config::{
    BatchingConfig, Device, RecognitionOptions, RecognizerConfig, SampleRate, StreamingConfig,
};
pub use nemo_speech_sys::UPSTREAM_REVISION;
pub use result::{Alternative, RecognitionResult, Word};
pub use stream::Stream;

use std::ffi::{CStr, CString};
use std::mem::size_of;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use nemo_speech_sys as sys;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Load(#[from] sys::LoadError),
    #[error("invalid {field}: {message}")]
    InvalidArgument {
        field: &'static str,
        message: &'static str,
    },
    #[error("{operation} failed with native status {status}: {message}")]
    Native {
        operation: &'static str,
        status: u32,
        message: String,
    },
    #[error("native ABI violation in {operation}: {message}")]
    Protocol {
        operation: &'static str,
        message: &'static str,
    },
    #[error("native {field} is not UTF-8: {source}")]
    Utf8 {
        field: &'static str,
        source: std::str::Utf8Error,
    },
    #[error("stream is {state}; it cannot accept this operation")]
    StreamState { state: &'static str },
}

impl Error {
    pub(crate) fn invalid(field: &'static str, message: &'static str) -> Self {
        Self::InvalidArgument { field, message }
    }
}

/// A loaded SDK. Clones retain the same library; recognizers and streams retain
/// it automatically even after the caller drops all runtime handles.
#[derive(Clone)]
pub struct Runtime {
    api: Arc<sys::Api>,
}

impl Runtime {
    /// Load an explicitly selected, trusted ASR shared library.
    ///
    /// # Safety
    /// The library and its dependencies execute native code. They must implement
    /// the pinned NeMo ASR v1 ABI (or an ABI-compatible revision), including its
    /// pointer validity, ownership and threading contracts. Matching symbol names
    /// or a version string alone cannot establish this. Use an absolute path to
    /// a verified SDK built from [`UPSTREAM_REVISION`].
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: the caller supplies a trusted, ABI-compatible native library.
        let api = unsafe { sys::Api::load(path)? };
        Ok(Self { api: Arc::new(api) })
    }

    /// Copy the native runtime version string. This is diagnostic information,
    /// not proof of ABI compatibility or of GPU operator coverage.
    pub fn version(&self) -> Result<String> {
        // SAFETY: the trusted API returns a NUL-terminated runtime-owned string.
        unsafe { copy_string((self.api.nemo_speech_asr_version)(), "version") }
    }

    pub fn create_recognizer(&self, config: &RecognizerConfig) -> Result<Recognizer> {
        let path = model_path(&config.model_path)?;
        let gpu = match config.device {
            Device::Cpu => -1,
            Device::Gpu(index) => config::native_int("gpu", index, false)?,
        };
        let backend = sys::nemo_speech_asr_backend_config {
            size: size_of::<sys::nemo_speech_asr_backend_config>(),
            gpu,
        };
        let model = sys::nemo_speech_asr_model_config {
            size: size_of::<sys::nemo_speech_asr_model_config>(),
            path: path.as_ptr(),
            ..Default::default()
        };
        let streaming = config
            .streaming
            .map(|s| -> Result<_> {
                s.validate()?;
                Ok(sys::nemo_speech_asr_streaming_config {
                    size: size_of::<sys::nemo_speech_asr_streaming_config>(),
                    chunk_size: s.chunk_size_seconds,
                    ctc_left_padding: s.ctc_left_padding_seconds,
                    ctc_right_padding: s.ctc_right_padding_seconds,
                    rnnt_right_context: s.rnnt_right_context,
                })
            })
            .transpose()?;
        let batching = config
            .batching
            .map(|b| -> Result<_> {
                Ok(sys::nemo_speech_asr_batching_config {
                    size: size_of::<sys::nemo_speech_asr_batching_config>(),
                    enable: true,
                    max_batch_size: config::native_int("max_batch_size", b.max_batch_size, true)?,
                    max_queue_delay_us: config::native_int(
                        "max_queue_delay_us",
                        b.max_queue_delay_us,
                        false,
                    )?,
                    max_queue_depth: config::native_int(
                        "max_queue_depth",
                        b.max_queue_depth,
                        true,
                    )?,
                    ingress_cohort_delay_us: 0, // Transport coordination belongs to the application.
                    state_arena_slots: config::native_int(
                        "state_arena_slots",
                        b.state_arena_slots,
                        true,
                    )?,
                })
            })
            .transpose()?;
        let cfg = sys::nemo_speech_asr_recognizer_config {
            size: size_of::<sys::nemo_speech_asr_recognizer_config>(),
            backend: &backend,
            model: &model,
            streaming: streaming.as_ref().map_or(ptr::null(), |s| s),
            batching: batching.as_ref().map_or(ptr::null(), |b| b),
            ..Default::default()
        };
        let mut handle = ptr::null_mut();
        // SAFETY: all sized config structs and C strings live through this call;
        // the native create function copies them and writes an owned handle.
        let status = unsafe { (self.api.nemo_speech_asr_create)(&cfg, &mut handle) };
        if let Err(error) = check_status(&self.api, status, "create") {
            if !handle.is_null() {
                // SAFETY: an unexpected returned handle still belongs to this call.
                unsafe { (self.api.nemo_speech_asr_destroy)(handle) };
            }
            return Err(error);
        }
        let handle = NonNull::new(handle).ok_or(Error::Protocol {
            operation: "create",
            message: "success returned a null recognizer",
        })?;
        Ok(Recognizer {
            inner: Arc::new(RecognizerInner {
                handle,
                api: Arc::clone(&self.api),
            }),
        })
    }
}

/// One shared model. Clone this handle to create independent streams on
/// dedicated blocking threads; cloning does not reload weights.
#[derive(Clone)]
pub struct Recognizer {
    pub(crate) inner: Arc<RecognizerInner>,
}

pub(crate) struct RecognizerInner {
    pub(crate) handle: NonNull<sys::nemo_speech_asr_recognizer>,
    pub(crate) api: Arc<sys::Api>,
}

// SAFETY: the pinned SDK explicitly supports independent concurrent calls on
// a shared recognizer (docs/sdk.md and src/asr/recognizer.h). Arc retains it
// until all calls and streams finish; stream handles are never shared.
unsafe impl Send for RecognizerInner {}
// SAFETY: same documented shared-recognizer contract as the Send implementation.
unsafe impl Sync for RecognizerInner {}

impl Drop for RecognizerInner {
    fn drop(&mut self) {
        // SAFETY: the last Arc is gone, so no calls or streams can still use it.
        unsafe { (self.api.nemo_speech_asr_destroy)(self.handle.as_ptr()) };
    }
}

impl Recognizer {
    pub fn create_stream(&self, options: &RecognitionOptions) -> Result<Stream> {
        Stream::create(Arc::clone(&self.inner), options)
    }

    /// Blocking whole-utterance recognition. Prefer streaming for live audio.
    pub fn recognize(
        &self,
        samples: &[f32],
        rate: SampleRate,
        options: &RecognitionOptions,
    ) -> Result<RecognitionResult> {
        validate_audio(samples)?;
        let options = NativeOptions::new(&self.inner.api, options)?;
        let mut handle = ptr::null_mut();
        // SAFETY: self retains the model/library; audio and options are valid
        // through the synchronous call, and the native result is caller-owned.
        let status = unsafe {
            (self.inner.api.nemo_speech_asr_recognize_f32)(
                self.inner.handle.as_ptr(),
                &options.raw,
                samples.as_ptr(),
                samples.len(),
                rate.raw(),
                &mut handle,
            )
        };
        let checked = check_status(&self.inner.api, status, "recognize_f32");
        let result = result::NativeResult::new(&self.inner.api, handle);
        checked?;
        result
            .ok_or(Error::Protocol {
                operation: "recognize_f32",
                message: "success returned a null result",
            })?
            .copy()
    }
}

pub(crate) struct NativeOptions {
    pub(crate) raw: sys::nemo_speech_asr_recognition_options,
    _language: Option<CString>,
}

impl NativeOptions {
    pub(crate) fn new(api: &sys::Api, options: &RecognitionOptions) -> Result<Self> {
        let language = options
            .language
            .as_deref()
            .map(|s| cstring(s.as_bytes(), "language"))
            .transpose()?;
        let max_alternatives =
            config::native_int("max_alternatives", options.max_alternatives, true)?;
        // SAFETY: a by-value POD with native defaults and its size populated.
        let mut raw = unsafe { (api.nemo_speech_asr_recognition_options_default)() };
        raw.size = size_of::<sys::nemo_speech_asr_recognition_options>();
        raw.language_code = language.as_ref().map_or(ptr::null(), |s| s.as_ptr());
        raw.enable_word_time_offsets = options.word_time_offsets;
        raw.verbatim_transcripts = options.verbatim_transcripts;
        raw.max_alternatives = max_alternatives;
        Ok(Self {
            raw,
            _language: language,
        })
    }
}

pub(crate) fn check_status(
    api: &sys::Api,
    status: sys::nemo_speech_asr_status,
    operation: &'static str,
) -> Result<()> {
    if status == sys::NEMO_SPEECH_ASR_OK {
        return Ok(());
    }
    // SAFETY: native TLS error string is valid until the next API call on this
    // thread. Copy it now, before invoking cleanup or any result accessors.
    let message = unsafe {
        let ptr = (api.nemo_speech_asr_last_error)();
        if ptr.is_null() {
            "native runtime returned no error message".into()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(Error::Native {
        operation,
        status,
        message,
    })
}

pub(crate) unsafe fn copy_string(
    ptr: *const std::ffi::c_char,
    field: &'static str,
) -> Result<String> {
    if ptr.is_null() {
        return Err(Error::Protocol {
            operation: field,
            message: "null string pointer",
        });
    }
    // SAFETY: caller guarantees a live NUL-terminated native string.
    let bytes = unsafe { CStr::from_ptr(ptr) };
    bytes
        .to_str()
        .map(str::to_owned)
        .map_err(|source| Error::Utf8 { field, source })
}

pub(crate) fn validate_audio(samples: &[f32]) -> Result<()> {
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(Error::invalid(
            "audio",
            "samples must be finite mono float32 values",
        ));
    }
    Ok(())
}

fn cstring(bytes: &[u8], field: &'static str) -> Result<CString> {
    CString::new(bytes).map_err(|_| Error::invalid(field, "contains an embedded NUL"))
}

fn model_path(path: &Path) -> Result<CString> {
    if path.as_os_str().is_empty() {
        return Err(Error::invalid("model_path", "must not be empty"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        cstring(path.as_os_str().as_bytes(), "model_path")
    }
    #[cfg(not(unix))]
    {
        cstring(
            path.to_str()
                .ok_or(Error::invalid(
                    "model_path",
                    "must be representable as UTF-8 on this platform",
                ))?
                .as_bytes(),
            "model_path",
        )
    }
}
