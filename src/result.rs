use std::ptr::NonNull;

use nemo_speech_sys as sys;

use crate::{copy_string, Result};

/// Owned recognition output. No strings borrow the native result or stream.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionResult {
    pub is_final: bool,
    /// Duration of audio processed by the native recognizer, in seconds.
    pub audio_processed_seconds: f32,
    /// Native 1-based input channel; mono recognition currently returns 1.
    pub channel_tag: i32,
    /// Ranked best first. Greedy decoding currently returns one alternative.
    pub alternatives: Vec<Alternative>,
}

impl RecognitionResult {
    pub fn transcript(&self) -> Option<&str> {
        self.alternatives.first().map(|a| a.transcript.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alternative {
    pub transcript: String,
    /// Native score, copied without normalization or calibration.
    pub confidence: f32,
    /// Present when word offsets are requested and supported by the model.
    pub words: Vec<Word>,
    pub language_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_time_ms: i32,
    pub end_time_ms: i32,
    pub confidence: f32,
    /// Native speaker tag: 0 = untagged, otherwise 1-based.
    pub speaker_tag: i32,
}

/// Private RAII guard, including when copying a malformed result fails.
pub(crate) struct NativeResult<'a> {
    api: &'a sys::Api,
    handle: NonNull<sys::nemo_speech_asr_result>,
}

impl<'a> NativeResult<'a> {
    pub(crate) fn new(api: &'a sys::Api, handle: *mut sys::nemo_speech_asr_result) -> Option<Self> {
        NonNull::new(handle).map(|handle| Self { api, handle })
    }

    pub(crate) fn copy(&self) -> Result<RecognitionResult> {
        let api = self.api;
        let handle = self.handle.as_ptr();
        // SAFETY: the guard owns a live result and retains the library. Counts
        // bound every accessor index; native strings remain valid until Drop.
        unsafe {
            let count = (api.nemo_speech_asr_result_alternative_count)(handle);
            let mut alternatives = Vec::with_capacity(count);
            for alt in 0..count {
                let transcript = copy_string(
                    (api.nemo_speech_asr_result_transcript)(handle, alt),
                    "transcript",
                )?;
                let word_count = (api.nemo_speech_asr_result_word_count)(handle, alt);
                let mut words = Vec::with_capacity(word_count);
                for i in 0..word_count {
                    words.push(Word {
                        text: copy_string(
                            (api.nemo_speech_asr_result_word_text)(handle, alt, i),
                            "word",
                        )?,
                        start_time_ms: (api.nemo_speech_asr_result_word_start_time)(handle, alt, i),
                        end_time_ms: (api.nemo_speech_asr_result_word_end_time)(handle, alt, i),
                        confidence: (api.nemo_speech_asr_result_word_confidence)(handle, alt, i),
                        speaker_tag: (api.nemo_speech_asr_result_word_speaker_tag)(handle, alt, i),
                    });
                }
                let language_count = (api.nemo_speech_asr_result_language_count)(handle, alt);
                let mut language_codes = Vec::with_capacity(language_count);
                for i in 0..language_count {
                    language_codes.push(copy_string(
                        (api.nemo_speech_asr_result_language_code)(handle, alt, i),
                        "language_code",
                    )?);
                }
                alternatives.push(Alternative {
                    transcript,
                    confidence: (api.nemo_speech_asr_result_confidence)(handle, alt),
                    words,
                    language_codes,
                });
            }
            Ok(RecognitionResult {
                is_final: (api.nemo_speech_asr_result_is_final)(handle),
                audio_processed_seconds: (api.nemo_speech_asr_result_audio_processed)(handle),
                channel_tag: (api.nemo_speech_asr_result_channel_tag)(handle),
                alternatives,
            })
        }
    }
}

impl Drop for NativeResult<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard owns the result exclusively and destroys it once.
        unsafe { (self.api.nemo_speech_asr_result_destroy)(self.handle.as_ptr()) };
    }
}
