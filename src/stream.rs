use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::Arc;

use nemo_speech_sys as sys;

use crate::result::NativeResult;
use crate::{
    check_status, validate_audio, Error, NativeOptions, RecognitionOptions, RecognitionResult,
    RecognizerInner, Result, SampleRate,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Open,
    Finished,
    Drained,
    Failed,
}

/// One independent blocking recognition session, driven on its creating thread.
///
/// After each push, pull results until [`next_result`](Self::next_result) returns
/// `None`. Call [`finish`](Self::finish) at EOF, then drain again for the final.
/// A final result may also mark an utterance endpoint before EOF.
///
/// Native operation failures poison the stream: its internal state may already
/// have changed. Drop it and open a new stream. Input validation failures leave
/// it usable. Dropping retains no pending audio or results.
///
/// Streams cannot be moved to another thread:
/// ```compile_fail
/// fn move_stream(stream: nemo_speech::Stream) {
///     std::thread::spawn(move || drop(stream));
/// }
/// ```
pub struct Stream {
    owner: Arc<RecognizerInner>,
    handle: NonNull<sys::nemo_speech_asr_stream>,
    state: State,
    rate: Option<SampleRate>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Stream {
    pub(crate) fn create(
        owner: Arc<RecognizerInner>,
        options: &RecognitionOptions,
    ) -> Result<Self> {
        let options = NativeOptions::new(&owner.api, options)?;
        let mut handle = ptr::null_mut();
        // SAFETY: the owner retains the recognizer/library, and native code
        // copies the options during this synchronous call.
        let status = unsafe {
            (owner.api.nemo_speech_asr_streaming_recognize)(
                owner.handle.as_ptr(),
                &options.raw,
                &mut handle,
            )
        };
        if let Err(error) = check_status(&owner.api, status, "streaming_recognize") {
            if !handle.is_null() {
                // SAFETY: any returned stream is owned by this call.
                unsafe { (owner.api.nemo_speech_asr_stream_close)(handle) };
            }
            return Err(error);
        }
        let handle = NonNull::new(handle).ok_or(Error::Protocol {
            operation: "streaming_recognize",
            message: "success returned a null stream",
        })?;
        Ok(Self {
            owner,
            handle,
            state: State::Open,
            rate: None,
            _thread_bound: PhantomData,
        })
    }

    /// Buffer finite mono float32 samples; decoding is driven by `next_result`.
    /// The explicit sample rate cannot change after the first nonempty push.
    /// Empty pushes are no-ops while open.
    pub fn push_audio(&mut self, samples: &[f32], rate: SampleRate) -> Result<()> {
        self.require_open()?;
        validate_audio(samples)?;
        if samples.is_empty() {
            return Ok(());
        }
        if self.rate.is_some_and(|previous| previous != rate) {
            return Err(Error::invalid(
                "sample_rate",
                "cannot change within a stream",
            ));
        }
        // SAFETY: mutable access serializes this live stream. Native code copies
        // the samples before returning; the model and library remain alive.
        let status = unsafe {
            (self.owner.api.nemo_speech_asr_stream_push_f32)(
                self.handle.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                rate.raw(),
            )
        };
        self.check(status, "stream_push_f32")?;
        self.rate = Some(rate);
        Ok(())
    }

    /// Pull one owned result, driving inference synchronously. While open,
    /// `None` means more audio is needed. After finish, it means fully drained.
    pub fn next_result(&mut self) -> Result<Option<RecognitionResult>> {
        if self.state == State::Failed {
            return Err(Error::StreamState { state: "failed" });
        }
        if self.state == State::Drained {
            return Ok(None);
        }
        let mut handle = ptr::null_mut();
        // SAFETY: this stream is live and exclusively borrowed; out receives a
        // separately owned result or NULL, retained by a guard below.
        let status = unsafe {
            (self.owner.api.nemo_speech_asr_stream_next)(self.handle.as_ptr(), &mut handle)
        };
        // Copy TLS error before the result guard can perform native cleanup.
        let checked = check_status(&self.owner.api, status, "stream_next");
        let native = NativeResult::new(&self.owner.api, handle);
        if let Err(error) = checked {
            self.state = State::Failed;
            return Err(error);
        }
        match native {
            Some(result) => match result.copy() {
                Ok(result) => Ok(Some(result)),
                Err(error) => {
                    self.state = State::Failed;
                    Err(error)
                }
            },
            None => {
                if self.state == State::Finished {
                    self.state = State::Drained;
                }
                Ok(None)
            }
        }
    }

    /// Flush the audio tail synchronously. Call `next_result` to retrieve the
    /// final. Repeated successful finish calls are no-ops and retain that final.
    pub fn finish(&mut self) -> Result<()> {
        match self.state {
            State::Finished | State::Drained => return Ok(()),
            State::Failed => return Err(Error::StreamState { state: "failed" }),
            State::Open => {}
        }
        // SAFETY: exclusive access to a live open stream with its owner retained.
        let status =
            unsafe { (self.owner.api.nemo_speech_asr_stream_finish)(self.handle.as_ptr()) };
        self.check(status, "stream_finish")?;
        self.state = State::Finished;
        Ok(())
    }

    /// Request an utterance endpoint, honored by the next inference pull.
    /// This does not close the stream; additional audio may follow.
    pub fn force_endpoint(&mut self) -> Result<()> {
        self.require_open()?;
        // SAFETY: exclusive access to a live open stream.
        let status =
            unsafe { (self.owner.api.nemo_speech_asr_stream_force_endpoint)(self.handle.as_ptr()) };
        self.check(status, "stream_force_endpoint")
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, State::Finished | State::Drained)
    }

    /// Destroy this stream and discard pending audio. This is equivalent to
    /// dropping it; it does not interrupt a native call already in progress.
    pub fn close(self) {}

    fn require_open(&self) -> Result<()> {
        match self.state {
            State::Open => Ok(()),
            State::Finished | State::Drained => Err(Error::StreamState { state: "finished" }),
            State::Failed => Err(Error::StreamState { state: "failed" }),
        }
    }

    fn check(
        &mut self,
        status: sys::nemo_speech_asr_status,
        operation: &'static str,
    ) -> Result<()> {
        check_status(&self.owner.api, status, operation).inspect_err(|_| self.state = State::Failed)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: exclusively owned stream is closed once, before the owner
        // Arc is released. This runs on the same thread that created it.
        unsafe { (self.owner.api.nemo_speech_asr_stream_close)(self.handle.as_ptr()) };
    }
}
