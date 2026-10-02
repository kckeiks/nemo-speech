# nemo-speech

Safe, in-process ASR bindings for NVIDIA's official
[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/4c101bc7113f49101a3e11d2c994c519f41939f6).
This crate wraps its ASR C ABI, alongside the raw
[nemo-speech-sys](https://github.com/kckeiks/nemo-speech-sys) crate. It targets revision
`4c101bc7113f49101a3e11d2c994c519f41939f6` (runtime version `0.1.0`).
It is separate from the community `nemotron-asr.cpp` runtime.

Normal Cargo builds need no native SDK, CUDA toolkit, bindgen, Clang, model, or
network access. Inference loads an explicitly selected native library. There is
no sidecar or native server dependency. Application servers select and integrate
the backend separately.

## API and ownership

`Runtime` retains the loaded SDK. `Recognizer` owns one shared model and is
`Send + Sync`; clone it to create independent streams on blocking threads.
`Stream` is deliberately neither `Send` nor `Sync`. Create, drive and destroy
each stream on the same thread, following the pinned SDK contract. Streams
retain their recognizer and SDK, so dropping the application's runtime or model
handle does not invalidate active streams. Returned transcripts, alternatives,
words and language codes are owned Rust values. Native result handles are
destroyed immediately after copying, including on errors.

```rust,no_run
use nemo_speech::{Device, RecognitionOptions, RecognizerConfig, Runtime,
    SampleRate, StreamingConfig};

// SAFETY: the operator verified this SDK and its dependencies implement the
// pinned ASR ABI and its ownership/threading contracts.
let runtime = unsafe {
    Runtime::load("/opt/nemo-speech/lib/libnemo_speech_asr_c.so.1")?
};
let mut config = RecognizerConfig::new("/models/nemotron-streaming.gguf");
config.device = Device::Gpu(0); // CPU is the default; unavailable GPU is an error.
config.streaming = Some(StreamingConfig {
    rnnt_right_context: 6, // Nemotron streaming 560 ms operating point.
    ..Default::default()
});
let model = runtime.create_recognizer(&config)?;
let mut stream = model.create_stream(&RecognitionOptions::default())?;
let pcm_f32 = vec![0.0; 8_960]; // Replace with mono 16 kHz float32 audio.
stream.push_audio(&pcm_f32, SampleRate::HZ_16000)?;
while let Some(result) = stream.next_result()? {
    println!("{:?}", result.transcript());
}
stream.finish()?;
while let Some(result) = stream.next_result()? {
    println!("final={}: {:?}", result.is_final, result.transcript());
}
# Ok::<(), nemo_speech::Error>(())
```

`push_audio` buffers audio; `next_result` and `finish` can perform inference and
block. While open, a `None` pull means more audio is needed. Drain after each
push, finish once at EOF, then drain for the final. The wrapper makes repeated
successful finish calls no-ops to preserve the pending final, and rejects
subsequent pushes or endpoint requests. `is_final` may also mark an utterance
endpoint while the stream remains open. `force_endpoint` requests that boundary
on the next pull. Closing/dropping discards pending audio and does **not**
interrupt an in-flight call; the C ABI has no ASR cancellation operation.

Native failures poison a stream because the native operation may have partially
changed its state. Reopen it to continue. Argument validation errors leave it
usable. Error codes and thread-local messages are copied before cleanup; unknown
future status codes remain representable. Finite audio, fixed stream rate,
embedded NULs, configuration ranges and native integer limits are checked before
calling the SDK. Explicit sample rates cover 8–96 kHz; the SDK resamples them.
The wrapper bounds CTC chunks to at least 1 ms, total chunk plus padding to 60
seconds, and RNNT lookahead to -1 through 1024 frames to avoid unchecked native
conversions and zero-step loops. Model support further limits valid lookahead.

The safe API exposes CPU/GPU selection, streaming geometry, neural batching,
language selection, word offsets, verbatim transcripts, and N-best
requests/results. Greedy decoding currently returns one alternative. Word times
are signed milliseconds, processed audio is seconds, and speaker tags are copied
without interpreting untagged zero. Additional native settings, including VAD,
beam decoding, context biasing, diarization and postprocessing models, remain
available through the raw crate. The pinned implementation ignores request IDs
and the interim-results flag, so the safe wrapper does not expose them as
functioning settings.

## Install a compatible SDK

The native SDK is installed separately from the Rust packages. Cargo builds the
bindings; it does not build or download NeMo-Speech.cpp. The current setup below
builds the pinned native runtime from source and installs it with CMake.

Choose a persistent installation directory, such as a project-local
`native/nemo-speech/` directory or an application-managed installation prefix.
There is **no required `/tmp` path**: validation used a temporary SDK
installation prefix. CMake's install step places the SDK's
libraries together; users do not copy individual dependencies into a temporary
file. Keep that installation intact and pass the absolute path of its
`lib/libnemo_speech_asr_c.so.1` to `Runtime::load`. The NVIDIA driver and required
CUDA runtime libraries must also be available on the target machine. A compatible
GGUF model is supplied separately.

Loading native code is the one unsafe API entry point. Supply an **absolute path
to a trusted library** built at the pin, with its dependencies installed.
Symbol presence, the path and the version string cannot prove ABI compatibility.
Use ordinary C enum layout; do not build with `-fshort-enums`. A default-options
struct is returned by value; future struct growth must be reviewed rather than
assuming binary compatibility from append-only pointer-based configurations.

For Linux CUDA, prepare Git, CMake >= 3.26, Ninja, a supported C++/CUDA compiler,
the CUDA toolkit/driver and SentencePiece development files. Follow the pinned
[build guide](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4c101bc7113f49101a3e11d2c994c519f41939f6/docs/build.md)
for prerequisites. From a separate upstream checkout:

```sh
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git
cd NeMo-Speech.cpp
git checkout 4c101bc7113f49101a3e11d2c994c519f41939f6
# The pinned configure helper also checks llama.cpp's miniaudio header for ASR,
# even when microphone capture is disabled.
git submodule update --init ggml llama.cpp
# Prefer private, pinned static SentencePiece on Linux if not already available.
bash scripts/build_sentencepiece_static.sh
# The helper applies required ggml patches. No native server is needed.
bash scripts/configure.sh cuda-asr \
    -DNEMO_SPEECH_BUILD_CLI=OFF \
    -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
    -DNEMO_SPEECH_BUILD_DIAR=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build --preset cuda-asr --parallel 4
cmake --install build/cuda-asr --prefix "$PWD/install"
```

Architecture `89` is for the 4090; choose the deployment architecture. A CPU
validation build can use `cpu-asr` and omit the CUDA architecture definition.
Keep the **whole installed SDK** together, including its dependent libraries
and applicable license notices. The Linux entry point is
`install/lib/libnemo_speech_asr_c.so.1`; the similarly named
`libnemo_speech_asr.so` is the internal C++ implementation. Models and companion
assets are separate. The wrapper does not download or convert weights; use a
compatible official GGUF, not the current parakeet-rs ONNX files. Linux x86-64
is the validated host; Windows/macOS execution is unvalidated.

## Streaming smoke example

Use a known spoken sample as **mono PCM16LE, 16 kHz, without a WAV header**:

```sh
ffmpeg -i sample.wav -ac 1 -ar 16000 -f s16le sample.pcm
cargo run --release -p nemo-speech --example stream_pcm -- \
    /opt/nemo-speech/lib/libnemo_speech_asr_c.so.1 \
    /models/nemotron-streaming.gguf sample.pcm 0
```

The last argument is `cpu` or a GPU index. The example uses right context `6`
and reads 560 ms chunks. Select a compatible Nemotron streaming model. It is a
blocking transcription example; it does not pace real-time audio or measure
capacity. For other models, configure their supported geometry in application
code.

## Verification

Run these commands from this package's directory. Host tests compile C fixtures
with `CC` (default `cc`), without an SDK or GPU:

```sh
cargo test -p nemo-speech
cargo clippy -p nemo-speech --all-targets -- -D warnings
cargo fmt -p nemo-speech -- --check
```

These cover shared-model independent streams, retained ownership, result copying, null
successes, native failures, UTF-8 errors, integer/input validation, stream state
and exact cleanup counts. A compile-fail doctest enforces the stream thread
restriction. Run the raw package's tests separately for every public C struct
layout/field and enum and eager symbol loading; see its
README under "Regeneration and verification". The C fixture tests obtain the
pinned header through the resolved raw dependency; they do not read a sibling
repository's files.

Manual tests require an operator-selected, trusted SDK. Load/version and native
model-error smoke (no model needed):

```sh
export NEMO_SPEECH_LIBRARY=/opt/nemo-speech/lib/libnemo_speech_asr_c.so.1
cargo test -p nemo-speech --test native_sdk native_sdk_loads -- --ignored
# Only against a CPU-only SDK: confirms explicit GPU selection fails.
cargo test -p nemo-speech --test native_sdk native_cpu_sdk -- --ignored
```

For real transcript validation with two distinct independent concurrent streams:

```sh
export NEMO_SPEECH_MODEL=/models/nemotron-streaming.gguf
export NEMO_SPEECH_PCM=/samples/speech-one.pcm
export NEMO_EXPECTED_TEXT='a phrase exclusive to the first recording'
export NEMO_SPEECH_PCM_2=/samples/speech-two.pcm
export NEMO_EXPECTED_TEXT_2='a different phrase exclusive to the second'
export NEMO_GPU=0 # Omit for CPU.
export NEMO_REPEATS=8 # Optional: default 1, maximum 100 per worker.
cargo test --release -p nemo-speech --test native_sdk native_shared_model -- --ignored --nocapture
```

This test uses Nemotron right context `6`, enables batching on one shared model,
creates each stream on its driving thread, and checks its own reference phrase
and absence of the other stream's phrase in the owned finals after dropping
public model/runtime handles. Pick phrases exclusive to their recording.
Word offsets are also requested and checked after closing each native stream.
Repeated runs reuse the shared recognizer and recreate streams to exercise state
resource cleanup. The test prints each final transcript for inspection.

Alternatively, pass the same inputs to the package-local runner:

```sh
bash scripts/validate_native.sh \
    /opt/nemo-speech/lib/libnemo_speech_asr_c.so.1 /models/nemotron-streaming.gguf \
    /samples/speech-one.pcm 'first exclusive reference phrase' \
    /samples/speech-two.pcm 'second exclusive reference phrase' 0 8
```

Use absolute input paths; the last arguments are `cpu` or a GPU index and an
optional repeat count. Loading the selected SDK still requires trusting its
native code and ABI, as described above.

## Integrating with an application server

Keep transport, admission and scheduling in the application. Implement
a backend adapter with one shared recognizer and a bounded set of dedicated
blocking workers. Create streams inside their worker and route each session's
commands back to that same worker. A Tokio task may migrate threads between
polls; do not place a native stream directly in it or execute blocking inference
on async executor threads.

Neural batching is **opt-in** (`config.batching = Some(...)`) and requires
overlapping independent native calls. One global inference mutex or one serial
worker would suppress batching. The C ABI does not enable the native server's
ingress cohort coordination; the application owns ingress. Bound queued work and admitted
streams. Provision `state_arena_slots` for stateful stream capacity, separately
from batch size: native arena exhaustion can wait up to 60 seconds before an
error, so application admission must prevent it. `BatchingConfig::default()`
is a wrapper preset (batch size/state slots 32), not a capacity claim or the
upstream disabled batching defaults.

These bindings route inference through the chosen native SDK. They do not
certify that every operation, resampling step, decoder loop or text operation
runs on the GPU. Validate operator placement with native diagnostics, then
measure transcription accuracy, streaming latency and sustained concurrency
with representative audio and consistent real-time criteria and soak durations.
Preserving stream state and selecting a GPU backend are prerequisites, not
throughput evidence.

## Standalone repository

This package has no inherited workspace dependencies. Its raw bindings dependency
comes from [nemo-speech-sys](https://github.com/kckeiks/nemo-speech-sys), pinned to
commit `c8bf763bc9f4520f58358feaf1b79740e6f2ec9f` in `Cargo.toml` and `Cargo.lock`.
No sibling checkout is required. Cargo fetches that Git revision on the first
build; subsequent offline builds work once the Rust dependencies are cached.

Source, C fixtures, SDK setup guide, validation runner, license, and tests are
package-local. No SDK binary, model weights, audio recordings, build output, or
application Git history are included. Supply the native SDK and inference inputs
separately as described above.
