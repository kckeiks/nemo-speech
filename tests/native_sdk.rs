//! Opt-in tests using an operator-selected, trusted SDK. Never auto-download it.
use nemo_speech::{
    BatchingConfig, Device, Error, RecognitionOptions, RecognizerConfig, Runtime, SampleRate,
    StreamingConfig,
};

fn runtime() -> Runtime {
    let path = std::env::var_os("NEMO_SPEECH_LIBRARY")
        .expect("set NEMO_SPEECH_LIBRARY to a trusted SDK's absolute C ABI library path");
    // SAFETY: manually running these ignored tests is the operator's assertion
    // that the selected library and its dependencies satisfy the pinned ABI.
    unsafe { Runtime::load(std::path::PathBuf::from(path)) }.expect("load SDK")
}

#[test]
#[ignore = "requires a trusted native SDK; see README"]
fn native_sdk_loads_and_returns_owned_model_error() {
    let runtime = runtime();
    assert_eq!(runtime.version().unwrap(), "nemo-speech-asr 0.1.0");
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does-not-exist.gguf");
    let error = match runtime.create_recognizer(&RecognizerConfig::new(missing)) {
        Err(error) => error,
        Ok(_) => panic!("missing model unexpectedly loaded"),
    };
    drop(runtime);
    match error {
        Error::Native {
            operation,
            status,
            message,
        } => {
            assert_eq!(operation, "create");
            assert_eq!(status, 3);
            assert!(message.contains("Failed to load"), "{message}");
        }
        other => panic!("{other}"),
    }
}

#[test]
#[ignore = "requires a trusted CPU-only SDK, specifically; see README"]
fn native_cpu_sdk_rejects_gpu_request() {
    let runtime = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut config = RecognizerConfig::new(dir.path().join("does-not-exist.gguf"));
    config.device = Device::Gpu(0);
    match runtime.create_recognizer(&config) {
        Err(Error::Native { message, .. }) => {
            assert!(message.contains("no matching GPU device"), "{message}")
        }
        _ => panic!("CPU-only SDK must fail explicit GPU selection"),
    }
}

#[test]
#[ignore = "requires SDK, streaming Nemotron GGUF, two PCM clips and reference phrases; see README"]
fn native_shared_model_transcribes_independent_streams() {
    let runtime = runtime();
    let model_path = std::env::var_os("NEMO_SPEECH_MODEL").expect("set NEMO_SPEECH_MODEL");
    let expected = [
        std::env::var("NEMO_EXPECTED_TEXT").expect("set NEMO_EXPECTED_TEXT"),
        std::env::var("NEMO_EXPECTED_TEXT_2").expect("set NEMO_EXPECTED_TEXT_2"),
    ]
    .map(|text| normalize(&text));
    assert!(
        expected.iter().all(|text| !text.is_empty()),
        "references must contain words"
    );
    assert!(
        !expected[0].contains(&expected[1]) && !expected[1].contains(&expected[0]),
        "choose distinct reference phrases exclusive to their respective recordings"
    );
    let samples = ["NEMO_SPEECH_PCM", "NEMO_SPEECH_PCM_2"].map(|key| {
        let path =
            std::env::var_os(key).unwrap_or_else(|| panic!("set {key} (mono PCM16LE 16kHz)"));
        let bytes = std::fs::read(path).unwrap();
        assert!(!bytes.is_empty() && bytes.len().is_multiple_of(2));
        bytes
            .chunks_exact(2)
            .map(|s| f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0)
            .collect::<Vec<f32>>()
    });
    let mut config = RecognizerConfig::new(std::path::PathBuf::from(model_path));
    if let Ok(gpu) = std::env::var("NEMO_GPU") {
        config.device = Device::Gpu(gpu.parse().expect("GPU index"));
    }
    config.streaming = Some(StreamingConfig {
        rnnt_right_context: 6,
        ..Default::default()
    });
    config.batching = Some(BatchingConfig {
        max_batch_size: 2,
        state_arena_slots: 2,
        ..Default::default()
    });
    let model = runtime.create_recognizer(&config).unwrap();
    let repeats = std::env::var("NEMO_REPEATS")
        .map(|value| value.parse::<usize>().expect("positive repeat count"))
        .unwrap_or(1);
    assert!(
        (1..=100).contains(&repeats),
        "repeat count must be 1 through 100"
    );
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = samples
        .into_iter()
        .map(|samples| {
            let model = model.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                // Meet before fallible native creation, so a failure cannot
                // strand the other thread at the barrier.
                barrier.wait();
                let mut outputs = Vec::new();
                for _ in 0..repeats {
                    let options = RecognitionOptions {
                        word_time_offsets: true,
                        ..Default::default()
                    };
                    let mut stream = model.create_stream(&options).unwrap();
                    let mut finals = Vec::new();
                    for chunk in samples.chunks(8_960) {
                        stream.push_audio(chunk, SampleRate::HZ_16000).unwrap();
                        while let Some(result) = stream.next_result().unwrap() {
                            if result.is_final {
                                finals.push(result);
                            }
                        }
                    }
                    stream.finish().unwrap();
                    stream.finish().unwrap();
                    while let Some(result) = stream.next_result().unwrap() {
                        if result.is_final {
                            finals.push(result);
                        }
                    }
                    assert!(!finals.is_empty(), "no final result");
                    assert!(stream.is_finished());
                    assert!(stream.next_result().unwrap().is_none());
                    // Return owned result metadata after the native stream closes.
                    stream.close();
                    outputs.push(finals);
                }
                outputs
            })
        })
        .collect();
    drop(model);
    drop(runtime);
    for (i, thread) in threads.into_iter().enumerate() {
        for (round, finals) in thread.join().unwrap().into_iter().enumerate() {
            let text = finals
                .iter()
                .filter_map(|result| result.transcript())
                .collect::<Vec<_>>()
                .join(" ");
            println!("stream={i} round={round} final={text:?}");
            assert!(
                normalize(&text).contains(&expected[i]),
                "expected phrase {:?} in {text:?}",
                expected[i]
            );
            assert!(
                !normalize(&text).contains(&expected[1 - i]),
                "other stream's reference leaked into {text:?}"
            );
            let mut word_count = 0;
            for result in finals {
                assert_eq!(result.channel_tag, 1);
                assert!(
                    result.audio_processed_seconds.is_finite()
                        && result.audio_processed_seconds > 0.0
                );
                for word in result.alternatives.iter().flat_map(|alt| &alt.words) {
                    assert!(
                        word.end_time_ms >= word.start_time_ms,
                        "invalid word interval: {word:?}"
                    );
                    word_count += 1;
                }
            }
            assert!(word_count > 0, "requested word offsets were absent");
        }
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
