#![cfg(unix)]

use std::path::Path;
use std::process::Command;

use nemo_speech::{
    BatchingConfig, Device, Error, RecognitionOptions, Recognizer, RecognizerConfig, Runtime,
    SampleRate, StreamingConfig,
};

struct Fixture {
    runtime: Runtime,
    library: libloading::Library,
    _dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("fixture.so");
        // Use the header shipped by the resolved raw dependency, independent
        // of its checkout location or whether it comes from Git/Cargo.
        std::fs::write(dir.path().join("asr.h"), nemo_speech_sys::ASR_HEADER)
            .expect("write pinned fixture header");
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .args([
                "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
            ])
            .arg("-I")
            .arg(dir.path())
            .arg(root.join("tests/fixtures/asr_fixture.c"))
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fixture compiler: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // SAFETY: this test just built the fixture implementing the pinned ABI.
        let runtime = unsafe { Runtime::load(&library) }.unwrap();
        // SAFETY: known fixture code, also retained by runtime.
        let library = unsafe { libloading::Library::new(library) }.unwrap();
        Self {
            runtime,
            library,
            _dir: dir,
        }
    }

    fn count(&self, kind: i32) -> i32 {
        // SAFETY: this known fixture exports the exact C signature below.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"fixture_count\0")
                .unwrap()(kind)
        }
    }

    fn value(&self, symbol: &[u8]) -> i32 {
        // SAFETY: callers select only known fixture functions of this signature.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn() -> i32>(symbol)
                .unwrap()()
        }
    }

    fn model(&self, mode: &str) -> Recognizer {
        self.runtime
            .create_recognizer(&RecognizerConfig::new(mode))
            .unwrap()
    }

    fn assert_no_live_handles(&self) {
        for kind in 9..=11 {
            assert_eq!(self.count(kind), 0, "live counter {kind}");
        }
    }
}

fn native_error<T>(result: nemo_speech::Result<T>, operation: &str, text: &str) {
    match result {
        Err(Error::Native {
            operation: actual,
            status,
            message,
        }) => {
            assert_eq!(actual, operation);
            assert_eq!(status, 3);
            assert_eq!(message, text);
        }
        _ => panic!("expected native {operation} failure"),
    }
}

#[test]
fn owned_results_keep_all_alternatives_words_and_language() {
    let fixture = Fixture::new();
    assert_eq!(fixture.runtime.version().unwrap(), "fixture-1.0");
    let model = fixture.model("multi-alternative");
    let options = RecognitionOptions {
        language: Some("es-ES".into()),
        word_time_offsets: true,
        max_alternatives: 2,
        ..Default::default()
    };
    let result = model
        .recognize(&[0.0; 8000], SampleRate::HZ_16000, &options)
        .unwrap();
    drop(model);
    fixture.assert_no_live_handles();
    assert_eq!(fixture.count(4), fixture.count(5));
    assert!(result.is_final);
    assert_eq!(result.channel_tag, 1);
    assert_eq!(result.audio_processed_seconds, 0.5);
    assert_eq!(result.transcript(), Some("hello world"));
    assert_eq!(result.alternatives.len(), 2);
    assert_eq!(result.alternatives[1].transcript, "yellow world");
    assert_eq!(result.alternatives[1].confidence, 0.6);
    for alternative in result.alternatives {
        assert_eq!(alternative.language_codes, ["es-ES"]);
        assert_eq!(alternative.words.len(), 2);
        assert_eq!(alternative.words[0].start_time_ms, 0);
        assert_eq!(alternative.words[0].end_time_ms, 250);
        assert_eq!(alternative.words[1].text, "world");
        assert_eq!(alternative.words[1].start_time_ms, 250);
        assert_eq!(alternative.words[1].end_time_ms, 500);
        assert_eq!(alternative.words[1].speaker_tag, 0);
    }
}

#[test]
fn stream_retains_model_and_library_after_public_handles_drop() {
    let fixture = Fixture::new();
    let model = fixture.model("good");
    let options = RecognitionOptions {
        language: Some("fr-FR".into()),
        ..Default::default()
    };
    let mut stream = model.create_stream(&options).unwrap();
    drop(options);
    drop(model);
    // Remove the only Runtime handle while retaining the fixture counter view.
    let Fixture {
        runtime,
        library,
        _dir,
    } = fixture;
    drop(runtime);
    stream
        .push_audio(&[0.0; 1600], SampleRate::HZ_16000)
        .unwrap();
    let interim = stream.next_result().unwrap().unwrap();
    assert!(!interim.is_final);
    assert_eq!(interim.alternatives[0].language_codes, ["fr-FR"]);
    stream.finish().unwrap();
    let final_result = stream.next_result().unwrap().unwrap();
    assert!(final_result.is_final);
    stream.close();
    // SAFETY: library remains loaded; fixture_count has the known signature.
    unsafe {
        let count = library
            .get::<unsafe extern "C" fn(i32) -> i32>(b"fixture_count\0")
            .unwrap();
        assert_eq!(count(1), 1);
        assert_eq!(count(3), 1);
        for kind in 9..=11 {
            assert_eq!(count(kind), 0);
        }
    }
    assert_eq!(final_result.transcript(), Some("hello world"));
}

#[test]
fn stream_retains_library_without_a_second_loader_handle() {
    let fixture = Fixture::new();
    let model = fixture.model("good");
    let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
    let Fixture {
        runtime,
        library,
        _dir,
    } = fixture;
    drop(library); // No counter-view library may mask an early library unload.
    drop(runtime);
    drop(model);
    stream.push_audio(&[0.0], SampleRate::HZ_16000).unwrap();
    assert_eq!(
        stream.next_result().unwrap().unwrap().transcript(),
        Some("hello world")
    );
    stream.finish().unwrap();
    assert!(stream.next_result().unwrap().unwrap().is_final);
    stream.close();
}

#[test]
fn unsafe_native_geometry_never_reaches_create() {
    let fixture = Fixture::new();
    for streaming in [
        StreamingConfig {
            chunk_size_seconds: f32::MIN_POSITIVE,
            ..Default::default()
        },
        StreamingConfig {
            chunk_size_seconds: f32::MAX,
            ..Default::default()
        },
        StreamingConfig {
            ctc_left_padding_seconds: f32::MAX,
            ..Default::default()
        },
        StreamingConfig {
            chunk_size_seconds: 30.0,
            ctc_left_padding_seconds: 30.0,
            ctc_right_padding_seconds: 0.001,
            ..Default::default()
        },
        StreamingConfig {
            rnnt_right_context: i32::MAX,
            ..Default::default()
        },
    ] {
        let mut config = RecognizerConfig::new("good");
        config.streaming = Some(streaming);
        assert!(matches!(
            fixture.runtime.create_recognizer(&config),
            Err(Error::InvalidArgument { .. })
        ));
    }
    assert_eq!(fixture.count(0), 0);
    // The operational limits are inclusive, with zero padding and -1 allowed.
    for streaming in [
        StreamingConfig {
            chunk_size_seconds: 0.001,
            ctc_left_padding_seconds: 0.0,
            ctc_right_padding_seconds: 0.0,
            rnnt_right_context: -1,
        },
        StreamingConfig {
            chunk_size_seconds: 60.0,
            ctc_left_padding_seconds: 0.0,
            ctc_right_padding_seconds: 0.0,
            rnnt_right_context: 1024,
        },
    ] {
        let mut config = RecognizerConfig::new("good");
        config.streaming = Some(streaming);
        drop(fixture.runtime.create_recognizer(&config).unwrap());
    }
    assert_eq!(fixture.count(0), 2);
    fixture.assert_no_live_handles();
}

#[test]
fn open_none_endpoint_and_idempotent_finish_have_distinct_meanings() {
    let fixture = Fixture::new();
    let model = fixture.model("good");
    let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
    assert!(stream.next_result().unwrap().is_none());
    assert!(!stream.is_finished());
    stream
        .push_audio(&[0.0; 100], SampleRate::HZ_16000)
        .unwrap();
    assert!(!stream.next_result().unwrap().unwrap().is_final);
    assert!(stream.next_result().unwrap().is_none());
    stream.force_endpoint().unwrap();
    assert!(stream.next_result().unwrap().unwrap().is_final);
    assert!(!stream.is_finished());
    stream
        .push_audio(&[0.0; 100], SampleRate::HZ_16000)
        .unwrap();
    assert!(!stream.next_result().unwrap().unwrap().is_final);
    stream.finish().unwrap();
    stream.finish().unwrap();
    assert!(stream.is_finished());
    assert_eq!(fixture.count(6), 1);
    assert!(matches!(
        stream.push_audio(&[], SampleRate::HZ_16000),
        Err(Error::StreamState { .. })
    ));
    assert!(matches!(
        stream.force_endpoint(),
        Err(Error::StreamState { .. })
    ));
    assert_eq!(fixture.count(7), 2);
    assert_eq!(fixture.count(13), 1);
    assert!(stream.next_result().unwrap().unwrap().is_final);
    assert!(stream.next_result().unwrap().is_none());
    let pulls = fixture.count(8);
    assert!(stream.next_result().unwrap().is_none());
    stream.finish().unwrap();
    assert_eq!(fixture.count(8), pulls);
    assert_eq!(fixture.count(6), 1);
    drop(stream);
    drop(model);
    fixture.assert_no_live_handles();
}

#[test]
fn native_stream_errors_poison_session_and_preserve_error_before_cleanup() {
    for mode in [
        "fail-push",
        "fail-next",
        "fail-next-with-result",
        "fail-finish",
        "fail-endpoint",
    ] {
        let fixture = Fixture::new();
        let model = fixture.model(mode);
        let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
        match mode {
            "fail-push" => native_error(
                stream.push_audio(&[0.0], SampleRate::HZ_16000),
                "stream_push_f32",
                "fixture push failure",
            ),
            "fail-next" => {
                native_error(stream.next_result(), "stream_next", "fixture next failure")
            }
            "fail-next-with-result" => {
                native_error(
                    stream.next_result(),
                    "stream_next",
                    "fixture next failure with result",
                );
                assert_eq!(fixture.count(4), 1);
                assert_eq!(fixture.count(5), 1);
            }
            "fail-finish" => {
                native_error(stream.finish(), "stream_finish", "fixture finish failure")
            }
            _ => native_error(
                stream.force_endpoint(),
                "stream_force_endpoint",
                "fixture endpoint failure",
            ),
        }
        assert!(matches!(
            stream.next_result(),
            Err(Error::StreamState { state: "failed" })
        ));
        assert!(matches!(
            stream.finish(),
            Err(Error::StreamState { state: "failed" })
        ));
        assert!(matches!(
            stream.force_endpoint(),
            Err(Error::StreamState { state: "failed" })
        ));
        assert!(matches!(
            stream.push_audio(&[0.0], SampleRate::HZ_16000),
            Err(Error::StreamState { state: "failed" })
        ));
        drop(stream);
        drop(model);
        fixture.assert_no_live_handles();
    }
}

#[test]
fn result_copy_failures_destroy_results_and_poison_only_stream() {
    for mode in ["invalid-utf8", "null-transcript"] {
        let fixture = Fixture::new();
        let model = fixture.model(mode);
        let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
        stream.push_audio(&[0.0], SampleRate::HZ_16000).unwrap();
        let result = stream.next_result();
        if mode == "invalid-utf8" {
            assert!(matches!(result, Err(Error::Utf8 { .. })));
        } else {
            assert!(matches!(result, Err(Error::Protocol { .. })));
        }
        assert_eq!(fixture.count(4), 1);
        assert_eq!(fixture.count(5), 1);
        assert!(matches!(
            stream.next_result(),
            Err(Error::StreamState { .. })
        ));
        assert!(model
            .recognize(&[0.0], SampleRate::HZ_16000, &RecognitionOptions::default())
            .is_err());
        assert_eq!(fixture.count(5), 2);
        drop(stream);
        drop(model);
        fixture.assert_no_live_handles();
    }
}

#[test]
fn allocating_call_failures_and_null_successes_are_reported() {
    let fixture = Fixture::new();
    native_error(
        fixture
            .runtime
            .create_recognizer(&RecognizerConfig::new("fail-create")),
        "create",
        "fixture create failure",
    );
    assert!(matches!(
        fixture
            .runtime
            .create_recognizer(&RecognizerConfig::new("null-create")),
        Err(Error::Protocol {
            operation: "create",
            ..
        })
    ));
    assert!(matches!(
        fixture
            .runtime
            .create_recognizer(&RecognizerConfig::new("unknown-status")),
        Err(Error::Native { status: 99, .. })
    ));
    for mode in ["fail-stream", "null-stream", "fail-offline", "null-offline"] {
        let model = fixture.model(mode);
        match mode {
            "fail-stream" => native_error(
                model.create_stream(&RecognitionOptions::default()),
                "streaming_recognize",
                "fixture stream failure",
            ),
            "null-stream" => assert!(matches!(
                model.create_stream(&RecognitionOptions::default()),
                Err(Error::Protocol { .. })
            )),
            "fail-offline" => native_error(
                model.recognize(&[0.0], SampleRate::HZ_16000, &RecognitionOptions::default()),
                "recognize_f32",
                "fixture offline failure",
            ),
            _ => assert!(matches!(
                model.recognize(&[0.0], SampleRate::HZ_16000, &RecognitionOptions::default()),
                Err(Error::Protocol { .. })
            )),
        }
    }
    fixture.assert_no_live_handles();
}

#[test]
fn validation_prevents_native_calls_without_poisoning_stream() {
    let fixture = Fixture::new();
    let model = fixture.model("good");
    let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
    assert!(stream
        .push_audio(&[f32::NAN], SampleRate::HZ_16000)
        .is_err());
    assert!(stream
        .push_audio(&[f32::INFINITY], SampleRate::HZ_16000)
        .is_err());
    stream
        .push_audio(&[], SampleRate::new(8000).unwrap())
        .unwrap();
    assert_eq!(fixture.count(7), 0);
    stream.push_audio(&[0.0], SampleRate::HZ_16000).unwrap();
    assert!(stream
        .push_audio(&[0.0], SampleRate::new(8000).unwrap())
        .is_err());
    assert_eq!(fixture.count(7), 1);
    stream.push_audio(&[0.0], SampleRate::HZ_16000).unwrap();
    for options in [
        RecognitionOptions {
            language: Some("en\0US".into()),
            ..Default::default()
        },
        RecognitionOptions {
            max_alternatives: 0,
            ..Default::default()
        },
        RecognitionOptions {
            max_alternatives: u32::MAX,
            ..Default::default()
        },
    ] {
        assert!(model.create_stream(&options).is_err());
        assert!(model
            .recognize(&[0.0], SampleRate::HZ_16000, &options)
            .is_err());
    }
    assert!(model
        .recognize(
            &[f32::NEG_INFINITY],
            SampleRate::HZ_16000,
            &RecognitionOptions::default()
        )
        .is_err());
    assert_eq!(fixture.count(2), 1);
    assert_eq!(fixture.count(12), 0);
    for hz in [0, 7999, 96001, u32::MAX] {
        assert!(SampleRate::new(hz).is_err());
    }
    assert_eq!(SampleRate::new(96000).unwrap().hz(), 96000);
}

#[test]
fn configurations_are_validated_and_explicitly_forwarded() {
    let fixture = Fixture::new();
    let mut config = RecognizerConfig::new("good");
    config.device = Device::Gpu(2);
    config.streaming = Some(StreamingConfig {
        rnnt_right_context: 6,
        ..Default::default()
    });
    config.batching = Some(BatchingConfig {
        max_batch_size: 21,
        state_arena_slots: 24,
        ..Default::default()
    });
    drop(fixture.runtime.create_recognizer(&config).unwrap());
    assert_eq!(fixture.value(b"fixture_gpu\0"), 2);
    assert_eq!(fixture.value(b"fixture_batch_size\0"), 21);
    assert_eq!(fixture.value(b"fixture_batch_enabled\0"), 1);
    assert_eq!(fixture.value(b"fixture_right_context\0"), 6);
    let creates = fixture.count(0);
    config.device = Device::Gpu(u32::MAX);
    assert!(fixture.runtime.create_recognizer(&config).is_err());
    config.device = Device::Cpu;
    for batching in [
        BatchingConfig {
            max_batch_size: 0,
            ..Default::default()
        },
        BatchingConfig {
            state_arena_slots: 0,
            ..Default::default()
        },
        BatchingConfig {
            max_queue_depth: 0,
            ..Default::default()
        },
        BatchingConfig {
            max_queue_delay_us: u32::MAX,
            ..Default::default()
        },
    ] {
        config.batching = Some(batching);
        assert!(fixture.runtime.create_recognizer(&config).is_err());
    }
    config.batching = None;
    for streaming in [
        StreamingConfig {
            chunk_size_seconds: f32::NAN,
            ..Default::default()
        },
        StreamingConfig {
            chunk_size_seconds: 0.0,
            ..Default::default()
        },
        StreamingConfig {
            ctc_left_padding_seconds: -1.0,
            ..Default::default()
        },
        StreamingConfig {
            ctc_right_padding_seconds: f32::INFINITY,
            ..Default::default()
        },
        StreamingConfig {
            rnnt_right_context: -2,
            ..Default::default()
        },
    ] {
        config.streaming = Some(streaming);
        assert!(fixture.runtime.create_recognizer(&config).is_err());
    }
    assert_eq!(fixture.count(0), creates);
    assert!(fixture
        .runtime
        .create_recognizer(&RecognizerConfig::new(""))
        .is_err());
    assert!(fixture
        .runtime
        .create_recognizer(&RecognizerConfig::new("model\0gguf"))
        .is_err());
    drop(
        fixture
            .runtime
            .create_recognizer(&RecognizerConfig::new("good"))
            .unwrap(),
    );
    assert_eq!(fixture.value(b"fixture_gpu\0"), -1);
    assert_eq!(fixture.value(b"fixture_batch_enabled\0"), 0);
    fixture.assert_no_live_handles();
}

#[test]
fn independent_threads_share_one_model_and_return_owned_outputs() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Runtime>();
    assert_send_sync::<Recognizer>();
    let fixture = Fixture::new();
    let model = fixture.model("good");
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let model = model.clone();
            std::thread::spawn(move || {
                let mut stream = model.create_stream(&RecognitionOptions::default()).unwrap();
                stream
                    .push_audio(&[0.0; 1000], SampleRate::HZ_16000)
                    .unwrap();
                while stream.next_result().unwrap().is_some() {}
                stream.finish().unwrap();
                stream.next_result().unwrap().unwrap()
            })
        })
        .collect();
    drop(model);
    for thread in threads {
        assert!(thread.join().unwrap().is_final);
    }
    assert_eq!(fixture.count(0), 1);
    assert_eq!(fixture.count(1), 1);
    assert_eq!(fixture.count(2), 8);
    assert_eq!(fixture.count(3), 8);
    assert_eq!(fixture.count(4), 16);
    assert_eq!(fixture.count(5), 16);
    fixture.assert_no_live_handles();
}
