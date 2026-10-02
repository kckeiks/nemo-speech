//! Blocking smoke example: mono PCM16LE, 16 kHz; no WAV header.
use std::error::Error;
use std::io::{self, Read};
use std::path::PathBuf;

use nemo_speech::{
    Device, RecognitionOptions, RecognizerConfig, Runtime, SampleRate, Stream, StreamingConfig,
};

fn drain(stream: &mut Stream) -> nemo_speech::Result<()> {
    while let Some(result) = stream.next_result()? {
        println!(
            "{}: {}",
            if result.is_final { "final" } else { "interim" },
            result.transcript().unwrap_or("")
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: stream_pcm /absolute/libnemo_speech_asr_c.so.1 model.gguf audio.pcm cpu|GPU_INDEX".into());
    }
    let library = PathBuf::from(&args[0]);
    let mut config = RecognizerConfig::new(PathBuf::from(&args[1]));
    let device = args[3]
        .to_str()
        .ok_or("device must be cpu or a GPU index")?;
    config.device = if device == "cpu" {
        Device::Cpu
    } else {
        Device::Gpu(device.parse()?)
    };
    // Nemotron streaming 560 ms geometry; select a compatible model.
    config.streaming = Some(StreamingConfig {
        rnnt_right_context: 6,
        ..Default::default()
    });
    // SAFETY: the operator must supply a trusted SDK built at UPSTREAM_REVISION
    // implementing the pinned ABI. See the crate README deployment contract.
    let runtime = unsafe { Runtime::load(library)? };
    eprintln!("{}", runtime.version()?);
    let model = runtime.create_recognizer(&config)?;
    let mut stream = model.create_stream(&RecognitionOptions {
        word_time_offsets: true,
        ..Default::default()
    })?;
    let mut audio = io::BufReader::new(std::fs::File::open(&args[2])?);
    // Read 560 ms at a time. Reads can be short, so fill the buffer explicitly.
    let mut bytes = vec![0_u8; 8_960 * 2];
    loop {
        let mut used = 0;
        while used < bytes.len() {
            match audio.read(&mut bytes[used..]) {
                Ok(0) => break,
                Ok(n) => used += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        if used == 0 {
            break;
        }
        if used % 2 != 0 {
            return Err("PCM16 input ends with an incomplete sample".into());
        }
        let samples: Vec<_> = bytes[..used]
            .chunks_exact(2)
            .map(|s| f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0)
            .collect();
        stream.push_audio(&samples, SampleRate::HZ_16000)?;
        drain(&mut stream)?;
    }
    stream.finish()?;
    drain(&mut stream)?;
    Ok(())
}
