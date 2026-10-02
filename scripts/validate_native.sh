#!/usr/bin/env bash
# Run opt-in streaming checks with an explicitly selected, trusted native SDK.
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "usage: bash scripts/validate_native.sh LIBRARY MODEL PCM1 REFERENCE1 PCM2 REFERENCE2 cpu|GPU_INDEX [REPEATS]" >&2
    exit 2
fi

crate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEMO_SPEECH_LIBRARY="$1"
export NEMO_SPEECH_MODEL="$2"
export NEMO_SPEECH_PCM="$3"
export NEMO_EXPECTED_TEXT="$4"
export NEMO_SPEECH_PCM_2="$5"
export NEMO_EXPECTED_TEXT_2="$6"
export NEMO_REPEATS="${8:-1}"

if [[ "$7" == cpu ]]; then
    unset NEMO_GPU
elif [[ "$7" =~ ^[0-9]+$ ]]; then
    export NEMO_GPU="$7"
else
    echo "device must be cpu or a nonnegative GPU index" >&2
    exit 2
fi

cd -- "$crate_dir"
cargo test --release -p nemo-speech --test native_sdk \
    native_shared_model -- --ignored --nocapture
