/* SPDX-License-Identifier: Apache-2.0
 * Deterministic ABI test double: no model files, SDK, or GPU required.
 * Compile with the pinned vendor directory on the include path.
 *
 * fixture_count kinds count calls (including failed calls), except allocation
 * counters 4/5, which count actual result allocation/destruction, and live
 * counters 9/10/11. Kinds: 0 create, 1 destroy, 2 stream create, 3 stream close,
 * 4 result create, 5 result destroy, 6 finish, 7 push, 8 next, 9 live models,
 * 10 live streams, 11 live results, 12 offline recognize, 13 force endpoint.
 * fixture_reset is valid only with no live handles and no concurrent calls.
 * Every stream copies its options; each nonempty push queues one interim
 * result, matching the pinned runtime's ignored interim_results option.
 * Finish queues one final. All cleanup calls overwrite
 * the thread-local error to expose wrappers that copy error messages too late.
 */
#include "asr.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Atomic int counts[14];
static _Atomic int last_gpu, last_batch_size, last_batch_enabled, last_right_context;
static _Thread_local char error_message[128];

struct nemo_speech_asr_recognizer { char *mode; };
struct nemo_speech_asr_stream {
    nemo_speech_asr_recognizer *recognizer;
    char *request_id, *language;
    bool word_offsets, diarization, finished;
    unsigned pending_interim, pending_final;
    size_t sample_count;
    int32_t sample_rate;
};
struct nemo_speech_asr_result {
    char *mode, *request_id, *language;
    bool final, word_offsets, diarization;
    float audio_processed;
};

static char *copy_string(const char *value) {
    if (!value) value = "";
    size_t length = strlen(value) + 1;
    char *copy = malloc(length);
    if (copy) memcpy(copy, value, length);
    return copy;
}
static nemo_speech_asr_status fail(const char *message) {
    snprintf(error_message, sizeof error_message, "%s", message);
    return NEMO_SPEECH_ASR_ERROR_RUNTIME;
}
static bool mode_is(const nemo_speech_asr_recognizer *recognizer, const char *mode) {
    return recognizer && strcmp(recognizer->mode, mode) == 0;
}

int fixture_count(int kind) { return kind >= 0 && kind < 14 ? atomic_load(&counts[kind]) : -1; }
void fixture_reset(void) {
    for (size_t i = 0; i < 14; ++i) atomic_store(&counts[i], 0);
    atomic_store(&last_gpu, 0);
    atomic_store(&last_batch_size, 0);
    atomic_store(&last_batch_enabled, 0);
    atomic_store(&last_right_context, 0);
    error_message[0] = '\0';
}
int fixture_gpu(void) { return atomic_load(&last_gpu); }
int fixture_batch_size(void) { return atomic_load(&last_batch_size); }
int fixture_batch_enabled(void) { return atomic_load(&last_batch_enabled); }
int fixture_right_context(void) { return atomic_load(&last_right_context); }
const char *fixture_stream_request_id(const nemo_speech_asr_stream *stream) { return stream->request_id; }
const char *fixture_stream_language(const nemo_speech_asr_stream *stream) { return stream->language; }
const char *fixture_result_request_id(const nemo_speech_asr_result *result) { return result->request_id; }

nemo_speech_asr_recognition_options nemo_speech_asr_recognition_options_default(void) {
    nemo_speech_asr_recognition_options options = {0};
    options.size = sizeof options;
    options.max_alternatives = 1;
    return options;
}

nemo_speech_asr_status nemo_speech_asr_create(const nemo_speech_asr_recognizer_config *cfg,
                                            nemo_speech_asr_recognizer **out) {
    atomic_fetch_add(&counts[0], 1);
    if (!out) return fail("missing recognizer output");
    *out = NULL;
    if (!cfg || !cfg->model || !cfg->model->path) return fail("missing model path");
    const char *mode = cfg->model->path;
    if (strcmp(mode, "fail-create") == 0) return fail("fixture create failure");
    if (strcmp(mode, "unknown-status") == 0) {
        fail("fixture unknown status");
        return (nemo_speech_asr_status)99;
    }
    if (strcmp(mode, "null-create") == 0) return NEMO_SPEECH_ASR_OK;
    nemo_speech_asr_recognizer *recognizer = calloc(1, sizeof *recognizer);
    if (!recognizer) return fail("fixture allocation failure");
    recognizer->mode = copy_string(mode);
    if (!recognizer->mode) { free(recognizer); return fail("fixture allocation failure"); }
    atomic_store(&last_gpu, cfg->backend ? cfg->backend->gpu : 0);
    atomic_store(&last_batch_enabled, cfg->batching ? cfg->batching->enable : 0);
    atomic_store(&last_batch_size, cfg->batching ? cfg->batching->max_batch_size : 0);
    atomic_store(&last_right_context, cfg->streaming ? cfg->streaming->rnnt_right_context : -1);
    atomic_fetch_add(&counts[9], 1);
    *out = recognizer;
    return NEMO_SPEECH_ASR_OK;
}
void nemo_speech_asr_destroy(nemo_speech_asr_recognizer *recognizer) {
    atomic_fetch_add(&counts[1], 1);
    if (recognizer) {
        free(recognizer->mode);
        free(recognizer);
        atomic_fetch_sub(&counts[9], 1);
    }
    fail("fixture recognizer cleanup replaced error");
}

static nemo_speech_asr_result *make_result(nemo_speech_asr_recognizer *recognizer,
                                          const char *request_id, const char *language,
                                          bool final, bool words, bool diarization,
                                          float audio_processed) {
    nemo_speech_asr_result *result = calloc(1, sizeof *result);
    if (!result) return NULL;
    result->mode = copy_string(recognizer->mode);
    result->request_id = copy_string(request_id);
    result->language = copy_string(language && *language ? language : "en-US");
    if (!result->mode || !result->request_id || !result->language) {
        free(result->mode); free(result->request_id); free(result->language); free(result);
        return NULL;
    }
    result->final = final;
    result->word_offsets = words;
    result->diarization = diarization;
    result->audio_processed = audio_processed;
    atomic_fetch_add(&counts[4], 1);
    atomic_fetch_add(&counts[11], 1);
    return result;
}

nemo_speech_asr_status nemo_speech_asr_recognize_f32(nemo_speech_asr_recognizer *recognizer,
    const nemo_speech_asr_recognition_options *options, const float *samples,
    size_t n_samples, int32_t sample_rate, nemo_speech_asr_result **out) {
    atomic_fetch_add(&counts[12], 1);
    if (!out) return fail("missing result output");
    *out = NULL;
    if (!recognizer || (!samples && n_samples)) return fail("invalid offline input");
    if (mode_is(recognizer, "fail-offline")) return fail("fixture offline failure");
    if (mode_is(recognizer, "null-offline")) return NEMO_SPEECH_ASR_OK;
    *out = make_result(recognizer, options ? options->request_id : NULL,
                       options ? options->language_code : NULL, true,
                       options && options->enable_word_time_offsets,
                       options && options->enable_speaker_diarization,
                       (float)n_samples / (sample_rate > 0 ? sample_rate : 16000));
    return *out ? NEMO_SPEECH_ASR_OK : fail("fixture allocation failure");
}

nemo_speech_asr_status nemo_speech_asr_streaming_recognize(nemo_speech_asr_recognizer *recognizer,
    const nemo_speech_asr_recognition_options *options, nemo_speech_asr_stream **out) {
    atomic_fetch_add(&counts[2], 1);
    if (!out) return fail("missing stream output");
    *out = NULL;
    if (!recognizer) return fail("missing recognizer");
    if (mode_is(recognizer, "fail-stream")) return fail("fixture stream failure");
    if (mode_is(recognizer, "null-stream")) return NEMO_SPEECH_ASR_OK;
    nemo_speech_asr_stream *stream = calloc(1, sizeof *stream);
    if (!stream) return fail("fixture allocation failure");
    stream->recognizer = recognizer;
    stream->request_id = copy_string(options ? options->request_id : NULL);
    stream->language = copy_string(options ? options->language_code : NULL);
    if (!stream->request_id || !stream->language) {
        free(stream->request_id); free(stream->language); free(stream);
        return fail("fixture allocation failure");
    }
    stream->word_offsets = options && options->enable_word_time_offsets;
    stream->diarization = options && options->enable_speaker_diarization;
    atomic_fetch_add(&counts[10], 1);
    *out = stream;
    return NEMO_SPEECH_ASR_OK;
}
nemo_speech_asr_status nemo_speech_asr_stream_push_f32(nemo_speech_asr_stream *stream,
    const float *samples, size_t n_samples, int32_t sample_rate) {
    atomic_fetch_add(&counts[7], 1);
    if (!stream || (!samples && n_samples) || stream->finished) return fail("invalid stream input");
    if (mode_is(stream->recognizer, "fail-push")) return fail("fixture push failure");
    if (stream->sample_count && stream->sample_rate != sample_rate) return fail("sample rate changed");
    stream->sample_rate = sample_rate;
    stream->sample_count += n_samples;
    if (n_samples) ++stream->pending_interim;
    return NEMO_SPEECH_ASR_OK;
}
nemo_speech_asr_status nemo_speech_asr_stream_force_endpoint(nemo_speech_asr_stream *stream) {
    atomic_fetch_add(&counts[13], 1);
    if (!stream || stream->finished) return fail("invalid stream endpoint");
    if (mode_is(stream->recognizer, "fail-endpoint")) return fail("fixture endpoint failure");
    ++stream->pending_final;
    return NEMO_SPEECH_ASR_OK;
}
nemo_speech_asr_status nemo_speech_asr_stream_finish(nemo_speech_asr_stream *stream) {
    atomic_fetch_add(&counts[6], 1);
    if (!stream || stream->finished) return fail("invalid stream finish");
    if (mode_is(stream->recognizer, "fail-finish")) return fail("fixture finish failure");
    stream->finished = true;
    ++stream->pending_final;
    return NEMO_SPEECH_ASR_OK;
}
nemo_speech_asr_status nemo_speech_asr_stream_next(nemo_speech_asr_stream *stream,
                                                 nemo_speech_asr_result **out) {
    atomic_fetch_add(&counts[8], 1);
    if (!out) return fail("missing result output");
    *out = NULL;
    if (!stream) return fail("missing stream");
    if (mode_is(stream->recognizer, "fail-next")) return fail("fixture next failure");
    bool error_with_result = mode_is(stream->recognizer, "fail-next-with-result");
    if (!stream->pending_interim && !stream->pending_final && !error_with_result) return NEMO_SPEECH_ASR_OK;
    bool final = !stream->pending_interim;
    *out = make_result(stream->recognizer, stream->request_id, stream->language, final,
                       stream->word_offsets, stream->diarization,
                       (float)stream->sample_count / (stream->sample_rate > 0 ? stream->sample_rate : 16000));
    if (!*out) return fail("fixture allocation failure");
    if (error_with_result) return fail("fixture next failure with result");
    if (final) --stream->pending_final; else --stream->pending_interim;
    return NEMO_SPEECH_ASR_OK;
}
void nemo_speech_asr_stream_close(nemo_speech_asr_stream *stream) {
    atomic_fetch_add(&counts[3], 1);
    if (stream) {
        free(stream->request_id); free(stream->language); free(stream);
        atomic_fetch_sub(&counts[10], 1);
    }
    fail("fixture stream cleanup replaced error");
}

bool nemo_speech_asr_result_is_final(const nemo_speech_asr_result *result) { return result->final; }
float nemo_speech_asr_result_audio_processed(const nemo_speech_asr_result *result) { return result->audio_processed; }
int32_t nemo_speech_asr_result_channel_tag(const nemo_speech_asr_result *result) { (void)result; return 1; }
size_t nemo_speech_asr_result_alternative_count(const nemo_speech_asr_result *result) {
    return strcmp(result->mode, "multi-alternative") == 0 ? 2 : 1;
}
const char *nemo_speech_asr_result_transcript(const nemo_speech_asr_result *result, size_t alt) {
    if (alt >= nemo_speech_asr_result_alternative_count(result) || strcmp(result->mode, "null-transcript") == 0) return NULL;
    if (strcmp(result->mode, "invalid-utf8") == 0) return "\xff\xfe";
    return alt == 0 ? "hello world" : "yellow world";
}
float nemo_speech_asr_result_confidence(const nemo_speech_asr_result *result, size_t alt) {
    (void)result; return alt == 0 ? 0.9f : 0.6f;
}
size_t nemo_speech_asr_result_word_count(const nemo_speech_asr_result *result, size_t alt) {
    return result->word_offsets && alt < nemo_speech_asr_result_alternative_count(result) ? 2 : 0;
}
const char *nemo_speech_asr_result_word_text(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    if (i >= nemo_speech_asr_result_word_count(result, alt)) return NULL;
    return i == 1 ? "world" : (alt == 0 ? "hello" : "yellow");
}
int32_t nemo_speech_asr_result_word_start_time(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    (void)result; (void)alt; return (int32_t)i * 250;
}
int32_t nemo_speech_asr_result_word_end_time(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    (void)result; (void)alt; return (int32_t)(i + 1) * 250;
}
float nemo_speech_asr_result_word_confidence(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    (void)result; (void)alt; (void)i; return 0.8f;
}
int32_t nemo_speech_asr_result_word_speaker_tag(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    (void)alt; (void)i; return result->diarization ? 2 : 0;
}
size_t nemo_speech_asr_result_language_count(const nemo_speech_asr_result *result, size_t alt) {
    return alt < nemo_speech_asr_result_alternative_count(result) ? 1 : 0;
}
const char *nemo_speech_asr_result_language_code(const nemo_speech_asr_result *result, size_t alt, size_t i) {
    return i == 0 && alt < nemo_speech_asr_result_alternative_count(result) ? result->language : NULL;
}
void nemo_speech_asr_result_destroy(nemo_speech_asr_result *result) {
    if (result) {
        free(result->mode); free(result->request_id); free(result->language); free(result);
        atomic_fetch_add(&counts[5], 1);
        atomic_fetch_sub(&counts[11], 1);
    }
    fail("fixture result cleanup replaced error");
}
const char *nemo_speech_asr_last_error(void) { return error_message; }
const char *nemo_speech_asr_version(void) { return "fixture-1.0"; }
