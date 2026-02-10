/**
 * @file rac_hybrid_voice_pipeline.cpp
 * @brief C API implementation for HybridVoicePipeline
 */

#include "rac/features/voice_agent/rac_hybrid_voice_pipeline.h"
#include "rac/features/voice_agent/hybrid_voice_pipeline.h"

#include <string>
#include <cstring>

namespace {
    // Store last result for returning pointers
    thread_local std::string g_last_asr_text;
    thread_local std::string g_last_llm_text;
    thread_local std::vector<float> g_last_tts_audio;
    thread_local std::string g_last_error;
}

extern "C" {

rac_hybrid_pipeline_handle_t rac_hybrid_pipeline_create() {
    return new rac::voice::HybridVoicePipeline();
}

void rac_hybrid_pipeline_destroy(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->cleanup();
        delete pipeline;
    }
}

int rac_hybrid_pipeline_initialize(
    rac_hybrid_pipeline_handle_t handle,
    const rac_hybrid_pipeline_config_t* config) {

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !config) {
        return -1;
    }

    rac::voice::HybridPipelineConfig cpp_config;

    // Memory config
    cpp_config.memory.llm_kv_cache_mb = config->memory.llm_kv_cache_mb;
    cpp_config.memory.asr_buffer_mb = config->memory.asr_buffer_mb;
    cpp_config.memory.tts_buffer_mb = config->memory.tts_buffer_mb;
    cpp_config.memory.scratch_mb = config->memory.scratch_mb;

    // Model config
    if (config->models.asr_model_path) {
        cpp_config.models.asr_model_path = config->models.asr_model_path;
    }
    if (config->models.asr_language) {
        cpp_config.models.asr_language = config->models.asr_language;
    }
    if (config->models.llm_model_path) {
        cpp_config.models.llm_model_path = config->models.llm_model_path;
    }
    if (config->models.system_prompt) {
        cpp_config.models.system_prompt = config->models.system_prompt;
    }
    cpp_config.models.llm_max_tokens = config->models.llm_max_tokens > 0 ? 
        config->models.llm_max_tokens : 256;
    cpp_config.models.llm_temperature = config->models.llm_temperature > 0 ?
        config->models.llm_temperature : 0.7f;
    if (config->models.tts_model_path) {
        cpp_config.models.tts_model_path = config->models.tts_model_path;
    }
    if (config->models.tts_voice_id) {
        cpp_config.models.tts_voice_id = config->models.tts_voice_id;
    }
    cpp_config.models.tts_sample_rate = config->models.tts_sample_rate > 0 ?
        config->models.tts_sample_rate : 22050;

    cpp_config.num_threads = config->num_threads;

    return pipeline->initialize(cpp_config) ? 0 : -1;
}

int rac_hybrid_pipeline_is_ready(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline && pipeline->is_ready() ? 1 : 0;
}

void rac_hybrid_pipeline_cleanup(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->cleanup();
    }
}

rac_pipeline_result_t rac_hybrid_pipeline_process(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples) {

    rac_pipeline_result_t result = {};

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        g_last_error = "Invalid handle";
        result.error_message = g_last_error.c_str();
        return result;
    }

    auto cpp_result = pipeline->process(audio, num_samples);

    // Store strings for returning pointers
    g_last_asr_text = cpp_result.asr.text;
    g_last_llm_text = cpp_result.llm.text;
    g_last_tts_audio = std::move(cpp_result.tts.audio);
    g_last_error = cpp_result.error_message;

    // Fill result
    result.asr.text = g_last_asr_text.c_str();
    result.asr.confidence = cpp_result.asr.confidence;
    result.asr.latency_ms = cpp_result.asr.latency_ms;

    result.llm.text = g_last_llm_text.c_str();
    result.llm.tokens_generated = cpp_result.llm.tokens_generated;
    result.llm.latency_ms = cpp_result.llm.latency_ms;

    result.tts.audio = g_last_tts_audio.data();
    result.tts.num_samples = g_last_tts_audio.size();
    result.tts.sample_rate = cpp_result.tts.sample_rate;
    result.tts.latency_ms = cpp_result.tts.latency_ms;

    result.total_latency_ms = cpp_result.total_latency_ms;
    result.success = cpp_result.success ? 1 : 0;
    result.error_message = g_last_error.c_str();

    return result;
}

int rac_hybrid_pipeline_process_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples,
    rac_asr_callback_fn on_asr_complete,
    rac_llm_token_callback_fn on_llm_token,
    rac_tts_chunk_callback_fn on_tts_chunk,
    rac_pipeline_complete_callback_fn on_complete,
    void* user_data) {

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        return -1;
    }

    rac::voice::PipelineCallbacks callbacks;

    if (on_asr_complete) {
        callbacks.on_asr_complete = [on_asr_complete, user_data](
            const rac::voice::ASRResult& result) {
            on_asr_complete(result.text.c_str(), result.confidence, user_data);
        };
    }

    if (on_llm_token) {
        callbacks.on_llm_token = [on_llm_token, user_data](const std::string& token) {
            on_llm_token(token.c_str(), user_data);
        };
    }

    if (on_tts_chunk) {
        callbacks.on_tts_chunk = [on_tts_chunk, user_data](
            const float* audio, size_t samples) {
            on_tts_chunk(audio, samples, user_data);
        };
    }

    if (on_complete) {
        callbacks.on_complete = [on_complete, user_data](
            const rac::voice::PipelineResult& result) {
            // Convert to C struct
            rac_pipeline_result_t c_result = {};
            // (simplified - would need proper string storage)
            c_result.success = result.success ? 1 : 0;
            c_result.total_latency_ms = result.total_latency_ms;
            on_complete(&c_result, user_data);
        };
    }

    auto result = pipeline->process_streaming(audio, num_samples, callbacks);
    return result.success ? 0 : -1;
}

rac_asr_result_t rac_hybrid_pipeline_transcribe(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples) {

    rac_asr_result_t result = {};

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        return result;
    }

    auto cpp_result = pipeline->transcribe(audio, num_samples);
    g_last_asr_text = cpp_result.text;

    result.text = g_last_asr_text.c_str();
    result.confidence = cpp_result.confidence;
    result.latency_ms = cpp_result.latency_ms;

    return result;
}

rac_llm_result_t rac_hybrid_pipeline_generate(
    rac_hybrid_pipeline_handle_t handle,
    const char* input) {

    rac_llm_result_t result = {};

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !input) {
        return result;
    }

    auto cpp_result = pipeline->generate_response(input);
    g_last_llm_text = cpp_result.text;

    result.text = g_last_llm_text.c_str();
    result.tokens_generated = cpp_result.tokens_generated;
    result.latency_ms = cpp_result.latency_ms;

    return result;
}

int rac_hybrid_pipeline_generate_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const char* input,
    rac_llm_token_callback_fn callback,
    void* user_data) {

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !input || !callback) {
        return -1;
    }

    auto wrapper = [callback, user_data](const std::string& token) {
        return callback(token.c_str(), user_data) != 0;
    };

    auto result = pipeline->generate_response_streaming(input, wrapper);
    return result.text.empty() ? -1 : 0;
}

rac_tts_result_t rac_hybrid_pipeline_synthesize(
    rac_hybrid_pipeline_handle_t handle,
    const char* text) {

    rac_tts_result_t result = {};

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !text) {
        return result;
    }

    auto cpp_result = pipeline->synthesize(text);
    g_last_tts_audio = std::move(cpp_result.audio);

    result.audio = g_last_tts_audio.data();
    result.num_samples = g_last_tts_audio.size();
    result.sample_rate = cpp_result.sample_rate;
    result.latency_ms = cpp_result.latency_ms;

    return result;
}

int rac_hybrid_pipeline_synthesize_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const char* text,
    rac_tts_chunk_callback_fn callback,
    void* user_data) {

    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !text || !callback) {
        return -1;
    }

    auto wrapper = [callback, user_data](const float* audio, size_t samples) {
        return callback(audio, samples, user_data) != 0;
    };

    auto result = pipeline->synthesize_streaming(text, wrapper);
    return result.audio.empty() ? -1 : 0;
}

void rac_hybrid_pipeline_reset_for_next_utterance(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->reset_for_next_utterance();
    }
}

size_t rac_hybrid_pipeline_get_total_memory(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline ? pipeline->get_total_memory_allocated() : 0;
}

size_t rac_hybrid_pipeline_get_memory_used(rac_hybrid_pipeline_handle_t handle) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline ? pipeline->get_memory_used() : 0;
}

int rac_hybrid_pipeline_reload_asr(rac_hybrid_pipeline_handle_t handle, const char* path) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline && path && pipeline->reload_asr_model(path) ? 0 : -1;
}

int rac_hybrid_pipeline_reload_llm(rac_hybrid_pipeline_handle_t handle, const char* path) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline && path && pipeline->reload_llm_model(path) ? 0 : -1;
}

int rac_hybrid_pipeline_reload_tts(rac_hybrid_pipeline_handle_t handle, const char* path) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline && path && pipeline->reload_tts_model(path) ? 0 : -1;
}

void rac_hybrid_pipeline_set_system_prompt(rac_hybrid_pipeline_handle_t handle, const char* prompt) {
    auto* pipeline = static_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline && prompt) {
        pipeline->set_system_prompt(prompt);
    }
}

}  // extern "C"
