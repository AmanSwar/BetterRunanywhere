/**
 * @file hybrid_voice_pipeline.cpp
 * @brief Implementation of HybridVoicePipeline
 *
 * Uses ONNX Runtime (Sherpa-ONNX) for ASR/TTS and llama.cpp for LLM,
 * all backed by UnifiedMemoryManager for zero-allocation inference.
 */

#include "rac/features/voice_agent/hybrid_voice_pipeline.h"
#include "rac/core/unified_memory_manager.h"

// Backend includes
#ifdef RAC_BACKEND_LLAMACPP
#include "backends/llamacpp/llamacpp_backend.h"
#else
namespace runanywhere {
class LlamaCppBackend {
public:
    bool is_initialized() const { return false; }
    void cleanup() {}
};
}
#endif

#ifdef RAC_BACKEND_ONNX
#include "backends/onnx/onnx_backend.h"
#else
namespace runanywhere {
class ONNXBackendNew {
public:
    bool is_initialized() const { return false; }
    void cleanup() {}
};
}
#endif

#include <chrono>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "HybridVoicePipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, "[HybridVoicePipeline] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[HybridVoicePipeline ERROR] " __VA_ARGS__)
#endif

namespace rac {
namespace voice {

// Helper to measure latency
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// Constructor / Destructor
// =============================================================================

HybridVoicePipeline::HybridVoicePipeline() = default;

HybridVoicePipeline::~HybridVoicePipeline() {
    cleanup();
}

// =============================================================================
// Initialization
// =============================================================================

bool HybridVoicePipeline::initialize(const HybridPipelineConfig& config) {
    if (initialized_) {
        LOGE("Pipeline already initialized\n");
        return false;
    }

    config_ = config;
    Timer timer;

    // Step 1: Initialize Unified Memory Manager
    LOGI("Initializing memory manager...\n");
    memory_manager_ = std::make_unique<rac::core::UnifiedMemoryManager>();

    rac::core::MemoryPoolConfig mem_config;
    mem_config.llm_kv_cache_size = config.memory.llm_kv_cache_mb * 1024 * 1024;
    mem_config.asr_buffer_size = config.memory.asr_buffer_mb * 1024 * 1024;
    mem_config.tts_buffer_size = config.memory.tts_buffer_mb * 1024 * 1024;
    mem_config.scratch_size = config.memory.scratch_mb * 1024 * 1024;

    if (!memory_manager_->initialize(mem_config)) {
        LOGE("Failed to initialize memory manager\n");
        return false;
    }

    LOGI("Memory pool allocated: %zu MB\n",
         memory_manager_->get_total_size() / (1024 * 1024));

    // Step 2: Initialize ONNX backend for ASR/TTS
#ifdef RAC_BACKEND_ONNX
    LOGI("Initializing ONNX backend for ASR/TTS...\n");
    onnx_ = std::make_unique<runanywhere::ONNXBackendNew>();

    nlohmann::json onnx_config;
    if (config.num_threads > 0) {
        onnx_config["num_threads"] = config.num_threads;
    }

    if (!onnx_->initialize(onnx_config)) {
        LOGE("Failed to initialize ONNX backend\n");
        return false;
    }

    // Wire up memory segments for pooled allocations
    onnx_->set_memory_segments(
        &memory_manager_->get_asr_segment(),
        &memory_manager_->get_tts_segment()
    );
    LOGI("ONNX memory pools configured (ASR + TTS)\n");

    // Load ASR/STT model
    if (!config.models.asr_model_path.empty()) {
        LOGI("Loading STT model: %s\n", config.models.asr_model_path.c_str());
        auto* stt = onnx_->get_stt();
        if (stt) {
            nlohmann::json stt_config;
            stt_config["language"] = config.models.asr_language;

            if (!stt->load_model(config.models.asr_model_path,
                                runanywhere::STTModelType::WHISPER,
                                stt_config)) {
                LOGE("Failed to load STT model\n");
                return false;
            }
        }
    }

    // Load TTS model
    if (!config.models.tts_model_path.empty()) {
        LOGI("Loading TTS model: %s\n", config.models.tts_model_path.c_str());
        auto* tts = onnx_->get_tts();
        if (tts) {
            nlohmann::json tts_config;

            if (!tts->load_model(config.models.tts_model_path,
                                runanywhere::TTSModelType::PIPER,
                                tts_config)) {
                LOGE("Failed to load TTS model\n");
                return false;
            }
        }
    }
#else
    LOGI("ONNX backend not available\n");
#endif

    // Step 3: Initialize llama.cpp for LLM
#ifdef RAC_BACKEND_LLAMACPP
    LOGI("Initializing llama.cpp backend...\n");
    llamacpp_ = std::make_unique<runanywhere::LlamaCppBackend>();

    nlohmann::json llm_config;
    if (config.num_threads > 0) {
        llm_config["n_threads"] = config.num_threads;
    }

    if (!llamacpp_->initialize(llm_config)) {
        LOGE("Failed to initialize llama.cpp backend\n");
        return false;
    }

    // Wire up memory segment for LLM KV cache
    llamacpp_->set_memory_segment(&memory_manager_->get_llm_segment());
    LOGI("LLM memory pool configured\n");

    // Load LLM model
    if (!config.models.llm_model_path.empty()) {
        LOGI("Loading LLM model: %s\n", config.models.llm_model_path.c_str());
        auto* text_gen = llamacpp_->get_text_generation();
        if (text_gen) {
            nlohmann::json model_config;
            model_config["n_ctx"] = 2048;

            if (!text_gen->load_model(config.models.llm_model_path, model_config)) {
                LOGE("Failed to load LLM model\n");
                return false;
            }
        }
    }

    system_prompt_ = config.models.system_prompt;
#else
    LOGI("llama.cpp backend not available, LLM disabled\n");
#endif

    initialized_ = true;
    LOGI("Pipeline initialized in %.2f ms\n", timer.elapsed_ms());
    return true;
}

bool HybridVoicePipeline::is_ready() const {
    if (!initialized_ || !memory_manager_) {
        return false;
    }

#ifdef RAC_BACKEND_LLAMACPP
    if (!llamacpp_ || !llamacpp_->is_initialized()) {
        return false;
    }
#endif

    return true;
}

void HybridVoicePipeline::cleanup() {
#ifdef RAC_BACKEND_LLAMACPP
    if (llamacpp_) {
        llamacpp_->cleanup();
        llamacpp_.reset();
    }
#endif

#ifdef RAC_BACKEND_ONNX
    if (onnx_) {
        onnx_->cleanup();
        onnx_.reset();
    }
#endif

    if (memory_manager_) {
        memory_manager_->cleanup();
        memory_manager_.reset();
    }

    initialized_ = false;
    LOGI("Pipeline cleaned up\n");
}

// =============================================================================
// Full Pipeline Processing
// =============================================================================

PipelineResult HybridVoicePipeline::process(const float* audio, size_t num_samples) {
    return process_streaming(audio, num_samples, PipelineCallbacks{});
}

PipelineResult HybridVoicePipeline::process_streaming(
    const float* audio, size_t num_samples,
    const PipelineCallbacks& callbacks) {

    PipelineResult result;
    Timer total_timer;

    if (!is_ready()) {
        result.error_message = "Pipeline not ready";
        return result;
    }

    // Step 1: ASR - Transcribe audio
    result.asr = transcribe(audio, num_samples);
    if (callbacks.on_asr_complete) {
        callbacks.on_asr_complete(result.asr);
    }

    if (result.asr.text.empty()) {
        result.error_message = "ASR produced no output";
        return result;
    }

    // Step 2: LLM - Generate response
    if (callbacks.on_llm_token) {
        result.llm = generate_response_streaming(result.asr.text,
            [&callbacks](const std::string& token) {
                callbacks.on_llm_token(token);
                return true;
            });
    } else {
        result.llm = generate_response(result.asr.text);
    }

    if (callbacks.on_llm_complete) {
        callbacks.on_llm_complete(result.llm);
    }

    if (result.llm.text.empty()) {
        result.error_message = "LLM produced no output";
        return result;
    }

    // Step 3: TTS - Synthesize audio
    if (callbacks.on_tts_chunk) {
        result.tts = synthesize_streaming(result.llm.text,
            [&callbacks](const float* a, size_t s) {
                callbacks.on_tts_chunk(a, s);
                return true;
            });
    } else {
        result.tts = synthesize(result.llm.text);
    }

    result.total_latency_ms = total_timer.elapsed_ms();
    result.success = !result.tts.audio.empty();

    if (callbacks.on_complete) {
        callbacks.on_complete(result);
    }

    LOGI("Pipeline complete: ASR=%.1fms LLM=%.1fms TTS=%.1fms Total=%.1fms\n",
         result.asr.latency_ms, result.llm.latency_ms,
         result.tts.latency_ms, result.total_latency_ms);

    return result;
}

// =============================================================================
// Individual Stage Processing
// =============================================================================

ASRResult HybridVoicePipeline::transcribe(const float* audio, size_t num_samples) {
    ASRResult result;
    Timer timer;

#ifdef RAC_BACKEND_ONNX
    if (onnx_) {
        auto* stt = onnx_->get_stt();
        if (stt && stt->is_model_loaded()) {
            runanywhere::STTRequest request;
            request.audio_samples.assign(audio, audio + num_samples);
            request.sample_rate = 16000;
            request.language = config_.models.asr_language;

            auto stt_result = stt->transcribe(request);
            result.text = stt_result.text;
            result.confidence = stt_result.confidence;
        } else {
            LOGE("STT model not loaded\n");
        }
    } else {
        LOGE("ONNX backend not available\n");
    }
#else
    LOGE("ONNX backend not compiled in\n");
    (void)audio;
    (void)num_samples;
#endif

    result.latency_ms = timer.elapsed_ms();
    return result;
}

LLMResult HybridVoicePipeline::generate_response(const std::string& input) {
    return generate_response_streaming(input, nullptr);
}

LLMResult HybridVoicePipeline::generate_response_streaming(
    const std::string& input,
    LLMStreamCallback callback) {

    LLMResult result;
    Timer timer;

#ifdef RAC_BACKEND_LLAMACPP
    if (llamacpp_ && llamacpp_->is_initialized()) {
        auto* text_gen = llamacpp_->get_text_generation();
        if (text_gen && text_gen->is_model_loaded()) {
            runanywhere::TextGenerationRequest request;
            request.prompt = input;
            request.system_prompt = system_prompt_;
            request.max_tokens = config_.models.llm_max_tokens;
            request.temperature = config_.models.llm_temperature;

            if (callback) {
                text_gen->generate_stream(request, callback);
                result.text = "";
            } else {
                auto gen_result = text_gen->generate(request);
                result.text = gen_result.text;
                result.tokens_generated = gen_result.tokens_generated;
            }
        } else {
            LOGE("LLM model not loaded\n");
        }
    } else {
        LOGE("LLM backend not available\n");
    }
#else
    LOGE("llama.cpp backend not compiled in\n");
    (void)input;
    (void)callback;
#endif

    result.latency_ms = timer.elapsed_ms();
    return result;
}

TTSResult HybridVoicePipeline::synthesize(const std::string& text) {
    TTSResult result;
    Timer timer;

#ifdef RAC_BACKEND_ONNX
    if (onnx_) {
        auto* tts = onnx_->get_tts();
        if (tts && tts->is_model_loaded()) {
            runanywhere::TTSRequest request;
            request.text = text;
            request.sample_rate = config_.models.tts_sample_rate;
            request.voice_id = config_.models.tts_voice_id;

            auto tts_result = tts->synthesize(request);
            result.audio = std::move(tts_result.audio_samples);
            result.sample_rate = tts_result.sample_rate;
        } else {
            LOGE("TTS model not loaded\n");
        }
    } else {
        LOGE("ONNX backend not available\n");
    }
#else
    LOGE("ONNX backend not compiled in\n");
    (void)text;
#endif

    result.latency_ms = timer.elapsed_ms();
    return result;
}

TTSResult HybridVoicePipeline::synthesize_streaming(
    const std::string& text,
    TTSStreamCallback callback) {

    // Sherpa-ONNX TTS doesn't have native streaming, so synthesize fully
    // then deliver via callback in chunks
    TTSResult result = synthesize(text);

    if (callback && !result.audio.empty()) {
        callback(result.audio.data(), result.audio.size());
    }

    return result;
}

// =============================================================================
// Memory Management
// =============================================================================

void HybridVoicePipeline::reset_for_next_utterance() {
    if (memory_manager_) {
        memory_manager_->reset_reusable_segments();
    }

#ifdef RAC_BACKEND_ONNX
    if (onnx_) {
        onnx_->reset_memory_pools();
    }
#endif
}

size_t HybridVoicePipeline::get_total_memory_allocated() const {
    return memory_manager_ ? memory_manager_->get_total_size() : 0;
}

size_t HybridVoicePipeline::get_memory_used() const {
    return memory_manager_ ? memory_manager_->get_total_used() : 0;
}

// =============================================================================
// Model Management
// =============================================================================

bool HybridVoicePipeline::reload_asr_model(const std::string& model_path) {
#ifdef RAC_BACKEND_ONNX
    if (!onnx_) return false;

    auto* stt = onnx_->get_stt();
    if (!stt) return false;

    stt->unload_model();
    memory_manager_->reset_asr_segment();

    nlohmann::json stt_config;
    stt_config["language"] = config_.models.asr_language;

    return stt->load_model(model_path, runanywhere::STTModelType::WHISPER, stt_config);
#else
    (void)model_path;
    return false;
#endif
}

bool HybridVoicePipeline::reload_llm_model(const std::string& model_path) {
#ifdef RAC_BACKEND_LLAMACPP
    if (!llamacpp_) return false;

    auto* text_gen = llamacpp_->get_text_generation();
    if (!text_gen) return false;

    text_gen->unload_model();

    nlohmann::json model_config;
    model_config["n_ctx"] = 2048;

    return text_gen->load_model(model_path, model_config);
#else
    (void)model_path;
    return false;
#endif
}

bool HybridVoicePipeline::reload_tts_model(const std::string& model_path) {
#ifdef RAC_BACKEND_ONNX
    if (!onnx_) return false;

    auto* tts = onnx_->get_tts();
    if (!tts) return false;

    tts->unload_model();
    memory_manager_->reset_tts_segment();

    nlohmann::json tts_config;

    return tts->load_model(model_path, runanywhere::TTSModelType::PIPER, tts_config);
#else
    (void)model_path;
    return false;
#endif
}

void HybridVoicePipeline::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

}  // namespace voice
}  // namespace rac
