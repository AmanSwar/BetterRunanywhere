/**
 * @file hybrid_voice_pipeline.h
 * @brief Hybrid Voice Pipeline - Unified ASR → LLM → TTS processing
 *
 * Coordinates:
 * - UnifiedMemoryManager for pre-allocated memory pools
 * - ONNX Runtime (via Sherpa-ONNX) for ASR and TTS
 * - llama.cpp for LLM inference
 *
 * Achieves zero runtime memory allocation during inference for optimal
 * mobile performance.
 */

#ifndef RAC_HYBRID_VOICE_PIPELINE_H
#define RAC_HYBRID_VOICE_PIPELINE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rac {
namespace core {
class UnifiedMemoryManager;
}
}

namespace runanywhere {
class LlamaCppBackend;
class LlamaCppTextGeneration;
class ONNXBackendNew;
}


namespace rac {
namespace voice {

// =============================================================================
// Pipeline Configuration
// =============================================================================

struct PipelineMemoryConfig {
    size_t llm_kv_cache_mb = 400;     // LLM KV cache (MB)
    size_t asr_buffer_mb = 100;       // ASR working memory (MB)
    size_t tts_buffer_mb = 50;        // TTS working memory (MB)
    size_t scratch_mb = 50;           // Shared scratch space (MB)
};

struct PipelineModelConfig {
    // ASR (Speech-to-Text) - Sherpa-ONNX Whisper
    std::string asr_model_path;       // Path to Whisper .onnx model directory
    std::string asr_language = "en";  // Language code

    // LLM (Text Generation) - llama.cpp
    std::string llm_model_path;       // Path to .gguf model
    std::string system_prompt;        // System prompt for LLM
    int llm_max_tokens = 256;         // Max response tokens
    float llm_temperature = 0.7f;     // Sampling temperature

    // TTS (Text-to-Speech) - Sherpa-ONNX Piper
    std::string tts_model_path;       // Path to Piper .onnx model directory
    std::string tts_voice_id;         // Voice identifier
    int tts_sample_rate = 22050;      // Audio output sample rate
};

struct HybridPipelineConfig {
    PipelineMemoryConfig memory;
    PipelineModelConfig models;
    int num_threads = 0;              // 0 = auto-detect
};


// =============================================================================
// Processing Results
// =============================================================================

struct ASRResult {
    std::string text;
    float confidence = 0.0f;
    double latency_ms = 0.0;
};

struct LLMResult {
    std::string text;
    int tokens_generated = 0;
    double latency_ms = 0.0;
};

struct TTSResult {
    std::vector<float> audio;         // Audio samples [-1.0, 1.0]
    int sample_rate = 22050;
    double latency_ms = 0.0;
};

struct PipelineResult {
    ASRResult asr;                    // Speech recognition result
    LLMResult llm;                    // LLM response
    TTSResult tts;                    // Synthesized audio
    double total_latency_ms = 0.0;    // End-to-end latency
    bool success = false;
    std::string error_message;
};

// =============================================================================
// Callbacks
// =============================================================================

// Called when LLM generates tokens (for streaming display)
using LLMStreamCallback = std::function<bool(const std::string& token)>;

// Called when TTS generates audio chunks (for streaming playback)
using TTSStreamCallback = std::function<bool(const float* audio, size_t samples)>;

// Pipeline stage callbacks for progress tracking
struct PipelineCallbacks {
    std::function<void(const ASRResult&)> on_asr_complete;
    std::function<void(const std::string& token)> on_llm_token;
    std::function<void(const LLMResult&)> on_llm_complete;
    std::function<void(const float*, size_t)> on_tts_chunk;
    std::function<void(const PipelineResult&)> on_complete;
};

// =============================================================================
// Hybrid Voice Pipeline
// =============================================================================

class HybridVoicePipeline {
public:
    HybridVoicePipeline();
    ~HybridVoicePipeline();

    // Non-copyable
    HybridVoicePipeline(const HybridVoicePipeline&) = delete;
    HybridVoicePipeline& operator=(const HybridVoicePipeline&) = delete;

    /**
     * @brief Initialize the pipeline with all components
     * @param config Pipeline configuration
     * @return true if all components initialized successfully
     */
    bool initialize(const HybridPipelineConfig& config);

    /**
     * @brief Check if pipeline is ready for processing
     */
    bool is_ready() const;

    /**
     * @brief Cleanup and release all resources
     */
    void cleanup();

    // =========================================================================
    // Full Pipeline Processing
    // =========================================================================

    /**
     * @brief Process audio through full pipeline: ASR → LLM → TTS
     * @param audio Audio samples (16kHz, mono, normalized)
     * @param num_samples Number of audio samples
     * @return Complete pipeline result
     */
    PipelineResult process(const float* audio, size_t num_samples);

    /**
     * @brief Process with streaming callbacks
     */
    PipelineResult process_streaming(
        const float* audio, size_t num_samples,
        const PipelineCallbacks& callbacks);

    // =========================================================================
    // Individual Stage Processing (for flexibility)
    // =========================================================================

    /**
     * @brief Transcribe audio to text (ASR only)
     */
    ASRResult transcribe(const float* audio, size_t num_samples);

    /**
     * @brief Generate LLM response from text
     */
    LLMResult generate_response(const std::string& input);

    /**
     * @brief Generate LLM response with streaming
     */
    LLMResult generate_response_streaming(
        const std::string& input,
        LLMStreamCallback callback);

    /**
     * @brief Synthesize audio from text (TTS only)
     */
    TTSResult synthesize(const std::string& text);

    /**
     * @brief Synthesize with streaming output
     */
    TTSResult synthesize_streaming(
        const std::string& text,
        TTSStreamCallback callback);

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * @brief Reset reusable memory segments between utterances
     *
     * Call this between processing different audio inputs to
     * clear ASR/TTS/scratch buffers. LLM KV cache is NOT cleared.
     */
    void reset_for_next_utterance();

    /**
     * @brief Get total allocated memory in bytes
     */
    size_t get_total_memory_allocated() const;

    /**
     * @brief Get currently used memory in bytes
     */
    size_t get_memory_used() const;

    // =========================================================================
    // Model Management
    // =========================================================================

    /**
     * @brief Reload ASR model with new path
     */
    bool reload_asr_model(const std::string& model_path);

    /**
     * @brief Reload LLM model with new path
     */
    bool reload_llm_model(const std::string& model_path);

    /**
     * @brief Reload TTS model with new path
     */
    bool reload_tts_model(const std::string& model_path);

    /**
     * @brief Update LLM system prompt
     */
    void set_system_prompt(const std::string& prompt);

private:
    bool initialized_ = false;
    HybridPipelineConfig config_;

    // Memory management
    std::unique_ptr<rac::core::UnifiedMemoryManager> memory_manager_;

    // ONNX backend for ASR/TTS (Sherpa-ONNX)
    std::unique_ptr<runanywhere::ONNXBackendNew> onnx_;

    // llama.cpp backend for LLM
    std::unique_ptr<runanywhere::LlamaCppBackend> llamacpp_;

    // Current system prompt
    std::string system_prompt_;
};


}  // namespace voice
}  // namespace rac

#endif  // RAC_HYBRID_VOICE_PIPELINE_H
