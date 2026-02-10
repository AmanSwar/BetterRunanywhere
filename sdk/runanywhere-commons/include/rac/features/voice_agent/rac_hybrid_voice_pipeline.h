/**
 * @file rac_hybrid_voice_pipeline.h
 * @brief C API for HybridVoicePipeline
 *
 * Exposes the hybrid voice pipeline to C-based consumers including
 * JNI (Android), Swift (iOS), and C FFI bindings.
 */

#ifndef RAC_HYBRID_VOICE_PIPELINE_API_H
#define RAC_HYBRID_VOICE_PIPELINE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Handle Types
// =============================================================================

typedef void* rac_hybrid_pipeline_handle_t;

// =============================================================================
// Configuration Structures
// =============================================================================

typedef struct {
    size_t llm_kv_cache_mb;
    size_t asr_buffer_mb;
    size_t tts_buffer_mb;
    size_t scratch_mb;
} rac_pipeline_memory_config_t;

typedef struct {
    const char* asr_model_path;
    const char* asr_language;
    const char* llm_model_path;
    const char* system_prompt;
    int llm_max_tokens;
    float llm_temperature;
    const char* tts_model_path;
    const char* tts_voice_id;
    int tts_sample_rate;
} rac_pipeline_model_config_t;

typedef struct {
    rac_pipeline_memory_config_t memory;
    rac_pipeline_model_config_t models;
    int num_threads;
} rac_hybrid_pipeline_config_t;

// =============================================================================
// Result Structures
// =============================================================================

typedef struct {
    const char* text;
    float confidence;
    double latency_ms;
} rac_asr_result_t;

typedef struct {
    const char* text;
    int tokens_generated;
    double latency_ms;
} rac_llm_result_t;

typedef struct {
    const float* audio;
    size_t num_samples;
    int sample_rate;
    double latency_ms;
} rac_tts_result_t;

typedef struct {
    rac_asr_result_t asr;
    rac_llm_result_t llm;
    rac_tts_result_t tts;
    double total_latency_ms;
    int success;
    const char* error_message;
} rac_pipeline_result_t;

// =============================================================================
// Callbacks
// =============================================================================

typedef void (*rac_asr_callback_fn)(const char* text, float confidence, void* user_data);
typedef int (*rac_llm_token_callback_fn)(const char* token, void* user_data);
typedef int (*rac_tts_chunk_callback_fn)(const float* audio, size_t samples, void* user_data);
typedef void (*rac_pipeline_complete_callback_fn)(const rac_pipeline_result_t* result, void* user_data);

// =============================================================================
// Lifecycle
// =============================================================================

/**
 * @brief Create a new hybrid pipeline instance
 */
rac_hybrid_pipeline_handle_t rac_hybrid_pipeline_create(void);

/**
 * @brief Destroy a pipeline instance
 */
void rac_hybrid_pipeline_destroy(rac_hybrid_pipeline_handle_t handle);

/**
 * @brief Initialize the pipeline
 */
int rac_hybrid_pipeline_initialize(
    rac_hybrid_pipeline_handle_t handle,
    const rac_hybrid_pipeline_config_t* config);

/**
 * @brief Check if pipeline is ready
 */
int rac_hybrid_pipeline_is_ready(rac_hybrid_pipeline_handle_t handle);

/**
 * @brief Cleanup and release resources
 */
void rac_hybrid_pipeline_cleanup(rac_hybrid_pipeline_handle_t handle);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Process audio through full pipeline
 * @note result.tts.audio is owned by the pipeline and valid until next call
 */
rac_pipeline_result_t rac_hybrid_pipeline_process(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples);

/**
 * @brief Process with streaming callbacks
 */
int rac_hybrid_pipeline_process_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples,
    rac_asr_callback_fn on_asr_complete,
    rac_llm_token_callback_fn on_llm_token,
    rac_tts_chunk_callback_fn on_tts_chunk,
    rac_pipeline_complete_callback_fn on_complete,
    void* user_data);

// =============================================================================
// Individual Stages
// =============================================================================

/**
 * @brief Transcribe audio only
 */
rac_asr_result_t rac_hybrid_pipeline_transcribe(
    rac_hybrid_pipeline_handle_t handle,
    const float* audio,
    size_t num_samples);

/**
 * @brief Generate LLM response only
 */
rac_llm_result_t rac_hybrid_pipeline_generate(
    rac_hybrid_pipeline_handle_t handle,
    const char* input);

/**
 * @brief Generate with streaming
 */
int rac_hybrid_pipeline_generate_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const char* input,
    rac_llm_token_callback_fn callback,
    void* user_data);

/**
 * @brief Synthesize audio from text
 */
rac_tts_result_t rac_hybrid_pipeline_synthesize(
    rac_hybrid_pipeline_handle_t handle,
    const char* text);

/**
 * @brief Synthesize with streaming
 */
int rac_hybrid_pipeline_synthesize_streaming(
    rac_hybrid_pipeline_handle_t handle,
    const char* text,
    rac_tts_chunk_callback_fn callback,
    void* user_data);

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief Reset reusable memory segments
 */
void rac_hybrid_pipeline_reset_for_next_utterance(rac_hybrid_pipeline_handle_t handle);

/**
 * @brief Get total allocated memory
 */
size_t rac_hybrid_pipeline_get_total_memory(rac_hybrid_pipeline_handle_t handle);

/**
 * @brief Get currently used memory
 */
size_t rac_hybrid_pipeline_get_memory_used(rac_hybrid_pipeline_handle_t handle);

// =============================================================================
// Model Management
// =============================================================================

int rac_hybrid_pipeline_reload_asr(rac_hybrid_pipeline_handle_t handle, const char* path);
int rac_hybrid_pipeline_reload_llm(rac_hybrid_pipeline_handle_t handle, const char* path);
int rac_hybrid_pipeline_reload_tts(rac_hybrid_pipeline_handle_t handle, const char* path);
void rac_hybrid_pipeline_set_system_prompt(rac_hybrid_pipeline_handle_t handle, const char* prompt);

#ifdef __cplusplus
}
#endif

#endif  // RAC_HYBRID_VOICE_PIPELINE_API_H
