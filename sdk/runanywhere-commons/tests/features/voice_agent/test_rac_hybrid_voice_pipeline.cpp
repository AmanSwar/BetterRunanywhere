/**
 * @file test_rac_hybrid_voice_pipeline.cpp
 * @brief Unit tests for C API of HybridVoicePipeline
 *
 * Tests the C API wrapper functions exposed for FFI and JNI integration.
 */

#include <gtest/gtest.h>
#include <cstring>

#include "rac/features/voice_agent/rac_hybrid_voice_pipeline.h"

class RacHybridVoicePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        handle_ = rac_hybrid_pipeline_create();
        ASSERT_NE(handle_, nullptr);
    }

    void TearDown() override {
        if (handle_) {
            rac_hybrid_pipeline_destroy(handle_);
            handle_ = nullptr;
        }
    }

    rac_hybrid_pipeline_handle_t handle_ = nullptr;
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, CreateAndDestroy) {
    rac_hybrid_pipeline_handle_t h = rac_hybrid_pipeline_create();
    ASSERT_NE(h, nullptr);
    rac_hybrid_pipeline_destroy(h);
}

TEST_F(RacHybridVoicePipelineTest, DestroyNull) {
    // Should not crash
    rac_hybrid_pipeline_destroy(nullptr);
}

TEST_F(RacHybridVoicePipelineTest, InitializeWithDefaultConfig) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;
    config.num_threads = 4;

    int result = rac_hybrid_pipeline_initialize(handle_, &config);
    // May fail without models, but should not crash
}

TEST_F(RacHybridVoicePipelineTest, InitializeWithNullConfig) {
    int result = rac_hybrid_pipeline_initialize(handle_, nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(RacHybridVoicePipelineTest, InitializeWithNullHandle) {
    rac_hybrid_pipeline_config_t config = {};
    int result = rac_hybrid_pipeline_initialize(nullptr, &config);
    EXPECT_EQ(result, -1);
}

TEST_F(RacHybridVoicePipelineTest, IsReadyBeforeInit) {
    EXPECT_EQ(rac_hybrid_pipeline_is_ready(handle_), 0);
}

TEST_F(RacHybridVoicePipelineTest, IsReadyNullHandle) {
    EXPECT_EQ(rac_hybrid_pipeline_is_ready(nullptr), 0);
}

TEST_F(RacHybridVoicePipelineTest, CleanupBeforeInit) {
    // Should not crash
    rac_hybrid_pipeline_cleanup(handle_);
}

TEST_F(RacHybridVoicePipelineTest, CleanupNullHandle) {
    rac_hybrid_pipeline_cleanup(nullptr);
}

// =============================================================================
// Processing Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, ProcessNullHandle) {
    float audio[100] = {0};
    auto result = rac_hybrid_pipeline_process(nullptr, audio, 100);
    EXPECT_EQ(result.success, 0);
}

TEST_F(RacHybridVoicePipelineTest, ProcessEmptyAudio) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        float audio[1] = {0};
        auto result = rac_hybrid_pipeline_process(handle_, audio, 0);
        // Should handle gracefully
    }
}

// =============================================================================
// Transcription Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, TranscribeNullHandle) {
    float audio[100] = {0};
    auto result = rac_hybrid_pipeline_transcribe(nullptr, audio, 100);
    EXPECT_EQ(result.text, nullptr);
}

TEST_F(RacHybridVoicePipelineTest, TranscribeEmptyAudio) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        float audio[1] = {0};
        auto result = rac_hybrid_pipeline_transcribe(handle_, audio, 0);
        // Should return valid result
    }
}

// =============================================================================
// Generation Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, GenerateNullHandle) {
    auto result = rac_hybrid_pipeline_generate(nullptr, "Hello");
    EXPECT_EQ(result.text, nullptr);
}

TEST_F(RacHybridVoicePipelineTest, GenerateNullInput) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        auto result = rac_hybrid_pipeline_generate(handle_, nullptr);
        // Should handle gracefully
    }
}

// =============================================================================
// Synthesis Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, SynthesizeNullHandle) {
    auto result = rac_hybrid_pipeline_synthesize(nullptr, "Hello");
    EXPECT_EQ(result.audio, nullptr);
    EXPECT_EQ(result.num_samples, 0);
}

TEST_F(RacHybridVoicePipelineTest, SynthesizeNullText) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        auto result = rac_hybrid_pipeline_synthesize(handle_, nullptr);
        // Should handle gracefully
    }
}

// =============================================================================
// Streaming Tests
// =============================================================================

static int test_llm_callback(const char* token, void* user_data) {
    int* count = static_cast<int*>(user_data);
    (*count)++;
    return 1;  // Continue
}

static int test_tts_callback(const float* audio, size_t samples, void* user_data) {
    int* count = static_cast<int*>(user_data);
    (*count)++;
    return 1;
}

TEST_F(RacHybridVoicePipelineTest, GenerateStreamingNullCallback) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        int result = rac_hybrid_pipeline_generate_streaming(handle_, "Hello", nullptr, nullptr);
        EXPECT_EQ(result, -1);
    }
}

TEST_F(RacHybridVoicePipelineTest, SynthesizeStreamingNullCallback) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        int result = rac_hybrid_pipeline_synthesize_streaming(handle_, "Hello", nullptr, nullptr);
        EXPECT_EQ(result, -1);
    }
}

// =============================================================================
// Memory Management Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, ResetForNextUtteranceNullHandle) {
    rac_hybrid_pipeline_reset_for_next_utterance(nullptr);
    // Should not crash
}

TEST_F(RacHybridVoicePipelineTest, GetTotalMemoryNullHandle) {
    size_t mem = rac_hybrid_pipeline_get_total_memory(nullptr);
    EXPECT_EQ(mem, 0);
}

TEST_F(RacHybridVoicePipelineTest, GetMemoryUsedNullHandle) {
    size_t mem = rac_hybrid_pipeline_get_memory_used(nullptr);
    EXPECT_EQ(mem, 0);
}

TEST_F(RacHybridVoicePipelineTest, GetMemoryAfterInit) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        size_t total = rac_hybrid_pipeline_get_total_memory(handle_);
        EXPECT_GT(total, 0);
    }
}

// =============================================================================
// Model Reload Tests
// =============================================================================

TEST_F(RacHybridVoicePipelineTest, ReloadASRNullHandle) {
    int result = rac_hybrid_pipeline_reload_asr(nullptr, "/path/to/model");
    EXPECT_EQ(result, -1);
}

TEST_F(RacHybridVoicePipelineTest, ReloadASRNullPath) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        int result = rac_hybrid_pipeline_reload_asr(handle_, nullptr);
        EXPECT_EQ(result, -1);
    }
}

TEST_F(RacHybridVoicePipelineTest, ReloadLLMInvalidPath) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        int result = rac_hybrid_pipeline_reload_llm(handle_, "/nonexistent/model.gguf");
        EXPECT_EQ(result, -1);
    }
}

TEST_F(RacHybridVoicePipelineTest, SetSystemPromptNullHandle) {
    rac_hybrid_pipeline_set_system_prompt(nullptr, "prompt");
    // Should not crash
}

TEST_F(RacHybridVoicePipelineTest, SetSystemPromptNullPrompt) {
    rac_hybrid_pipeline_config_t config = {};
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (rac_hybrid_pipeline_initialize(handle_, &config) == 0) {
        rac_hybrid_pipeline_set_system_prompt(handle_, nullptr);
        // Should not crash
    }
}
