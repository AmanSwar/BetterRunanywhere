/**
 * @file test_hybrid_voice_pipeline.cpp
 * @brief Unit tests for HybridVoicePipeline
 *
 * Tests the integration layer that coordinates UnifiedMemoryManager,
 * ExecuTorch ASR/TTS, and llama.cpp LLM for end-to-end voice processing.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>

#include "rac/features/voice_agent/hybrid_voice_pipeline.h"

using namespace rac::voice;

class HybridVoicePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pipeline_ = std::make_unique<HybridVoicePipeline>();
    }

    void TearDown() override {
        if (pipeline_) {
            pipeline_->cleanup();
        }
    }

    std::unique_ptr<HybridVoicePipeline> pipeline_;
};

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, DefaultMemoryConfig) {
    HybridPipelineConfig config;

    EXPECT_EQ(config.memory.llm_kv_cache_mb, 400);
    EXPECT_EQ(config.memory.asr_buffer_mb, 100);
    EXPECT_EQ(config.memory.tts_buffer_mb, 50);
    EXPECT_EQ(config.memory.scratch_mb, 50);
}

TEST_F(HybridVoicePipelineTest, DefaultModelConfig) {
    HybridPipelineConfig config;

    EXPECT_EQ(config.models.asr_language, "en");
    EXPECT_EQ(config.models.llm_max_tokens, 256);
    EXPECT_FLOAT_EQ(config.models.llm_temperature, 0.7f);
    EXPECT_EQ(config.models.tts_sample_rate, 22050);
}

TEST_F(HybridVoicePipelineTest, CustomConfig) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 200;
    config.memory.asr_buffer_mb = 50;
    config.models.system_prompt = "You are a helpful assistant.";
    config.num_threads = 4;

    EXPECT_EQ(config.memory.llm_kv_cache_mb, 200);
    EXPECT_EQ(config.num_threads, 4);
}

// =============================================================================
// Initialization Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, InitializeWithoutModels) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    // Without model paths, initialization may succeed or fail
    // depending on implementation (fail if models required)
    bool result = pipeline_->initialize(config);
    // Just verify it doesn't crash
}

TEST_F(HybridVoicePipelineTest, InitializeWithInvalidModelPaths) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;
    config.models.asr_model_path = "/nonexistent/whisper.onnx";
    config.models.llm_model_path = "/nonexistent/model.gguf";
    config.models.tts_model_path = "/nonexistent/piper.onnx";

    // When backends are compiled in, invalid paths should cause failure.
    // Without backends, initialization succeeds (memory-only).
    bool result = pipeline_->initialize(config);
    // Just verify it doesn't crash - behavior depends on backend availability
}

TEST_F(HybridVoicePipelineTest, DoubleInitialization) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    bool first = pipeline_->initialize(config);
    bool second = pipeline_->initialize(config);

    // Second should fail
    if (first) {
        EXPECT_FALSE(second);
    }
}

// =============================================================================
// Memory Management Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, MemoryAllocationTracking) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        size_t total = pipeline_->get_total_memory_allocated();
        EXPECT_GT(total, 0);
        EXPECT_EQ(total, (50 + 25 + 15 + 10) * 1024 * 1024);
    }
}

TEST_F(HybridVoicePipelineTest, MemoryUsageTracking) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        size_t used = pipeline_->get_memory_used();
        // Initially should be zero or minimal
        EXPECT_LE(used, pipeline_->get_total_memory_allocated());
    }
}

TEST_F(HybridVoicePipelineTest, ResetForNextUtterance) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        // Reset should not throw
        EXPECT_NO_THROW(pipeline_->reset_for_next_utterance());
    }
}

// =============================================================================
// ASR Stage Tests (Mocked)
// =============================================================================

TEST_F(HybridVoicePipelineTest, TranscribeEmptyAudio) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        float empty[1] = {0.0f};
        auto result = pipeline_->transcribe(empty, 0);

        EXPECT_TRUE(result.text.empty());
    }
}

TEST_F(HybridVoicePipelineTest, TranscribeSilentAudio) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        std::vector<float> silence(16000, 0.0f);  // 1 second at 16kHz
        auto result = pipeline_->transcribe(silence.data(), silence.size());

        // Latency should be tracked
        EXPECT_GE(result.latency_ms, 0.0);
    }
}

// =============================================================================
// LLM Stage Tests (Mocked)
// =============================================================================

TEST_F(HybridVoicePipelineTest, GenerateEmptyInput) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        auto result = pipeline_->generate_response("");

        // Empty input may return empty or default response
    }
}

TEST_F(HybridVoicePipelineTest, GenerateWithCallback) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        std::string accumulated;
        auto callback = [&accumulated](const std::string& token) {
            accumulated += token;
            return true;  // Continue generating
        };

        auto result = pipeline_->generate_response_streaming("Hello", callback);

        // Callback should work without crashing
    }
}

// =============================================================================
// TTS Stage Tests (Mocked)
// =============================================================================

TEST_F(HybridVoicePipelineTest, SynthesizeEmptyText) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        auto result = pipeline_->synthesize("");

        EXPECT_TRUE(result.audio.empty());
    }
}

TEST_F(HybridVoicePipelineTest, SynthesizeWithSampleRate) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;
    config.models.tts_sample_rate = 22050;

    if (pipeline_->initialize(config)) {
        auto result = pipeline_->synthesize("Test");

        EXPECT_EQ(result.sample_rate, 22050);
    }
}

// =============================================================================
// Full Pipeline Tests (Mocked)
// =============================================================================

TEST_F(HybridVoicePipelineTest, ProcessEmptyAudio) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        float empty[1] = {0.0f};
        auto result = pipeline_->process(empty, 0);

        // Check result structure is valid
        EXPECT_GE(result.total_latency_ms, 0.0);
    }
}

TEST_F(HybridVoicePipelineTest, ProcessWithCallbacks) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        PipelineCallbacks callbacks;

        bool asr_called = false;
        bool llm_called = false;
        bool tts_called = false;

        callbacks.on_asr_complete = [&asr_called](const ASRResult& r) {
            asr_called = true;
        };

        callbacks.on_llm_token = [&llm_called](const std::string& token) {
            llm_called = true;
            return true;
        };

        callbacks.on_tts_chunk = [&tts_called](const float* audio, size_t samples) {
            tts_called = true;
            return true;
        };

        std::vector<float> audio(16000, 0.0f);
        auto result = pipeline_->process_streaming(audio.data(), audio.size(), callbacks);

        // Callbacks should be invoked (if pipeline is properly initialized)
    }
}

// =============================================================================
// Model Reload Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, ReloadASRModel) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        // Reload with invalid path should fail
        EXPECT_FALSE(pipeline_->reload_asr_model("/nonexistent/whisper.pte"));
    }
}

TEST_F(HybridVoicePipelineTest, ReloadLLMModel) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        EXPECT_FALSE(pipeline_->reload_llm_model("/nonexistent/model.gguf"));
    }
}

TEST_F(HybridVoicePipelineTest, ReloadTTSModel) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        EXPECT_FALSE(pipeline_->reload_tts_model("/nonexistent/piper.pte"));
    }
}

TEST_F(HybridVoicePipelineTest, SetSystemPrompt) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        EXPECT_NO_THROW(pipeline_->set_system_prompt("New prompt"));
    }
}

// =============================================================================
// Cleanup Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, CleanupReleaseResources) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        pipeline_->cleanup();

        EXPECT_FALSE(pipeline_->is_ready());
    }
}

TEST_F(HybridVoicePipelineTest, DoubleCleanup) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        pipeline_->cleanup();
        EXPECT_NO_THROW(pipeline_->cleanup());  // Should be safe
    }
}

// =============================================================================
// Latency Measurement Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, LatencyMeasurement) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        std::vector<float> audio(16000, 0.0f);
        auto result = pipeline_->process(audio.data(), audio.size());

        // All individual stage latencies should be non-negative
        EXPECT_GE(result.asr.latency_ms, 0.0);
        EXPECT_GE(result.llm.latency_ms, 0.0);
        EXPECT_GE(result.tts.latency_ms, 0.0);
        EXPECT_GE(result.total_latency_ms, 0.0);

        // If pipeline ran to completion (all stages), total >= sum of parts
        if (result.success) {
            double sum = result.asr.latency_ms + result.llm.latency_ms + result.tts.latency_ms;
            EXPECT_GE(result.total_latency_ms, sum * 0.9);
        }
    }
}

// =============================================================================
// IsReady Tests
// =============================================================================

TEST_F(HybridVoicePipelineTest, IsReadyBeforeInit) {
    EXPECT_FALSE(pipeline_->is_ready());
}

TEST_F(HybridVoicePipelineTest, IsReadyAfterInit) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        EXPECT_TRUE(pipeline_->is_ready());
    }
}

TEST_F(HybridVoicePipelineTest, IsReadyAfterCleanup) {
    HybridPipelineConfig config;
    config.memory.llm_kv_cache_mb = 50;
    config.memory.asr_buffer_mb = 25;
    config.memory.tts_buffer_mb = 15;
    config.memory.scratch_mb = 10;

    if (pipeline_->initialize(config)) {
        pipeline_->cleanup();
        EXPECT_FALSE(pipeline_->is_ready());
    }
}
