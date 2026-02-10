/**
 * @file test_llamacpp_pooled_buffer.cpp
 * @brief Unit tests for LlamaCppPooledBuffer
 *
 * Tests the bump allocator used by the llama.cpp backend for KV cache
 * and scratch memory. LlamaCppPooledBuffer has no dependency on actual
 * llama.cpp - it only wraps a MemorySegment from UnifiedMemoryManager.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdint>

#include "rac/core/unified_memory_manager.h"
#include "backends/llamacpp/llamacpp_pooled_buffer.h"

class LlamaCppPooledBufferTest : public ::testing::Test {
protected:
    static constexpr size_t kLLMSize     = 4 * 1024 * 1024;  // 4 MB
    static constexpr size_t kASRSize     = 1 * 1024 * 1024;  // 1 MB
    static constexpr size_t kTTSSize     = 1 * 1024 * 1024;  // 1 MB
    static constexpr size_t kScratchSize = 1 * 1024 * 1024;  // 1 MB

    void SetUp() override {
        manager_ = std::make_unique<rac::core::UnifiedMemoryManager>();

        rac::core::MemoryPoolConfig config;
        config.llm_kv_cache_size = kLLMSize;
        config.asr_buffer_size   = kASRSize;
        config.tts_buffer_size   = kTTSSize;
        config.scratch_size      = kScratchSize;

        ASSERT_TRUE(manager_->initialize(config));

        auto& llm_segment = const_cast<rac::core::MemorySegment&>(
            manager_->get_llm_segment());
        buffer_ = std::make_unique<runanywhere::LlamaCppPooledBuffer>(llm_segment);
    }

    void TearDown() override {
        buffer_.reset();
        manager_.reset();
    }

    std::unique_ptr<rac::core::UnifiedMemoryManager> manager_;
    std::unique_ptr<runanywhere::LlamaCppPooledBuffer> buffer_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(LlamaCppPooledBufferTest, ConstructFromSegment) {
    EXPECT_NE(buffer_->get_base(), nullptr);
    EXPECT_EQ(buffer_->get_size(), kLLMSize);
    EXPECT_EQ(buffer_->get_used(), 0u);
    EXPECT_EQ(buffer_->get_available(), kLLMSize);
    EXPECT_EQ(buffer_->get_allocation_count(), 0);
}

// =============================================================================
// Allocation Tests
// =============================================================================

TEST_F(LlamaCppPooledBufferTest, AllocateReturnsAlignedPointer) {
    void* ptr = buffer_->allocate(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST_F(LlamaCppPooledBufferTest, AllocateRespectsBounds) {
    // Allocate 3 MB from 4 MB buffer
    void* ptr1 = buffer_->allocate(3 * 1024 * 1024, 64);
    ASSERT_NE(ptr1, nullptr);

    // Try to allocate 2 MB more - should fail
    void* ptr2 = buffer_->allocate(2 * 1024 * 1024, 64);
    EXPECT_EQ(ptr2, nullptr);
}

TEST_F(LlamaCppPooledBufferTest, MultipleAllocationsTrackCount) {
    void* ptr1 = buffer_->allocate(1024, 64);
    void* ptr2 = buffer_->allocate(2048, 64);
    void* ptr3 = buffer_->allocate(512, 64);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);

    // All pointers should be different
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);

    EXPECT_EQ(buffer_->get_allocation_count(), 3);
    EXPECT_GE(buffer_->get_used(), 1024u + 2048 + 512);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(LlamaCppPooledBufferTest, ResetRestoresFullCapacity) {
    buffer_->allocate(2 * 1024 * 1024, 64);
    EXPECT_GT(buffer_->get_used(), 0u);
    EXPECT_GT(buffer_->get_allocation_count(), 0);

    buffer_->reset();

    EXPECT_EQ(buffer_->get_used(), 0u);
    EXPECT_EQ(buffer_->get_available(), kLLMSize);
    EXPECT_EQ(buffer_->get_allocation_count(), 0);
}

TEST_F(LlamaCppPooledBufferTest, AllocateAfterReset) {
    void* ptr1 = buffer_->allocate(1024, 64);
    ASSERT_NE(ptr1, nullptr);

    buffer_->reset();

    void* ptr2 = buffer_->allocate(1024, 64);
    ASSERT_NE(ptr2, nullptr);

    // After reset, the bump allocator starts from offset 0 again,
    // so with the same alignment the pointer should be the same
    EXPECT_EQ(ptr1, ptr2);
}

// =============================================================================
// KV Cache Calculation Tests
// =============================================================================

TEST_F(LlamaCppPooledBufferTest, CalculateKVCacheSize) {
    // Small model: n_ctx=32, n_layer=2, n_embd=64, n_head_kv=2
    size_t kv_size = runanywhere::LlamaCppPooledBuffer::calculate_kv_cache_size(
        32, 2, 64, 2);

    // Formula: n_ctx * n_layer * n_embd * 2 * sizeof(float) * 1.1
    // = 32 * 2 * 64 * 2 * 4 * 1.1 = 36044.8 -> ~36045 bytes
    EXPECT_GT(kv_size, 0u);

    // Verify it's roughly in the expected range
    size_t base = 32UL * 2 * 64 * 2 * sizeof(float);
    size_t with_margin = base + (base / 10);
    EXPECT_EQ(kv_size, with_margin);
}

TEST_F(LlamaCppPooledBufferTest, ReserveKVCacheForSmallModel) {
    // Small model that fits in our 4 MB buffer
    void* kv_ptr = buffer_->reserve_kv_cache(32, 2, 64, 2);
    ASSERT_NE(kv_ptr, nullptr);

    // Should have consumed some memory
    EXPECT_GT(buffer_->get_used(), 0u);
    EXPECT_EQ(buffer_->get_allocation_count(), 1);

    // Pointer should be aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(kv_ptr) % 64, 0u);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(LlamaCppPooledBufferTest, ConcurrentAllocations) {
    constexpr int kNumThreads = 4;
    constexpr int kAllocsPerThread = 50;
    constexpr size_t kAllocSize = 256;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([this, &success_count, &failure_count]() {
            for (int i = 0; i < kAllocsPerThread; ++i) {
                void* ptr = buffer_->allocate(kAllocSize, 64);
                if (ptr) {
                    // Verify pointer is within buffer bounds
                    auto addr = reinterpret_cast<uintptr_t>(ptr);
                    auto base = reinterpret_cast<uintptr_t>(buffer_->get_base());
                    EXPECT_GE(addr, base);
                    EXPECT_LT(addr + kAllocSize, base + buffer_->get_size());
                    success_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    failure_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load() + failure_count.load(),
              kNumThreads * kAllocsPerThread);
    EXPECT_GT(success_count.load(), 0);
    EXPECT_LE(buffer_->get_used(), buffer_->get_size());
}
