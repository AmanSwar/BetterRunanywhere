/**
 * @file test_unified_memory_manager.cpp
 * @brief Unit tests for MemoryPoolConfig, MemorySegment, and UnifiedMemoryManager
 *
 * Tests memory pool allocation, segment management, thread safety,
 * and edge cases for the unified memory management system.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <cstdint>

#include "rac/core/unified_memory_manager.h"

using namespace rac::core;

// =============================================================================
// A. MemoryPoolConfig Tests
// =============================================================================

TEST(MemoryPoolConfigTest, DefaultValues) {
    MemoryPoolConfig config;
    EXPECT_EQ(config.llm_kv_cache_size, 400UL * 1024 * 1024);
    EXPECT_EQ(config.asr_buffer_size, 100UL * 1024 * 1024);
    EXPECT_EQ(config.tts_buffer_size, 50UL * 1024 * 1024);
    EXPECT_EQ(config.scratch_size, 50UL * 1024 * 1024);
    EXPECT_EQ(config.alignment, 16384UL);
}

TEST(MemoryPoolConfigTest, TotalSizeCalculation) {
    MemoryPoolConfig config;
    // 400 + 100 + 50 + 50 = 600 MB
    size_t expected = (400UL + 100 + 50 + 50) * 1024 * 1024;
    EXPECT_EQ(config.total_size(), expected);
}

TEST(MemoryPoolConfigTest, CustomConfigTotalSize) {
    MemoryPoolConfig config;
    config.llm_kv_cache_size = 10 * 1024 * 1024;   // 10 MB
    config.asr_buffer_size   =  5 * 1024 * 1024;   //  5 MB
    config.tts_buffer_size   =  3 * 1024 * 1024;   //  3 MB
    config.scratch_size      =  2 * 1024 * 1024;   //  2 MB

    size_t expected = (10 + 5 + 3 + 2) * 1024UL * 1024;
    EXPECT_EQ(config.total_size(), expected);
}

// =============================================================================
// B. MemorySegment Standalone Tests
// =============================================================================

class MemorySegmentTest : public ::testing::Test {
protected:
    static constexpr size_t kBufSize = 8192;
    alignas(128) uint8_t buf_[kBufSize]{};
    MemorySegment seg_;

    void SetUp() override {
        // Reset segment to a valid state for each test
        seg_.base = buf_;
        seg_.size = kBufSize;
        seg_.used.store(0, std::memory_order_release);
    }
};

TEST_F(MemorySegmentTest, DefaultConstructedIsInvalid) {
    MemorySegment default_seg;
    EXPECT_FALSE(default_seg.is_valid());
}

TEST_F(MemorySegmentTest, PopulatedSegmentIsValid) {
    EXPECT_TRUE(seg_.is_valid());
}

TEST_F(MemorySegmentTest, AllocateReturnsAlignedPointer) {
    void* ptr = seg_.allocate(128, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST_F(MemorySegmentTest, AllocateReturnsNullptrWhenExhausted) {
    // Fill the segment
    void* ptr = seg_.allocate(kBufSize, 8);
    ASSERT_NE(ptr, nullptr);

    // Next allocation should fail
    EXPECT_EQ(seg_.allocate(1, 8), nullptr);
}

TEST_F(MemorySegmentTest, AllocateZeroBytesReturnsNullptr) {
    EXPECT_EQ(seg_.allocate(0, 8), nullptr);
}

TEST_F(MemorySegmentTest, ResetMakesSpaceAvailable) {
    seg_.allocate(4096, 8);
    EXPECT_LT(seg_.available(), kBufSize);

    seg_.reset();
    EXPECT_EQ(seg_.available(), kBufSize);
    EXPECT_EQ(seg_.used.load(std::memory_order_acquire), 0u);
}

TEST_F(MemorySegmentTest, AvailableTracksUsage) {
    EXPECT_EQ(seg_.available(), kBufSize);

    seg_.allocate(1024, 8);
    // available should have decreased (exact amount depends on alignment padding)
    EXPECT_LE(seg_.available(), kBufSize - 1024);
    EXPECT_GT(seg_.available(), 0u);
}

TEST_F(MemorySegmentTest, ConcurrentAllocations) {
    // Use a larger buffer for concurrent test
    static constexpr size_t kLargeBuf = 1024 * 1024;  // 1 MB
    std::vector<uint8_t> large_buf(kLargeBuf + 128);
    // Align manually
    uintptr_t raw = reinterpret_cast<uintptr_t>(large_buf.data());
    uintptr_t aligned = (raw + 127) & ~uintptr_t(127);

    MemorySegment seg;
    seg.base = reinterpret_cast<void*>(aligned);
    seg.size = kLargeBuf;
    seg.used.store(0, std::memory_order_release);

    constexpr int kNumThreads = 4;
    constexpr int kAllocsPerThread = 100;
    constexpr size_t kAllocSize = 256;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&seg, &success_count, &failure_count]() {
            for (int i = 0; i < kAllocsPerThread; ++i) {
                void* ptr = seg.allocate(kAllocSize, 8);
                if (ptr) {
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

    // All threads completed
    EXPECT_EQ(success_count + failure_count, kNumThreads * kAllocsPerThread);
    // Some should succeed (1MB / 256 bytes = up to ~4096 allocations possible, we do 400)
    EXPECT_GT(success_count.load(), 0);
    // Used should be consistent
    EXPECT_LE(seg.used.load(std::memory_order_acquire), kLargeBuf);
}

// =============================================================================
// C-I. UnifiedMemoryManager Tests
// =============================================================================

class UnifiedMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager_ = std::make_unique<UnifiedMemoryManager>();
    }

    void TearDown() override {
        // Destructor calls cleanup() automatically
        manager_.reset();
    }

    // Helper: small config to keep test memory usage low
    MemoryPoolConfig small_config() {
        MemoryPoolConfig config;
        config.llm_kv_cache_size = 4 * 1024 * 1024;   // 4 MB
        config.asr_buffer_size   = 2 * 1024 * 1024;   // 2 MB
        config.tts_buffer_size   = 1 * 1024 * 1024;   // 1 MB
        config.scratch_size      = 1 * 1024 * 1024;   // 1 MB
        return config;
    }

    std::unique_ptr<UnifiedMemoryManager> manager_;
};

// ---------------------------------------------------------------------------
// C. Initialization Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, InitializeWithDefaultConfig) {
    ASSERT_TRUE(manager_->initialize());
    EXPECT_TRUE(manager_->is_initialized());
}

TEST_F(UnifiedMemoryManagerTest, InitializeWithCustomConfig) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));
    EXPECT_TRUE(manager_->is_initialized());

    const auto& stored = manager_->get_config();
    EXPECT_EQ(stored.llm_kv_cache_size, config.llm_kv_cache_size);
    EXPECT_EQ(stored.asr_buffer_size, config.asr_buffer_size);
    EXPECT_EQ(stored.tts_buffer_size, config.tts_buffer_size);
    EXPECT_EQ(stored.scratch_size, config.scratch_size);
}

TEST_F(UnifiedMemoryManagerTest, DoubleInitializationFails) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));
    EXPECT_FALSE(manager_->initialize(config));
}

TEST_F(UnifiedMemoryManagerTest, InitializeWithZeroSizesFails) {
    MemoryPoolConfig config;
    config.llm_kv_cache_size = 0;
    config.asr_buffer_size = 0;
    config.tts_buffer_size = 0;
    config.scratch_size = 0;

    EXPECT_FALSE(manager_->initialize(config));
    EXPECT_FALSE(manager_->is_initialized());
}

// ---------------------------------------------------------------------------
// D. Segment Layout Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, AllSegmentsAreValidAfterInit) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    EXPECT_TRUE(manager_->get_llm_segment().is_valid());
    EXPECT_TRUE(manager_->get_asr_segment().is_valid());
    EXPECT_TRUE(manager_->get_tts_segment().is_valid());
    EXPECT_TRUE(manager_->get_scratch_segment().is_valid());
}

TEST_F(UnifiedMemoryManagerTest, SegmentsHaveCorrectSizes) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    EXPECT_EQ(manager_->get_llm_segment().size, config.llm_kv_cache_size);
    EXPECT_EQ(manager_->get_asr_segment().size, config.asr_buffer_size);
    EXPECT_EQ(manager_->get_tts_segment().size, config.tts_buffer_size);
    EXPECT_EQ(manager_->get_scratch_segment().size, config.scratch_size);
}

TEST_F(UnifiedMemoryManagerTest, SegmentsAreContiguous) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    const auto& llm     = manager_->get_llm_segment();
    const auto& asr     = manager_->get_asr_segment();
    const auto& tts     = manager_->get_tts_segment();
    const auto& scratch = manager_->get_scratch_segment();

    auto* llm_end = static_cast<const uint8_t*>(llm.base) + llm.size;
    auto* asr_end = static_cast<const uint8_t*>(asr.base) + asr.size;
    auto* tts_end = static_cast<const uint8_t*>(tts.base) + tts.size;

    EXPECT_EQ(llm_end, static_cast<const uint8_t*>(asr.base));
    EXPECT_EQ(asr_end, static_cast<const uint8_t*>(tts.base));
    EXPECT_EQ(tts_end, static_cast<const uint8_t*>(scratch.base));
}

TEST_F(UnifiedMemoryManagerTest, PoolIs16KBAligned) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    const auto& llm = manager_->get_llm_segment();
    // LLM segment starts at the pool base, which is posix_memalign'd to 16KB
    EXPECT_EQ(reinterpret_cast<uintptr_t>(llm.base) % 16384, 0u);
}

TEST_F(UnifiedMemoryManagerTest, AllSegmentsHaveZeroUsedAfterInit) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    EXPECT_EQ(manager_->get_llm_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_asr_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_tts_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_scratch_segment().used.load(std::memory_order_acquire), 0u);
}

// ---------------------------------------------------------------------------
// E. Allocation Within Segments Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, AllocateFromSegmentReturnsAlignedPointer) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& scratch = const_cast<MemorySegment&>(manager_->get_scratch_segment());
    void* ptr = scratch.allocate(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST_F(UnifiedMemoryManagerTest, AllocateAlignmentVariations) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& scratch = const_cast<MemorySegment&>(manager_->get_scratch_segment());

    void* ptr16  = scratch.allocate(256, 16);
    void* ptr64  = scratch.allocate(256, 64);
    void* ptr128 = scratch.allocate(256, 128);

    ASSERT_NE(ptr16, nullptr);
    ASSERT_NE(ptr64, nullptr);
    ASSERT_NE(ptr128, nullptr);

    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr16) % 16, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr64) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr128) % 128, 0u);
}

TEST_F(UnifiedMemoryManagerTest, AllocationExceedsSegmentSize) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    // scratch is 1 MB, try to allocate 2 MB
    auto& scratch = const_cast<MemorySegment&>(manager_->get_scratch_segment());
    void* ptr = scratch.allocate(2 * 1024 * 1024, 8);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(UnifiedMemoryManagerTest, MultipleAllocationsTrackUsed) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& asr = const_cast<MemorySegment&>(manager_->get_asr_segment());

    void* ptr1 = asr.allocate(1024, 8);
    void* ptr2 = asr.allocate(2048, 8);
    void* ptr3 = asr.allocate(512, 8);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);

    // All pointers are different
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);

    // Used should be at least the sum of allocations
    EXPECT_GE(asr.used.load(std::memory_order_acquire), 1024u + 2048 + 512);
}

// ---------------------------------------------------------------------------
// F. Reset Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, ResetASRSegment) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& asr = const_cast<MemorySegment&>(manager_->get_asr_segment());
    asr.allocate(1024, 8);
    EXPECT_GT(asr.used.load(std::memory_order_acquire), 0u);

    manager_->reset_asr_segment();
    EXPECT_EQ(manager_->get_asr_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_asr_segment().available(), config.asr_buffer_size);
}

TEST_F(UnifiedMemoryManagerTest, ResetTTSSegment) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& tts = const_cast<MemorySegment&>(manager_->get_tts_segment());
    tts.allocate(1024, 8);
    EXPECT_GT(tts.used.load(std::memory_order_acquire), 0u);

    manager_->reset_tts_segment();
    EXPECT_EQ(manager_->get_tts_segment().used.load(std::memory_order_acquire), 0u);
}

TEST_F(UnifiedMemoryManagerTest, ResetScratchSegment) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& scratch = const_cast<MemorySegment&>(manager_->get_scratch_segment());
    scratch.allocate(1024, 8);
    EXPECT_GT(scratch.used.load(std::memory_order_acquire), 0u);

    manager_->reset_scratch();
    EXPECT_EQ(manager_->get_scratch_segment().used.load(std::memory_order_acquire), 0u);
}

TEST_F(UnifiedMemoryManagerTest, ResetReusableSegmentsPreservesLLM) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    // Allocate in all segments
    const_cast<MemorySegment&>(manager_->get_llm_segment()).allocate(1024, 8);
    const_cast<MemorySegment&>(manager_->get_asr_segment()).allocate(1024, 8);
    const_cast<MemorySegment&>(manager_->get_tts_segment()).allocate(1024, 8);
    const_cast<MemorySegment&>(manager_->get_scratch_segment()).allocate(1024, 8);

    manager_->reset_reusable_segments();

    // LLM should NOT be reset
    EXPECT_GT(manager_->get_llm_segment().used.load(std::memory_order_acquire), 0u);

    // ASR, TTS, scratch should be reset
    EXPECT_EQ(manager_->get_asr_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_tts_segment().used.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(manager_->get_scratch_segment().used.load(std::memory_order_acquire), 0u);
}

// ---------------------------------------------------------------------------
// G. Statistics Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, GetTotalSizeMatchesConfig) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    EXPECT_EQ(manager_->get_total_size(), config.total_size());
}

TEST_F(UnifiedMemoryManagerTest, GetTotalUsedAggregatesAllSegments) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    // Initially zero
    EXPECT_EQ(manager_->get_total_used(), 0u);

    // Allocate from multiple segments
    const_cast<MemorySegment&>(manager_->get_llm_segment()).allocate(1024, 8);
    const_cast<MemorySegment&>(manager_->get_asr_segment()).allocate(512, 8);
    const_cast<MemorySegment&>(manager_->get_scratch_segment()).allocate(256, 8);

    size_t total_used = manager_->get_total_used();
    EXPECT_GE(total_used, 1024u + 512 + 256);

    // Verify it equals the sum of individual segment usage
    size_t manual_sum =
        manager_->get_llm_segment().used.load(std::memory_order_acquire) +
        manager_->get_asr_segment().used.load(std::memory_order_acquire) +
        manager_->get_tts_segment().used.load(std::memory_order_acquire) +
        manager_->get_scratch_segment().used.load(std::memory_order_acquire);
    EXPECT_EQ(total_used, manual_sum);
}

// ---------------------------------------------------------------------------
// H. Cleanup and Re-initialization Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, CleanupReleasesMemory) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));
    EXPECT_TRUE(manager_->is_initialized());

    manager_->cleanup();
    EXPECT_FALSE(manager_->is_initialized());
}

TEST_F(UnifiedMemoryManagerTest, ReinitializeAfterCleanup) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    manager_->cleanup();
    EXPECT_FALSE(manager_->is_initialized());

    // Reinitialize with different config
    MemoryPoolConfig config2;
    config2.llm_kv_cache_size = 2 * 1024 * 1024;
    config2.asr_buffer_size   = 1 * 1024 * 1024;
    config2.tts_buffer_size   = 1 * 1024 * 1024;
    config2.scratch_size      = 1 * 1024 * 1024;

    ASSERT_TRUE(manager_->initialize(config2));
    EXPECT_TRUE(manager_->is_initialized());
    EXPECT_EQ(manager_->get_config().llm_kv_cache_size, config2.llm_kv_cache_size);
    EXPECT_TRUE(manager_->get_llm_segment().is_valid());
}

TEST_F(UnifiedMemoryManagerTest, DoubleCleanupIsSafe) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    manager_->cleanup();
    // Second cleanup should not crash
    manager_->cleanup();
    EXPECT_FALSE(manager_->is_initialized());
}

// ---------------------------------------------------------------------------
// I. Thread Safety Tests
// ---------------------------------------------------------------------------

TEST_F(UnifiedMemoryManagerTest, ConcurrentSegmentAllocations) {
    auto config = small_config();
    ASSERT_TRUE(manager_->initialize(config));

    auto& scratch = const_cast<MemorySegment&>(manager_->get_scratch_segment());

    constexpr int kNumThreads = 4;
    constexpr int kAllocsPerThread = 100;
    constexpr size_t kAllocSize = 64;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&scratch, &success_count, &failure_count]() {
            for (int i = 0; i < kAllocsPerThread; ++i) {
                void* ptr = scratch.allocate(kAllocSize, 8);
                if (ptr) {
                    // Verify pointer is within segment bounds
                    auto addr = reinterpret_cast<uintptr_t>(ptr);
                    auto base = reinterpret_cast<uintptr_t>(scratch.base);
                    EXPECT_GE(addr, base);
                    EXPECT_LT(addr + kAllocSize, base + scratch.size);
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

    EXPECT_EQ(success_count + failure_count, kNumThreads * kAllocsPerThread);
    EXPECT_GT(success_count.load(), 0);
    // Used should not exceed segment size
    EXPECT_LE(scratch.used.load(std::memory_order_acquire), scratch.size);
}
