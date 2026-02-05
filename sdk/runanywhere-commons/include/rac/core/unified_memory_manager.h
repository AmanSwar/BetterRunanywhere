/**
 * @file unified_memory_manager.h
 * @brief Unified Memory Manager for Voice AI Pipeline
 *
 * Provides a single pre-allocated memory pool that is partitioned into segments
 * for different inference engines (llama.cpp KV cache, ExecuTorch ASR/TTS buffers).
 * This eliminates runtime malloc/free during inference for optimal mobile performance.
 *
 * Memory Layout:
 * ┌──────────────────────────────────────────────────────────────────┐
 * │                        Total Pool                                │
 * ├─────────────────────┬─────────────────┬───────────────┬──────────┤
 * │  LLM KV Cache       │  ASR Buffer     │  TTS Buffer   │ Scratch  │
 * │  (llama.cpp)        │  (ExecuTorch)   │  (ExecuTorch) │ (shared) │
 * └─────────────────────┴─────────────────┴───────────────┴──────────┘
 */

#ifndef RAC_CORE_UNIFIED_MEMORY_MANAGER_H
#define RAC_CORE_UNIFIED_MEMORY_MANAGER_H

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <mutex>

namespace rac {
namespace core {


struct MemoryPoolConfig {
    // LLM KV cache size for llama.cpp -> Default= 400 MB
    size_t llm_kv_cache_size = 400 * 1024 * 1024;

    // ASR/STT model buffer size for ExecuTorch (currently whisper models) -> Default= 100 MB
    size_t asr_buffer_size = 100 * 1024 * 1024;

    // TTS model buffer size for ExecuTorch (currently piper models) -> Default= 50 MB
    size_t tts_buffer_size = 50 * 1024 * 1024;

    //extra space for temporary allocations -> Default= 50 MB
    size_t scratch_size = 50 * 1024 * 1024;

    // Memory alignment (16KB for Android 15+ compliance) -> Default= 16384
    size_t alignment = 16384;

    size_t total_size() const {
        return llm_kv_cache_size + asr_buffer_size + tts_buffer_size + scratch_size;
    }
};

// memory segment handle with bound checkings
struct MemorySegment {
    void* base = nullptr;
    size_t size = 0;
    std::atomic<size_t> used{0};  //for bump allocator within segment

    bool is_valid() const { return base != nullptr && size > 0; }

    // Reset usage counter (does not clear memory)
    void reset() { used.store(0, std::memory_order_release); }

    // Bump allocate within segment (thread-safe)
    void* allocate(size_t bytes, size_t align = 8);

    // Get remaining available space
    size_t available() const {
        return size - used.load(std::memory_order_acquire);
    }
};
/*
// Unified Memory Manager
//
// Allocates a single contiguous memory pool at initialization and partitions
// it into segments for each inference engine. 
 *
 * Usage:
 * @code
 *   UnifiedMemoryManager mgr;
 *   MemoryPoolConfig config;
 *   config.llm_kv_cache_size = 512 * 1024 * 1024;  // 512 MB
 *
 *   if (!mgr.initialize(config)) {
 *       // Handle allocation failure
 *   }
 *
 *   // Pass segments to backends
 *   llama_cpp_backend.set_kv_buffer(mgr.get_llm_segment());
 *   executorch_backend.set_asr_buffer(mgr.get_asr_segment());
 *   executorch_backend.set_tts_buffer(mgr.get_tts_segment());
 *
 *   // After each utterance, reset scratch and reusable segments
 *   mgr.reset_scratch();
 * @endcode
 */
class UnifiedMemoryManager {
public:
    UnifiedMemoryManager() = default;
    ~UnifiedMemoryManager();

    // Non-copyable, non-movable (owns raw memory)
    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager(UnifiedMemoryManager&&) = delete;
    UnifiedMemoryManager& operator=(UnifiedMemoryManager&&) = delete;

    /**
     * @brief Initialize the memory pool
     * @param config Pool configuration
     * @return true if allocation succeeded, false otherwise
     *
     * Allocates a single contiguous block and partitions into segments.
     * Must be called before using any segment accessors.
     */
    bool initialize(const MemoryPoolConfig& config = MemoryPoolConfig{});

    /**
     * @brief Check if memory pool is initialized
     */
    bool is_initialized() const { return pool_ != nullptr; }

    /**
     * @brief Release all memory
     *
     * After cleanup, the manager can be re-initialized with a new config.
     */
    void cleanup();

    
    // Get LLM KV cache segment (for llama.cpp)
    const MemorySegment& get_llm_segment() const { return llm_segment_; }

    // Get ASR buffer segment
    const MemorySegment& get_asr_segment() const { return asr_segment_; }

    // Get TTS buffer segment
    const MemorySegment& get_tts_segment() const { return tts_segment_; }

    // Get scratch segment (shared temporary space)
    const MemorySegment& get_scratch_segment() const { return scratch_segment_; }


    // Reset ASR segment for next utterance (zero-cost)
    void reset_asr_segment() { asr_segment_.reset(); }

    // Reset TTS segment for next synthesis (zero-cost)
    void reset_tts_segment() { tts_segment_.reset(); }

    // Reset scratch segment (zero-cost)
    void reset_scratch() { scratch_segment_.reset(); }

    // Reset all reusable segments (ASR, TTS, scratch) - NOT LLM KV cache
    void reset_reusable_segments();

   
    // Get configuration used for initialization
    const MemoryPoolConfig& get_config() const { return config_; }

    // Get total allocated pool size
    size_t get_total_size() const { return config_.total_size(); }

    // Get total bytes currently in use across all segments
    size_t get_total_used() const;

private:
    void* pool_ = nullptr;
    MemoryPoolConfig config_;

    MemorySegment llm_segment_;
    MemorySegment asr_segment_;
    MemorySegment tts_segment_;
    MemorySegment scratch_segment_;

    mutable std::mutex mutex_;
};

}  // namespace core
}  // namespace rac

#endif  // RAC_CORE_UNIFIED_MEMORY_MANAGER_H
