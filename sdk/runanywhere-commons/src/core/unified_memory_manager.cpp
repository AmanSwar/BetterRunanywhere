/**
 * @file unified_memory_manager.cpp
 * @brief Implementation of UnifiedMemoryManager
 */

#include "rac/core/unified_memory_manager.h"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "UnifiedMemoryManager"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, "[UnifiedMemoryManager] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[UnifiedMemoryManager ERROR] " __VA_ARGS__)
#endif

namespace rac {
namespace core {

// =============================================================================
// MemorySegment Implementation
// =============================================================================
// args
// bytes : number of bytes to be allocated
// align : bytes it should be aligned with
void* MemorySegment::allocate(size_t bytes, size_t align) {

    //sanity chekc
    if (!is_valid() || bytes == 0) {
        return nullptr;
    }

    // Thread-safe bump allocation using atomic compare-exchange
    //ensure that we get the most up to date `used` count from other threads that
    // just fininhsed the allocations
    size_t current = used.load(std::memory_order_acquire);
    
    size_t aligned_offset;
    size_t new_used;

    do {
        //calculate aligned offset
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(base) + current;
        
        //ensure that the memory is aligned to the required alignment 
        // using ~(align -1) -> basically ensures that its aligned to 
        // 16 bytes
        // so if lets say my base address ptr is pointing at
        // 0x1000 -> this will work . Why ? cuz divisible by 16
        // but 0x1018 won't work cuz not divisible by 16
        // hence we need to find nearest aligned address 
        uintptr_t aligned_addr = (base_addr + align - 1) & ~(align - 1);

        aligned_offset = aligned_addr - reinterpret_cast<uintptr_t>(base);

        new_used = aligned_offset + bytes;

        if (new_used > size) {
            return nullptr;
        }
    } while (!used.compare_exchange_weak(
        current, new_used,
        std::memory_order_release, // if current == new used : 
        std::memory_order_relaxed
    ));

    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + aligned_offset);
}

// =============================================================================
// UnifiedMemoryManager Implementation
// =============================================================================

UnifiedMemoryManager::~UnifiedMemoryManager() {
    cleanup();
}

bool UnifiedMemoryManager::initialize(const MemoryPoolConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_ != nullptr) {
        LOGE("Memory pool already initialized. Call cleanup() first.\n");
        return false;
    }

    if (config.total_size() == 0) {
        LOGE("Invalid configuration: total size is 0\n");
        return false;
    }

    config_ = config;
    size_t total = config_.total_size();


//Credit : Claude Opus 4.5 
    // Allocate with alignment
    // Use posix_memalign on POSIX systems, _aligned_malloc on Windows
#if defined(_WIN32)
    pool_ = _aligned_malloc(total, config_.alignment);
#elif defined(__ANDROID__) || defined(__linux__) || defined(__APPLE__)
    int result = posix_memalign(&pool_, config_.alignment, total);
    if (result != 0) {
        pool_ = nullptr;
    }
#else
    // Fallback: over-allocate and manually align
    // Note: This leaks the original pointer, so prefer platform-specific APIs
    void* raw = std::malloc(total + config_.alignment);
    if (raw) {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + config_.alignment) & 
                            ~(config_.alignment - 1);
        pool_ = reinterpret_cast<void*>(aligned);
    }
#endif

    if (pool_ == nullptr) {
        LOGE("Failed to allocate %zu bytes with %zu alignment\n", 
             total, config_.alignment);
        return false;
    }

    
    //zero-initialize the entire pool
    std::memset(pool_, 0, total);

    //partition into segments
    uint8_t* ptr = static_cast<uint8_t*>(pool_);

    // ptr -> llm_segment_ptr -> ptr += llm_segment -> ptr -> asr_segment_ptr -> ptr += asr_segment -> ptr -> tts_segment_ptr -> ptr += tts_segment -> ptr -> scratch_segment_ptr
    llm_segment_.base = ptr;
    llm_segment_.size = config_.llm_kv_cache_size;
    llm_segment_.used.store(0, std::memory_order_release);
    ptr += config_.llm_kv_cache_size;

    asr_segment_.base = ptr;
    asr_segment_.size = config_.asr_buffer_size;
    asr_segment_.used.store(0, std::memory_order_release);
    ptr += config_.asr_buffer_size;

    tts_segment_.base = ptr;
    tts_segment_.size = config_.tts_buffer_size;
    tts_segment_.used.store(0, std::memory_order_release);
    ptr += config_.tts_buffer_size;

    scratch_segment_.base = ptr;
    scratch_segment_.size = config_.scratch_size;
    scratch_segment_.used.store(0, std::memory_order_release);

    LOGI("Initialized memory pool:\n");
    LOGI("  Total: %zu MB\n", total / (1024 * 1024));
    LOGI("  LLM KV Cache: %zu MB @ %p\n", 
         config_.llm_kv_cache_size / (1024 * 1024), llm_segment_.base);
    LOGI("  ASR Buffer: %zu MB @ %p\n", 
         config_.asr_buffer_size / (1024 * 1024), asr_segment_.base);
    LOGI("  TTS Buffer: %zu MB @ %p\n", 
         config_.tts_buffer_size / (1024 * 1024), tts_segment_.base);
    LOGI("  Scratch: %zu MB @ %p\n", 
         config_.scratch_size / (1024 * 1024), scratch_segment_.base);

    return true;
}

void UnifiedMemoryManager::cleanup() {

    // wait for all threads to finish using the memory pool
    // we do not want other threads to use allocate() while one of them is
    // freeing the pool -> will cause segfault
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_ == nullptr) {
        return;
    }

#if defined(_WIN32)
    _aligned_free(pool_);
#else
    std::free(pool_);
#endif

    pool_ = nullptr;

    // Reset all segments manually (can't assign because std::atomic is non-copyable)
    llm_segment_.base = nullptr;
    llm_segment_.size = 0;
    llm_segment_.used.store(0, std::memory_order_release);

    asr_segment_.base = nullptr;
    asr_segment_.size = 0;
    asr_segment_.used.store(0, std::memory_order_release);

    tts_segment_.base = nullptr;
    tts_segment_.size = 0;
    tts_segment_.used.store(0, std::memory_order_release);

    scratch_segment_.base = nullptr;
    scratch_segment_.size = 0;
    scratch_segment_.used.store(0, std::memory_order_release);

    LOGI("Memory pool cleaned up\n");
}

void UnifiedMemoryManager::reset_reusable_segments() {
    asr_segment_.reset();
    tts_segment_.reset();
    scratch_segment_.reset();
}

size_t UnifiedMemoryManager::get_total_used() const {
    return llm_segment_.used.load(std::memory_order_acquire) +
           asr_segment_.used.load(std::memory_order_acquire) +
           tts_segment_.used.load(std::memory_order_acquire) +
           scratch_segment_.used.load(std::memory_order_acquire);
}

}  // namespace core
}  // namespace rac
