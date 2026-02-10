/**
 * @file llamacpp_pooled_buffer.cpp
 * @brief Implementation of LlamaCppPooledBuffer
 */

#include "rac/core/unified_memory_manager.h"
#include "llamacpp_pooled_buffer.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "LlamaCppPooledBuffer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, "[LlamaCppPooledBuffer] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[LlamaCppPooledBuffer ERROR] " __VA_ARGS__)
#endif

namespace runanywhere {

LlamaCppPooledBuffer::LlamaCppPooledBuffer(rac::core::MemorySegment& segment)
    : base_(static_cast<uint8_t*>(segment.base))
    , size_(segment.size)
    , offset_(0)
    , allocation_count_(0) {
    
    LOGI("Initialized with pool size: %zu MB\n", size_ / (1024 * 1024));
}

void* LlamaCppPooledBuffer::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align the current offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    
    // Check if we have enough space
    if (aligned_offset + size > size_) {
        LOGE("Pool exhausted! Requested %zu bytes, available %zu bytes\n",
             size, size_ - aligned_offset);
        return nullptr;
    }

    void* ptr = base_ + aligned_offset;
    offset_ = aligned_offset + size;
    allocation_count_++;

    return ptr;
}

void LlamaCppPooledBuffer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    offset_ = 0;
    allocation_count_ = 0;
}

size_t LlamaCppPooledBuffer::calculate_kv_cache_size(
    int n_ctx, int n_layer, int n_embd, int n_head_kv) {
    
    // KV cache stores both keys and values for each layer
    // Size per layer = 2 * n_ctx * head_dim * n_head_kv * sizeof(float)
    // where head_dim = n_embd / n_head
    //
    // Simplified: KV cache ≈ 2 * n_ctx * n_layer * (n_embd * n_head_kv / n_head) * sizeof(float)
    // For safety, assume n_head_kv = n_head (full attention)
    
    // Approximate formula (conservative estimate):
    // KV cache size ≈ 4 * n_ctx * n_layer * n_embd * sizeof(float) / n_head_ratio
    
    // Typical models: n_head_kv = n_head (GQA: n_head_kv < n_head)
    // Using conservative estimate: 2 * n_ctx * n_layer * n_embd * 2 (K+V) * sizeof(float16)
    
    size_t kv_per_token = static_cast<size_t>(n_layer) * n_embd * 2 * sizeof(float);
    size_t total = static_cast<size_t>(n_ctx) * kv_per_token;
    
    // Add 10% margin for alignment and metadata
    return total + (total / 10);
}

void* LlamaCppPooledBuffer::reserve_kv_cache(
    int n_ctx, int n_layer, int n_embd, int n_head_kv) {
    
    size_t kv_size = calculate_kv_cache_size(n_ctx, n_layer, n_embd, n_head_kv);
    
    LOGI("Reserving KV cache: n_ctx=%d, n_layer=%d, n_embd=%d -> %zu MB\n",
         n_ctx, n_layer, n_embd, kv_size / (1024 * 1024));
    
    void* ptr = allocate(kv_size, 64);  // 64-byte aligned for SIMD
    
    if (!ptr) {
        LOGE("Failed to reserve KV cache of %zu bytes\n", kv_size);
    }
    
    return ptr;
}

}  // namespace runanywhere
