/**
 * @file llamacpp_pooled_buffer.h
 * @brief Custom llama.cpp buffer using UnifiedMemoryManager
 *
 * This provides a pre-allocated memory region for llama.cpp operations,
 * particularly for the KV cache which is the largest memory consumer.
 * Eliminates runtime allocations during token generation.
 */

#ifndef RAC_LLAMACPP_POOLED_BUFFER_H
#define RAC_LLAMACPP_POOLED_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rac {
namespace core {
struct MemorySegment;  // Forward declaration
}
}

namespace runanywhere {

/**
 * @brief Pooled buffer for llama.cpp operations
 *
 * Provides memory from UnifiedMemoryManager for:
 * - KV cache (largest consumer)
 * - Scratch space for token processing
 * - Temporary buffers for inference
 *
 * Uses bump allocation - allocations advance offset, reset() resets to start.
 */
class LlamaCppPooledBuffer {
public:
    /**
     * @brief Construct buffer with a memory segment
     * @param segment Pre-allocated segment from UnifiedMemoryManager
     */
    explicit LlamaCppPooledBuffer(rac::core::MemorySegment& segment);
    ~LlamaCppPooledBuffer() = default;

    // Non-copyable
    LlamaCppPooledBuffer(const LlamaCppPooledBuffer&) = delete;
    LlamaCppPooledBuffer& operator=(const LlamaCppPooledBuffer&) = delete;

    /**
     * @brief Get base pointer to the memory pool
     * @return Raw pointer to start of buffer
     */
    void* get_base() { return base_; }
    const void* get_base() const { return base_; }

    /**
     * @brief Get total buffer size
     * @return Size in bytes
     */
    size_t get_size() const { return size_; }

    /**
     * @brief Allocate memory from the pool
     * @param size Number of bytes to allocate
     * @param alignment Alignment requirement (default 64 for cache lines)
     * @return Pointer to allocated memory, or nullptr if exhausted
     */
    void* allocate(size_t size, size_t alignment = 64);

    /**
     * @brief Reset allocator for next inference
     * Resets offset to 0, making all memory available again.
     */
    void reset();

    /**
     * @brief Get current memory usage
     * @return Bytes currently allocated
     */
    size_t get_used() const { return offset_; }

    /**
     * @brief Get available memory
     * @return Bytes remaining in pool
     */
    size_t get_available() const { return size_ - offset_; }

    /**
     * @brief Get number of allocations since last reset
     */
    int get_allocation_count() const { return allocation_count_; }

    // =========================================================================
    // Specialized allocation helpers for llama.cpp
    // =========================================================================

    /**
     * @brief Reserve space for KV cache
     * @param n_ctx Context size (tokens)
     * @param n_layer Number of transformer layers
     * @param n_embd Embedding dimension
     * @param n_head_kv Number of KV heads
     * @return Pointer to KV cache memory, or nullptr if insufficient space
     */
    void* reserve_kv_cache(int n_ctx, int n_layer, int n_embd, int n_head_kv);

    /**
     * @brief Get size needed for KV cache
     */
    static size_t calculate_kv_cache_size(int n_ctx, int n_layer, int n_embd, int n_head_kv);

private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
    int allocation_count_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace runanywhere

#endif  // RAC_LLAMACPP_POOLED_BUFFER_H
