

#ifndef RAC_ONNX_POOLED_ALLOCATOR_H
#define RAC_ONNX_POOLED_ALLOCATOR_H

#include <onnxruntime_c_api.h>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rac {
namespace core {
struct MemorySegment;  
}
}

namespace runanywhere {


class OnnxPooledAllocator {
public:
    
    explicit OnnxPooledAllocator(rac::core::MemorySegment& segment);
    ~OnnxPooledAllocator();

    // Non-copyable
    OnnxPooledAllocator(const OnnxPooledAllocator&) = delete;
    OnnxPooledAllocator& operator=(const OnnxPooledAllocator&) = delete;

    /**
     * @brief Get the OrtAllocator pointer for ONNX Runtime
     * @return Pointer to use with OrtSessionOptions or OrtValue creation
     */
    OrtAllocator* get_ort_allocator() { return &ort_allocator_; }

   
    void reset();

    
    size_t get_used() const { return offset_; }

   
    size_t get_capacity() const { return size_; }

   
    int get_allocation_count() const { return allocation_count_; }

private:
    // OrtAllocator callback implementations
    static void* ORT_API_CALL Alloc(OrtAllocator* this_, size_t size);
    static void ORT_API_CALL Free(OrtAllocator* this_, void* p);
    static const OrtMemoryInfo* ORT_API_CALL Info(const OrtAllocator* this_);

    // Internal allocate with alignment
    void* allocate_aligned(size_t size, size_t alignment = 64);

    // Memory pool
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
    int allocation_count_ = 0;

    // ORT structures
    OrtAllocator ort_allocator_;
    OrtMemoryInfo* memory_info_ = nullptr;

    mutable std::mutex mutex_;
};

}  // namespace runanywhere

#endif  // RAC_ONNX_POOLED_ALLOCATOR_H
