/**
 * @file onnx_pooled_allocator.cpp
 * @brief Implementation of OnnxPooledAllocator
 */

#include "onnx_pooled_allocator.h"
#include "rac/core/unified_memory_manager.h"

#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "OnnxPooledAllocator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, "[OnnxPooledAllocator] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[OnnxPooledAllocator ERROR] " __VA_ARGS__)
#endif

namespace runanywhere {

OnnxPooledAllocator::OnnxPooledAllocator(rac::core::MemorySegment& segment)
    : base_(static_cast<uint8_t*>(segment.base))
    , size_(segment.size)
    , offset_(0)
    , allocation_count_(0) {

    // Initialize OrtAllocator structure
    ort_allocator_.version = ORT_API_VERSION;
    ort_allocator_.Alloc = &OnnxPooledAllocator::Alloc;
    ort_allocator_.Free = &OnnxPooledAllocator::Free;
    ort_allocator_.Info = &OnnxPooledAllocator::Info;
    ort_allocator_.Reserve = nullptr;  

    // Create OrtMemoryInfo for CPU allocator
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (ort_api) {
        OrtStatus* status = ort_api->CreateMemoryInfo(
            "PooledCPU",
            OrtAllocatorType::OrtDeviceAllocator,
            0,  // device_id
            OrtMemType::OrtMemTypeDefault,
            &memory_info_
        );
        if (status != nullptr) {
            LOGE("Failed to create OrtMemoryInfo: %s\n", ort_api->GetErrorMessage(status));
            ort_api->ReleaseStatus(status);
            memory_info_ = nullptr;
        }
    }

    LOGI("Initialized with pool size: %zu MB\n", size_ / (1024 * 1024));
}

OnnxPooledAllocator::~OnnxPooledAllocator() {
    if (memory_info_) {
        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (ort_api) {
            ort_api->ReleaseMemoryInfo(memory_info_);
        }
    }
    LOGI("Destroyed. Total allocations: %d\n", allocation_count_);
}

void OnnxPooledAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    offset_ = 0;
    allocation_count_ = 0;
}

void* OnnxPooledAllocator::allocate_aligned(size_t size, size_t alignment) {
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

// Static OrtAllocator callbacks

void* ORT_API_CALL OnnxPooledAllocator::Alloc(OrtAllocator* this_, size_t size) {
    // Get our class instance from the OrtAllocator pointer
    // The OrtAllocator is the first member, so we can cast directly
    auto* self = reinterpret_cast<OnnxPooledAllocator*>(
        reinterpret_cast<uint8_t*>(this_) - offsetof(OnnxPooledAllocator, ort_allocator_)
    );
    
    return self->allocate_aligned(size);
}

void ORT_API_CALL OnnxPooledAllocator::Free(OrtAllocator* this_, void* p) {
    // No-op: We use bump allocation, memory is freed by reset()
    // This is safe because ONNX Runtime calls Free() at the end of inference
    (void)this_;
    (void)p;
}

const OrtMemoryInfo* ORT_API_CALL OnnxPooledAllocator::Info(const OrtAllocator* this_) {
    auto* self = reinterpret_cast<const OnnxPooledAllocator*>(
        reinterpret_cast<const uint8_t*>(this_) - offsetof(OnnxPooledAllocator, ort_allocator_)
    );
    return self->memory_info_;
}

}  // namespace runanywhere
