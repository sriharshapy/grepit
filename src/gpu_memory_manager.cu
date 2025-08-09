#include "memory_manager.h"
#include "cuda_utils.h"
#include "hpc_regex.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <iostream>

namespace hpc_regex {

// GPU Memory Block for pool management
struct GPUMemoryBlock {
    void* device_ptr;
    void* host_ptr;
    size_t size;
    bool is_free;
    bool is_unified;
    bool is_pinned;
    std::chrono::high_resolution_clock::time_point allocated_time;
    
    GPUMemoryBlock(void* dev_ptr, void* host_ptr, size_t s, bool unified = false, bool pinned = false)
        : device_ptr(dev_ptr), host_ptr(host_ptr), size(s), is_free(true), 
          is_unified(unified), is_pinned(pinned),
          allocated_time(std::chrono::high_resolution_clock::now()) {}
};

// GPUMemoryManager Implementation
class GPUMemoryManager::Impl {
public:
    explicit Impl(int device_id, AllocationStrategy strategy)
        : device_id_(device_id), strategy_(strategy), pool_size_(512 * 1024 * 1024),
          total_allocated_(0), peak_usage_(0) {
        
        // Set CUDA device
        CUDA_CHECK(cudaSetDevice(device_id_));
        
        // Get device properties
        CUDA_CHECK(cudaGetDeviceProperties(&device_props_, device_id_));
        
        // Initialize memory streams for async operations
        CUDA_CHECK(cudaStreamCreate(&memory_stream_));
        
        std::cout << "GPU Memory Manager initialized on device " << device_id_ 
                  << " (" << device_props_.name << ")" << std::endl;
    }
    
    ~Impl() {
        cleanup();
        if (memory_stream_) {
            cudaStreamDestroy(memory_stream_);
        }
    }
    
    void* allocate(size_t size) {
        return allocate_device(size);
    }
    
    void deallocate(void* ptr) {
        deallocate_device(ptr);
    }
    
    void* allocate_device(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size == 0) return nullptr;
        
        // Align size to 256 bytes for optimal memory access
        size_t aligned_size = (size + 255) & ~255;
        
        void* ptr = nullptr;
        
        // Try to allocate from pool first
        if (strategy_ == AllocationStrategy::POOL) {
            ptr = allocate_from_device_pool(aligned_size);
        }
        
        // Fall back to direct allocation
        if (!ptr) {
            if (strategy_ == AllocationStrategy::UNIFIED) {
                CUDA_CHECK(cudaMallocManaged(&ptr, aligned_size));
            } else {
                CUDA_CHECK(cudaMalloc(&ptr, aligned_size));
            }
        }
        
        if (ptr) {
            device_allocations_[ptr] = aligned_size;
            total_allocated_ += aligned_size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);
        }
        
        return ptr;
    }
    
    void* allocate_host(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size == 0) return nullptr;
        
        size_t aligned_size = (size + 255) & ~255;
        void* ptr = nullptr;
        
        if (strategy_ == AllocationStrategy::PINNED) {
            CUDA_CHECK(cudaMallocHost(&ptr, aligned_size));
        } else {
            ptr = std::malloc(aligned_size);
        }
        
        if (ptr) {
            host_allocations_[ptr] = aligned_size;
        }
        
        return ptr;
    }
    
    void* allocate_unified(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size == 0) return nullptr;
        
        size_t aligned_size = (size + 255) & ~255;
        void* ptr = nullptr;
        
        CUDA_CHECK(cudaMallocManaged(&ptr, aligned_size));
        
        if (ptr) {
            unified_allocations_[ptr] = aligned_size;
            total_allocated_ += aligned_size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);
        }
        
        return ptr;
    }
    
    void deallocate_device(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = device_allocations_.find(ptr);
        if (it != device_allocations_.end()) {
            size_t size = it->second;
            device_allocations_.erase(it);
            total_allocated_ -= size;
            
            if (strategy_ == AllocationStrategy::POOL) {
                return_to_device_pool(ptr, size);
            } else {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    }
    
    void deallocate_host(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = host_allocations_.find(ptr);
        if (it != host_allocations_.end()) {
            host_allocations_.erase(it);
            
            if (strategy_ == AllocationStrategy::PINNED) {
                CUDA_CHECK(cudaFreeHost(ptr));
            } else {
                std::free(ptr);
            }
        }
    }
    
    void deallocate_unified(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = unified_allocations_.find(ptr);
        if (it != unified_allocations_.end()) {
            size_t size = it->second;
            unified_allocations_.erase(it);
            total_allocated_ -= size;
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    
    void copy_host_to_device(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, memory_stream_));
    }
    
    void copy_device_to_host(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, memory_stream_));
    }
    
    void copy_device_to_device(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, memory_stream_));
    }
    
    size_t get_total_allocated() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_;
    }
    
    size_t get_peak_usage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_usage_;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Free all device allocations
        for (auto& [ptr, size] : device_allocations_) {
            cudaFree(ptr);
        }
        device_allocations_.clear();
        
        // Free all host allocations
        for (auto& [ptr, size] : host_allocations_) {
            if (strategy_ == AllocationStrategy::PINNED) {
                cudaFreeHost(ptr);
            } else {
                std::free(ptr);
            }
        }
        host_allocations_.clear();
        
        // Free all unified allocations
        for (auto& [ptr, size] : unified_allocations_) {
            cudaFree(ptr);
        }
        unified_allocations_.clear();
        
        // Clean up memory pools
        for (auto& block : device_memory_pool_) {
            cudaFree(block.device_ptr);
        }
        device_memory_pool_.clear();
        
        total_allocated_ = 0;
    }
    
    void set_device(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (device_id != device_id_) {
            cleanup();
            device_id_ = device_id;
            CUDA_CHECK(cudaSetDevice(device_id_));
            CUDA_CHECK(cudaGetDeviceProperties(&device_props_, device_id_));
        }
    }
    
    int get_device() const {
        return device_id_;
    }
    
    void set_allocation_strategy(AllocationStrategy strategy) {
        std::lock_guard<std::mutex> lock(mutex_);
        strategy_ = strategy;
    }
    
    void set_pool_size(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_size_ = size;
    }
    
    void defragment_pool() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Coalesce adjacent free blocks
        std::sort(device_memory_pool_.begin(), device_memory_pool_.end(),
            [](const GPUMemoryBlock& a, const GPUMemoryBlock& b) {
                return a.device_ptr < b.device_ptr;
            });
        
        for (auto it = device_memory_pool_.begin(); it != device_memory_pool_.end(); ) {
            if (it->is_free && it + 1 != device_memory_pool_.end() && (it + 1)->is_free) {
                char* end_ptr = static_cast<char*>(it->device_ptr) + it->size;
                if (end_ptr == (it + 1)->device_ptr) {
                    // Adjacent blocks, merge them
                    it->size += (it + 1)->size;
                    it = device_memory_pool_.erase(it + 1) - 1;
                    continue;
                }
            }
            ++it;
        }
    }
    
    float get_fragmentation_ratio() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (device_memory_pool_.empty()) return 0.0f;
        
        size_t free_blocks = 0;
        size_t total_free_size = 0;
        size_t largest_free_block = 0;
        
        for (const auto& block : device_memory_pool_) {
            if (block.is_free) {
                ++free_blocks;
                total_free_size += block.size;
                largest_free_block = std::max(largest_free_block, block.size);
            }
        }
        
        if (total_free_size == 0) return 0.0f;
        
        return 1.0f - (static_cast<float>(largest_free_block) / total_free_size);
    }

private:
    void* allocate_from_device_pool(size_t size) {
        // Find a suitable free block
        for (auto& block : device_memory_pool_) {
            if (block.is_free && block.size >= size) {
                block.is_free = false;
                block.allocated_time = std::chrono::high_resolution_clock::now();
                
                // Split block if it's significantly larger
                if (block.size > size + 1024) { // 1KB threshold
                    size_t remaining_size = block.size - size;
                    void* remaining_ptr = static_cast<char*>(block.device_ptr) + size;
                    device_memory_pool_.emplace_back(remaining_ptr, nullptr, remaining_size);
                    block.size = size;
                }
                
                return block.device_ptr;
            }
        }
        
        // No suitable block found, allocate new one if pool has space
        size_t current_pool_usage = 0;
        for (const auto& block : device_memory_pool_) {
            current_pool_usage += block.size;
        }
        
        if (current_pool_usage + size <= pool_size_) {
            void* ptr = nullptr;
            cudaError_t result = cudaMalloc(&ptr, size);
            if (result == cudaSuccess) {
                device_memory_pool_.emplace_back(ptr, nullptr, size, false, false);
                device_memory_pool_.back().is_free = false;
                return ptr;
            }
        }
        
        return nullptr;
    }
    
    void return_to_device_pool(void* ptr, size_t size) {
        // Find the block and mark it as free
        for (auto& block : device_memory_pool_) {
            if (block.device_ptr == ptr) {
                block.is_free = true;
                
                // Defragment periodically
                static int return_count = 0;
                if (++return_count % 100 == 0) {
                    defragment_pool();
                }
                return;
            }
        }
        
        // If not found in pool, just free it directly
        cudaFree(ptr);
    }
    
    mutable std::mutex mutex_;
    int device_id_;
    AllocationStrategy strategy_;
    size_t pool_size_;
    size_t total_allocated_;
    size_t peak_usage_;
    
    cudaDeviceProp device_props_;
    cudaStream_t memory_stream_;
    
    std::unordered_map<void*, size_t> device_allocations_;
    std::unordered_map<void*, size_t> host_allocations_;
    std::unordered_map<void*, size_t> unified_allocations_;
    
    std::vector<GPUMemoryBlock> device_memory_pool_;
};

// GPUMemoryManager public interface
GPUMemoryManager::GPUMemoryManager(int device_id, AllocationStrategy strategy)
    : pImpl(std::make_unique<Impl>(device_id, strategy)) {}

GPUMemoryManager::~GPUMemoryManager() = default;

void* GPUMemoryManager::allocate(size_t size) {
    return pImpl->allocate(size);
}

void GPUMemoryManager::deallocate(void* ptr) {
    pImpl->deallocate(ptr);
}

void* GPUMemoryManager::allocate_device(size_t size) {
    return pImpl->allocate_device(size);
}

void* GPUMemoryManager::allocate_host(size_t size) {
    return pImpl->allocate_host(size);
}

void* GPUMemoryManager::allocate_unified(size_t size) {
    return pImpl->allocate_unified(size);
}

void GPUMemoryManager::deallocate_device(void* ptr) {
    pImpl->deallocate_device(ptr);
}

void GPUMemoryManager::deallocate_host(void* ptr) {
    pImpl->deallocate_host(ptr);
}

void GPUMemoryManager::deallocate_unified(void* ptr) {
    pImpl->deallocate_unified(ptr);
}

void GPUMemoryManager::copy_host_to_device(void* dst, const void* src, size_t size) {
    pImpl->copy_host_to_device(dst, src, size);
}

void GPUMemoryManager::copy_device_to_host(void* dst, const void* src, size_t size) {
    pImpl->copy_device_to_host(dst, src, size);
}

void GPUMemoryManager::copy_device_to_device(void* dst, const void* src, size_t size) {
    pImpl->copy_device_to_device(dst, src, size);
}

size_t GPUMemoryManager::get_total_allocated() const {
    return pImpl->get_total_allocated();
}

size_t GPUMemoryManager::get_peak_usage() const {
    return pImpl->get_peak_usage();
}

void GPUMemoryManager::cleanup() {
    pImpl->cleanup();
}

void GPUMemoryManager::set_device(int device_id) {
    pImpl->set_device(device_id);
}

int GPUMemoryManager::get_device() const {
    return pImpl->get_device();
}

void GPUMemoryManager::set_allocation_strategy(AllocationStrategy strategy) {
    pImpl->set_allocation_strategy(strategy);
}

void GPUMemoryManager::set_pool_size(size_t size) {
    pImpl->set_pool_size(size);
}

void GPUMemoryManager::defragment_pool() {
    pImpl->defragment_pool();
}

float GPUMemoryManager::get_fragmentation_ratio() const {
    return pImpl->get_fragmentation_ratio();
}

} // namespace hpc_regex