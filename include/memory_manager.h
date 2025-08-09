#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace hpc_regex {

// Memory allocation strategies
enum class AllocationStrategy {
    STANDARD,
    POOL,
    PINNED,
    UNIFIED
};

// Base memory manager interface
class MemoryManager {
public:
    virtual ~MemoryManager() = default;
    
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual size_t get_total_allocated() const = 0;
    virtual size_t get_peak_usage() const = 0;
    virtual void cleanup() = 0;
};

// CPU memory manager with alignment and pooling
class CPUMemoryManager : public MemoryManager {
public:
    explicit CPUMemoryManager(size_t alignment = 32);
    ~CPUMemoryManager() override;
    
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    size_t get_total_allocated() const override;
    size_t get_peak_usage() const override;
    void cleanup() override;
    
    // CPU-specific methods
    void set_alignment(size_t alignment);
    void enable_pooling(bool enable);
    void set_pool_size(size_t size);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// GPU memory manager with CUDA memory management
class GPUMemoryManager : public MemoryManager {
public:
    explicit GPUMemoryManager(int device_id = 0, AllocationStrategy strategy = AllocationStrategy::POOL);
    ~GPUMemoryManager() override;
    
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    size_t get_total_allocated() const override;
    size_t get_peak_usage() const override;
    void cleanup() override;
    
    // GPU-specific methods
    void* allocate_device(size_t size);
    void* allocate_host(size_t size);
    void* allocate_unified(size_t size);
    
    void deallocate_device(void* ptr);
    void deallocate_host(void* ptr);
    void deallocate_unified(void* ptr);
    
    void copy_host_to_device(void* dst, const void* src, size_t size);
    void copy_device_to_host(void* dst, const void* src, size_t size);
    void copy_device_to_device(void* dst, const void* src, size_t size);
    
    void set_device(int device_id);
    int get_device() const;
    void set_allocation_strategy(AllocationStrategy strategy);
    
    // Memory pool management
    void set_pool_size(size_t size);
    void defragment_pool();
    float get_fragmentation_ratio() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Smart pointer wrappers for automatic memory management
template<typename T>
class managed_ptr {
public:
    managed_ptr(T* ptr, MemoryManager* manager) 
        : ptr_(ptr), manager_(manager) {}
    
    ~managed_ptr() {
        if (ptr_ && manager_) {
            manager_->deallocate(ptr_);
        }
    }
    
    // Move semantics
    managed_ptr(managed_ptr&& other) noexcept 
        : ptr_(other.ptr_), manager_(other.manager_) {
        other.ptr_ = nullptr;
        other.manager_ = nullptr;
    }
    
    managed_ptr& operator=(managed_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && manager_) {
                manager_->deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            manager_ = other.manager_;
            other.ptr_ = nullptr;
            other.manager_ = nullptr;
        }
        return *this;
    }
    
    // Disable copy semantics
    managed_ptr(const managed_ptr&) = delete;
    managed_ptr& operator=(const managed_ptr&) = delete;
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
    MemoryManager* manager_;
};

// Memory pool for efficient allocation/deallocation
class MemoryPool {
public:
    explicit MemoryPool(size_t block_size, size_t initial_blocks = 16);
    ~MemoryPool();
    
    void* allocate();
    void deallocate(void* ptr);
    void expand(size_t additional_blocks);
    void shrink();
    
    size_t get_block_size() const;
    size_t get_total_blocks() const;
    size_t get_free_blocks() const;
    size_t get_memory_usage() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Memory statistics and monitoring
struct MemoryStats {
    size_t total_allocated;
    size_t peak_usage;
    size_t current_usage;
    size_t allocation_count;
    size_t deallocation_count;
    double fragmentation_ratio;
    std::chrono::microseconds total_alloc_time;
    std::chrono::microseconds total_dealloc_time;
};

class MemoryMonitor {
public:
    static MemoryMonitor& instance();
    
    void register_manager(const std::string& name, MemoryManager* manager);
    void unregister_manager(const std::string& name);
    
    MemoryStats get_stats(const std::string& name) const;
    std::vector<std::pair<std::string, MemoryStats>> get_all_stats() const;
    
    void reset_stats();
    void print_summary() const;

private:
    MemoryMonitor() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MemoryManager*> managers_;
};

} // namespace hpc_regex