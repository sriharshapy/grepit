#include "memory_manager.h"
#include "hpc_regex.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <list>
#include <mutex>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hpc_regex {

// Memory block structure for pool allocation
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool is_free;
    std::chrono::high_resolution_clock::time_point allocated_time;
    
    MemoryBlock(void* p, size_t s, bool free = true) 
        : ptr(p), size(s), is_free(free), allocated_time(std::chrono::high_resolution_clock::now()) {}
};

// CPUMemoryManager Implementation
class CPUMemoryManager::Impl {
public:
    explicit Impl(size_t alignment) 
        : alignment_(alignment), pooling_enabled_(false), pool_size_(0),
          total_allocated_(0), peak_usage_(0) {
        // Ensure alignment is power of 2
        if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
            alignment_ = 32; // Default to 32-byte alignment
        }
    }
    
    ~Impl() {
        cleanup();
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size == 0) return nullptr;
        
        // Align size to boundary
        size_t aligned_size = align_size(size);
        
        void* ptr = nullptr;
        
        if (pooling_enabled_) {
            ptr = allocate_from_pool(aligned_size);
        }
        
        if (!ptr) {
            ptr = allocate_aligned(aligned_size);
            if (!ptr) {
                throw MemoryException("Failed to allocate " + std::to_string(size) + " bytes");
            }
        }
        
        // Track allocation
        allocations_[ptr] = aligned_size;
        total_allocated_ += aligned_size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            return; // Not our allocation
        }
        
        size_t size = it->second;
        allocations_.erase(it);
        total_allocated_ -= size;
        
        if (pooling_enabled_) {
            return_to_pool(ptr, size);
        } else {
            free_aligned(ptr);
        }
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
        
        // Free all tracked allocations
        for (auto& [ptr, size] : allocations_) {
            free_aligned(ptr);
        }
        allocations_.clear();
        
        // Clean up memory pool
        for (auto& block : memory_pool_) {
            free_aligned(block.ptr);
        }
        memory_pool_.clear();
        
        total_allocated_ = 0;
    }
    
    void set_alignment(size_t alignment) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (alignment > 0 && (alignment & (alignment - 1)) == 0) {
            alignment_ = alignment;
        }
    }
    
    void enable_pooling(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        pooling_enabled_ = enable;
        if (!enable) {
            // Clean up pool when disabling
            for (auto& block : memory_pool_) {
                if (block.is_free) {
                    free_aligned(block.ptr);
                }
            }
            memory_pool_.erase(
                std::remove_if(memory_pool_.begin(), memory_pool_.end(),
                    [](const MemoryBlock& block) { return block.is_free; }),
                memory_pool_.end());
        }
    }
    
    void set_pool_size(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_size_ = size;
    }

private:
    size_t align_size(size_t size) const {
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }
    
    void* allocate_aligned(size_t size) {
#ifdef _WIN32
        return _aligned_malloc(size, alignment_);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment_, size) != 0) {
            return nullptr;
        }
        return ptr;
#endif
    }
    
    void free_aligned(void* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
    
    void* allocate_from_pool(size_t size) {
        // Find a suitable free block
        for (auto& block : memory_pool_) {
            if (block.is_free && block.size >= size) {
                block.is_free = false;
                block.allocated_time = std::chrono::high_resolution_clock::now();
                
                // Split block if it's significantly larger
                if (block.size > size + alignment_) {
                    size_t remaining_size = block.size - size;
                    void* remaining_ptr = static_cast<char*>(block.ptr) + size;
                    memory_pool_.emplace_back(remaining_ptr, remaining_size, true);
                    block.size = size;
                }
                
                return block.ptr;
            }
        }
        
        // No suitable block found, allocate new one
        if (pool_size_ == 0 || memory_pool_.size() * average_block_size() < pool_size_) {
            void* ptr = allocate_aligned(size);
            if (ptr) {
                memory_pool_.emplace_back(ptr, size, false);
                return ptr;
            }
        }
        
        return nullptr;
    }
    
    void return_to_pool(void* ptr, size_t size) {
        // Find the block and mark it as free
        for (auto& block : memory_pool_) {
            if (block.ptr == ptr) {
                block.is_free = true;
                
                // Try to coalesce adjacent free blocks
                coalesce_free_blocks();
                return;
            }
        }
        
        // If not found in pool, just free it
        free_aligned(ptr);
    }
    
    void coalesce_free_blocks() {
        // Sort blocks by address for coalescing
        std::sort(memory_pool_.begin(), memory_pool_.end(),
            [](const MemoryBlock& a, const MemoryBlock& b) {
                return a.ptr < b.ptr;
            });
        
        // Coalesce adjacent free blocks
        for (auto it = memory_pool_.begin(); it != memory_pool_.end(); ) {
            if (it->is_free && it + 1 != memory_pool_.end() && (it + 1)->is_free) {
                char* end_ptr = static_cast<char*>(it->ptr) + it->size;
                if (end_ptr == (it + 1)->ptr) {
                    // Adjacent blocks, merge them
                    it->size += (it + 1)->size;
                    it = memory_pool_.erase(it + 1) - 1;
                    continue;
                }
            }
            ++it;
        }
    }
    
    size_t average_block_size() const {
        if (memory_pool_.empty()) return alignment_;
        size_t total = 0;
        for (const auto& block : memory_pool_) {
            total += block.size;
        }
        return total / memory_pool_.size();
    }
    
    mutable std::mutex mutex_;
    size_t alignment_;
    bool pooling_enabled_;
    size_t pool_size_;
    size_t total_allocated_;
    size_t peak_usage_;
    
    std::unordered_map<void*, size_t> allocations_;
    std::vector<MemoryBlock> memory_pool_;
};

// CPUMemoryManager public interface
CPUMemoryManager::CPUMemoryManager(size_t alignment) 
    : pImpl(std::make_unique<Impl>(alignment)) {}

CPUMemoryManager::~CPUMemoryManager() = default;

void* CPUMemoryManager::allocate(size_t size) {
    return pImpl->allocate(size);
}

void CPUMemoryManager::deallocate(void* ptr) {
    pImpl->deallocate(ptr);
}

size_t CPUMemoryManager::get_total_allocated() const {
    return pImpl->get_total_allocated();
}

size_t CPUMemoryManager::get_peak_usage() const {
    return pImpl->get_peak_usage();
}

void CPUMemoryManager::cleanup() {
    pImpl->cleanup();
}

void CPUMemoryManager::set_alignment(size_t alignment) {
    pImpl->set_alignment(alignment);
}

void CPUMemoryManager::enable_pooling(bool enable) {
    pImpl->enable_pooling(enable);
}

void CPUMemoryManager::set_pool_size(size_t size) {
    pImpl->set_pool_size(size);
}

// MemoryPool Implementation
class MemoryPool::Impl {
public:
    explicit Impl(size_t block_size, size_t initial_blocks)
        : block_size_(block_size), total_blocks_(0), free_blocks_(0) {
        expand(initial_blocks);
    }
    
    ~Impl() {
        cleanup();
    }
    
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_list_.empty()) {
            expand(std::max(total_blocks_ / 2, size_t(8))); // Grow by 50% or 8 blocks
        }
        
        if (free_list_.empty()) {
            throw MemoryException("Memory pool exhausted");
        }
        
        void* ptr = free_list_.back();
        free_list_.pop_back();
        --free_blocks_;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Verify pointer belongs to our pool
        if (blocks_.find(ptr) != blocks_.end()) {
            free_list_.push_back(ptr);
            ++free_blocks_;
        }
    }
    
    void expand(size_t additional_blocks) {
        for (size_t i = 0; i < additional_blocks; ++i) {
            void* ptr = std::aligned_alloc(32, block_size_);
            if (!ptr) {
                throw MemoryException("Failed to expand memory pool");
            }
            
            blocks_.insert(ptr);
            free_list_.push_back(ptr);
            ++total_blocks_;
            ++free_blocks_;
        }
    }
    
    void shrink() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Free some blocks if we have too many free
        size_t blocks_to_free = std::min(free_blocks_ / 2, free_blocks_ - 1);
        
        for (size_t i = 0; i < blocks_to_free && !free_list_.empty(); ++i) {
            void* ptr = free_list_.back();
            free_list_.pop_back();
            blocks_.erase(ptr);
            std::free(ptr);
            --total_blocks_;
            --free_blocks_;
        }
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (void* ptr : blocks_) {
            std::free(ptr);
        }
        
        blocks_.clear();
        free_list_.clear();
        total_blocks_ = 0;
        free_blocks_ = 0;
    }
    
    size_t get_block_size() const { return block_size_; }
    size_t get_total_blocks() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return total_blocks_; 
    }
    size_t get_free_blocks() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_; 
    }
    size_t get_memory_usage() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return total_blocks_ * block_size_; 
    }

private:
    mutable std::mutex mutex_;
    size_t block_size_;
    size_t total_blocks_;
    size_t free_blocks_;
    
    std::unordered_set<void*> blocks_;
    std::vector<void*> free_list_;
};

// MemoryPool public interface
MemoryPool::MemoryPool(size_t block_size, size_t initial_blocks)
    : pImpl(std::make_unique<Impl>(block_size, initial_blocks)) {}

MemoryPool::~MemoryPool() = default;

void* MemoryPool::allocate() {
    return pImpl->allocate();
}

void MemoryPool::deallocate(void* ptr) {
    pImpl->deallocate(ptr);
}

void MemoryPool::expand(size_t additional_blocks) {
    pImpl->expand(additional_blocks);
}

void MemoryPool::shrink() {
    pImpl->shrink();
}

size_t MemoryPool::get_block_size() const {
    return pImpl->get_block_size();
}

size_t MemoryPool::get_total_blocks() const {
    return pImpl->get_total_blocks();
}

size_t MemoryPool::get_free_blocks() const {
    return pImpl->get_free_blocks();
}

size_t MemoryPool::get_memory_usage() const {
    return pImpl->get_memory_usage();
}

// MemoryMonitor Implementation
class MemoryMonitor::Impl {
public:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MemoryManager*> managers_;
};

MemoryMonitor& MemoryMonitor::instance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::register_manager(const std::string& name, MemoryManager* manager) {
    if (!pImpl) {
        pImpl = std::make_unique<Impl>();
    }
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->managers_[name] = manager;
}

void MemoryMonitor::unregister_manager(const std::string& name) {
    if (!pImpl) return;
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->managers_.erase(name);
}

MemoryStats MemoryMonitor::get_stats(const std::string& name) const {
    MemoryStats stats{};
    if (!pImpl) return stats;
    
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    auto it = pImpl->managers_.find(name);
    if (it != pImpl->managers_.end()) {
        stats.total_allocated = it->second->get_total_allocated();
        stats.peak_usage = it->second->get_peak_usage();
        stats.current_usage = stats.total_allocated;
    }
    return stats;
}

std::vector<std::pair<std::string, MemoryStats>> MemoryMonitor::get_all_stats() const {
    std::vector<std::pair<std::string, MemoryStats>> all_stats;
    if (!pImpl) return all_stats;
    
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    for (const auto& [name, manager] : pImpl->managers_) {
        MemoryStats stats{};
        stats.total_allocated = manager->get_total_allocated();
        stats.peak_usage = manager->get_peak_usage();
        stats.current_usage = stats.total_allocated;
        all_stats.emplace_back(name, stats);
    }
    return all_stats;
}

void MemoryMonitor::reset_stats() {
    // Implementation depends on more detailed tracking
}

void MemoryMonitor::print_summary() const {
    auto all_stats = get_all_stats();
    std::cout << "\n=== Memory Usage Summary ===\n";
    for (const auto& [name, stats] : all_stats) {
        std::cout << name << ":\n";
        std::cout << "  Current Usage: " << stats.current_usage / 1024 << " KB\n";
        std::cout << "  Peak Usage: " << stats.peak_usage / 1024 << " KB\n";
        std::cout << "  Total Allocated: " << stats.total_allocated / 1024 << " KB\n\n";
    }
}

} // namespace hpc_regex