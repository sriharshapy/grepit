#include "regex_engine.h"
#include "hpc_regex.h"
#include "memory_manager.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

// Forward declaration of CUDA kernels
namespace hpc_regex {
namespace cuda_kernels {
    __global__ void regex_match_dp_kernel(
        const char* text, size_t text_length,
        const char* pattern, size_t pattern_length,
        bool* dp_table, size_t table_width, bool* result);
    
    __global__ void regex_match_dp_shared_kernel(
        const char* text, size_t text_length,
        const char* pattern, size_t pattern_length, bool* result);
    
    __global__ void regex_find_all_kernel(
        const char* text, size_t text_length,
        const char* pattern, size_t pattern_length,
        int* match_positions, int* match_lengths,
        int* match_count, int max_matches);
    
    __global__ void regex_batch_match_kernel(
        const char** texts, size_t* text_lengths, int num_texts,
        const char* pattern, size_t pattern_length, bool* results);
}
}

namespace hpc_regex {

// GPURegexEngine Implementation
class GPURegexEngine::Impl {
public:
    explicit Impl(int device_id) 
        : device_id_(device_id), max_text_length_(1024 * 1024), max_pattern_length_(256) {
        
        // Initialize CUDA device
        cuda::set_device(device_id_);
        device_info_ = cuda::get_device_info(device_id_);
        
        // Initialize memory manager
        memory_manager_ = std::make_unique<GPUMemoryManager>(device_id_, AllocationStrategy::POOL);
        
        // Create CUDA streams for async operations
        compute_stream_ = cuda::create_stream();
        memory_stream_ = cuda::create_stream();
        
        // Pre-allocate GPU memory for DP table and buffers
        allocate_gpu_buffers();
        
        std::cout << "GPU Regex Engine initialized on device " << device_id_ 
                  << " (" << device_info_.name << ")" << std::endl;
    }
    
    ~Impl() {
        cleanup();
        if (compute_stream_) cuda::destroy_stream(compute_stream_);
        if (memory_stream_) cuda::destroy_stream(memory_stream_);
    }
    
    MatchResult match(const std::string& pattern, const std::string& text) {
        if (text.length() > max_text_length_ || pattern.length() > max_pattern_length_) {
            throw HPCRegexException("Text or pattern exceeds maximum length");
        }
        
        if (text.empty() || pattern.empty()) {
            return MatchResult();
        }
        
        // Preprocess pattern
        std::string processed_pattern = preprocess_pattern(pattern);
        
        // Choose kernel based on problem size
        bool result = false;
        if (text.length() <= 1024 && pattern.length() <= 64) {
            result = match_small_shared(processed_pattern, text);
        } else {
            result = match_large_global(processed_pattern, text);
        }
        
        if (result) {
            return MatchResult(true, 0, text.length() - 1, text);
        }
        
        return MatchResult();
    }
    
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) {
        if (text.length() > max_text_length_ || pattern.length() > max_pattern_length_) {
            throw HPCRegexException("Text or pattern exceeds maximum length");
        }
        
        if (text.empty() || pattern.empty()) {
            return {};
        }
        
        std::string processed_pattern = preprocess_pattern(pattern);
        return find_all_matches_gpu(processed_pattern, text);
    }
    
    void set_max_text_length(size_t length) {
        if (length != max_text_length_) {
            max_text_length_ = length;
            allocate_gpu_buffers(); // Reallocate buffers
        }
    }
    
    void set_max_pattern_length(size_t length) {
        if (length != max_pattern_length_) {
            max_pattern_length_ = length;
            allocate_gpu_buffers(); // Reallocate buffers
        }
    }
    
    size_t get_memory_usage() const {
        return memory_manager_->get_total_allocated();
    }
    
    void cleanup() {
        free_gpu_buffers();
        memory_manager_->cleanup();
    }
    
    void set_device(int device_id) {
        if (device_id != device_id_) {
            cleanup();
            device_id_ = device_id;
            cuda::set_device(device_id_);
            device_info_ = cuda::get_device_info(device_id_);
            memory_manager_->set_device(device_id_);
            allocate_gpu_buffers();
        }
    }
    
    int get_device() const {
        return device_id_;
    }
    
    void set_memory_pool_size(size_t size) {
        memory_manager_->set_pool_size(size);
    }

private:
    void allocate_gpu_buffers() {
        free_gpu_buffers();
        
        // Allocate DP table (largest memory requirement)
        size_t table_size = (max_text_length_ + 1) * (max_pattern_length_ + 1) * sizeof(bool);
        d_dp_table_ = memory_manager_->allocate_device(table_size);
        
        // Allocate text and pattern buffers
        d_text_ = memory_manager_->allocate_device(max_text_length_ + 1);
        d_pattern_ = memory_manager_->allocate_device(max_pattern_length_ + 1);
        
        // Allocate result buffer
        d_result_ = memory_manager_->allocate_device(sizeof(bool));
        
        // Allocate buffers for find_all operation
        const size_t max_matches = 1000;
        d_match_positions_ = memory_manager_->allocate_device(max_matches * sizeof(int));
        d_match_lengths_ = memory_manager_->allocate_device(max_matches * sizeof(int));
        d_match_count_ = memory_manager_->allocate_device(sizeof(int));
        
        // Allocate pinned host memory for faster transfers
        h_result_ = memory_manager_->allocate_host(sizeof(bool));
        h_match_count_ = memory_manager_->allocate_host(sizeof(int));
        h_match_positions_ = memory_manager_->allocate_host(max_matches * sizeof(int));
        h_match_lengths_ = memory_manager_->allocate_host(max_matches * sizeof(int));
    }
    
    void free_gpu_buffers() {
        if (d_dp_table_) { memory_manager_->deallocate_device(d_dp_table_); d_dp_table_ = nullptr; }
        if (d_text_) { memory_manager_->deallocate_device(d_text_); d_text_ = nullptr; }
        if (d_pattern_) { memory_manager_->deallocate_device(d_pattern_); d_pattern_ = nullptr; }
        if (d_result_) { memory_manager_->deallocate_device(d_result_); d_result_ = nullptr; }
        if (d_match_positions_) { memory_manager_->deallocate_device(d_match_positions_); d_match_positions_ = nullptr; }
        if (d_match_lengths_) { memory_manager_->deallocate_device(d_match_lengths_); d_match_lengths_ = nullptr; }
        if (d_match_count_) { memory_manager_->deallocate_device(d_match_count_); d_match_count_ = nullptr; }
        
        if (h_result_) { memory_manager_->deallocate_host(h_result_); h_result_ = nullptr; }
        if (h_match_count_) { memory_manager_->deallocate_host(h_match_count_); h_match_count_ = nullptr; }
        if (h_match_positions_) { memory_manager_->deallocate_host(h_match_positions_); h_match_positions_ = nullptr; }
        if (h_match_lengths_) { memory_manager_->deallocate_host(h_match_lengths_); h_match_lengths_ = nullptr; }
    }
    
    bool match_small_shared(const std::string& pattern, const std::string& text) {
        // Copy data to GPU
        memory_manager_->copy_host_to_device(d_text_, text.c_str(), text.length() + 1);
        memory_manager_->copy_host_to_device(d_pattern_, pattern.c_str(), pattern.length() + 1);
        
        // Initialize result
        bool init_result = false;
        memory_manager_->copy_host_to_device(d_result_, &init_result, sizeof(bool));
        
        // Calculate kernel parameters
        dim3 block_size = cuda::calculate_optimal_block_size(device_id_, 
            (text.length() + 1) * (pattern.length() + 1) * sizeof(bool));
        dim3 grid_size = cuda::calculate_grid_size(1, block_size.x);
        
        size_t shared_memory_size = (text.length() + 1) * (pattern.length() + 1) * sizeof(bool);
        
        // Launch kernel
        cuda_kernels::regex_match_dp_shared_kernel<<<grid_size, block_size, 
            shared_memory_size, compute_stream_>>>(
            static_cast<char*>(d_text_), text.length(),
            static_cast<char*>(d_pattern_), pattern.length(),
            static_cast<bool*>(d_result_)
        );
        
        CUDA_CHECK_LAST();
        
        // Copy result back
        memory_manager_->copy_device_to_host(h_result_, d_result_, sizeof(bool));
        cuda::synchronize_stream(compute_stream_);
        
        return *static_cast<bool*>(h_result_);
    }
    
    bool match_large_global(const std::string& pattern, const std::string& text) {
        // Copy data to GPU
        memory_manager_->copy_host_to_device(d_text_, text.c_str(), text.length() + 1);
        memory_manager_->copy_host_to_device(d_pattern_, pattern.c_str(), pattern.length() + 1);
        
        // Initialize DP table and result
        cuda::memset_device(d_dp_table_, 0, 
            (max_text_length_ + 1) * (max_pattern_length_ + 1) * sizeof(bool));
        
        bool init_result = false;
        memory_manager_->copy_host_to_device(d_result_, &init_result, sizeof(bool));
        
        // Calculate kernel parameters
        size_t total_elements = text.length() * pattern.length();
        dim3 block_size = cuda::calculate_optimal_block_size(device_id_);
        dim3 grid_size = cuda::calculate_grid_size(total_elements, block_size.x);
        
        // Launch kernel
        cuda_kernels::regex_match_dp_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
            static_cast<char*>(d_text_), text.length(),
            static_cast<char*>(d_pattern_), pattern.length(),
            static_cast<bool*>(d_dp_table_), max_pattern_length_ + 1,
            static_cast<bool*>(d_result_)
        );
        
        CUDA_CHECK_LAST();
        
        // Copy result back
        memory_manager_->copy_device_to_host(h_result_, d_result_, sizeof(bool));
        cuda::synchronize_stream(compute_stream_);
        
        return *static_cast<bool*>(h_result_);
    }
    
    std::vector<MatchResult> find_all_matches_gpu(const std::string& pattern, const std::string& text) {
        // Copy data to GPU
        memory_manager_->copy_host_to_device(d_text_, text.c_str(), text.length() + 1);
        memory_manager_->copy_host_to_device(d_pattern_, pattern.c_str(), pattern.length() + 1);
        
        // Initialize counters
        int init_count = 0;
        memory_manager_->copy_host_to_device(d_match_count_, &init_count, sizeof(int));
        
        // Calculate kernel parameters
        dim3 block_size = cuda::calculate_optimal_block_size(device_id_);
        dim3 grid_size = cuda::calculate_grid_size(text.length(), block_size.x);
        
        const int max_matches = 1000;
        
        // Launch find_all kernel
        cuda_kernels::regex_find_all_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
            static_cast<char*>(d_text_), text.length(),
            static_cast<char*>(d_pattern_), pattern.length(),
            static_cast<int*>(d_match_positions_), static_cast<int*>(d_match_lengths_),
            static_cast<int*>(d_match_count_), max_matches
        );
        
        CUDA_CHECK_LAST();
        
        // Copy results back
        memory_manager_->copy_device_to_host(h_match_count_, d_match_count_, sizeof(int));
        cuda::synchronize_stream(compute_stream_);
        
        int num_matches = *static_cast<int*>(h_match_count_);
        num_matches = std::min(num_matches, max_matches);
        
        if (num_matches > 0) {
            memory_manager_->copy_device_to_host(h_match_positions_, d_match_positions_, 
                num_matches * sizeof(int));
            memory_manager_->copy_device_to_host(h_match_lengths_, d_match_lengths_, 
                num_matches * sizeof(int));
            cuda::synchronize_stream(compute_stream_);
        }
        
        // Build result vector
        std::vector<MatchResult> results;
        int* positions = static_cast<int*>(h_match_positions_);
        int* lengths = static_cast<int*>(h_match_lengths_);
        
        for (int i = 0; i < num_matches; ++i) {
            size_t start = positions[i];
            size_t length = lengths[i];
            std::string matched_text = text.substr(start, length);
            results.emplace_back(true, start, start + length - 1, matched_text);
        }
        
        return results;
    }
    
    std::string preprocess_pattern(const std::string& pattern) {
        // Same preprocessing as CPU version
        std::string result = pattern;
        
        for (size_t i = 0; i < result.length(); ++i) {
            if (result[i] == '\\' && i + 1 < result.length()) {
                char next = result[i + 1];
                switch (next) {
                    case 'd':
                        result.replace(i, 2, "[0-9]");
                        break;
                    case 's':
                        result.replace(i, 2, "[ \t\n\r]");
                        break;
                    case 'w':
                        result.replace(i, 2, "[a-zA-Z0-9_]");
                        break;
                    default:
                        result.erase(i, 1);
                        break;
                }
            }
        }
        
        return result;
    }
    
    int device_id_;
    size_t max_text_length_;
    size_t max_pattern_length_;
    
    cuda::DeviceInfo device_info_;
    std::unique_ptr<GPUMemoryManager> memory_manager_;
    
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    
    // GPU memory buffers
    void* d_dp_table_;
    void* d_text_;
    void* d_pattern_;
    void* d_result_;
    void* d_match_positions_;
    void* d_match_lengths_;
    void* d_match_count_;
    
    // Host memory buffers (pinned)
    void* h_result_;
    void* h_match_count_;
    void* h_match_positions_;
    void* h_match_lengths_;
};

// GPURegexEngine public interface
GPURegexEngine::GPURegexEngine(int device_id) : pImpl(std::make_unique<Impl>(device_id)) {}

GPURegexEngine::~GPURegexEngine() = default;

MatchResult GPURegexEngine::match(const std::string& pattern, const std::string& text) {
    return pImpl->match(pattern, text);
}

std::vector<MatchResult> GPURegexEngine::find_all(const std::string& pattern, const std::string& text) {
    return pImpl->find_all(pattern, text);
}

void GPURegexEngine::set_max_text_length(size_t length) {
    pImpl->set_max_text_length(length);
}

void GPURegexEngine::set_max_pattern_length(size_t length) {
    pImpl->set_max_pattern_length(length);
}

size_t GPURegexEngine::get_memory_usage() const {
    return pImpl->get_memory_usage();
}

void GPURegexEngine::cleanup() {
    pImpl->cleanup();
}

void GPURegexEngine::set_device(int device_id) {
    pImpl->set_device(device_id);
}

int GPURegexEngine::get_device() const {
    return pImpl->get_device();
}

void GPURegexEngine::set_memory_pool_size(size_t size) {
    pImpl->set_memory_pool_size(size);
}

} // namespace hpc_regex