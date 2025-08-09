#pragma once

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace hpc_regex {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw CudaException(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while(0)

// Device information structure
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int major_compute_capability;
    int minor_compute_capability;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    size_t shared_memory_per_block;
    size_t max_texture_memory;
    bool unified_addressing;
    bool concurrent_kernels;
};

// Device management functions
int get_device_count();
DeviceInfo get_device_info(int device_id);
void set_device(int device_id);
int get_current_device();
void print_all_devices();
void synchronize_device();

// Memory utilities
void* allocate_device(size_t size);
void* allocate_host(size_t size);
void* allocate_unified(size_t size);
void* allocate_pinned(size_t size);

void free_device(void* ptr);
void free_host(void* ptr);
void free_unified(void* ptr);
void free_pinned(void* ptr);

void memcpy_host_to_device(void* dst, const void* src, size_t size);
void memcpy_device_to_host(void* dst, const void* src, size_t size);
void memcpy_device_to_device(void* dst, const void* src, size_t size);
void memcpy_async(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream);

void memset_device(void* ptr, int value, size_t size);
void memset_async(void* ptr, int value, size_t size, cudaStream_t stream);

// Stream management
cudaStream_t create_stream();
void destroy_stream(cudaStream_t stream);
void synchronize_stream(cudaStream_t stream);

// Event management
cudaEvent_t create_event();
void destroy_event(cudaEvent_t event);
void record_event(cudaEvent_t event, cudaStream_t stream = 0);
void synchronize_event(cudaEvent_t event);
float elapsed_time(cudaEvent_t start, cudaEvent_t stop);

// Grid and block dimension helpers
dim3 calculate_grid_size(size_t total_elements, size_t block_size);
dim3 calculate_optimal_block_size(int device_id, size_t shared_memory_per_thread = 0);

// Performance utilities
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    
    void start();
    void stop();
    float elapsed_ms() const;
    void reset();

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool timing_started_;
};

// Memory bandwidth benchmark
struct BandwidthResult {
    float host_to_device_bandwidth;  // GB/s
    float device_to_host_bandwidth;  // GB/s
    float device_to_device_bandwidth; // GB/s
};

BandwidthResult benchmark_memory_bandwidth(size_t data_size = 1024 * 1024 * 100); // 100MB default

// Occupancy calculator
struct OccupancyInfo {
    int max_active_blocks;
    int max_active_threads;
    float occupancy_ratio;
    int optimal_block_size;
};

OccupancyInfo calculate_occupancy(void* kernel_func, int block_size, size_t shared_memory_per_block = 0);

// Texture memory utilities
template<typename T>
class TextureWrapper {
public:
    TextureWrapper(const T* data, size_t size);
    ~TextureWrapper();
    
    cudaTextureObject_t get_texture() const { return texture_; }
    size_t get_size() const { return size_; }

private:
    cudaTextureObject_t texture_;
    cudaArray_t array_;
    size_t size_;
};

// Constant memory utilities
template<typename T>
void copy_to_constant_memory(const T* host_data, size_t size, const void* symbol);

// Shared memory utilities
template<typename T>
__device__ T* get_shared_memory();

// Cooperative groups utilities (for newer CUDA versions)
#if __CUDACC_VER_MAJOR__ >= 9
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

void launch_cooperative_kernel(void* kernel, dim3 grid, dim3 block, 
                              void** args, size_t shared_mem = 0, 
                              cudaStream_t stream = 0);
#endif

// Error handling utilities
std::string get_cuda_error_string(cudaError_t error);
void check_cuda_error(cudaError_t error, const char* file, int line);
void check_last_cuda_error(const char* file, int line);

#define CUDA_CHECK_LAST() check_last_cuda_error(__FILE__, __LINE__)

} // namespace cuda
} // namespace hpc_regex