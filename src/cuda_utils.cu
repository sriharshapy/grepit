#include "cuda_utils.h"
#include "hpc_regex.h"
#include <iostream>
#include <vector>
#include <sstream>

namespace hpc_regex {
namespace cuda {

// Device management functions
int get_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    DeviceInfo info;
    info.device_id = device_id;
    info.name = props.name;
    info.total_memory = total_mem;
    info.free_memory = free_mem;
    info.major_compute_capability = props.major;
    info.minor_compute_capability = props.minor;
    info.multiprocessor_count = props.multiProcessorCount;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;
    info.warp_size = props.warpSize;
    info.shared_memory_per_block = props.sharedMemPerBlock;
    info.max_texture_memory = props.maxTexture1D;
    info.unified_addressing = props.unifiedAddressing;
    info.concurrent_kernels = props.concurrentKernels;
    
    return info;
}

void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int get_current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void print_all_devices() {
    int device_count = get_device_count();
    std::cout << "Available CUDA Devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        DeviceInfo info = get_device_info(i);
        std::cout << "\nDevice " << i << ": " << info.name << std::endl;
        std::cout << "  Compute Capability: " << info.major_compute_capability 
                  << "." << info.minor_compute_capability << std::endl;
        std::cout << "  Total Memory: " << info.total_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "  Free Memory: " << info.free_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << info.multiprocessor_count << std::endl;
        std::cout << "  Max Threads per Block: " << info.max_threads_per_block << std::endl;
        std::cout << "  Warp Size: " << info.warp_size << std::endl;
        std::cout << "  Shared Memory per Block: " << info.shared_memory_per_block / 1024 << " KB" << std::endl;
    }
}

void synchronize_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Memory utilities
void* allocate_device(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void* allocate_host(size_t size) {
    return std::malloc(size);
}

void* allocate_unified(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

void* allocate_pinned(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void free_device(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void free_host(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

void free_unified(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void free_pinned(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void memcpy_host_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void memcpy_device_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void memcpy_device_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void memcpy_async(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
}

void memset_device(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

void memset_async(void* ptr, int value, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream));
}

// Stream management
cudaStream_t create_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

void destroy_stream(cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void synchronize_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Event management
cudaEvent_t create_event() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

void destroy_event(cudaEvent_t event) {
    if (event) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
}

void record_event(cudaEvent_t event, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void synchronize_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
}

float elapsed_time(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}

// Grid and block dimension helpers
dim3 calculate_grid_size(size_t total_elements, size_t block_size) {
    size_t grid_size = (total_elements + block_size - 1) / block_size;
    
    // Handle very large grids
    if (grid_size <= 65535) {
        return dim3(static_cast<unsigned int>(grid_size));
    } else {
        // Use 2D grid for very large problems
        size_t grid_x = std::min(grid_size, size_t(65535));
        size_t grid_y = (grid_size + grid_x - 1) / grid_x;
        return dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
    }
}

dim3 calculate_optimal_block_size(int device_id, size_t shared_memory_per_thread) {
    DeviceInfo info = get_device_info(device_id);
    
    // Calculate optimal block size based on device properties
    size_t max_shared_memory = info.shared_memory_per_block;
    size_t max_threads = info.max_threads_per_block;
    
    // If shared memory is constraining factor
    if (shared_memory_per_thread > 0) {
        size_t max_threads_by_shared_mem = max_shared_memory / shared_memory_per_thread;
        max_threads = std::min(max_threads, max_threads_by_shared_mem);
    }
    
    // Round down to multiple of warp size
    size_t warp_size = info.warp_size;
    size_t optimal_threads = (max_threads / warp_size) * warp_size;
    
    // Common block sizes, prefer 256 for most kernels
    std::vector<size_t> common_sizes = {64, 128, 256, 512, 1024};
    for (auto size : common_sizes) {
        if (size <= optimal_threads) {
            return dim3(static_cast<unsigned int>(size));
        }
    }
    
    return dim3(static_cast<unsigned int>(optimal_threads));
}

// CudaTimer implementation
CudaTimer::CudaTimer() : timing_started_(false) {
    start_event_ = create_event();
    stop_event_ = create_event();
}

CudaTimer::~CudaTimer() {
    destroy_event(start_event_);
    destroy_event(stop_event_);
}

void CudaTimer::start() {
    record_event(start_event_);
    timing_started_ = true;
}

void CudaTimer::stop() {
    if (timing_started_) {
        record_event(stop_event_);
        synchronize_event(stop_event_);
        timing_started_ = false;
    }
}

float CudaTimer::elapsed_ms() const {
    if (timing_started_) {
        return 0.0f; // Timer still running
    }
    return elapsed_time(start_event_, stop_event_);
}

void CudaTimer::reset() {
    timing_started_ = false;
}

// Memory bandwidth benchmark
BandwidthResult benchmark_memory_bandwidth(size_t data_size) {
    BandwidthResult result{};
    
    // Allocate host and device memory
    void* host_data = allocate_pinned(data_size);
    void* device_data1 = allocate_device(data_size);
    void* device_data2 = allocate_device(data_size);
    
    // Initialize host data
    std::memset(host_data, 0xAB, data_size);
    
    CudaTimer timer;
    const int iterations = 10;
    
    try {
        // Benchmark Host to Device
        float total_time = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            memcpy_host_to_device(device_data1, host_data, data_size);
            synchronize_device();
            timer.stop();
            total_time += timer.elapsed_ms();
        }
        result.host_to_device_bandwidth = (data_size * iterations / (1024.0f * 1024.0f * 1024.0f)) / (total_time / 1000.0f);
        
        // Benchmark Device to Host
        total_time = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            memcpy_device_to_host(host_data, device_data1, data_size);
            synchronize_device();
            timer.stop();
            total_time += timer.elapsed_ms();
        }
        result.device_to_host_bandwidth = (data_size * iterations / (1024.0f * 1024.0f * 1024.0f)) / (total_time / 1000.0f);
        
        // Benchmark Device to Device
        total_time = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            memcpy_device_to_device(device_data2, device_data1, data_size);
            synchronize_device();
            timer.stop();
            total_time += timer.elapsed_ms();
        }
        result.device_to_device_bandwidth = (data_size * iterations / (1024.0f * 1024.0f * 1024.0f)) / (total_time / 1000.0f);
        
    } catch (...) {
        // Clean up on error
        free_pinned(host_data);
        free_device(device_data1);
        free_device(device_data2);
        throw;
    }
    
    // Clean up
    free_pinned(host_data);
    free_device(device_data1);
    free_device(device_data2);
    
    return result;
}

// Occupancy calculator (simplified version)
OccupancyInfo calculate_occupancy(void* kernel_func, int block_size, size_t shared_memory_per_block) {
    OccupancyInfo info{};
    
    int device = get_current_device();
    DeviceInfo device_info = get_device_info(device);
    
    // Simplified occupancy calculation
    // In a real implementation, you'd use cudaOccupancyMaxActiveBlocksPerMultiprocessor
    
    int max_blocks_per_sm = device_info.max_threads_per_multiprocessor / block_size;
    
    // Consider shared memory limitations
    if (shared_memory_per_block > 0) {
        int max_blocks_by_shared_mem = static_cast<int>(device_info.shared_memory_per_block / shared_memory_per_block);
        max_blocks_per_sm = std::min(max_blocks_per_sm, max_blocks_by_shared_mem);
    }
    
    info.max_active_blocks = max_blocks_per_sm * device_info.multiprocessor_count;
    info.max_active_threads = info.max_active_blocks * block_size;
    
    int theoretical_max_threads = device_info.multiprocessor_count * device_info.max_threads_per_multiprocessor;
    info.occupancy_ratio = static_cast<float>(info.max_active_threads) / theoretical_max_threads;
    
    // Find optimal block size (simplified)
    info.optimal_block_size = block_size;
    
    return info;
}

// Error handling utilities
std::string get_cuda_error_string(cudaError_t error) {
    return std::string(cudaGetErrorString(error));
}

void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << " - " << get_cuda_error_string(error);
        throw CudaException(oss.str());
    }
}

void check_last_cuda_error(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    check_cuda_error(error, file, line);
}

#if __CUDACC_VER_MAJOR__ >= 9
void launch_cooperative_kernel(void* kernel, dim3 grid, dim3 block, 
                              void** args, size_t shared_mem, 
                              cudaStream_t stream) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid, block, args, shared_mem, stream));
}
#endif

} // namespace cuda
} // namespace hpc_regex