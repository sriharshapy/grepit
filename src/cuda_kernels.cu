#include "cuda_utils.h"
#include "hpc_regex.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace hpc_regex {
namespace cuda_kernels {

namespace cg = cooperative_groups;

// CUDA kernel for dynamic programming regex matching
__global__ void regex_match_dp_kernel(
    const char* text, 
    size_t text_length,
    const char* pattern, 
    size_t pattern_length,
    bool* dp_table,
    size_t table_width,
    bool* result
) {
    // Calculate thread position in the DP table
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple cells in a diagonal-wise manner
    // to maintain dependencies in the DP algorithm
    
    // Initialize the DP table
    if (tid == 0) {
        dp_table[0] = true; // Empty pattern matches empty text
        *result = false;
    }
    
    __syncthreads();
    
    // Handle patterns like a*, a*b*, etc. that can match empty string
    if (tid < (pattern_length / 2)) {
        int j = (tid + 1) * 2;
        if (j <= pattern_length && pattern[j - 1] == '*') {
            dp_table[j] = dp_table[j - 2];
        }
    }
    
    __syncthreads();
    
    // Process the DP table in diagonal order to respect dependencies
    for (int diag = 1; diag <= text_length + pattern_length; ++diag) {
        // Calculate the range of valid (i, j) pairs for this diagonal
        int start_i = max(1, diag - (int)pattern_length);
        int end_i = min((int)text_length, diag - 1);
        
        // Distribute work among threads
        for (int idx = tid; idx <= end_i - start_i; idx += total_threads) {
            int i = start_i + idx;
            int j = diag - i;
            
            if (i <= text_length && j <= pattern_length && j >= 1) {
                char text_char = text[i - 1];
                char pattern_char = pattern[j - 1];
                
                bool match = false;
                
                if (pattern_char == '*') {
                    // Kleene star - can match zero or more of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j - 2];
                        
                        // Zero occurrences
                        match = dp_table[(i) * table_width + (j - 2)];
                        
                        // One or more occurrences
                        if (prev_char == '.' || prev_char == text_char) {
                            match = match || dp_table[(i - 1) * table_width + j];
                        }
                    }
                } else if (pattern_char == '.' || pattern_char == text_char) {
                    // Direct match or wildcard
                    match = dp_table[(i - 1) * table_width + (j - 1)];
                } else if (pattern_char == '+') {
                    // One or more of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j - 2];
                        if (prev_char == '.' || prev_char == text_char) {
                            match = dp_table[(i - 1) * table_width + (j - 2)] || 
                                   dp_table[(i - 1) * table_width + j];
                        }
                    }
                } else if (pattern_char == '?') {
                    // Zero or one of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j - 2];
                        // Zero occurrences
                        match = dp_table[i * table_width + (j - 2)];
                        // One occurrence
                        if (prev_char == '.' || prev_char == text_char) {
                            match = match || dp_table[(i - 1) * table_width + (j - 2)];
                        }
                    }
                }
                
                dp_table[i * table_width + j] = match;
            }
        }
        
        __syncthreads();
    }
    
    // Check final result
    if (tid == 0) {
        *result = dp_table[text_length * table_width + pattern_length];
    }
}

// Optimized kernel using shared memory for small patterns
__global__ void regex_match_dp_shared_kernel(
    const char* text, 
    size_t text_length,
    const char* pattern, 
    size_t pattern_length,
    bool* result
) {
    extern __shared__ bool shared_dp[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Each block processes a chunk of text
    size_t chunk_size = (text_length + gridDim.x - 1) / gridDim.x;
    size_t start_pos = bid * chunk_size;
    size_t end_pos = min(start_pos + chunk_size, text_length);
    size_t local_text_length = end_pos - start_pos;
    
    if (start_pos >= text_length) return;
    
    // Initialize shared memory DP table
    size_t table_width = pattern_length + 1;
    size_t table_size = (local_text_length + 1) * table_width;
    
    // Initialize with false
    for (int i = tid; i < table_size; i += blockDim.x) {
        shared_dp[i] = false;
    }
    
    __syncthreads();
    
    if (tid == 0) {
        shared_dp[0] = true; // Empty pattern matches empty text
        
        // Handle patterns like a*, a*b*, etc.
        for (size_t j = 2; j <= pattern_length; j += 2) {
            if (pattern[j - 1] == '*') {
                shared_dp[j] = shared_dp[j - 2];
            }
        }
    }
    
    __syncthreads();
    
    // Fill the DP table
    for (size_t i = 1; i <= local_text_length; ++i) {
        for (size_t j = tid + 1; j <= pattern_length; j += blockDim.x) {
            char text_char = text[start_pos + i - 1];
            char pattern_char = pattern[j - 1];
            
            bool match = false;
            
            if (pattern_char == '*') {
                if (j >= 2) {
                    char prev_char = pattern[j - 2];
                    match = shared_dp[i * table_width + (j - 2)];
                    if (prev_char == '.' || prev_char == text_char) {
                        match = match || shared_dp[(i - 1) * table_width + j];
                    }
                }
            } else if (pattern_char == '.' || pattern_char == text_char) {
                match = shared_dp[(i - 1) * table_width + (j - 1)];
            }
            
            shared_dp[i * table_width + j] = match;
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        bool block_result = shared_dp[local_text_length * table_width + pattern_length];
        if (block_result) {
            atomicOr((int*)result, 1);
        }
    }
}

// Kernel for finding all matches in parallel
__global__ void regex_find_all_kernel(
    const char* text,
    size_t text_length,
    const char* pattern,
    size_t pattern_length,
    int* match_positions,
    int* match_lengths,
    int* match_count,
    int max_matches
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread checks different starting positions
    for (size_t start_pos = tid; start_pos < text_length; start_pos += total_threads) {
        // Try different match lengths starting from this position
        for (size_t len = 1; len <= min(text_length - start_pos, size_t(256)); ++len) {
            if (start_pos + len > text_length) break;
            
            // Quick pattern matching for this substring
            bool match = simple_match_device(text + start_pos, len, pattern, pattern_length);
            
            if (match) {
                // Record the match
                int match_idx = atomicAdd(match_count, 1);
                if (match_idx < max_matches) {
                    match_positions[match_idx] = static_cast<int>(start_pos);
                    match_lengths[match_idx] = static_cast<int>(len);
                }
                
                // Skip ahead to avoid overlapping matches
                start_pos += len - 1;
                break;
            }
        }
    }
}

// Device function for simple pattern matching
__device__ bool simple_match_device(
    const char* text, 
    size_t text_len, 
    const char* pattern, 
    size_t pattern_len
) {
    // Simplified regex matching for basic patterns
    size_t i = 0, j = 0;
    
    while (i < text_len && j < pattern_len) {
        if (pattern[j] == '.' || pattern[j] == text[i]) {
            i++;
            j++;
        } else if (j + 1 < pattern_len && pattern[j + 1] == '*') {
            // Handle Kleene star
            char star_char = pattern[j];
            j += 2; // Skip char and *
            
            // Try zero occurrences
            bool zero_match = simple_match_device(text + i, text_len - i, pattern + j, pattern_len - j);
            if (zero_match) return true;
            
            // Try one or more occurrences
            while (i < text_len && (star_char == '.' || star_char == text[i])) {
                i++;
                bool more_match = simple_match_device(text + i, text_len - i, pattern + j, pattern_len - j);
                if (more_match) return true;
            }
            return false;
        } else {
            return false;
        }
    }
    
    // Handle remaining pattern (e.g., trailing a*)
    while (j + 1 < pattern_len && pattern[j + 1] == '*') {
        j += 2;
    }
    
    return j >= pattern_len;
}

// Batch processing kernel for multiple texts
__global__ void regex_batch_match_kernel(
    const char** texts,
    size_t* text_lengths,
    int num_texts,
    const char* pattern,
    size_t pattern_length,
    bool* results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_texts) {
        // Each thread processes one text
        const char* text = texts[tid];
        size_t text_length = text_lengths[tid];
        
        // Allocate temporary DP table (limited size)
        bool dp_table[MAX_PATTERN_LENGTH + 1][MAX_TEXT_CHUNK_LENGTH + 1];
        
        if (text_length <= MAX_TEXT_CHUNK_LENGTH && pattern_length <= MAX_PATTERN_LENGTH) {
            // Initialize
            for (int i = 0; i <= text_length; ++i) {
                for (int j = 0; j <= pattern_length; ++j) {
                    dp_table[j][i] = false;
                }
            }
            
            dp_table[0][0] = true;
            
            // Basic DP algorithm
            for (int i = 0; i <= text_length; ++i) {
                for (int j = 1; j <= pattern_length; ++j) {
                    char pattern_char = pattern[j - 1];
                    
                    if (i > 0) {
                        char text_char = text[i - 1];
                        
                        if (pattern_char == '.' || pattern_char == text_char) {
                            dp_table[j][i] = dp_table[j - 1][i - 1];
                        } else if (pattern_char == '*' && j >= 2) {
                            char prev_char = pattern[j - 2];
                            dp_table[j][i] = dp_table[j - 2][i];
                            if (prev_char == '.' || prev_char == text_char) {
                                dp_table[j][i] = dp_table[j][i] || dp_table[j][i - 1];
                            }
                        }
                    } else if (pattern_char == '*' && j >= 2) {
                        dp_table[j][i] = dp_table[j - 2][i];
                    }
                }
            }
            
            results[tid] = dp_table[pattern_length][text_length];
        } else {
            // Fallback for large inputs
            results[tid] = false;
        }
    }
}

// Constants for kernel limitations
#define MAX_PATTERN_LENGTH 256
#define MAX_TEXT_CHUNK_LENGTH 1024

// Texture memory optimization for large texts
texture<char, 1, cudaReadModeElementType> text_texture;

__global__ void regex_match_texture_kernel(
    size_t text_length,
    const char* pattern,
    size_t pattern_length,
    bool* result
) {
    extern __shared__ bool shared_dp[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    size_t chunk_size = (text_length + gridDim.x - 1) / gridDim.x;
    size_t start_pos = bid * chunk_size;
    size_t end_pos = min(start_pos + chunk_size, text_length);
    
    if (start_pos >= text_length) return;
    
    // Use texture memory to read text data
    for (size_t i = start_pos; i < end_pos; ++i) {
        char text_char = tex1Dfetch(text_texture, i);
        
        // Simplified matching logic using texture reads
        // Full implementation would use the same DP logic as above
        // but reading from texture memory for better cache performance
    }
}

// Cooperative groups version for modern GPUs
__global__ void regex_match_cooperative_kernel(
    const char* text,
    size_t text_length,
    const char* pattern,
    size_t pattern_length,
    bool* dp_table,
    size_t table_width,
    bool* result
) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int tid = grid.thread_rank();
    
    if (tid == 0) {
        dp_table[0] = true;
        *result = false;
    }
    
    grid.sync();
    
    // Warp-cooperative processing of DP table
    for (size_t i = 1; i <= text_length; ++i) {
        for (size_t j = warp.thread_rank() + 1; j <= pattern_length; j += warp.size()) {
            char text_char = text[i - 1];
            char pattern_char = pattern[j - 1];
            
            bool match = false;
            
            if (pattern_char == '.' || pattern_char == text_char) {
                match = dp_table[(i - 1) * table_width + (j - 1)];
            } else if (pattern_char == '*' && j >= 2) {
                char prev_char = pattern[j - 2];
                match = dp_table[i * table_width + (j - 2)];
                if (prev_char == '.' || prev_char == text_char) {
                    match = match || dp_table[(i - 1) * table_width + j];
                }
            }
            
            dp_table[i * table_width + j] = match;
        }
        
        block.sync();
    }
    
    if (tid == 0) {
        *result = dp_table[text_length * table_width + pattern_length];
    }
}

} // namespace cuda_kernels
} // namespace hpc_regex