#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace hpc_regex {

// Forward declarations
class RegexEngine;
class MemoryManager;
class BenchmarkResults;

// Match result structure
struct MatchResult {
    bool found;
    size_t start_pos;
    size_t end_pos;
    std::string matched_text;
    
    MatchResult() : found(false), start_pos(0), end_pos(0) {}
    MatchResult(bool f, size_t start, size_t end, const std::string& text)
        : found(f), start_pos(start), end_pos(end), matched_text(text) {}
};

// Configuration for regex matching
struct RegexConfig {
    bool use_gpu = true;
    size_t max_text_length = 1024 * 1024;  // 1MB default
    size_t max_pattern_length = 256;
    int gpu_device_id = 0;
    size_t gpu_memory_pool_size = 512 * 1024 * 1024;  // 512MB
    bool enable_caching = true;
    
    RegexConfig() = default;
};

// Main HPC Regex Library Class
class HPCRegex {
public:
    explicit HPCRegex(const RegexConfig& config = RegexConfig());
    ~HPCRegex();

    // Core regex matching functions
    MatchResult match(const std::string& pattern, const std::string& text);
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text);
    
    // Batch processing for large datasets
    std::vector<MatchResult> batch_match(const std::string& pattern, 
                                       const std::vector<std::string>& texts);
    
    // Performance and benchmarking
    BenchmarkResults benchmark(const std::string& pattern, const std::string& text,
                              int iterations = 100);
    
    // Configuration management
    void set_config(const RegexConfig& config);
    RegexConfig get_config() const;
    
    // Memory and resource management
    void cleanup();
    size_t get_memory_usage() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Benchmark results structure
class BenchmarkResults {
public:
    struct TimingInfo {
        std::chrono::microseconds cpu_time;
        std::chrono::microseconds gpu_time;
        std::chrono::microseconds memory_transfer_time;
        double speedup_factor;
        size_t iterations;
    };
    
    TimingInfo timing;
    size_t memory_used_cpu;
    size_t memory_used_gpu;
    bool correctness_verified;
    
    void print_summary() const;
    std::string to_csv() const;
};

// Utility functions
namespace utils {
    std::string generate_test_text(size_t length, int seed = 42);
    std::vector<std::string> load_test_patterns();
    bool validate_pattern(const std::string& pattern);
    void print_device_info();
}

// Exception classes
class HPCRegexException : public std::exception {
public:
    explicit HPCRegexException(const std::string& message) : msg_(message) {}
    const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

class CudaException : public HPCRegexException {
public:
    explicit CudaException(const std::string& message) 
        : HPCRegexException("CUDA Error: " + message) {}
};

class MemoryException : public HPCRegexException {
public:
    explicit MemoryException(const std::string& message) 
        : HPCRegexException("Memory Error: " + message) {}
};

} // namespace hpc_regex