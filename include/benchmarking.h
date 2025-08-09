#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace hpc_regex {

// Forward declarations
class RegexEngine;
struct MatchResult;

// Timing utilities
class Timer {
public:
    Timer();
    void start();
    void stop();
    void reset();
    
    std::chrono::microseconds elapsed_microseconds() const;
    std::chrono::milliseconds elapsed_milliseconds() const;
    double elapsed_seconds() const;
    
    bool is_running() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

// Performance metrics
struct PerformanceMetrics {
    std::chrono::microseconds execution_time;
    std::chrono::microseconds memory_transfer_time;
    size_t memory_usage_bytes;
    size_t peak_memory_bytes;
    double throughput_mb_per_sec;
    bool correctness_verified;
    
    PerformanceMetrics();
};

// Benchmark configuration
struct BenchmarkConfig {
    int iterations = 100;
    bool warmup_enabled = true;
    int warmup_iterations = 10;
    bool verify_correctness = true;
    bool measure_memory_transfer = true;
    bool enable_profiling = false;
    std::string output_format = "table"; // "table", "csv", "json"
    std::string output_file = "";
    
    BenchmarkConfig() = default;
};

// Test case for benchmarking
struct TestCase {
    std::string name;
    std::string pattern;
    std::string text;
    std::vector<MatchResult> expected_results;
    
    TestCase(const std::string& n, const std::string& p, const std::string& t)
        : name(n), pattern(p), text(t) {}
};

// Benchmark results for a single test case
struct BenchmarkResult {
    std::string test_name;
    std::string engine_name;
    PerformanceMetrics cpu_metrics;
    PerformanceMetrics gpu_metrics;
    double speedup_factor;
    bool correctness_passed;
    
    BenchmarkResult(const std::string& test, const std::string& engine)
        : test_name(test), engine_name(engine), speedup_factor(0.0), correctness_passed(false) {}
};

// Main benchmarking class
class Benchmarker {
public:
    explicit Benchmarker(const BenchmarkConfig& config = BenchmarkConfig());
    ~Benchmarker();
    
    // Add test cases
    void add_test_case(const TestCase& test_case);
    void add_test_cases(const std::vector<TestCase>& test_cases);
    void load_test_cases_from_file(const std::string& filename);
    
    // Generate synthetic test cases
    void generate_synthetic_tests(const std::vector<std::string>& patterns,
                                 const std::vector<size_t>& text_sizes,
                                 int seed = 42);
    
    // Run benchmarks
    BenchmarkResult benchmark_single(const TestCase& test_case,
                                   RegexEngine* cpu_engine,
                                   RegexEngine* gpu_engine);
    
    std::vector<BenchmarkResult> benchmark_all(RegexEngine* cpu_engine,
                                              RegexEngine* gpu_engine);
    
    // Comparative benchmarks
    void compare_engines(const std::vector<RegexEngine*>& engines);
    void scalability_test(const std::string& pattern, 
                         const std::vector<size_t>& text_sizes);
    
    // Results management
    void save_results(const std::string& filename) const;
    void print_summary() const;
    void print_detailed_results() const;
    
    // Configuration
    void set_config(const BenchmarkConfig& config);
    BenchmarkConfig get_config() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Profiling utilities
class Profiler {
public:
    static Profiler& instance();
    
    void start_profiling(const std::string& name);
    void end_profiling(const std::string& name);
    void mark_event(const std::string& name);
    
    void print_profile() const;
    void save_profile(const std::string& filename) const;
    void reset();

private:
    Profiler() = default;
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// RAII profiling helper
class ProfileScope {
public:
    explicit ProfileScope(const std::string& name);
    ~ProfileScope();

private:
    std::string name_;
};

#define PROFILE_SCOPE(name) ProfileScope _prof_scope(name)

// Performance testing utilities
namespace perf_utils {
    // Generate test data
    std::string generate_random_text(size_t length, const std::string& charset = "", int seed = 42);
    std::string generate_structured_text(size_t length, const std::vector<std::string>& patterns, int seed = 42);
    std::vector<std::string> generate_regex_patterns(int count, int complexity_level = 3);
    
    // Text analysis
    double calculate_pattern_density(const std::string& pattern, const std::string& text);
    size_t estimate_memory_usage(const std::string& pattern, const std::string& text);
    
    // Performance prediction
    double predict_cpu_time(const std::string& pattern, size_t text_length);
    double predict_gpu_time(const std::string& pattern, size_t text_length, int device_id = 0);
    
    // System information
    void print_system_info();
    void print_cuda_info();
    size_t get_available_memory();
    int get_cpu_core_count();
}

// Statistical analysis of benchmark results
class StatisticalAnalyzer {
public:
    StatisticalAnalyzer();
    
    void add_measurement(double value);
    void add_measurements(const std::vector<double>& values);
    
    double mean() const;
    double median() const;
    double stddev() const;
    double min() const;
    double max() const;
    double percentile(double p) const;
    
    void print_statistics() const;
    std::string to_string() const;
    
    void reset();

private:
    std::vector<double> measurements_;
};

} // namespace hpc_regex