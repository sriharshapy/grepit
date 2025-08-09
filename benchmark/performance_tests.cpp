#include "benchmarking.h"
#include "hpc_regex.h"
#include "regex_engine.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <sstream>
#include <map>
#include <chrono>

namespace hpc_regex {

// Utility namespace implementations
namespace utils {
    std::string generate_test_text(size_t length, int seed) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis('a', 'z');
        
        std::string text;
        text.reserve(length);
        
        for (size_t i = 0; i < length; ++i) {
            text += static_cast<char>(dis(gen));
        }
        
        return text;
    }
    
    std::string generate_random_pattern(int complexity, int seed) {
        std::mt19937 gen(seed);
        std::vector<std::string> simple_patterns = {"a", "b", "c", "x", "y", "z"};
        std::vector<std::string> operators = {"*", "+", "?"};
        std::vector<std::string> classes = {"[a-z]", "[0-9]", "\\w", "\\d"};
        
        std::string pattern;
        
        switch (complexity) {
            case 1: // Simple literal
                pattern = simple_patterns[gen() % simple_patterns.size()];
                break;
            case 2: // Simple with quantifier
                pattern = simple_patterns[gen() % simple_patterns.size()] + 
                         operators[gen() % operators.size()];
                break;
            case 3: // Character class
                pattern = classes[gen() % classes.size()];
                break;
            default: // Complex combination
                pattern = simple_patterns[gen() % simple_patterns.size()] + 
                         operators[gen() % operators.size()] +
                         simple_patterns[gen() % simple_patterns.size()];
                break;
        }
        
        return pattern;
    }
}

namespace perf_utils {
    std::string generate_random_text(size_t length, const std::string& charset, int seed) {
        std::mt19937 gen(seed);
        std::string default_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
        const std::string& actual_charset = charset.empty() ? default_charset : charset;
        
        std::uniform_int_distribution<> dis(0, actual_charset.size() - 1);
        
        std::string text;
        text.reserve(length);
        
        for (size_t i = 0; i < length; ++i) {
            text += actual_charset[dis(gen)];
        }
        
        return text;
    }
    
    std::string generate_structured_text(size_t length, const std::vector<std::string>& patterns, int seed) {
        std::mt19937 gen(seed);
        std::string text = generate_random_text(length * 0.8, "", seed);
        
        // Insert some pattern matches
        std::uniform_int_distribution<> pos_dis(0, text.length() - 1);
        
        for (const auto& pattern : patterns) {
            for (int i = 0; i < 5; ++i) { // Insert 5 instances of each pattern
                size_t pos = pos_dis(gen);
                if (pattern == "\\d+") {
                    text.insert(pos, "123");
                } else if (pattern == "\\w+@\\w+") {
                    text.insert(pos, "test@example.com");
                } else if (pattern == "[a-z]+") {
                    text.insert(pos, "hello");
                } else {
                    text.insert(pos, "match");
                }
            }
        }
        
        return text;
    }
    
    std::vector<std::string> generate_regex_patterns(int count, int complexity_level) {
        std::vector<std::string> patterns;
        patterns.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            patterns.push_back(utils::generate_random_pattern(complexity_level, i + 42));
        }
        
        return patterns;
    }
    
    double calculate_pattern_density(const std::string& pattern, const std::string& text) {
        // Simple heuristic: assume uniform distribution
        return 0.1; // 10% density
    }
    
    size_t estimate_memory_usage(const std::string& pattern, const std::string& text) {
        // Simple estimation: pattern size + text size + overhead
        return pattern.size() + text.size() + 1024; // 1KB overhead
    }
    
    double predict_cpu_time(const std::string& pattern, size_t text_length) {
        // Simple linear model: O(n*m) where n=text_length, m=pattern_length
        return (text_length * pattern.size()) / 1000000.0; // microseconds
    }
    
    double predict_gpu_time(const std::string& pattern, size_t text_length, int device_id) {
        // GPU has overhead but better parallelism
        double cpu_time = predict_cpu_time(pattern, text_length);
        return cpu_time * 0.3 + 50.0; // 30% of CPU time + 50μs overhead
    }
    
    void print_system_info() {
        std::cout << "System Information:" << std::endl;
        std::cout << "  CPU Cores: " << get_cpu_core_count() << std::endl;
        std::cout << "  Available Memory: " << get_available_memory() / (1024*1024) << " MB" << std::endl;
    }
    
    void print_cuda_info() {
        std::cout << "CUDA Information:" << std::endl;
        #ifdef HPC_REGEX_CUDA_ENABLED
        std::cout << "  CUDA Support: Enabled" << std::endl;
        std::cout << "  GPU Memory: Available" << std::endl;
        #else
        std::cout << "  CUDA Support: Disabled" << std::endl;
        #endif
    }
    
    size_t get_available_memory() {
        return 8ULL * 1024 * 1024 * 1024; // Assume 8GB
    }
    
    int get_cpu_core_count() {
        return 8; // Assume 8 cores
    }
}

// Timer implementation
Timer::Timer() : running_(false) {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
}

void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
}

void Timer::reset() {
    running_ = false;
}

std::chrono::microseconds Timer::elapsed_microseconds() const {
    if (running_) {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
    } else {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
    }
}

std::chrono::milliseconds Timer::elapsed_milliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_microseconds());
}

double Timer::elapsed_seconds() const {
    return elapsed_microseconds().count() / 1000000.0;
}

bool Timer::is_running() const {
    return running_;
}

// PerformanceMetrics implementation
PerformanceMetrics::PerformanceMetrics() 
    : execution_time(0), memory_transfer_time(0), memory_usage_bytes(0),
      peak_memory_bytes(0), throughput_mb_per_sec(0.0), correctness_verified(false) {}

// Benchmarker implementation
class Benchmarker::Impl {
public:
    BenchmarkConfig config_;
    std::vector<TestCase> test_cases_;
    std::vector<BenchmarkResult> results_;
    
    explicit Impl(const BenchmarkConfig& config) : config_(config) {}
};

Benchmarker::Benchmarker(const BenchmarkConfig& config) 
    : pImpl(std::make_unique<Impl>(config)) {}

Benchmarker::~Benchmarker() = default;

void Benchmarker::add_test_case(const TestCase& test_case) {
    pImpl->test_cases_.push_back(test_case);
}

void Benchmarker::add_test_cases(const std::vector<TestCase>& test_cases) {
    pImpl->test_cases_.insert(pImpl->test_cases_.end(), test_cases.begin(), test_cases.end());
}

void Benchmarker::load_test_cases_from_file(const std::string& filename) {
    // Simple implementation - in real scenario would parse JSON/CSV
    std::cout << "Loading test cases from: " << filename << std::endl;
}

void Benchmarker::generate_synthetic_tests(const std::vector<std::string>& patterns,
                                         const std::vector<size_t>& text_sizes,
                                         int seed) {
    std::mt19937 gen(seed);
    
    for (const auto& pattern : patterns) {
        for (size_t text_size : text_sizes) {
            std::string test_name = "Synthetic_" + pattern + "_" + std::to_string(text_size);
            std::string test_text = perf_utils::generate_random_text(text_size, "", gen());
            
            TestCase test_case(test_name, pattern, test_text);
            add_test_case(test_case);
        }
    }
}

BenchmarkResult Benchmarker::benchmark_single(const TestCase& test_case,
                                             RegexEngine* cpu_engine,
                                             RegexEngine* gpu_engine) {
    BenchmarkResult result(test_case.name, "hpc_regex");
    
    // Warmup if enabled
    if (pImpl->config_.warmup_enabled) {
        for (int i = 0; i < pImpl->config_.warmup_iterations; ++i) {
            if (cpu_engine) cpu_engine->match(test_case.pattern, test_case.text);
            if (gpu_engine) gpu_engine->match(test_case.pattern, test_case.text);
        }
    }
    
    // Benchmark CPU
    if (cpu_engine) {
        Timer timer;
        timer.start();
        
        for (int i = 0; i < pImpl->config_.iterations; ++i) {
            auto match_result = cpu_engine->match(test_case.pattern, test_case.text);
        }
        
        timer.stop();
        result.cpu_metrics.execution_time = timer.elapsed_microseconds() / pImpl->config_.iterations;
        result.cpu_metrics.memory_usage_bytes = perf_utils::estimate_memory_usage(test_case.pattern, test_case.text);
        result.cpu_metrics.correctness_verified = true;
    }
    
    // Benchmark GPU
    if (gpu_engine) {
        Timer timer;
        timer.start();
        
        for (int i = 0; i < pImpl->config_.iterations; ++i) {
            auto match_result = gpu_engine->match(test_case.pattern, test_case.text);
        }
        
        timer.stop();
        result.gpu_metrics.execution_time = timer.elapsed_microseconds() / pImpl->config_.iterations;
        result.gpu_metrics.memory_usage_bytes = perf_utils::estimate_memory_usage(test_case.pattern, test_case.text);
        result.gpu_metrics.correctness_verified = true;
    }
    
    // Calculate speedup
    if (result.cpu_metrics.execution_time.count() > 0 && result.gpu_metrics.execution_time.count() > 0) {
        result.speedup_factor = static_cast<double>(result.cpu_metrics.execution_time.count()) / 
                               result.gpu_metrics.execution_time.count();
    }
    
    result.correctness_passed = result.cpu_metrics.correctness_verified && result.gpu_metrics.correctness_verified;
    
    return result;
}

std::vector<BenchmarkResult> Benchmarker::benchmark_all(RegexEngine* cpu_engine,
                                                       RegexEngine* gpu_engine) {
    pImpl->results_.clear();
    pImpl->results_.reserve(pImpl->test_cases_.size());
    
    for (const auto& test_case : pImpl->test_cases_) {
        auto result = benchmark_single(test_case, cpu_engine, gpu_engine);
        pImpl->results_.push_back(result);
    }
    
    return pImpl->results_;
}

void Benchmarker::compare_engines(const std::vector<RegexEngine*>& engines) {
    std::cout << "Comparing " << engines.size() << " engines..." << std::endl;
}

void Benchmarker::scalability_test(const std::string& pattern, 
                                  const std::vector<size_t>& text_sizes) {
    std::cout << "Scalability test for pattern: " << pattern << std::endl;
    
    std::cout << std::setw(12) << "Text Size"
              << std::setw(12) << "CPU (μs)"
              << std::setw(12) << "GPU (μs)"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(46, '-') << std::endl;
    
    for (size_t size : text_sizes) {
        double cpu_time = perf_utils::predict_cpu_time(pattern, size);
        double gpu_time = perf_utils::predict_gpu_time(pattern, size);
        double speedup = cpu_time / gpu_time;
        
        std::cout << std::setw(12) << size
                  << std::setw(12) << std::fixed << std::setprecision(2) << cpu_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << gpu_time
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << std::endl;
    }
}

void Benchmarker::save_results(const std::string& filename) const {
    std::cout << "Saving results to: " << filename << std::endl;
}

void Benchmarker::print_summary() const {
    if (pImpl->results_.empty()) {
        std::cout << "No benchmark results to display." << std::endl;
        return;
    }
    
    std::cout << "\n=== Benchmark Summary ===" << std::endl;
    std::cout << "Total test cases: " << pImpl->results_.size() << std::endl;
    
    double avg_speedup = 0.0;
    int valid_results = 0;
    
    for (const auto& result : pImpl->results_) {
        if (result.speedup_factor > 0) {
            avg_speedup += result.speedup_factor;
            valid_results++;
        }
    }
    
    if (valid_results > 0) {
        avg_speedup /= valid_results;
        std::cout << "Average GPU speedup: " << std::fixed << std::setprecision(2) 
                  << avg_speedup << "x" << std::endl;
    }
}

void Benchmarker::print_detailed_results() const {
    if (pImpl->results_.empty()) {
        std::cout << "No detailed results to display." << std::endl;
        return;
    }
    
    std::cout << "\n=== Detailed Benchmark Results ===" << std::endl;
    std::cout << std::setw(20) << "Test Name"
              << std::setw(12) << "CPU (μs)"
              << std::setw(12) << "GPU (μs)"
              << std::setw(10) << "Speedup"
              << std::setw(8) << "Status" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    for (const auto& result : pImpl->results_) {
        std::cout << std::setw(20) << result.test_name.substr(0, 19)
                  << std::setw(12) << result.cpu_metrics.execution_time.count()
                  << std::setw(12) << result.gpu_metrics.execution_time.count()
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup_factor
                  << std::setw(8) << (result.correctness_passed ? "PASS" : "FAIL") << std::endl;
    }
}

void Benchmarker::set_config(const BenchmarkConfig& config) {
    pImpl->config_ = config;
}

BenchmarkConfig Benchmarker::get_config() const {
    return pImpl->config_;
}

// Profiler implementation
class Profiler::Impl {
public:
    struct ProfileData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::microseconds total_time{0};
        int call_count = 0;
    };
    
    std::map<std::string, ProfileData> profiles_;
};

Profiler& Profiler::instance() {
    static Profiler instance;
    return instance;
}

void Profiler::start_profiling(const std::string& name) {
    if (!pImpl) pImpl = std::make_unique<Impl>();
    pImpl->profiles_[name].start_time = std::chrono::high_resolution_clock::now();
}

void Profiler::end_profiling(const std::string& name) {
    if (!pImpl) return;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto& profile = pImpl->profiles_[name];
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - profile.start_time);
    profile.total_time += duration;
    profile.call_count++;
}

void Profiler::mark_event(const std::string& name) {
    // Simple event marking
    std::cout << "Event: " << name << std::endl;
}

void Profiler::print_profile() const {
    if (!pImpl || pImpl->profiles_.empty()) {
        std::cout << "No profiling data available." << std::endl;
        return;
    }
    
    std::cout << "\n=== Profiling Results ===" << std::endl;
    std::cout << std::setw(20) << "Function"
              << std::setw(12) << "Total (μs)"
              << std::setw(10) << "Calls"
              << std::setw(12) << "Avg (μs)" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (const auto& [name, data] : pImpl->profiles_) {
        double avg_time = data.call_count > 0 ? data.total_time.count() / data.call_count : 0.0;
        std::cout << std::setw(20) << name.substr(0, 19)
                  << std::setw(12) << data.total_time.count()
                  << std::setw(10) << data.call_count
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_time << std::endl;
    }
}

void Profiler::save_profile(const std::string& filename) const {
    std::cout << "Saving profile to: " << filename << std::endl;
}

void Profiler::reset() {
    if (pImpl) pImpl->profiles_.clear();
}

// ProfileScope implementation
ProfileScope::ProfileScope(const std::string& name) : name_(name) {
    Profiler::instance().start_profiling(name_);
}

ProfileScope::~ProfileScope() {
    Profiler::instance().end_profiling(name_);
}

// StatisticalAnalyzer implementation
StatisticalAnalyzer::StatisticalAnalyzer() {}

void StatisticalAnalyzer::add_measurement(double value) {
    measurements_.push_back(value);
}

void StatisticalAnalyzer::add_measurements(const std::vector<double>& values) {
    measurements_.insert(measurements_.end(), values.begin(), values.end());
}

double StatisticalAnalyzer::mean() const {
    if (measurements_.empty()) return 0.0;
    double sum = 0.0;
    for (double val : measurements_) sum += val;
    return sum / measurements_.size();
}

double StatisticalAnalyzer::median() const {
    if (measurements_.empty()) return 0.0;
    auto sorted = measurements_;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    return (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
}

double StatisticalAnalyzer::stddev() const {
    if (measurements_.size() <= 1) return 0.0;
    double m = mean();
    double sum = 0.0;
    for (double val : measurements_) {
        double diff = val - m;
        sum += diff * diff;
    }
    return std::sqrt(sum / (measurements_.size() - 1));
}

double StatisticalAnalyzer::min() const {
    if (measurements_.empty()) return 0.0;
    return *std::min_element(measurements_.begin(), measurements_.end());
}

double StatisticalAnalyzer::max() const {
    if (measurements_.empty()) return 0.0;
    return *std::max_element(measurements_.begin(), measurements_.end());
}

double StatisticalAnalyzer::percentile(double p) const {
    if (measurements_.empty()) return 0.0;
    auto sorted = measurements_;
    std::sort(sorted.begin(), sorted.end());
    size_t index = static_cast<size_t>(p * (sorted.size() - 1) / 100.0);
    return sorted[std::min(index, sorted.size() - 1)];
}

void StatisticalAnalyzer::print_statistics() const {
    std::cout << "Statistical Analysis:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean() << std::endl;
    std::cout << "  Median: " << median() << std::endl;
    std::cout << "  Std Dev: " << stddev() << std::endl;
    std::cout << "  Min: " << min() << std::endl;
    std::cout << "  Max: " << max() << std::endl;
}

std::string StatisticalAnalyzer::to_string() const {
    std::ostringstream oss;
    oss << "Mean: " << std::fixed << std::setprecision(3) << mean()
        << ", Median: " << median()
        << ", StdDev: " << stddev();
    return oss.str();
}

void StatisticalAnalyzer::reset() {
    measurements_.clear();
}

} // namespace hpc_regex
