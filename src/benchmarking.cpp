#include "benchmarking.h"
#include "regex_engine.h"
#include "hpc_regex.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

namespace hpc_regex {

// Timer Implementation
Timer::Timer() : running_(false) {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
}

void Timer::stop() {
    if (running_) {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }
}

void Timer::reset() {
    running_ = false;
}

std::chrono::microseconds Timer::elapsed_microseconds() const {
    if (running_) {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
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

// PerformanceMetrics Implementation
PerformanceMetrics::PerformanceMetrics() 
    : execution_time(0), memory_transfer_time(0), memory_usage_bytes(0),
      peak_memory_bytes(0), throughput_mb_per_sec(0.0), correctness_verified(false) {}

// Benchmarker Implementation
class Benchmarker::Impl {
public:
    explicit Impl(const BenchmarkConfig& config) : config_(config) {}
    
    void add_test_case(const TestCase& test_case) {
        test_cases_.push_back(test_case);
    }
    
    void add_test_cases(const std::vector<TestCase>& test_cases) {
        test_cases_.insert(test_cases_.end(), test_cases.begin(), test_cases.end());
    }
    
    void load_test_cases_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw HPCRegexException("Cannot open test case file: " + filename);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Parse CSV format: name,pattern,text
            std::istringstream iss(line);
            std::string name, pattern, text;
            
            if (std::getline(iss, name, ',') &&
                std::getline(iss, pattern, ',') &&
                std::getline(iss, text)) {
                test_cases_.emplace_back(name, pattern, text);
            }
        }
    }
    
    void generate_synthetic_tests(const std::vector<std::string>& patterns,
                                 const std::vector<size_t>& text_sizes,
                                 int seed) {
        std::mt19937 gen(seed);
        
        for (const auto& pattern : patterns) {
            for (size_t text_size : text_sizes) {
                std::string test_name = "synthetic_" + pattern + "_" + std::to_string(text_size);
                std::string test_text = perf_utils::generate_random_text(text_size, "", seed);
                test_cases_.emplace_back(test_name, pattern, test_text);
            }
        }
    }
    
    BenchmarkResult benchmark_single(const TestCase& test_case,
                                   RegexEngine* cpu_engine,
                                   RegexEngine* gpu_engine) {
        BenchmarkResult result(test_case.name, "HPCRegex");
        
        // Warmup if enabled
        if (config_.warmup_enabled) {
            for (int i = 0; i < config_.warmup_iterations; ++i) {
                cpu_engine->match(test_case.pattern, test_case.text);
                if (gpu_engine) {
                    gpu_engine->match(test_case.pattern, test_case.text);
                }
            }
        }
        
        // Benchmark CPU
        Timer cpu_timer;
        MatchResult cpu_result;
        
        cpu_timer.start();
        for (int i = 0; i < config_.iterations; ++i) {
            cpu_result = cpu_engine->match(test_case.pattern, test_case.text);
        }
        cpu_timer.stop();
        
        result.cpu_metrics.execution_time = cpu_timer.elapsed_microseconds() / config_.iterations;
        result.cpu_metrics.memory_usage_bytes = cpu_engine->get_memory_usage();
        result.cpu_metrics.throughput_mb_per_sec = calculate_throughput(
            test_case.text.length(), result.cpu_metrics.execution_time);
        
        // Benchmark GPU if available
        if (gpu_engine) {
            Timer gpu_timer;
            MatchResult gpu_result;
            
            gpu_timer.start();
            for (int i = 0; i < config_.iterations; ++i) {
                gpu_result = gpu_engine->match(test_case.pattern, test_case.text);
            }
            gpu_timer.stop();
            
            result.gpu_metrics.execution_time = gpu_timer.elapsed_microseconds() / config_.iterations;
            result.gpu_metrics.memory_usage_bytes = gpu_engine->get_memory_usage();
            result.gpu_metrics.throughput_mb_per_sec = calculate_throughput(
                test_case.text.length(), result.gpu_metrics.execution_time);
            
            // Calculate speedup
            if (result.gpu_metrics.execution_time.count() > 0) {
                result.speedup_factor = static_cast<double>(result.cpu_metrics.execution_time.count()) /
                                       result.gpu_metrics.execution_time.count();
            }
            
            // Verify correctness
            if (config_.verify_correctness) {
                result.correctness_passed = verify_correctness(cpu_result, gpu_result);
                result.cpu_metrics.correctness_verified = result.correctness_passed;
                result.gpu_metrics.correctness_verified = result.correctness_passed;
            }
        } else {
            result.speedup_factor = 1.0;
            result.correctness_passed = true;
        }
        
        return result;
    }
    
    std::vector<BenchmarkResult> benchmark_all(RegexEngine* cpu_engine,
                                              RegexEngine* gpu_engine) {
        std::vector<BenchmarkResult> results;
        results.reserve(test_cases_.size());
        
        std::cout << "Running " << test_cases_.size() << " benchmark tests..." << std::endl;
        
        for (size_t i = 0; i < test_cases_.size(); ++i) {
            std::cout << "Test " << (i + 1) << "/" << test_cases_.size() 
                      << ": " << test_cases_[i].name << std::endl;
            
            BenchmarkResult result = benchmark_single(test_cases_[i], cpu_engine, gpu_engine);
            results.push_back(result);
            
            // Print progress
            if (gpu_engine) {
                std::cout << "  CPU: " << result.cpu_metrics.execution_time.count() << "μs, "
                          << "GPU: " << result.gpu_metrics.execution_time.count() << "μs, "
                          << "Speedup: " << std::fixed << std::setprecision(2) 
                          << result.speedup_factor << "x" << std::endl;
            } else {
                std::cout << "  CPU: " << result.cpu_metrics.execution_time.count() << "μs" << std::endl;
            }
        }
        
        return results;
    }
    
    void compare_engines(const std::vector<RegexEngine*>& engines) {
        // Implementation for comparing multiple engines
        // This would run the same test cases on different engine implementations
        std::cout << "Multi-engine comparison not fully implemented yet" << std::endl;
    }
    
    void scalability_test(const std::string& pattern, 
                         const std::vector<size_t>& text_sizes) {
        std::cout << "\n=== Scalability Test: Pattern '" << pattern << "' ===" << std::endl;
        std::cout << std::setw(12) << "Text Size" << std::setw(15) << "CPU Time (μs)" 
                  << std::setw(15) << "GPU Time (μs)" << std::setw(12) << "Speedup" << std::endl;
        std::cout << std::string(54, '-') << std::endl;
        
        for (size_t text_size : text_sizes) {
            std::string test_text = perf_utils::generate_random_text(text_size);
            TestCase test_case("scalability_" + std::to_string(text_size), pattern, test_text);
            
            // Create temporary engines for this test
            CPURegexEngine cpu_engine;
            GPURegexEngine gpu_engine;
            
            BenchmarkResult result = benchmark_single(test_case, &cpu_engine, &gpu_engine);
            
            std::cout << std::setw(12) << text_size 
                      << std::setw(15) << result.cpu_metrics.execution_time.count()
                      << std::setw(15) << result.gpu_metrics.execution_time.count()
                      << std::setw(12) << std::fixed << std::setprecision(2) 
                      << result.speedup_factor << std::endl;
        }
    }
    
    void save_results(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw HPCRegexException("Cannot open output file: " + filename);
        }
        
        if (config_.output_format == "csv") {
            save_results_csv(file);
        } else if (config_.output_format == "json") {
            save_results_json(file);
        } else {
            save_results_table(file);
        }
    }
    
    void print_summary() const {
        if (results_.empty()) {
            std::cout << "No benchmark results available." << std::endl;
            return;
        }
        
        std::cout << "\n=== Benchmark Summary ===" << std::endl;
        
        // Calculate statistics
        double total_cpu_time = 0, total_gpu_time = 0;
        double max_speedup = 0, min_speedup = std::numeric_limits<double>::max();
        int correctness_passed = 0;
        
        for (const auto& result : results_) {
            total_cpu_time += result.cpu_metrics.execution_time.count();
            total_gpu_time += result.gpu_metrics.execution_time.count();
            max_speedup = std::max(max_speedup, result.speedup_factor);
            min_speedup = std::min(min_speedup, result.speedup_factor);
            if (result.correctness_passed) ++correctness_passed;
        }
        
        double avg_speedup = total_cpu_time / total_gpu_time;
        
        std::cout << "Total Tests: " << results_.size() << std::endl;
        std::cout << "Correctness Passed: " << correctness_passed << "/" << results_.size() << std::endl;
        std::cout << "Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
        std::cout << "Max Speedup: " << max_speedup << "x" << std::endl;
        std::cout << "Min Speedup: " << min_speedup << "x" << std::endl;
        std::cout << "Total CPU Time: " << total_cpu_time / 1000 << " ms" << std::endl;
        std::cout << "Total GPU Time: " << total_gpu_time / 1000 << " ms" << std::endl;
    }
    
    void print_detailed_results() const {
        std::cout << "\n=== Detailed Benchmark Results ===" << std::endl;
        std::cout << std::setw(20) << "Test Name" 
                  << std::setw(12) << "CPU (μs)" 
                  << std::setw(12) << "GPU (μs)"
                  << std::setw(10) << "Speedup"
                  << std::setw(12) << "Correctness" << std::endl;
        std::cout << std::string(66, '-') << std::endl;
        
        for (const auto& result : results_) {
            std::cout << std::setw(20) << result.test_name.substr(0, 19)
                      << std::setw(12) << result.cpu_metrics.execution_time.count()
                      << std::setw(12) << result.gpu_metrics.execution_time.count()
                      << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup_factor
                      << std::setw(12) << (result.correctness_passed ? "PASS" : "FAIL") << std::endl;
        }
    }
    
    void set_config(const BenchmarkConfig& config) {
        config_ = config;
    }
    
    BenchmarkConfig get_config() const {
        return config_;
    }

private:
    double calculate_throughput(size_t data_size, std::chrono::microseconds time) {
        if (time.count() == 0) return 0.0;
        double seconds = time.count() / 1000000.0;
        double mb = data_size / (1024.0 * 1024.0);
        return mb / seconds;
    }
    
    bool verify_correctness(const MatchResult& cpu_result, const MatchResult& gpu_result) {
        if (cpu_result.found != gpu_result.found) return false;
        if (cpu_result.found) {
            return (cpu_result.start_pos == gpu_result.start_pos) &&
                   (cpu_result.end_pos == gpu_result.end_pos);
        }
        return true;
    }
    
    void save_results_csv(std::ofstream& file) const {
        file << "Test Name,CPU Time (μs),GPU Time (μs),Speedup,CPU Memory (bytes),GPU Memory (bytes),Correctness\n";
        for (const auto& result : results_) {
            file << result.test_name << ","
                 << result.cpu_metrics.execution_time.count() << ","
                 << result.gpu_metrics.execution_time.count() << ","
                 << result.speedup_factor << ","
                 << result.cpu_metrics.memory_usage_bytes << ","
                 << result.gpu_metrics.memory_usage_bytes << ","
                 << (result.correctness_passed ? "PASS" : "FAIL") << "\n";
        }
    }
    
    void save_results_json(std::ofstream& file) const {
        file << "{\n  \"benchmark_results\": [\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            file << "    {\n";
            file << "      \"test_name\": \"" << result.test_name << "\",\n";
            file << "      \"cpu_time_us\": " << result.cpu_metrics.execution_time.count() << ",\n";
            file << "      \"gpu_time_us\": " << result.gpu_metrics.execution_time.count() << ",\n";
            file << "      \"speedup\": " << result.speedup_factor << ",\n";
            file << "      \"correctness\": " << (result.correctness_passed ? "true" : "false") << "\n";
            file << "    }";
            if (i < results_.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ]\n}\n";
    }
    
    void save_results_table(std::ofstream& file) const {
        file << "Benchmark Results\n";
        file << "=================\n\n";
        file << std::setw(20) << "Test Name" 
             << std::setw(12) << "CPU (μs)" 
             << std::setw(12) << "GPU (μs)"
             << std::setw(10) << "Speedup"
             << std::setw(12) << "Correctness" << "\n";
        file << std::string(66, '-') << "\n";
        
        for (const auto& result : results_) {
            file << std::setw(20) << result.test_name
                 << std::setw(12) << result.cpu_metrics.execution_time.count()
                 << std::setw(12) << result.gpu_metrics.execution_time.count()
                 << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup_factor
                 << std::setw(12) << (result.correctness_passed ? "PASS" : "FAIL") << "\n";
        }
    }
    
    BenchmarkConfig config_;
    std::vector<TestCase> test_cases_;
    std::vector<BenchmarkResult> results_;
};

// Benchmarker public interface
Benchmarker::Benchmarker(const BenchmarkConfig& config) 
    : pImpl(std::make_unique<Impl>(config)) {}

Benchmarker::~Benchmarker() = default;

void Benchmarker::add_test_case(const TestCase& test_case) {
    pImpl->add_test_case(test_case);
}

void Benchmarker::add_test_cases(const std::vector<TestCase>& test_cases) {
    pImpl->add_test_cases(test_cases);
}

void Benchmarker::load_test_cases_from_file(const std::string& filename) {
    pImpl->load_test_cases_from_file(filename);
}

void Benchmarker::generate_synthetic_tests(const std::vector<std::string>& patterns,
                                         const std::vector<size_t>& text_sizes,
                                         int seed) {
    pImpl->generate_synthetic_tests(patterns, text_sizes, seed);
}

BenchmarkResult Benchmarker::benchmark_single(const TestCase& test_case,
                                             RegexEngine* cpu_engine,
                                             RegexEngine* gpu_engine) {
    return pImpl->benchmark_single(test_case, cpu_engine, gpu_engine);
}

std::vector<BenchmarkResult> Benchmarker::benchmark_all(RegexEngine* cpu_engine,
                                                       RegexEngine* gpu_engine) {
    return pImpl->benchmark_all(cpu_engine, gpu_engine);
}

void Benchmarker::compare_engines(const std::vector<RegexEngine*>& engines) {
    pImpl->compare_engines(engines);
}

void Benchmarker::scalability_test(const std::string& pattern, 
                                 const std::vector<size_t>& text_sizes) {
    pImpl->scalability_test(pattern, text_sizes);
}

void Benchmarker::save_results(const std::string& filename) const {
    pImpl->save_results(filename);
}

void Benchmarker::print_summary() const {
    pImpl->print_summary();
}

void Benchmarker::print_detailed_results() const {
    pImpl->print_detailed_results();
}

void Benchmarker::set_config(const BenchmarkConfig& config) {
    pImpl->set_config(config);
}

BenchmarkConfig Benchmarker::get_config() const {
    return pImpl->get_config();
}

// Performance utilities implementation
namespace perf_utils {
    std::string generate_random_text(size_t length, const std::string& charset, int seed) {
        std::mt19937 gen(seed);
        
        std::string default_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
        const std::string& actual_charset = charset.empty() ? default_charset : charset;
        
        std::uniform_int_distribution<> dis(0, actual_charset.length() - 1);
        
        std::string result;
        result.reserve(length);
        
        for (size_t i = 0; i < length; ++i) {
            result += actual_charset[dis(gen)];
        }
        
        return result;
    }
    
    std::string generate_structured_text(size_t length, const std::vector<std::string>& patterns, int seed) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> pattern_dis(0, patterns.size() - 1);
        std::uniform_int_distribution<> insert_dis(0, 10);
        
        std::string result = generate_random_text(length, "", seed);
        
        // Insert pattern matches at random positions
        for (size_t i = 0; i < length / 50 && i < patterns.size() * 5; ++i) {
            if (insert_dis(gen) < 3) { // 30% chance to insert a pattern
                size_t pos = gen() % (result.length() - 10);
                const std::string& pattern = patterns[pattern_dis(gen)];
                if (pos + pattern.length() < result.length()) {
                    result.replace(pos, pattern.length(), pattern);
                }
            }
        }
        
        return result;
    }
    
    std::vector<std::string> generate_regex_patterns(int count, int complexity_level) {
        std::vector<std::string> patterns;
        
        // Basic patterns
        std::vector<std::string> basic = {"a", "ab", "abc", "a*", "a+", "a?"};
        
        // Medium patterns
        std::vector<std::string> medium = {"a.b", "[a-z]*", "\\d+", "\\w+", "a*b+", "(a|b)*"};
        
        // Complex patterns
        std::vector<std::string> complex = {
            "\\w+@\\w+\\.\\w+", "[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
            "\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}", "(\\d{4}-\\d{2}-\\d{2})",
            "((a|b)*c)+", "a*b*c*d*", "[A-Z][a-z]+\\s+[A-Z][a-z]+"
        };
        
        if (complexity_level <= 1) {
            patterns.insert(patterns.end(), basic.begin(), basic.end());
        }
        if (complexity_level <= 2) {
            patterns.insert(patterns.end(), medium.begin(), medium.end());
        }
        if (complexity_level >= 3) {
            patterns.insert(patterns.end(), complex.begin(), complex.end());
        }
        
        // Return first 'count' patterns or all if count is larger
        if (count < patterns.size()) {
            patterns.resize(count);
        }
        
        return patterns;
    }
    
    void print_system_info() {
        std::cout << "\n=== System Information ===" << std::endl;
        std::cout << "CPU Cores: " << std::thread::hardware_concurrency() << std::endl;
        
        // Add more system info as needed
        #ifdef _WIN32
        std::cout << "Operating System: Windows" << std::endl;
        #elif __linux__
        std::cout << "Operating System: Linux" << std::endl;
        #elif __APPLE__
        std::cout << "Operating System: macOS" << std::endl;
        #endif
    }
    
    void print_cuda_info() {
        try {
            utils::print_device_info();
        } catch (const std::exception& e) {
            std::cout << "CUDA information not available: " << e.what() << std::endl;
        }
    }
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
    std::vector<double> sorted = measurements_;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        return sorted[n/2];
    }
}

double StatisticalAnalyzer::stddev() const {
    if (measurements_.size() < 2) return 0.0;
    double m = mean();
    double sum = 0.0;
    for (double val : measurements_) {
        sum += (val - m) * (val - m);
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
    std::vector<double> sorted = measurements_;
    std::sort(sorted.begin(), sorted.end());
    
    double index = p * (sorted.size() - 1) / 100.0;
    size_t lower = static_cast<size_t>(index);
    size_t upper = lower + 1;
    
    if (upper >= sorted.size()) return sorted.back();
    if (lower == upper) return sorted[lower];
    
    double weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

void StatisticalAnalyzer::print_statistics() const {
    std::cout << to_string() << std::endl;
}

std::string StatisticalAnalyzer::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Statistics (n=" << measurements_.size() << "):\n";
    oss << "  Mean: " << mean() << "\n";
    oss << "  Median: " << median() << "\n";
    oss << "  Std Dev: " << stddev() << "\n";
    oss << "  Min: " << min() << "\n";
    oss << "  Max: " << max() << "\n";
    oss << "  95th %ile: " << percentile(95) << "\n";
    oss << "  99th %ile: " << percentile(99);
    return oss.str();
}

void StatisticalAnalyzer::reset() {
    measurements_.clear();
}

} // namespace hpc_regex