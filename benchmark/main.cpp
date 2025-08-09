#include "hpc_regex.h"
#include "benchmarking.h"
#include "regex_engine.h"
#include <iostream>
#include <vector>
#include <string>

using namespace hpc_regex;

void run_basic_benchmarks() {
    std::cout << "\n=== Basic Performance Benchmarks ===" << std::endl;
    
    // Create benchmark configuration
    BenchmarkConfig config;
    config.iterations = 50;
    config.warmup_enabled = true;
    config.warmup_iterations = 5;
    config.verify_correctness = true;
    config.output_format = "table";
    
    Benchmarker benchmarker(config);
    
    // Add basic test cases
    std::vector<std::string> patterns = {
        "a*",
        "a+b*",
        "abc",
        "a.b",
        "[a-z]*",
        "\\d+",
        "\\w+@\\w+"
    };
    
    std::vector<size_t> text_sizes = {100, 1000, 10000};
    
    // Generate synthetic test cases
    benchmarker.generate_synthetic_tests(patterns, text_sizes, 42);
    
    // Create engines
    CPURegexEngine cpu_engine;
    GPURegexEngine gpu_engine;
    
    // Run benchmarks
    auto results = benchmarker.benchmark_all(&cpu_engine, &gpu_engine);
    
    // Print results
    benchmarker.print_summary();
    benchmarker.print_detailed_results();
}

void run_scalability_tests() {
    std::cout << "\n=== Scalability Tests ===" << std::endl;
    
    BenchmarkConfig config;
    config.iterations = 20;
    config.verify_correctness = true;
    
    Benchmarker benchmarker(config);
    
    // Test patterns of varying complexity
    std::vector<std::string> test_patterns = {
        "a*",           // Simple
        "a*b+c*",       // Medium
        "\\w+@\\w+\\.\\w+"  // Complex
    };
    
    std::vector<size_t> text_sizes = {
        1000, 5000, 10000, 50000, 100000
    };
    
    for (const auto& pattern : test_patterns) {
        benchmarker.scalability_test(pattern, text_sizes);
    }
}

void run_memory_benchmarks() {
    std::cout << "\n=== Memory Performance Tests ===" << std::endl;
    
    RegexConfig config;
    config.use_gpu = true;
    config.max_text_length = 1024 * 1024;  // 1MB
    config.gpu_memory_pool_size = 256 * 1024 * 1024;  // 256MB
    
    HPCRegex regex(config);
    
    // Test different text sizes
    std::vector<size_t> sizes = {1000, 10000, 100000, 500000};
    
    for (size_t size : sizes) {
        std::string large_text = utils::generate_test_text(size, 42);
        std::string pattern = "a*b+";
        
        std::cout << "Testing text size: " << size << " bytes" << std::endl;
        
        BenchmarkResults results = regex.benchmark(pattern, large_text, 10);
        
        std::cout << "  Memory usage: " << regex.get_memory_usage() / 1024 << " KB" << std::endl;
        results.print_summary();
        std::cout << std::endl;
    }
}

void run_pattern_complexity_tests() {
    std::cout << "\n=== Pattern Complexity Tests ===" << std::endl;
    
    // Test patterns of increasing complexity
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Simple literal", "hello"},
        {"Kleene star", "a*"},
        {"Plus operator", "a+"},
        {"Wildcard", "a.b"},
        {"Character class", "[a-z]+"},
        {"Escaped sequences", "\\d+"},
        {"Complex email", "\\w+@\\w+\\.\\w+"},
        {"Multiple operators", "a*b+c*"},
        {"Alternation", "(a|b)*c"},
        {"Nested quantifiers", "(a+b*)*"}
    };
    
    std::string test_text = utils::generate_test_text(10000, 42);
    
    // Insert some pattern matches
    test_text += "hello world 123 test@example.com aabbcc";
    
    RegexConfig config;
    config.use_gpu = true;
    HPCRegex regex(config);
    
    std::cout << std::setw(20) << "Pattern Type" 
              << std::setw(25) << "Pattern"
              << std::setw(12) << "CPU (μs)"
              << std::setw(12) << "GPU (μs)"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(79, '-') << std::endl;
    
    for (const auto& [description, pattern] : test_cases) {
        BenchmarkResults results = regex.benchmark(pattern, test_text, 20);
        
        std::cout << std::setw(20) << description.substr(0, 19)
                  << std::setw(25) << pattern.substr(0, 24)
                  << std::setw(12) << results.timing.cpu_time.count()
                  << std::setw(12) << results.timing.gpu_time.count()
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << results.timing.speedup_factor << std::endl;
    }
}

void run_correctness_tests() {
    std::cout << "\n=== Correctness Verification Tests ===" << std::endl;
    
    RegexConfig config;
    config.use_gpu = true;
    HPCRegex regex(config);
    
    // Test cases with known results
    std::vector<std::tuple<std::string, std::string, bool>> test_cases = {
        {"hello", "hello world", true},
        {"hello", "hi there", false},
        {"a*", "aaaa", true},
        {"a*", "bbbb", true},  // a* matches empty string
        {"a+", "aaaa", true},
        {"a+", "bbbb", false},
        {"a.b", "acb", true},
        {"a.b", "ab", false},
        {"[a-z]+", "hello", true},
        {"[a-z]+", "123", false},
        {"\\d+", "123", true},
        {"\\d+", "abc", false}
    };
    
    int passed = 0;
    int total = test_cases.size();
    
    for (const auto& [pattern, text, expected] : test_cases) {
        BenchmarkResults results = regex.benchmark(pattern, text, 1);
        
        if (results.correctness_verified) {
            MatchResult cpu_result = regex.match(pattern, text);
            bool actual = cpu_result.found;
            
            if (actual == expected) {
                std::cout << "✓ PASS: " << pattern << " vs \"" << text << "\"" << std::endl;
                ++passed;
            } else {
                std::cout << "✗ FAIL: " << pattern << " vs \"" << text 
                          << "\" (expected " << expected << ", got " << actual << ")" << std::endl;
            }
        } else {
            std::cout << "✗ FAIL: " << pattern << " vs \"" << text 
                      << "\" (CPU/GPU mismatch)" << std::endl;
        }
    }
    
    std::cout << "\nCorrectness Tests: " << passed << "/" << total 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * passed / total) << "%)" << std::endl;
}

void print_system_info() {
    std::cout << "=== HPC Regex Library Benchmarks ===" << std::endl;
    
    // Print system information
    perf_utils::print_system_info();
    perf_utils::print_cuda_info();
    
    // Print library configuration
    std::cout << "\n=== Library Configuration ===" << std::endl;
    RegexConfig config;
    std::cout << "Max Text Length: " << config.max_text_length / 1024 << " KB" << std::endl;
    std::cout << "Max Pattern Length: " << config.max_pattern_length << " chars" << std::endl;
    std::cout << "GPU Memory Pool: " << config.gpu_memory_pool_size / (1024*1024) << " MB" << std::endl;
    std::cout << "Caching Enabled: " << (config.enable_caching ? "Yes" : "No") << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        print_system_info();
        
        // Check command line arguments
        std::string test_type = "all";
        if (argc > 1) {
            test_type = argv[1];
        }
        
        if (test_type == "all" || test_type == "basic") {
            run_basic_benchmarks();
        }
        
        if (test_type == "all" || test_type == "scalability") {
            run_scalability_tests();
        }
        
        if (test_type == "all" || test_type == "memory") {
            run_memory_benchmarks();
        }
        
        if (test_type == "all" || test_type == "complexity") {
            run_pattern_complexity_tests();
        }
        
        if (test_type == "all" || test_type == "correctness") {
            run_correctness_tests();
        }
        
        if (test_type == "help") {
            std::cout << "Usage: " << argv[0] << " [test_type]" << std::endl;
            std::cout << "test_type options:" << std::endl;
            std::cout << "  all         - Run all tests (default)" << std::endl;
            std::cout << "  basic       - Basic performance benchmarks" << std::endl;
            std::cout << "  scalability - Scalability tests" << std::endl;
            std::cout << "  memory      - Memory performance tests" << std::endl;
            std::cout << "  complexity  - Pattern complexity tests" << std::endl;
            std::cout << "  correctness - Correctness verification" << std::endl;
            std::cout << "  help        - Show this help message" << std::endl;
        }
        
        std::cout << "\n=== Benchmark Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}