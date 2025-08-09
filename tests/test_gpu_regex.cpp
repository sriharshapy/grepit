#include "hpc_regex.h"
#include "regex_engine.h"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>

using namespace hpc_regex;

// Test framework functions (defined in test_main.cpp)
class TestSuite {
public:
    static void TestSuite::assert_true(bool condition, const std::string& message);
    static void assert_equals(const std::string& expected, const std::string& actual, const std::string& message);
};

void test_gpu_basic_matching() {
    std::cout << "Testing basic GPU regex matching..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        GPURegexEngine engine;
        
        // Test simple literal matching
        {
            auto result = engine.match("hello", "hello world");
            TestSuite::assert_true(result.found, "GPU: Simple literal match should succeed");
            TestSuite::assert_true(result.start_pos == 0, "GPU: Match should start at position 0");
        }
        
        // Test pattern not found
        {
            auto result = engine.match("xyz", "hello world");
            TestSuite::assert_true(!result.found, "GPU: Non-matching pattern should not be found");
        }
        
        // Test Kleene star
        {
            auto result = engine.match("a*", "aaab");
            TestSuite::assert_true(result.found, "GPU: Kleene star should match");
        }
        
        // Test plus operator
        {
            auto result = engine.match("a+", "aaab");
            TestSuite::assert_true(result.found, "GPU: Plus operator should match one or more");
        }
        
        {
            auto result = engine.match("a+", "bbb");
            TestSuite::assert_true(!result.found, "GPU: Plus operator should not match zero occurrences");
        }
        
        // Test wildcard
        {
            auto result = engine.match("a.b", "acb");
            TestSuite::assert_true(result.found, "GPU: Wildcard should match any character");
        }
        
        {
            auto result = engine.match("a.b", "ab");
            TestSuite::assert_true(!result.found, "GPU: Wildcard should require exactly one character");
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU basic matching tests" << std::endl;
    #endif
}

void test_gpu_find_all() {
    std::cout << "Testing GPU find_all functionality..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        GPURegexEngine engine;
        
        // Test finding multiple matches
        {
            auto results = engine.find_all("a", "banana");
            TestSuite::assert_true(results.size() == 3, "GPU: Should find 3 'a' characters in 'banana'");
        }
        
        // Test finding no matches
        {
            auto results = engine.find_all("x", "banana");
            TestSuite::assert_true(results.size() == 0, "GPU: Should find no 'x' characters in 'banana'");
        }
        
        // Test pattern with quantifiers
        {
            auto results = engine.find_all("a+", "aaa bbb aaa");
            TestSuite::assert_true(results.size() == 2, "GPU: Should find 2 sequences of 'a+' in 'aaa bbb aaa'");
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU find_all tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU find_all tests" << std::endl;
    #endif
}

void test_gpu_performance() {
    std::cout << "Testing GPU regex performance characteristics..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        GPURegexEngine engine;
        
        // Test with increasingly large text
        std::vector<size_t> sizes = {1000, 10000, 50000};
        
        for (size_t size : sizes) {
            std::string large_text(size, 'a');
            large_text += "target";
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine.match("target", large_text);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            TestSuite::assert_true(result.found, "GPU: Should find target in large text");
            std::cout << "  Text size " << size << ": " << duration.count() << " μs" << std::endl;
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU performance tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU performance tests" << std::endl;
    #endif
}

void test_gpu_memory_management() {
    std::cout << "Testing GPU regex memory management..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        GPURegexEngine engine;
        
        // Test memory usage reporting
        size_t initial_memory = engine.get_memory_usage();
        TestSuite::assert_true(initial_memory >= 0, "GPU: Memory usage should be non-negative");
        
        // Test large pattern/text
        std::string large_pattern(100, 'a');
        large_pattern += "*";
        std::string large_text(50000, 'a');
        
        auto result = engine.match(large_pattern, large_text);
        size_t after_memory = engine.get_memory_usage();
        
        TestSuite::assert_true(after_memory >= initial_memory, "GPU: Memory usage should increase with large data");
        
        // Test memory pool settings
        engine.set_memory_pool_size(1024 * 1024 * 128); // 128MB
        
        // Test cleanup
        engine.cleanup();
        size_t final_memory = engine.get_memory_usage();
        std::cout << "  GPU Memory: initial=" << initial_memory 
                  << ", after=" << after_memory 
                  << ", final=" << final_memory << std::endl;
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU memory tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU memory tests" << std::endl;
    #endif
}

void test_gpu_device_management() {
    std::cout << "Testing GPU device management..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        // Test default device
        {
            GPURegexEngine engine;
            TestSuite::assert_true(engine.get_device() >= 0, "GPU: Default device should be valid");
        }
        
        // Test specific device selection
        {
            GPURegexEngine engine(0);
            TestSuite::assert_true(engine.get_device() == 0, "GPU: Should use specified device");
        }
        
        // Test device switching
        {
            GPURegexEngine engine;
            int original_device = engine.get_device();
            engine.set_device(0);
            TestSuite::assert_true(engine.get_device() == 0, "GPU: Should switch to specified device");
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU device tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU device tests" << std::endl;
    #endif
}

void test_gpu_vs_cpu_correctness() {
    std::cout << "Testing GPU vs CPU correctness..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        CPURegexEngine cpu_engine;
        GPURegexEngine gpu_engine;
        
        // Test cases for correctness comparison
        std::vector<std::pair<std::string, std::string>> test_cases = {
            {"hello", "hello world"},
            {"a*", "aaab"},
            {"a+", "aaab"},
            {"a.b", "acb"},
            {"xyz", "hello world"},
            {"target", "find the target here"}
        };
        
        for (const auto& [pattern, text] : test_cases) {
            auto cpu_result = cpu_engine.match(pattern, text);
            auto gpu_result = gpu_engine.match(pattern, text);
            
            bool matches_agree = (cpu_result.found == gpu_result.found);
            TestSuite::assert_true(matches_agree, 
                       "GPU and CPU results should agree for pattern '" + pattern + "'");
            
            if (cpu_result.found && gpu_result.found) {
                bool positions_agree = (cpu_result.start_pos == gpu_result.start_pos && 
                                      cpu_result.end_pos == gpu_result.end_pos);
                TestSuite::assert_true(positions_agree, 
                           "GPU and CPU match positions should agree for pattern '" + pattern + "'");
            }
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU vs CPU tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU vs CPU correctness tests" << std::endl;
    #endif
}

void test_gpu_edge_cases() {
    std::cout << "Testing GPU regex edge cases..." << std::endl;
    
    #ifdef HPC_REGEX_CUDA_ENABLED
    try {
        GPURegexEngine engine;
        
        // Test empty pattern
        {
            auto result = engine.match("", "hello");
            TestSuite::assert_true(result.found, "GPU: Empty pattern should match (matches empty string)");
        }
        
        // Test empty text
        {
            auto result = engine.match("a", "");
            TestSuite::assert_true(!result.found, "GPU: Pattern should not match empty text");
        }
        
        // Test both empty
        {
            auto result = engine.match("", "");
            TestSuite::assert_true(result.found, "GPU: Empty pattern should match empty text");
        }
        
        // Test very long pattern
        {
            std::string long_pattern(200, 'a');
            auto result = engine.match(long_pattern, "hello");
            TestSuite::assert_true(!result.found, "GPU: Very long pattern should not match short text");
        }
        
        // Test memory limits
        {
            engine.set_max_text_length(1024);
            engine.set_max_pattern_length(100);
            
            auto result = engine.match("test", "test string");
            TestSuite::assert_true(result.found, "GPU: Should work within memory limits");
        }
        
    } catch (const CudaException& e) {
        std::cout << "CUDA not available, skipping GPU edge case tests: " << e.what() << std::endl;
    }
    #else
    std::cout << "CUDA support not enabled, skipping GPU edge case tests" << std::endl;
    #endif
}

// Main test function called from test_main.cpp
void test_gpu_regex() {
    test_gpu_basic_matching();
    test_gpu_find_all();
    test_gpu_performance();
    test_gpu_memory_management();
    test_gpu_device_management();
    test_gpu_vs_cpu_correctness();
    test_gpu_edge_cases();
    
    std::cout << "GPU regex tests completed." << std::endl;
}
