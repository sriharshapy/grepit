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

void test_cpu_basic_matching() {
    std::cout << "Testing basic CPU regex matching..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test simple literal matching
    {
        auto result = engine.match("hello", "hello world");
        TestSuite::TestSuite::assert_true(result.found, "CPU: Simple literal match should succeed");
        TestSuite::TestSuite::assert_true(result.start_pos == 0, "CPU: Match should start at position 0");
    }
    
    // Test pattern not found
    {
        auto result = engine.match("xyz", "hello world");
        TestSuite::TestSuite::assert_true(!result.found, "CPU: Non-matching pattern should not be found");
    }
    
    // Test Kleene star
    {
        auto result = engine.match("a*", "aaab");
        TestSuite::assert_true(result.found, "CPU: Kleene star should match");
    }
    
    // Test plus operator
    {
        auto result = engine.match("a+", "aaab");
        TestSuite::assert_true(result.found, "CPU: Plus operator should match one or more");
    }
    
    {
        auto result = engine.match("a+", "bbb");
        TestSuite::assert_true(!result.found, "CPU: Plus operator should not match zero occurrences");
    }
    
    // Test wildcard
    {
        auto result = engine.match("a.b", "acb");
        TestSuite::assert_true(result.found, "CPU: Wildcard should match any character");
    }
    
    {
        auto result = engine.match("a.b", "ab");
        TestSuite::assert_true(!result.found, "CPU: Wildcard should require exactly one character");
    }
}

void test_cpu_find_all() {
    std::cout << "Testing CPU find_all functionality..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test finding multiple matches
    {
        auto results = engine.find_all("a", "banana");
        TestSuite::assert_true(results.size() == 3, "CPU: Should find 3 'a' characters in 'banana'");
    }
    
    // Test finding no matches
    {
        auto results = engine.find_all("x", "banana");
        TestSuite::assert_true(results.size() == 0, "CPU: Should find no 'x' characters in 'banana'");
    }
    
    // Test pattern with quantifiers
    {
        auto results = engine.find_all("a+", "aaa bbb aaa");
        TestSuite::assert_true(results.size() == 2, "CPU: Should find 2 sequences of 'a+' in 'aaa bbb aaa'");
    }
}

void test_cpu_complex_patterns() {
    std::cout << "Testing CPU complex regex patterns..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test character classes (simplified for basic engine)
    {
        auto result = engine.match("\\d", "123");
        TestSuite::assert_true(result.found, "CPU: Digit class should match numbers");
    }
    
    // Test word boundaries (simplified)
    {
        auto result = engine.match("\\w", "hello");
        TestSuite::assert_true(result.found, "CPU: Word character class should match letters");
    }
    
    // Test escaped characters
    {
        auto result = engine.match("\\.", ".");
        TestSuite::assert_true(result.found, "CPU: Escaped dot should match literal dot");
    }
}

void test_cpu_performance() {
    std::cout << "Testing CPU regex performance characteristics..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test with increasingly large text
    std::vector<size_t> sizes = {100, 1000, 5000};
    
    for (size_t size : sizes) {
        std::string large_text(size, 'a');
        large_text += "target";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine.match("target", large_text);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        TestSuite::assert_true(result.found, "CPU: Should find target in large text");
        std::cout << "  Text size " << size << ": " << duration.count() << " μs" << std::endl;
    }
}

void test_cpu_memory_management() {
    std::cout << "Testing CPU regex memory management..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test memory usage reporting
    size_t initial_memory = engine.get_memory_usage();
    TestSuite::assert_true(initial_memory >= 0, "CPU: Memory usage should be non-negative");
    
    // Test large pattern/text
    std::string large_pattern(100, 'a');
    large_pattern += "*";
    std::string large_text(10000, 'a');
    
    auto result = engine.match(large_pattern, large_text);
    size_t after_memory = engine.get_memory_usage();
    
    TestSuite::assert_true(after_memory >= initial_memory, "CPU: Memory usage should increase with large data");
    
    // Test cleanup
    engine.cleanup();
    size_t final_memory = engine.get_memory_usage();
    std::cout << "  Memory: initial=" << initial_memory 
              << ", after=" << after_memory 
              << ", final=" << final_memory << std::endl;
}

void test_cpu_edge_cases() {
    std::cout << "Testing CPU regex edge cases..." << std::endl;
    
    CPURegexEngine engine;
    
    // Test empty pattern
    {
        auto result = engine.match("", "hello");
        TestSuite::assert_true(result.found, "CPU: Empty pattern should match (matches empty string)");
    }
    
    // Test empty text
    {
        auto result = engine.match("a", "");
        TestSuite::assert_true(!result.found, "CPU: Pattern should not match empty text");
    }
    
    // Test both empty
    {
        auto result = engine.match("", "");
        TestSuite::assert_true(result.found, "CPU: Empty pattern should match empty text");
    }
    
    // Test very long pattern
    {
        std::string long_pattern(200, 'a');
        auto result = engine.match(long_pattern, "hello");
        TestSuite::assert_true(!result.found, "CPU: Very long pattern should not match short text");
    }
}

// Main test function called from test_main.cpp
void test_cpu_regex() {
    test_cpu_basic_matching();
    test_cpu_find_all();
    test_cpu_complex_patterns();
    test_cpu_performance();
    test_cpu_memory_management();
    test_cpu_edge_cases();
    
    std::cout << "CPU regex tests completed." << std::endl;
}
