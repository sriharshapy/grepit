#include "hpc_regex.h"
#include <iostream>

using namespace hpc_regex;

int main() {
    try {
        // Create HPC Regex with GPU acceleration
        RegexConfig config;
        config.use_gpu = true;
        HPCRegex regex(config);
        
        // Test basic patterns
        std::cout << "=== Simple Test Cases ===" << std::endl;
        
        // Test 1: Simple literal
        {
            std::string pattern = "hello";
            std::string text = "hello world";
            MatchResult result = regex.match(pattern, text);
            std::cout << "Test 1 - Literal 'hello': " << (result.found ? "PASS" : "FAIL") << std::endl;
        }
        
        // Test 2: Wildcard
        {
            std::string pattern = "h.llo";
            std::string text = "hello world";
            MatchResult result = regex.match(pattern, text);
            std::cout << "Test 2 - Wildcard 'h.llo': " << (result.found ? "PASS" : "FAIL") << std::endl;
        }
        
        // Test 3: Kleene star
        {
            std::string pattern = "a*";
            std::string text = "aaabbb";
            MatchResult result = regex.match(pattern, text);
            std::cout << "Test 3 - Kleene star 'a*': " << (result.found ? "PASS" : "FAIL") << std::endl;
        }
        
        // Test 4: Performance test
        {
            std::string pattern = "\\d+";
            std::string large_text = utils::generate_test_text(10000, 42);
            large_text += " 12345 ";
            
            std::cout << "\n=== Performance Test ===" << std::endl;
            BenchmarkResults results = regex.benchmark(pattern, large_text, 10);
            results.print_summary();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}