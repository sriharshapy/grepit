#include "hpc_regex.h"
#include <iostream>
#include <vector>
#include <string>

using namespace hpc_regex;

void basic_matching_example() {
    std::cout << "=== Basic Regex Matching Example ===" << std::endl;
    
    // Create HPC Regex instance with default configuration
    RegexConfig config;
    config.use_gpu = true;  // Enable GPU acceleration
    HPCRegex regex(config);
    
    // Test cases
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"hello", "hello world"},
        {"a*", "aaabbb"},
        {"[0-9]+", "abc123def"},
        {"\\w+@\\w+", "user@example.com"},
        {"a.b", "acb"}
    };
    
    for (const auto& [pattern, text] : test_cases) {
        std::cout << "\nPattern: \"" << pattern << "\"" << std::endl;
        std::cout << "Text: \"" << text << "\"" << std::endl;
        
        MatchResult result = regex.match(pattern, text);
        
        if (result.found) {
            std::cout << "✓ Match found at position " << result.start_pos 
                      << "-" << result.end_pos << std::endl;
            std::cout << "  Matched text: \"" << result.matched_text << "\"" << std::endl;
        } else {
            std::cout << "✗ No match found" << std::endl;
        }
    }
}

void find_all_example() {
    std::cout << "\n=== Find All Matches Example ===" << std::endl;
    
    HPCRegex regex;
    
    std::string pattern = "\\d+";
    std::string text = "There are 123 apples, 456 oranges, and 789 bananas.";
    
    std::cout << "Pattern: \"" << pattern << "\"" << std::endl;
    std::cout << "Text: \"" << text << "\"" << std::endl;
    
    std::vector<MatchResult> matches = regex.find_all(pattern, text);
    
    std::cout << "Found " << matches.size() << " matches:" << std::endl;
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];
        std::cout << "  Match " << (i + 1) << ": \"" << match.matched_text 
                  << "\" at position " << match.start_pos << "-" << match.end_pos << std::endl;
    }
}

void batch_processing_example() {
    std::cout << "\n=== Batch Processing Example ===" << std::endl;
    
    HPCRegex regex;
    
    std::string pattern = "error|ERROR|Error";
    std::vector<std::string> log_lines = {
        "INFO: Application started successfully",
        "ERROR: Failed to connect to database",
        "WARN: Low memory warning",
        "Error: Invalid configuration file",
        "INFO: Processing completed",
        "ERROR: Timeout occurred"
    };
    
    std::cout << "Searching for pattern: \"" << pattern << "\"" << std::endl;
    std::cout << "In " << log_lines.size() << " log lines:" << std::endl;
    
    std::vector<MatchResult> results = regex.batch_match(pattern, log_lines);
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Line " << (i + 1) << ": ";
        if (results[i].found) {
            std::cout << "✓ ERROR found - \"" << log_lines[i] << "\"" << std::endl;
        } else {
            std::cout << "✗ No error - \"" << log_lines[i] << "\"" << std::endl;
        }
    }
}

void performance_comparison_example() {
    std::cout << "\n=== Performance Comparison Example ===" << std::endl;
    
    // Create a moderately large text
    std::string large_text = utils::generate_test_text(50000, 42);
    std::string pattern = "a*b+c*";
    
    std::cout << "Benchmarking pattern: \"" << pattern << "\"" << std::endl;
    std::cout << "Text size: " << large_text.length() << " characters" << std::endl;
    
    // Test with GPU enabled
    RegexConfig gpu_config;
    gpu_config.use_gpu = true;
    HPCRegex gpu_regex(gpu_config);
    
    // Test with CPU only
    RegexConfig cpu_config;
    cpu_config.use_gpu = false;
    HPCRegex cpu_regex(cpu_config);
    
    // Run benchmarks
    std::cout << "\nRunning performance comparison (50 iterations)..." << std::endl;
    
    BenchmarkResults gpu_results = gpu_regex.benchmark(pattern, large_text, 50);
    std::cout << "\nGPU Results:" << std::endl;
    gpu_results.print_summary();
    
    BenchmarkResults cpu_results = cpu_regex.benchmark(pattern, large_text, 50);
    std::cout << "\nCPU Results:" << std::endl;
    cpu_results.print_summary();
    
    // Compare memory usage
    std::cout << "\nMemory Usage Comparison:" << std::endl;
    std::cout << "GPU Memory: " << gpu_regex.get_memory_usage() / 1024 << " KB" << std::endl;
    std::cout << "CPU Memory: " << cpu_regex.get_memory_usage() / 1024 << " KB" << std::endl;
}

void configuration_example() {
    std::cout << "\n=== Configuration Example ===" << std::endl;
    
    // Create custom configuration
    RegexConfig config;
    config.use_gpu = true;
    config.max_text_length = 2 * 1024 * 1024;  // 2MB
    config.max_pattern_length = 512;
    config.gpu_device_id = 0;
    config.gpu_memory_pool_size = 1024 * 1024 * 1024;  // 1GB
    config.enable_caching = true;
    
    std::cout << "Creating HPCRegex with custom configuration:" << std::endl;
    std::cout << "  GPU enabled: " << (config.use_gpu ? "Yes" : "No") << std::endl;
    std::cout << "  Max text length: " << config.max_text_length / 1024 << " KB" << std::endl;
    std::cout << "  Max pattern length: " << config.max_pattern_length << " chars" << std::endl;
    std::cout << "  GPU memory pool: " << config.gpu_memory_pool_size / (1024*1024) << " MB" << std::endl;
    
    HPCRegex regex(config);
    
    // Test with a larger text
    std::string large_pattern = "\\w+@[\\w.-]+\\.\\w+";
    std::string large_text = utils::generate_test_text(100000, 42);
    large_text += " Contact us at support@example.com or admin@test.org for help.";
    
    auto matches = regex.find_all(large_pattern, large_text);
    std::cout << "\nFound " << matches.size() << " email addresses in " 
              << large_text.length() << " character text." << std::endl;
    
    for (const auto& match : matches) {
        std::cout << "  Email: " << match.matched_text << std::endl;
    }
}

void error_handling_example() {
    std::cout << "\n=== Error Handling Example ===" << std::endl;
    
    try {
        // Test with invalid configuration
        RegexConfig config;
        config.max_text_length = 10;  // Very small limit
        config.max_pattern_length = 5;
        
        HPCRegex regex(config);
        
        // Try to process text that exceeds limits
        std::string long_text = "This text is definitely longer than 10 characters";
        std::string pattern = "longer_pattern_than_5_chars";
        
        std::cout << "Attempting to process text longer than configured limit..." << std::endl;
        MatchResult result = regex.match(pattern, long_text);
        
    } catch (const HPCRegexException& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }
    
    try {
        // Test with invalid pattern
        HPCRegex regex;
        std::string invalid_pattern = "[unclosed bracket";
        std::string text = "test text";
        
        if (!utils::validate_pattern(invalid_pattern)) {
            std::cout << "✓ Pattern validation caught invalid pattern: \"" 
                      << invalid_pattern << "\"" << std::endl;
        }
        
    } catch (const HPCRegexException& e) {
        std::cout << "✓ Caught pattern exception: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::cout << "HPC Regex Library - Basic Usage Examples" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Print device information
        std::cout << "\nAvailable devices:" << std::endl;
        utils::print_device_info();
        
        // Run examples
        basic_matching_example();
        find_all_example();
        batch_processing_example();
        performance_comparison_example();
        configuration_example();
        error_handling_example();
        
        std::cout << "\n=== All Examples Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}