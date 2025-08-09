#include "hpc_regex.h"
#include "text_processor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

using namespace hpc_regex;

void large_text_benchmark() {
    std::cout << "=== Large Text Processing Benchmark ===" << std::endl;
    
    // Generate different sized texts for testing
    std::vector<size_t> text_sizes = {
        100 * 1024,      // 100KB
        1024 * 1024,     // 1MB  
        5 * 1024 * 1024, // 5MB
        10 * 1024 * 1024 // 10MB
    };
    
    std::vector<std::string> test_patterns = {
        "\\d+",                    // Simple digits
        "\\w+@\\w+\\.\\w+",        // Email pattern
        "[A-Z][a-z]+\\s+[A-Z][a-z]+", // Name pattern
        "(error|ERROR|fail|FAIL)" // Error pattern
    };
    
    RegexConfig config;
    config.use_gpu = true;
    config.max_text_length = 20 * 1024 * 1024; // 20MB
    HPCRegex regex(config);
    
    std::cout << std::setw(10) << "Size" 
              << std::setw(25) << "Pattern"
              << std::setw(12) << "CPU (ms)" 
              << std::setw(12) << "GPU (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Matches" << std::endl;
    std::cout << std::string(79, '-') << std::endl;
    
    for (size_t size : text_sizes) {
        std::string large_text = perf_utils::generate_structured_text(size, test_patterns, 42);
        
        for (const auto& pattern : test_patterns) {
            // Benchmark the pattern matching
            auto start = std::chrono::high_resolution_clock::now();
            BenchmarkResults results = regex.benchmark(pattern, large_text, 5);
            auto end = std::chrono::high_resolution_clock::now();
            
            // Count actual matches
            auto matches = regex.find_all(pattern, large_text);
            
            std::cout << std::setw(10) << (size / 1024) << "K"
                      << std::setw(25) << pattern.substr(0, 24)
                      << std::setw(12) << results.timing.cpu_time.count() / 1000
                      << std::setw(12) << results.timing.gpu_time.count() / 1000
                      << std::setw(10) << std::fixed << std::setprecision(2) 
                      << results.timing.speedup_factor
                      << std::setw(10) << matches.size() << std::endl;
        }
        std::cout << std::endl;
    }
}

void memory_efficiency_test() {
    std::cout << "\n=== Memory Efficiency Test ===" << std::endl;
    
    RegexConfig config;
    config.use_gpu = true;
    config.gpu_memory_pool_size = 512 * 1024 * 1024; // 512MB pool
    HPCRegex regex(config);
    
    // Test memory usage with increasingly large texts
    std::vector<size_t> sizes = {1024, 10*1024, 100*1024, 1024*1024, 5*1024*1024};
    
    std::cout << std::setw(12) << "Text Size" 
              << std::setw(15) << "Memory Used" 
              << std::setw(15) << "Peak Memory"
              << std::setw(12) << "Efficiency" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (size_t size : sizes) {
        std::string text = utils::generate_test_text(size, 42);
        std::string pattern = "a+b*c+";
        
        // Clear any previous allocations
        regex.cleanup();
        
        // Measure memory before
        size_t initial_memory = regex.get_memory_usage();
        
        // Perform matching
        MatchResult result = regex.match(pattern, text);
        
        // Measure memory after
        size_t final_memory = regex.get_memory_usage();
        size_t memory_used = final_memory - initial_memory;
        
        // Calculate efficiency (text size / memory used)
        double efficiency = static_cast<double>(size) / (memory_used + 1);
        
        std::cout << std::setw(12) << (size / 1024) << "KB"
                  << std::setw(15) << (memory_used / 1024) << "KB"
                  << std::setw(15) << (final_memory / 1024) << "KB"
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << efficiency << std::endl;
    }
}

void chunked_processing_example() {
    std::cout << "\n=== Chunked Processing Example ===" << std::endl;
    
    // Simulate processing a very large text by chunking
    const size_t total_size = 50 * 1024 * 1024; // 50MB
    const size_t chunk_size = 1024 * 1024;      // 1MB chunks
    
    std::cout << "Simulating processing of " << (total_size / (1024*1024)) 
              << "MB text in " << (chunk_size / 1024) << "KB chunks" << std::endl;
    
    RegexConfig config;
    config.use_gpu = true;
    config.max_text_length = chunk_size * 2; // Allow for overlap
    HPCRegex regex(config);
    
    std::string pattern = "\\d{3}-\\d{3}-\\d{4}"; // Phone number pattern
    
    // Create text processor for chunking
    TextProcessor processor(chunk_size);
    
    size_t total_matches = 0;
    size_t chunks_processed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate chunk processing
    for (size_t offset = 0; offset < total_size; offset += chunk_size) {
        size_t current_chunk_size = std::min(chunk_size, total_size - offset);
        
        // Generate chunk data (in real usage, this would be read from file/stream)
        std::string chunk_text = perf_utils::generate_random_text(current_chunk_size, "", 
                                                                  static_cast<int>(offset));
        
        // Add some phone numbers to chunks for testing
        if (chunks_processed % 10 == 0) {
            chunk_text += " 555-123-4567 ";
        }
        
        // Process chunk
        auto matches = regex.find_all(pattern, chunk_text);
        total_matches += matches.size();
        chunks_processed++;
        
        // Print progress
        if (chunks_processed % 10 == 0) {
            double progress = 100.0 * offset / total_size;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                      << progress << "% (" << chunks_processed << " chunks, " 
                      << total_matches << " matches)" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nChunked processing complete:" << std::endl;
    std::cout << "  Total chunks: " << chunks_processed << std::endl;
    std::cout << "  Total matches: " << total_matches << std::endl;
    std::cout << "  Processing time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << (total_size / (1024.0 * 1024.0)) / (elapsed.count() / 1000.0) 
              << " MB/s" << std::endl;
}

void parallel_batch_processing() {
    std::cout << "\n=== Parallel Batch Processing Example ===" << std::endl;
    
    // Create multiple texts to process in batch
    std::vector<std::string> texts;
    const size_t num_texts = 100;
    const size_t text_size = 10000;
    
    std::cout << "Generating " << num_texts << " texts of " << text_size 
              << " characters each..." << std::endl;
    
    for (size_t i = 0; i < num_texts; ++i) {
        std::string text = utils::generate_test_text(text_size, static_cast<int>(i));
        // Add some pattern matches
        text += " error_" + std::to_string(i) + " ";
        texts.push_back(text);
    }
    
    RegexConfig config;
    config.use_gpu = true;
    HPCRegex regex(config);
    
    std::string pattern = "error_\\d+";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process batch
    std::vector<MatchResult> results = regex.batch_match(pattern, texts);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Count matches
    size_t total_matches = 0;
    for (const auto& result : results) {
        if (result.found) total_matches++;
    }
    
    std::cout << "Batch processing results:" << std::endl;
    std::cout << "  Texts processed: " << texts.size() << std::endl;
    std::cout << "  Total matches: " << total_matches << std::endl;
    std::cout << "  Processing time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "  Rate: " << std::fixed << std::setprecision(2) 
              << (1000.0 * texts.size() / elapsed.count()) << " texts/second" << std::endl;
    
    // Compare with sequential processing
    std::cout << "\nComparing with sequential processing..." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    size_t sequential_matches = 0;
    for (const auto& text : texts) {
        MatchResult result = regex.match(pattern, text);
        if (result.found) sequential_matches++;
    }
    end = std::chrono::high_resolution_clock::now();
    auto sequential_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  Sequential time: " << sequential_time.count() << " ms" << std::endl;
    std::cout << "  Sequential matches: " << sequential_matches << std::endl;
    std::cout << "  Batch speedup: " << std::fixed << std::setprecision(2) 
              << (static_cast<double>(sequential_time.count()) / elapsed.count()) << "x" << std::endl;
}

void real_world_patterns_test() {
    std::cout << "\n=== Real-World Pattern Performance ===" << std::endl;
    
    // Real-world regex patterns
    std::vector<std::pair<std::string, std::string>> patterns = {
        {"Email", "\\w+([.-]?\\w+)*@\\w+([.-]?\\w+)*(\\.\\w{2,3})+"},
        {"URL", "https?://[\\w.-]+/[\\w./?#@&=+%-]*"},
        {"Phone", "\\(?\\d{3}\\)?[-. ]?\\d{3}[-. ]?\\d{4}"},
        {"IP Address", "\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}"},
        {"Date", "\\d{1,2}/\\d{1,2}/\\d{4}"},
        {"Credit Card", "\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}"},
        {"SSN", "\\d{3}-\\d{2}-\\d{4}"},
        {"Hex Color", "#[0-9a-fA-F]{6}"}
    };
    
    // Generate realistic test text
    std::string test_text = R"(
        Contact Information:
        Email: john.doe@example.com, jane.smith@company.org
        Phone: (555) 123-4567, 555.987.6543
        Website: https://www.example.com/page
        
        Server Logs:
        192.168.1.1 - Access granted
        10.0.0.50 - Connection timeout
        
        Dates: 12/25/2023, 01/01/2024
        Card: 1234-5678-9012-3456
        SSN: 123-45-6789
        Color: #FF5733, #3498DB
    )" + utils::generate_test_text(50000, 42);
    
    RegexConfig config;
    config.use_gpu = true;
    HPCRegex regex(config);
    
    std::cout << std::setw(15) << "Pattern Type" 
              << std::setw(12) << "Matches"
              << std::setw(12) << "CPU (μs)"
              << std::setw(12) << "GPU (μs)"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(61, '-') << std::endl;
    
    for (const auto& [name, pattern] : patterns) {
        BenchmarkResults results = regex.benchmark(pattern, test_text, 10);
        auto matches = regex.find_all(pattern, test_text);
        
        std::cout << std::setw(15) << name
                  << std::setw(12) << matches.size()
                  << std::setw(12) << results.timing.cpu_time.count()
                  << std::setw(12) << results.timing.gpu_time.count()
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << results.timing.speedup_factor << std::endl;
    }
}

int main() {
    try {
        std::cout << "HPC Regex Library - Large Text Processing Examples" << std::endl;
        std::cout << "===================================================" << std::endl;
        
        large_text_benchmark();
        memory_efficiency_test();
        chunked_processing_example();
        parallel_batch_processing();
        real_world_patterns_test();
        
        std::cout << "\n=== Large Text Processing Examples Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}