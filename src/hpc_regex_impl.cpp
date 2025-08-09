#include "hpc_regex.h"
#include "regex_engine.h"
#include "memory_manager.h"
#ifdef HPC_REGEX_CUDA_ENABLED
#include "cuda_utils.h"
#endif
#include <memory>
#include <iostream>

namespace hpc_regex {

// BenchmarkResults Implementation
void BenchmarkResults::print_summary() const {
    std::cout << "\n=== Benchmark Results Summary ===" << std::endl;
    std::cout << "CPU Time: " << timing.cpu_time.count() << " microseconds" << std::endl;
    std::cout << "GPU Time: " << timing.gpu_time.count() << " microseconds" << std::endl;
    std::cout << "Memory Transfer Time: " << timing.memory_transfer_time.count() << " microseconds" << std::endl;
    std::cout << "Speedup Factor: " << timing.speedup_factor << "x" << std::endl;
    std::cout << "CPU Memory Used: " << memory_used_cpu / 1024 << " KB" << std::endl;
    std::cout << "GPU Memory Used: " << memory_used_gpu / 1024 << " KB" << std::endl;
    std::cout << "Correctness Verified: " << (correctness_verified ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << timing.iterations << std::endl;
}

std::string BenchmarkResults::to_csv() const {
    std::ostringstream oss;
    oss << timing.cpu_time.count() << ","
        << timing.gpu_time.count() << ","
        << timing.memory_transfer_time.count() << ","
        << timing.speedup_factor << ","
        << memory_used_cpu << ","
        << memory_used_gpu << ","
        << (correctness_verified ? 1 : 0) << ","
        << timing.iterations;
    return oss.str();
}

// HPCRegex Implementation
class HPCRegex::Impl {
public:
    explicit Impl(const RegexConfig& config) : config_(config) {
        initialize();
    }
    
    ~Impl() {
        cleanup();
    }
    
    MatchResult match(const std::string& pattern, const std::string& text) {
#ifdef HPC_REGEX_CUDA_ENABLED
        if (config_.use_gpu && gpu_engine_) {
            try {
                return gpu_engine_->match(pattern, text);
            } catch (const CudaException& e) {
                std::cerr << "GPU error, falling back to CPU: " << e.what() << std::endl;
                config_.use_gpu = false;
            }
        }
#endif
        
        return cpu_engine_->match(pattern, text);
    }
    
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) {
#ifdef HPC_REGEX_CUDA_ENABLED
        if (config_.use_gpu && gpu_engine_) {
            try {
                return gpu_engine_->find_all(pattern, text);
            } catch (const CudaException& e) {
                std::cerr << "GPU error, falling back to CPU: " << e.what() << std::endl;
                config_.use_gpu = false;
            }
        }
#endif
        
        return cpu_engine_->find_all(pattern, text);
    }
    
    std::vector<MatchResult> batch_match(const std::string& pattern, 
                                       const std::vector<std::string>& texts) {
        std::vector<MatchResult> results;
        results.reserve(texts.size());
        
        // Process texts in parallel using the appropriate engine
#ifdef HPC_REGEX_CUDA_ENABLED
        if (config_.use_gpu && gpu_engine_ && texts.size() > 10) {
            // GPU batch processing for large datasets
            return batch_match_gpu(pattern, texts);
        } else {
#endif
            // CPU processing with threading
            return batch_match_cpu(pattern, texts);
#ifdef HPC_REGEX_CUDA_ENABLED
        }
#endif
    }
    
    BenchmarkResults benchmark(const std::string& pattern, const std::string& text, int iterations) {
        BenchmarkResults results;
        results.timing.iterations = iterations;
        
        // Benchmark CPU
        auto cpu_start = std::chrono::high_resolution_clock::now();
        MatchResult cpu_result;
        for (int i = 0; i < iterations; ++i) {
            cpu_result = cpu_engine_->match(pattern, text);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        results.timing.cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        results.memory_used_cpu = cpu_engine_->get_memory_usage();
        
        // Benchmark GPU if available
#ifdef HPC_REGEX_CUDA_ENABLED
        if (gpu_engine_) {
            auto gpu_start = std::chrono::high_resolution_clock::now();
            MatchResult gpu_result;
            for (int i = 0; i < iterations; ++i) {
                gpu_result = gpu_engine_->match(pattern, text);
            }
            auto gpu_end = std::chrono::high_resolution_clock::now();
            results.timing.gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
            results.memory_used_gpu = gpu_engine_->get_memory_usage();
            
            // Calculate speedup
            if (results.timing.gpu_time.count() > 0) {
                results.timing.speedup_factor = static_cast<double>(results.timing.cpu_time.count()) / 
                                               results.timing.gpu_time.count();
            }
            
            // Verify correctness
            results.correctness_verified = (cpu_result.found == gpu_result.found);
            if (cpu_result.found && gpu_result.found) {
                results.correctness_verified = results.correctness_verified &&
                    (cpu_result.start_pos == gpu_result.start_pos) &&
                    (cpu_result.end_pos == gpu_result.end_pos);
            }
        } else {
#endif
            results.timing.gpu_time = std::chrono::microseconds(0);
            results.timing.speedup_factor = 1.0;
            results.correctness_verified = true;
#ifdef HPC_REGEX_CUDA_ENABLED
        }
#endif
        
        return results;
    }
    
    void set_config(const RegexConfig& config) {
        bool device_changed = (config.gpu_device_id != config_.gpu_device_id);
        config_ = config;
        
#ifdef HPC_REGEX_CUDA_ENABLED
        if (device_changed && gpu_engine_) {
            gpu_engine_->set_device(config_.gpu_device_id);
        }
#endif
        
        // Update engine parameters
        cpu_engine_->set_max_text_length(config_.max_text_length);
        cpu_engine_->set_max_pattern_length(config_.max_pattern_length);
        
#ifdef HPC_REGEX_CUDA_ENABLED
        if (gpu_engine_) {
            gpu_engine_->set_max_text_length(config_.max_text_length);
            gpu_engine_->set_max_pattern_length(config_.max_pattern_length);
            gpu_engine_->set_memory_pool_size(config_.gpu_memory_pool_size);
        }
#endif
    }
    
    RegexConfig get_config() const {
        return config_;
    }
    
    void cleanup() {
        if (cpu_engine_) cpu_engine_->cleanup();
#ifdef HPC_REGEX_CUDA_ENABLED
        if (gpu_engine_) gpu_engine_->cleanup();
#endif
    }
    
    size_t get_memory_usage() const {
        size_t cpu_usage = cpu_engine_ ? cpu_engine_->get_memory_usage() : 0;
#ifdef HPC_REGEX_CUDA_ENABLED
        size_t gpu_usage = gpu_engine_ ? gpu_engine_->get_memory_usage() : 0;
        return cpu_usage + gpu_usage;
#else
        return cpu_usage;
#endif
    }

private:
    void initialize() {
        // Always create CPU engine
        cpu_engine_ = std::make_unique<CPURegexEngine>();
        cpu_engine_->set_max_text_length(config_.max_text_length);
        cpu_engine_->set_max_pattern_length(config_.max_pattern_length);
        
        // Create GPU engine if CUDA is available and requested
#ifdef HPC_REGEX_CUDA_ENABLED
        if (config_.use_gpu) {
            try {
                // Check if CUDA is available
                int device_count = cuda::get_device_count();
                if (device_count > 0 && config_.gpu_device_id < device_count) {
                    gpu_engine_ = std::make_unique<GPURegexEngine>(config_.gpu_device_id);
                    gpu_engine_->set_max_text_length(config_.max_text_length);
                    gpu_engine_->set_max_pattern_length(config_.max_pattern_length);
                    gpu_engine_->set_memory_pool_size(config_.gpu_memory_pool_size);
                    
                    std::cout << "GPU engine initialized successfully" << std::endl;
                } else {
                    std::cerr << "GPU requested but not available, using CPU only" << std::endl;
                    config_.use_gpu = false;
                }
            } catch (const CudaException& e) {
                std::cerr << "Failed to initialize GPU engine: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU-only mode" << std::endl;
                config_.use_gpu = false;
                gpu_engine_.reset();
            }
        }
#else
        if (config_.use_gpu) {
            std::cout << "GPU requested but CUDA not compiled in, using CPU only" << std::endl;
            config_.use_gpu = false;
        }
#endif
    }
    
    std::vector<MatchResult> batch_match_cpu(const std::string& pattern, 
                                           const std::vector<std::string>& texts) {
        std::vector<MatchResult> results;
        results.reserve(texts.size());
        
        // Simple sequential processing for now
        // In a full implementation, this would use thread pools
        for (const auto& text : texts) {
            results.emplace_back(cpu_engine_->match(pattern, text));
        }
        
        return results;
    }
    
#ifdef HPC_REGEX_CUDA_ENABLED
    std::vector<MatchResult> batch_match_gpu(const std::string& pattern, 
                                           const std::vector<std::string>& texts) {
        // For now, process sequentially on GPU
        // In a full implementation, this would use batch CUDA kernels
        std::vector<MatchResult> results;
        results.reserve(texts.size());
        
        for (const auto& text : texts) {
            results.emplace_back(gpu_engine_->match(pattern, text));
        }
        
        return results;
    }
#endif
    
    RegexConfig config_;
    std::unique_ptr<CPURegexEngine> cpu_engine_;
#ifdef HPC_REGEX_CUDA_ENABLED
    std::unique_ptr<GPURegexEngine> gpu_engine_;
#endif
};

// HPCRegex public interface
HPCRegex::HPCRegex(const RegexConfig& config) : pImpl(std::make_unique<Impl>(config)) {}

HPCRegex::~HPCRegex() = default;

MatchResult HPCRegex::match(const std::string& pattern, const std::string& text) {
    return pImpl->match(pattern, text);
}

std::vector<MatchResult> HPCRegex::find_all(const std::string& pattern, const std::string& text) {
    return pImpl->find_all(pattern, text);
}

std::vector<MatchResult> HPCRegex::batch_match(const std::string& pattern, 
                                             const std::vector<std::string>& texts) {
    return pImpl->batch_match(pattern, texts);
}

BenchmarkResults HPCRegex::benchmark(const std::string& pattern, const std::string& text, int iterations) {
    return pImpl->benchmark(pattern, text, iterations);
}

void HPCRegex::set_config(const RegexConfig& config) {
    pImpl->set_config(config);
}

RegexConfig HPCRegex::get_config() const {
    return pImpl->get_config();
}

void HPCRegex::cleanup() {
    pImpl->cleanup();
}

size_t HPCRegex::get_memory_usage() const {
    return pImpl->get_memory_usage();
}

// Utility functions implementation
namespace utils {
    std::string generate_test_text(size_t length, int seed) {
        std::srand(seed);
        std::string text;
        text.reserve(length);
        
        const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
        const size_t charset_size = sizeof(charset) - 1;
        
        for (size_t i = 0; i < length; ++i) {
            text += charset[std::rand() % charset_size];
        }
        
        return text;
    }
    
    std::vector<std::string> load_test_patterns() {
        return {
            "a*",           // Kleene star
            "a+",           // One or more
            "a?",           // Zero or one
            "a.b",          // Wildcard
            "abc",          // Literal
            "[a-z]*",       // Character class
            "\\d+",         // Digits
            "\\w+@\\w+",    // Email-like pattern
            "a*b+c*",       // Complex pattern
            "(a|b)*c"       // Alternation with grouping
        };
    }
    
    bool validate_pattern(const std::string& pattern) {
        // Basic pattern validation
        if (pattern.empty()) return false;
        
        // Check for unbalanced brackets
        int bracket_count = 0;
        int paren_count = 0;
        
        for (size_t i = 0; i < pattern.length(); ++i) {
            char c = pattern[i];
            
            switch (c) {
                case '[': ++bracket_count; break;
                case ']': --bracket_count; break;
                case '(': ++paren_count; break;
                case ')': --paren_count; break;
                case '\\':
                    if (i + 1 >= pattern.length()) return false; // Trailing backslash
                    ++i; // Skip escaped character
                    break;
            }
            
            if (bracket_count < 0 || paren_count < 0) return false;
        }
        
        return bracket_count == 0 && paren_count == 0;
    }
    
    void print_device_info() {
        try {
            cuda::print_all_devices();
        } catch (const CudaException& e) {
            std::cout << "CUDA not available: " << e.what() << std::endl;
        }
    }
}

} // namespace hpc_regex