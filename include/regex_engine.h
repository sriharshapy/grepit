#pragma once

#include <string>
#include <vector>
#include <memory>

namespace hpc_regex {

// Forward declaration
struct MatchResult;

// Abstract base class for regex engines
class RegexEngine {
public:
    virtual ~RegexEngine() = default;
    
    virtual MatchResult match(const std::string& pattern, const std::string& text) = 0;
    virtual std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) = 0;
    
    virtual void set_max_text_length(size_t length) = 0;
    virtual void set_max_pattern_length(size_t length) = 0;
    
    virtual size_t get_memory_usage() const = 0;
    virtual void cleanup() = 0;
};

// CPU-based regex engine using dynamic programming
class CPURegexEngine : public RegexEngine {
public:
    CPURegexEngine();
    ~CPURegexEngine() override;
    
    MatchResult match(const std::string& pattern, const std::string& text) override;
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) override;
    
    void set_max_text_length(size_t length) override;
    void set_max_pattern_length(size_t length) override;
    
    size_t get_memory_usage() const override;
    void cleanup() override;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// GPU-based regex engine using CUDA
class GPURegexEngine : public RegexEngine {
public:
    explicit GPURegexEngine(int device_id = 0);
    ~GPURegexEngine() override;
    
    MatchResult match(const std::string& pattern, const std::string& text) override;
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) override;
    
    void set_max_text_length(size_t length) override;
    void set_max_pattern_length(size_t length) override;
    
    size_t get_memory_usage() const override;
    void cleanup() override;
    
    // GPU-specific methods
    void set_device(int device_id);
    int get_device() const;
    void set_memory_pool_size(size_t size);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Dynamic Programming Implementation Details
namespace dp {
    // State transition table for regex matching
    struct StateTable {
        std::vector<std::vector<bool>> table;
        size_t rows;
        size_t cols;
        
        StateTable(size_t text_len, size_t pattern_len);
        void reset();
        bool& at(size_t i, size_t j);
        const bool& at(size_t i, size_t j) const;
    };
    
    // CPU implementation of dynamic programming regex
    bool match_dp_cpu(const std::string& pattern, const std::string& text, 
                      StateTable& table);
    
    // Find all matches using sliding window
    std::vector<MatchResult> find_all_dp_cpu(const std::string& pattern, 
                                            const std::string& text);
}

} // namespace hpc_regex