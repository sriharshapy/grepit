#include "regex_engine.h"
#include "hpc_regex.h"
#include "memory_manager.h"
#include <algorithm>
#include <thread>
#include <future>
#include <iostream>
#include <cstring>

namespace hpc_regex {

// Implementation of StateTable
namespace dp {
    StateTable::StateTable(size_t text_len, size_t pattern_len) 
        : rows(text_len + 1), cols(pattern_len + 1) {
        table.resize(rows, std::vector<bool>(cols, false));
    }
    
    void StateTable::reset() {
        for (auto& row : table) {
            std::fill(row.begin(), row.end(), false);
        }
    }
    
    bool& StateTable::at(size_t i, size_t j) {
        return table[i][j];
    }
    
    const bool& StateTable::at(size_t i, size_t j) const {
        return table[i][j];
    }
    
    // Core dynamic programming algorithm for regex matching
    bool match_dp_cpu(const std::string& pattern, const std::string& text, StateTable& table) {
        const size_t m = text.length();
        const size_t n = pattern.length();
        
        if (table.rows != m + 1 || table.cols != n + 1) {
            table = StateTable(m, n);
        } else {
            table.reset();
        }
        
        // Initialize base cases
        table.at(0, 0) = true;  // Empty pattern matches empty text
        
        // Handle patterns like a*, a*b*, etc. that can match empty string
        for (size_t j = 2; j <= n; j += 2) {
            if (pattern[j-1] == '*') {
                table.at(0, j) = table.at(0, j-2);
            }
        }
        
        // Fill the DP table
        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                char text_char = text[i-1];
                char pattern_char = pattern[j-1];
                
                if (pattern_char == '*') {
                    // Kleene star - can match zero or more of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j-2];
                        
                        // Zero occurrences
                        table.at(i, j) = table.at(i, j-2);
                        
                        // One or more occurrences
                        if (prev_char == '.' || prev_char == text_char) {
                            table.at(i, j) = table.at(i, j) || table.at(i-1, j);
                        }
                    }
                } else if (pattern_char == '.' || pattern_char == text_char) {
                    // Direct match or wildcard
                    table.at(i, j) = table.at(i-1, j-1);
                } else if (pattern_char == '+') {
                    // One or more of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j-2];
                        if (prev_char == '.' || prev_char == text_char) {
                            table.at(i, j) = table.at(i-1, j-2) || table.at(i-1, j);
                        }
                    }
                } else if (pattern_char == '?') {
                    // Zero or one of previous character
                    if (j >= 2) {
                        char prev_char = pattern[j-2];
                        // Zero occurrences
                        table.at(i, j) = table.at(i, j-2);
                        // One occurrence
                        if (prev_char == '.' || prev_char == text_char) {
                            table.at(i, j) = table.at(i, j) || table.at(i-1, j-2);
                        }
                    }
                }
                // If no match, table.at(i, j) remains false
            }
        }
        
        return table.at(m, n);
    }
    
    // Find all matches using sliding window approach
    std::vector<MatchResult> find_all_dp_cpu(const std::string& pattern, const std::string& text) {
        std::vector<MatchResult> results;
        const size_t text_len = text.length();
        const size_t pattern_len = pattern.length();
        
        if (text_len == 0 || pattern_len == 0) {
            return results;
        }
        
        StateTable table(text_len, pattern_len);
        
        // For efficiency, we'll search for matches starting at each position
        for (size_t start = 0; start < text_len; ++start) {
            for (size_t end = start + 1; end <= text_len; ++end) {
                std::string substring = text.substr(start, end - start);
                
                if (match_dp_cpu(pattern, substring, table)) {
                    MatchResult result(true, start, end - 1, substring);
                    results.push_back(result);
                    
                    // Avoid overlapping matches for this implementation
                    start = end - 1;
                    break;
                }
            }
        }
        
        return results;
    }
}

// CPURegexEngine Implementation
class CPURegexEngine::Impl {
public:
    Impl() : max_text_length_(1024 * 1024), max_pattern_length_(256), 
             memory_manager_(std::make_unique<CPUMemoryManager>()) {
        // Pre-allocate memory for state table
        state_table_ = std::make_unique<dp::StateTable>(max_text_length_, max_pattern_length_);
    }
    
    ~Impl() = default;
    
    MatchResult match(const std::string& pattern, const std::string& text) {
        if (text.length() > max_text_length_ || pattern.length() > max_pattern_length_) {
            throw HPCRegexException("Text or pattern exceeds maximum length");
        }
        
        if (text.empty() || pattern.empty()) {
            return MatchResult();
        }
        
        // Preprocess pattern if needed
        std::string processed_pattern = preprocess_pattern(pattern);
        
        // Use dynamic programming for matching
        bool found = dp::match_dp_cpu(processed_pattern, text, *state_table_);
        
        if (found) {
            // For exact match, the entire text matches
            return MatchResult(true, 0, text.length() - 1, text);
        }
        
        return MatchResult();
    }
    
    std::vector<MatchResult> find_all(const std::string& pattern, const std::string& text) {
        if (text.length() > max_text_length_ || pattern.length() > max_pattern_length_) {
            throw HPCRegexException("Text or pattern exceeds maximum length");
        }
        
        if (text.empty() || pattern.empty()) {
            return {};
        }
        
        std::string processed_pattern = preprocess_pattern(pattern);
        return dp::find_all_dp_cpu(processed_pattern, text);
    }
    
    void set_max_text_length(size_t length) {
        max_text_length_ = length;
        // Reallocate state table if needed
        if (length > state_table_->rows - 1) {
            state_table_ = std::make_unique<dp::StateTable>(length, max_pattern_length_);
        }
    }
    
    void set_max_pattern_length(size_t length) {
        max_pattern_length_ = length;
        // Reallocate state table if needed
        if (length > state_table_->cols - 1) {
            state_table_ = std::make_unique<dp::StateTable>(max_text_length_, length);
        }
    }
    
    size_t get_memory_usage() const {
        size_t table_memory = state_table_->rows * state_table_->cols * sizeof(bool);
        size_t manager_memory = memory_manager_->get_total_allocated();
        return table_memory + manager_memory;
    }
    
    void cleanup() {
        memory_manager_->cleanup();
        state_table_->reset();
    }

private:
    std::string preprocess_pattern(const std::string& pattern) {
        // Basic pattern preprocessing
        std::string result = pattern;
        
        // Handle escape sequences
        for (size_t i = 0; i < result.length(); ++i) {
            if (result[i] == '\\' && i + 1 < result.length()) {
                char next = result[i + 1];
                switch (next) {
                    case 'd':
                        result.replace(i, 2, "[0-9]");
                        break;
                    case 's':
                        result.replace(i, 2, "[ \t\n\r]");
                        break;
                    case 'w':
                        result.replace(i, 2, "[a-zA-Z0-9_]");
                        break;
                    default:
                        result.erase(i, 1); // Remove escape character
                        break;
                }
            }
        }
        
        return result;
    }
    
    size_t max_text_length_;
    size_t max_pattern_length_;
    std::unique_ptr<dp::StateTable> state_table_;
    std::unique_ptr<CPUMemoryManager> memory_manager_;
};

// CPURegexEngine public interface
CPURegexEngine::CPURegexEngine() : pImpl(std::make_unique<Impl>()) {}

CPURegexEngine::~CPURegexEngine() = default;

MatchResult CPURegexEngine::match(const std::string& pattern, const std::string& text) {
    return pImpl->match(pattern, text);
}

std::vector<MatchResult> CPURegexEngine::find_all(const std::string& pattern, const std::string& text) {
    return pImpl->find_all(pattern, text);
}

void CPURegexEngine::set_max_text_length(size_t length) {
    pImpl->set_max_text_length(length);
}

void CPURegexEngine::set_max_pattern_length(size_t length) {
    pImpl->set_max_pattern_length(length);
}

size_t CPURegexEngine::get_memory_usage() const {
    return pImpl->get_memory_usage();
}

void CPURegexEngine::cleanup() {
    pImpl->cleanup();
}

} // namespace hpc_regex