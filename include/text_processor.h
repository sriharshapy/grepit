#pragma once

#include <string>
#include <vector>
#include <memory>

namespace hpc_regex {

// Text preprocessing options
struct PreprocessingOptions {
    bool normalize_whitespace = false;
    bool convert_to_lowercase = false;
    bool remove_special_chars = false;
    bool escape_regex_chars = false;
    std::string encoding = "utf-8";
    
    PreprocessingOptions() = default;
};

// Text chunk for parallel processing
struct TextChunk {
    const char* data;
    size_t size;
    size_t offset;
    size_t chunk_id;
    
    TextChunk(const char* d, size_t s, size_t o, size_t id)
        : data(d), size(s), offset(o), chunk_id(id) {}
};

// Text processor for large text handling
class TextProcessor {
public:
    explicit TextProcessor(size_t chunk_size = 1024 * 1024); // 1MB default
    ~TextProcessor();
    
    // Text preprocessing
    std::string preprocess(const std::string& text, const PreprocessingOptions& options);
    void preprocess_inplace(std::string& text, const PreprocessingOptions& options);
    
    // Text chunking for parallel processing
    std::vector<TextChunk> create_chunks(const std::string& text, size_t overlap = 0);
    std::vector<TextChunk> create_balanced_chunks(const std::string& text, int num_chunks, size_t overlap = 0);
    
    // Memory-mapped file processing
    bool open_file(const std::string& filename);
    void close_file();
    bool is_file_open() const;
    size_t get_file_size() const;
    const char* get_file_data() const;
    
    std::vector<TextChunk> create_file_chunks(size_t overlap = 0);
    
    // Streaming text processing
    class StreamProcessor {
    public:
        explicit StreamProcessor(size_t buffer_size = 64 * 1024);
        ~StreamProcessor();
        
        bool open_stream(const std::string& filename);
        void close_stream();
        
        bool read_next_chunk(std::string& chunk, size_t overlap = 0);
        bool has_more_data() const;
        size_t get_position() const;
        void reset_position();
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
    
    // Configuration
    void set_chunk_size(size_t size);
    size_t get_chunk_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Parallel text processing utilities
namespace parallel {
    // Thread pool for CPU parallel processing
    class ThreadPool {
    public:
        explicit ThreadPool(size_t num_threads = 0); // 0 = auto-detect
        ~ThreadPool();
        
        template<typename Func, typename... Args>
        auto submit(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>;
        
        void wait_all();
        void shutdown();
        
        size_t get_thread_count() const;
        size_t get_queue_size() const;
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
    
    // Parallel text processing functions
    template<typename ProcessFunc>
    std::vector<typename std::result_of<ProcessFunc(const TextChunk&)>::type>
    process_chunks_parallel(const std::vector<TextChunk>& chunks, ProcessFunc func, size_t num_threads = 0);
    
    // Work-stealing scheduler for load balancing
    class WorkStealingScheduler {
    public:
        explicit WorkStealingScheduler(size_t num_workers = 0);
        ~WorkStealingScheduler();
        
        template<typename Task>
        void submit_task(Task&& task);
        
        void wait_completion();
        void shutdown();
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
}

// Text encoding utilities
namespace encoding {
    bool is_valid_utf8(const std::string& text);
    std::string convert_to_utf8(const std::string& text, const std::string& from_encoding);
    std::string normalize_unicode(const std::string& text);
    
    // Character classification
    bool is_ascii(char c);
    bool is_whitespace(char c);
    bool is_regex_metachar(char c);
    
    // Escape/unescape utilities
    std::string escape_regex(const std::string& text);
    std::string unescape_regex(const std::string& text);
}

// Large file utilities
namespace file_utils {
    size_t get_file_size(const std::string& filename);
    bool file_exists(const std::string& filename);
    std::string get_file_extension(const std::string& filename);
    
    // Memory-mapped file wrapper
    class MemoryMappedFile {
    public:
        explicit MemoryMappedFile(const std::string& filename);
        ~MemoryMappedFile();
        
        bool is_open() const;
        const char* data() const;
        size_t size() const;
        
        // Iterator support for chunk processing
        class iterator {
        public:
            iterator(const char* ptr, size_t chunk_size, size_t remaining);
            
            TextChunk operator*() const;
            iterator& operator++();
            bool operator!=(const iterator& other) const;
            
        private:
            const char* ptr_;
            size_t chunk_size_;
            size_t remaining_;
            size_t chunk_id_;
        };
        
        iterator begin(size_t chunk_size = 1024 * 1024) const;
        iterator end() const;
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
}

} // namespace hpc_regex