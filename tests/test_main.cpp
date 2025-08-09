#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <iomanip>

// Test framework utilities
class TestSuite {
public:
    static void run_all_tests();
    static void assert_true(bool condition, const std::string& message);
    static void assert_equals(const std::string& expected, const std::string& actual, const std::string& message);
    static void print_results();
    
private:
    static int total_tests_;
    static int passed_tests_;
    static std::string current_test_;
};

int TestSuite::total_tests_ = 0;
int TestSuite::passed_tests_ = 0;
std::string TestSuite::current_test_ = "";

void TestSuite::assert_true(bool condition, const std::string& message) {
    total_tests_++;
    if (condition) {
        passed_tests_++;
        std::cout << "✓ PASS: " << message << std::endl;
    } else {
        std::cout << "✗ FAIL: " << message << std::endl;
    }
}

void TestSuite::assert_equals(const std::string& expected, const std::string& actual, const std::string& message) {
    total_tests_++;
    if (expected == actual) {
        passed_tests_++;
        std::cout << "✓ PASS: " << message << std::endl;
    } else {
        std::cout << "✗ FAIL: " << message << " (expected: '" << expected << "', got: '" << actual << "')" << std::endl;
    }
}

void TestSuite::print_results() {
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Total tests: " << total_tests_ << std::endl;
    std::cout << "Passed: " << passed_tests_ << std::endl;
    std::cout << "Failed: " << (total_tests_ - passed_tests_) << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
              << (100.0 * passed_tests_ / total_tests_) << "%" << std::endl;
}

// Forward declarations of test functions
void test_cpu_regex();
void test_gpu_regex();
void test_memory_manager();

void TestSuite::run_all_tests() {
    std::cout << "=== HPC Regex Test Suite ===" << std::endl;
    std::cout << "Starting comprehensive testing...\n" << std::endl;
    
    try {
        std::cout << "--- CPU Regex Tests ---" << std::endl;
        test_cpu_regex();
        
        std::cout << "\n--- GPU Regex Tests ---" << std::endl;
        test_gpu_regex();
        
        std::cout << "\n--- Memory Manager Tests ---" << std::endl;
        test_memory_manager();
        
    } catch (const std::exception& e) {
        std::cout << "Exception during testing: " << e.what() << std::endl;
    }
    
    print_results();
}

int main(int argc, char* argv[]) {
    std::string test_type = "all";
    if (argc > 1) {
        test_type = argv[1];
    }
    
    if (test_type == "all") {
        TestSuite::run_all_tests();
    } else if (test_type == "cpu") {
        std::cout << "Running CPU-only tests..." << std::endl;
        test_cpu_regex();
    } else if (test_type == "gpu") {
        std::cout << "Running GPU-only tests..." << std::endl;
        test_gpu_regex();
    } else if (test_type == "memory") {
        std::cout << "Running memory tests..." << std::endl;
        test_memory_manager();
    } else if (test_type == "help") {
        std::cout << "Usage: " << argv[0] << " [test_type]" << std::endl;
        std::cout << "test_type options:" << std::endl;
        std::cout << "  all     - Run all tests (default)" << std::endl;
        std::cout << "  cpu     - Run CPU regex tests only" << std::endl;
        std::cout << "  gpu     - Run GPU regex tests only" << std::endl;
        std::cout << "  memory  - Run memory manager tests only" << std::endl;
        std::cout << "  help    - Show this help message" << std::endl;
        return 0;
    } else {
        std::cout << "Unknown test type: " << test_type << std::endl;
        std::cout << "Use 'help' for available options." << std::endl;
        return 1;
    }
    
    TestSuite::print_results();
    
    return 0;
}
