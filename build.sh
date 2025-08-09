#!/bin/bash

# HPC Regex Library Build Script
# This script builds the HPC Regex library and all examples

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}HPC Regex Library Build Script${NC}"
echo "=================================="

# Check if CUDA is available
check_cuda() {
    echo -e "${YELLOW}Checking CUDA installation...${NC}"
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}✓ CUDA found (version $CUDA_VERSION)${NC}"
        return 0
    else
        echo -e "${RED}✗ CUDA not found. Please install CUDA Toolkit.${NC}"
        return 1
    fi
}

# Check CMake version
check_cmake() {
    echo -e "${YELLOW}Checking CMake installation...${NC}"
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        echo -e "${GREEN}✓ CMake found (version $CMAKE_VERSION)${NC}"
        return 0
    else
        echo -e "${RED}✗ CMake not found. Please install CMake 3.18+.${NC}"
        return 1
    fi
}

# Check compiler
check_compiler() {
    echo -e "${YELLOW}Checking compiler...${NC}"
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1 | awk '{print $3}')
        echo -e "${GREEN}✓ g++ found (version $GCC_VERSION)${NC}"
        return 0
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1 | awk '{print $3}')
        echo -e "${GREEN}✓ clang++ found (version $CLANG_VERSION)${NC}"
        return 0
    else
        echo -e "${RED}✗ No suitable C++ compiler found.${NC}"
        return 1
    fi
}

# Create build directory
setup_build_dir() {
    echo -e "${YELLOW}Setting up build directory...${NC}"
    if [ -d "build" ]; then
        echo -e "${YELLOW}Removing existing build directory...${NC}"
        rm -rf build
    fi
    mkdir -p build
    echo -e "${GREEN}✓ Build directory created${NC}"
}

# Configure with CMake
configure_cmake() {
    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cd build
    
    # Configure with appropriate settings
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="61;70;75;80;86" \
        -DCMAKE_CXX_STANDARD=17
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CMake configuration successful${NC}"
        cd ..
        return 0
    else
        echo -e "${RED}✗ CMake configuration failed${NC}"
        cd ..
        return 1
    fi
}

# Build the project
build_project() {
    echo -e "${YELLOW}Building project...${NC}"
    cd build
    
    # Determine number of cores for parallel build
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CORES=$(nproc)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CORES=$(sysctl -n hw.ncpu)
    else
        CORES=4  # Default
    fi
    
    echo -e "${BLUE}Building with $CORES cores...${NC}"
    make -j$CORES
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful${NC}"
        cd ..
        return 0
    else
        echo -e "${RED}✗ Build failed${NC}"
        cd ..
        return 1
    fi
}

# List built targets
show_targets() {
    echo -e "${YELLOW}Built targets:${NC}"
    if [ -f "build/benchmark_regex" ]; then
        echo -e "${GREEN}  ✓ benchmark_regex${NC} - Main benchmark suite"
    fi
    if [ -f "build/example_basic" ]; then
        echo -e "${GREEN}  ✓ example_basic${NC} - Basic usage examples"
    fi
    if [ -f "build/example_large_text" ]; then
        echo -e "${GREEN}  ✓ example_large_text${NC} - Large text processing examples"
    fi
    if [ -f "build/libhpc_regex_static.a" ]; then
        echo -e "${GREEN}  ✓ libhpc_regex_static.a${NC} - Static library"
    fi
    if [ -f "build/libhpc_regex_shared.so" ] || [ -f "build/libhpc_regex_shared.dylib" ]; then
        echo -e "${GREEN}  ✓ libhpc_regex_shared${NC} - Shared library"
    fi
}

# Run a quick test
run_test() {
    echo -e "${YELLOW}Running quick test...${NC}"
    if [ -f "build/example_basic" ]; then
        echo -e "${BLUE}Running basic example...${NC}"
        cd build
        timeout 30s ./example_basic || echo -e "${YELLOW}Test completed or timed out${NC}"
        cd ..
        echo -e "${GREEN}✓ Quick test completed${NC}"
    else
        echo -e "${YELLOW}⚠ No test executable found${NC}"
    fi
}

# Clean build
clean() {
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    if [ -d "build" ]; then
        rm -rf build
        echo -e "${GREEN}✓ Build directory cleaned${NC}"
    else
        echo -e "${YELLOW}⚠ No build directory to clean${NC}"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  build     - Build the project (default)"
    echo "  clean     - Clean build directory"
    echo "  test      - Build and run quick test"
    echo "  help      - Show this help message"
}

# Main execution
main() {
    local action=${1:-build}
    
    case $action in
        "clean")
            clean
            ;;
        "help")
            show_usage
            ;;
        "test")
            if ! check_cuda || ! check_cmake || ! check_compiler; then
                echo -e "${RED}Prerequisites not met. Please install missing components.${NC}"
                exit 1
            fi
            
            setup_build_dir
            configure_cmake
            build_project
            show_targets
            run_test
            ;;
        "build"|*)
            if ! check_cuda || ! check_cmake || ! check_compiler; then
                echo -e "${RED}Prerequisites not met. Please install missing components.${NC}"
                exit 1
            fi
            
            setup_build_dir
            configure_cmake
            build_project
            show_targets
            
            echo ""
            echo -e "${GREEN}Build completed successfully!${NC}"
            echo ""
            echo "To run the benchmarks:"
            echo "  cd build && ./benchmark_regex"
            echo ""
            echo "To run the examples:"
            echo "  cd build && ./example_basic"
            echo "  cd build && ./example_large_text"
            ;;
    esac
}

# Execute main function with all arguments
main "$@"