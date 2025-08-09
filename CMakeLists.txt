cmake_minimum_required(VERSION 3.18)
project(hpc_regex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Try to find CUDA (optional) - use modern CMake CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Set CUDA architecture
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    add_definitions(-DHPC_REGEX_CUDA_ENABLED)
    message(STATUS "CUDA found - GPU acceleration enabled")
    set(CUDA_FOUND TRUE)
else()
    message(STATUS "CUDA not found - building CPU-only version")
    set(CUDA_FOUND FALSE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

# Include directories
include_directories(include)

# Source files
set(CPU_SOURCES
    src/regex_engine_cpu.cpp
    src/memory_manager.cpp
    src/benchmarking.cpp
    src/hpc_regex_impl.cpp
)

# Add CUDA sources only if CUDA is available
set(ALL_SOURCES ${CPU_SOURCES})
if(CUDA_FOUND)
    set(CUDA_SOURCES
        src/regex_engine_gpu.cu
        src/cuda_kernels.cu
        src/gpu_memory_manager.cu
        src/cuda_utils.cu
    )
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

set(HEADER_FILES
    include/hpc_regex.h
    include/regex_engine.h
    include/memory_manager.h
    include/benchmarking.h
    include/text_processor.h
    include/cuda_utils.h
)

# Create static library
add_library(hpc_regex_static STATIC ${ALL_SOURCES})
target_link_libraries(hpc_regex_static Threads::Threads)
if(CUDA_FOUND)
    set_property(TARGET hpc_regex_static PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Create shared library
add_library(hpc_regex_shared SHARED ${ALL_SOURCES})
target_link_libraries(hpc_regex_shared Threads::Threads)
if(CUDA_FOUND)
    set_property(TARGET hpc_regex_shared PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Benchmark executable
add_executable(benchmark_regex
    benchmark/main.cpp
    benchmark/performance_tests.cpp
)
target_link_libraries(benchmark_regex hpc_regex_static)

# Example programs
add_executable(example_basic examples/basic_usage.cpp)
target_link_libraries(example_basic hpc_regex_static)

add_executable(example_large_text examples/large_text_processing.cpp)
target_link_libraries(example_large_text hpc_regex_static)

# Test suite
add_executable(test_suite
    tests/test_main.cpp
    tests/test_cpu_regex.cpp
    tests/test_gpu_regex.cpp
    tests/test_memory_manager.cpp
)
target_link_libraries(test_suite hpc_regex_static)

# Install targets
install(TARGETS hpc_regex_static hpc_regex_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${HEADER_FILES} DESTINATION include/hpc_regex)