#!/bin/bash

echo "Testing CMake configuration..."

# Create build directory
mkdir -p build
cd build

# Run CMake configuration
echo "Running cmake configuration..."
cmake .. 2>&1

# Check if CMake succeeded
if [ $? -eq 0 ]; then
    echo "✓ CMake configuration successful!"
    
    # Try to build (just configuration check)
    echo "Testing build system generation..."
    make --dry-run > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Build system generation successful!"
    else
        echo "✗ Build system generation failed"
        exit 1
    fi
else
    echo "✗ CMake configuration failed"
    exit 1
fi

echo "Build test completed successfully!"
