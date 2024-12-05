cmake_minimum_required(VERSION 3.24)

project(GPUFHE LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

enable_language(CUDA)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Specify the CUDA flags to generate debug symbols
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")  # For host (C++ code)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")  # For CUDA (device) code

    # Optionally disable optimizations for debugging
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")  # Disable optimizations for C++ code
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")  # Disable optimizations for CUDA code
endif()

set(SOURCE_FILES src/bgvfhe_gpu.cu src/sub/poly_eqs.cu src/sub/poly.cu src/sub/cleartext_encoding_cpu.cpp src/sub/cleartext_encoding_gpu.cu src/sub/cypertext_eqs.cu)
add_executable(GPUFHEApp ${SOURCE_FILES})

set_target_properties(GPUFHEApp PROPERTIES CUDA_ARCHITECTURES 80) # 80 = rtx 3070
target_include_directories(GPUFHEApp PRIVATE include/)

# ide kell majd linkelni az új library-ket