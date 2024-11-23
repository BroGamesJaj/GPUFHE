cmake_minimum_required(VERSION 3.24)

project(GPUFHE LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

enable_language(CUDA)

set(SOURCE_FILES src/bgvfhe_gpu.cu)
add_executable(GPUFHEApp ${SOURCE_FILES})

set_target_properties(GPUFHEApp PROPERTIES CUDA_ARCHITECTURES native) # 80 = rtx 3070
target_include_directories(GPUFHEApp PRIVATE include/)

# ide kell majd linkelni az új library-ket