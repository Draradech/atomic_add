cmake_minimum_required(VERSION 3.24)

project(cuda_atomic_add_bench LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cuda-atomic-add-bench main.cu)
target_compile_features(cuda-atomic-add-bench PRIVATE cuda_std_17)
set_target_properties(cuda-atomic-add-bench PROPERTIES
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(cuda-atomic-add-bench PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)
