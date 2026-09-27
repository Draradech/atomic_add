cmake_minimum_required(VERSION 3.24)

project(vulkan_atomic_add_single LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc)
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator)
if(NOT GLSLC_EXECUTABLE AND NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "Install glslc or glslangValidator to compile the compute shader")
endif()

set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/atomic_add.comp")
set(SHADER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/atomic_add.comp.spv")
if(GLSLANG_VALIDATOR_EXECUTABLE)
    add_custom_command(
        OUTPUT "${SHADER_BINARY}"
        COMMAND "${GLSLANG_VALIDATOR_EXECUTABLE}" -V -o "${SHADER_BINARY}" -g "${SHADER_SOURCE}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM
    )
else()
    add_custom_command(
        OUTPUT "${SHADER_BINARY}"
        COMMAND "${GLSLC_EXECUTABLE}" -o "${SHADER_BINARY}" "${SHADER_SOURCE}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM
    )
endif()

add_custom_target(vulkan-atomic-single-shader DEPENDS "${SHADER_BINARY}")
add_executable(vulkan-atomic-add-single main.cpp)
add_dependencies(vulkan-atomic-add-single vulkan-atomic-single-shader)
target_compile_features(vulkan-atomic-add-single PRIVATE cxx_std_17)
target_compile_definitions(vulkan-atomic-add-single PRIVATE ATOMIC_SHADER_PATH="${SHADER_BINARY}")
target_link_libraries(vulkan-atomic-add-single PRIVATE Vulkan::Vulkan)
