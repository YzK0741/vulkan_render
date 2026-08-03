cmake_minimum_required(VERSION 4.3)
project(vulkan_render)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(utility STATIC)

target_sources(utility
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        utility/utility.cppm
        utility/thread_pool.cppm
        PRIVATE
        utility/utility.cpp
        utility/thread_pool.cpp
)

add_library(vulkan_lib STATIC)

target_sources(vulkan_lib
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        vulkan/core/core.cppm
        vulkan/core/vma/vma.cppm
        vulkan/core/handles/handles.cppm
        vulkan/core/init_utils/init_utils.cppm
        vulkan/core/pipeline/pipeline.cppm
        vulkan/model/model.cppm
        vulkan/runtime.cppm
        vulkan/core/pipeline/spirv_parser.cppm
    PRIVATE
        vulkan/core/core.cpp
        vulkan/core/vma/vma.cpp
        vulkan/core/handles/handles.cpp
        vulkan/core/init_utils/init_util.cpp
        vulkan/core/pipeline/pipeline.cpp
        vulkan/model/model.cpp
        vulkan/core/pipeline/spirv_parser.cpp
)

add_library(gltf_loader STATIC)

target_sources(gltf_loader
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        gltf_loader/gltf_loader.cppm

    PRIVATE
        gltf_loader/gltf_loader.cpp
)

target_link_libraries(vulkan_lib PRIVATE utility)

add_executable(vulkan_render main.cpp)

target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
        glfw
        vulkan_lib
        utility
        OpenSSL::Crypto
)

target_compile_options(vulkan_render PRIVATE
        $<$<CONFIG:Release>:
        -O3
        -fno-rtti
        -fno-exceptions
        -ffunction-sections
        -fdata-sections
        -flto
        -march=native
        -fno-unwind-tables
        -fno-asynchronous-unwind-tables
        -fmerge-all-constants
        >
        $<$<CONFIG:Debug>:
        -g
        -fno-omit-frame-pointer
        -fstandalone-debug
        >
        -Wall
        -Wextra
        -Werror
)

target_compile_options(vulkan_lib PRIVATE
        $<$<CONFIG:Release>:
        -O3
        -fno-rtti
        -ffunction-sections
        -fdata-sections
        -flto
        -march=native
        -fno-unwind-tables
        -fno-asynchronous-unwind-tables
        -fmerge-all-constants
        >
        $<$<CONFIG:Debug>:
        -g
        >
        -fno-exceptions
        -Wall
        -Wextra
        -Werror
)

target_compile_definitions(vulkan_lib PRIVATE
        $<$<CONFIG:Debug>:
        _DEBUG
        >
)