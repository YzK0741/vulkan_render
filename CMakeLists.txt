cmake_minimum_required(VERSION 4.3)
project(vulkan_render)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_FLAGS "-std=c++23")
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenSSL REQUIRED)

set(COMPILE_OPTIONS
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
        >
        -fno-exceptions
        -Wall
        -Wextra
        -Werror
        -Wno-reserved-module-identifier
)

add_library(std_lib STATIC)

target_sources(std_lib
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        std/std.cppm
)

add_library(utility STATIC)

target_sources(utility
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        utility/utility.cppm
        utility/data_block.cppm
        utility/BVH/bvh.cppm
        utility/better_pmr/better_pmr.cppm
        PRIVATE
        utility/utility.cpp
        utility/BVH/bvh.cpp
        utility/better_pmr/better_pmr.cpp
)

add_library(thread_pool STATIC)

target_sources(thread_pool PRIVATE utility/thread_pool/thread_pool.cpp)

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
        vulkan/runtime.cpp
        third_party/spirv-reflect/spirv_reflect.c
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

target_link_libraries(gltf_loader
        PRIVATE
        fastgltf
        std_lib
)

target_link_libraries(vulkan_lib
        Vulkan::Vulkan
        glfw
        utility
        ${CMAKE_DL_LIBS}
)

add_executable(vulkan_render main.cpp)

target_link_libraries(vulkan_lib
        std_lib
)

target_link_libraries(utility
        std_lib
)

target_link_libraries(thread_pool
        std_lib
)

target_link_libraries(${PROJECT_NAME}
        utility
        vulkan_lib
        gltf_loader
        OpenSSL::Crypto
        thread_pool
        std_lib
)

target_compile_options(std_lib PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(vulkan_render PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(vulkan_lib PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(utility PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(thread_pool PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(gltf_loader PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_definitions(vulkan_lib PRIVATE
        $<$<CONFIG:Debug>:
        _DEBUG
        >
)

# ---------------------------------------------------------------------------
# clang-format: run automatically before compiling
# 仅在源码比 stamp 新（源码有改动）时才执行格式化，避免每次构建都重排、
# 导致源码 mtime 变化从而无限触发重编译
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXECUTABLE clang-format)

if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE PROJECT_FORMAT_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/gltf_loader/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.cpp*)

    set(CLANG_FORMAT_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.clang-format.stamp")

    add_custom_command(
            OUTPUT "${CLANG_FORMAT_STAMP}"
            COMMAND ${CLANG_FORMAT_EXECUTABLE} -i --style=file ${PROJECT_FORMAT_SOURCES}
            COMMAND ${CMAKE_COMMAND} -E touch "${CLANG_FORMAT_STAMP}"
            DEPENDS ${PROJECT_FORMAT_SOURCES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Formatting sources with clang-format"
            VERBATIM)

    add_custom_target(clang-format DEPENDS "${CLANG_FORMAT_STAMP}")

    # 所有项目目标在编译前先完成格式化
    foreach(_tgt utility thread_pool vulkan_lib gltf_loader vulkan_render)
        add_dependencies(${_tgt} clang-format)
    endforeach()

    # 只检查不修改（可用于 CI）
    add_custom_target(clang-format-check
            COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run -Werror --style=file ${PROJECT_FORMAT_SOURCES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Checking sources with clang-format (dry-run)"
            VERBATIM)
else()
    message(WARNING "clang-format not found, skipping auto-format before build")
endif()