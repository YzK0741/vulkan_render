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

# Startup configuration: parses config.toml (+ --config / argv merging) into app_settings.
# Uses toml++ (header-only, system package mingw-w64-clang-x86_64-tomlplusplus; its headers
# live under the clang64 include root, no extra include dirs needed). Pure CPU, no Vulkan.
add_library(app_config STATIC)

target_sources(app_config
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        app_config/app_config.cppm

    PRIVATE
        app_config/app_config.cpp
)

add_library(vulkan_lib STATIC)

target_sources(vulkan_lib
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        vulkan/core/core.cppm
        vulkan/core/filter/filter.cppm
        vulkan/core/vma/vma.cppm
        vulkan/core/handles/handles.cppm
        vulkan/core/init_utils/init_utils.cppm
        vulkan/core/pipeline/pipeline.cppm
        vulkan/math/math.cppm
        vulkan/runtime.cppm
        vulkan/core/pipeline/spirv_parser.cppm
        vulkan/runtime/scene_tree/scene_tree.cppm
        vulkan/gui/gui.cppm
    PRIVATE
        vulkan/core/core.cpp
        vulkan/core/filter/filter.cpp
        vulkan/core/vma/vma.cpp
        vulkan/core/handles/handles.cpp
        vulkan/core/init_utils/init_util.cpp
        vulkan/core/pipeline/pipeline.cpp
        vulkan/math/math.cpp
        vulkan/core/pipeline/spirv_parser.cpp
        vulkan/runtime.cpp
        vulkan/runtime/scene_tree/scene_tree.cpp
        vulkan/gui/gui.cpp
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

# Dear ImGui (vendored under third_party/imgui, see the sources there): the Vulkan + GLFW
# backends are compiled as plain C++ (imgui is not a module and does not follow the project's
# -Werror policy — upstream code triggers warnings we do not own). No clang-format: the
# PROJECT_FORMAT_SOURCES glob only covers main/gltf_loader/utility/vulkan, never third_party.
add_library(imgui STATIC)

target_sources(imgui
    PRIVATE
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends
)

target_link_libraries(imgui
    PUBLIC
        Vulkan::Vulkan
        glfw
)

target_link_libraries(gltf_loader
        PRIVATE
        fastgltf
        std_lib
        utility
)

target_link_libraries(vulkan_lib
        Vulkan::Vulkan
        glfw
        utility
        glm::glm
        ${CMAKE_DL_LIBS}
        std_lib
        imgui
)

add_executable(vulkan_render main.cpp)

# Release builds link as a Windows GUI-subsystem executable: no console window appears when
# running. On the MSYS2 clang64 / MinGW-w64 toolchain WIN32_EXECUTABLE maps to -mwindows; the CRT
# still calls main(), only the subsystem changes. Debug keeps the console for terminal log output.
set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE $<$<CONFIG:Release>:ON>)

# utility.cpp implements the hash functions (md5 / sha256 / blake2) on top of OpenSSL, so the
# dependency is declared here, on the library that actually uses the EVP_* symbols. It propagates
# to consumers (vulkan_lib / gltf_loader / vulkan_render) at link time, so the executable no
# longer needs to link OpenSSL itself.
# better_pmr (utility.better_pmr) routes std::pmr allocations through mimalloc;
# the library lives in the clang64 default lib path, so the bare name links it
target_link_libraries(utility
        std_lib
        OpenSSL::Crypto
        mimalloc
)

target_link_libraries(thread_pool
        std_lib
)

target_link_libraries(app_config
        utility
        std_lib
)

target_link_libraries(${PROJECT_NAME}
        utility
        vulkan_lib
        gltf_loader
        app_config
        std_lib
        glm::glm
        imgui
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

target_compile_options(app_config PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_definitions(vulkan_lib PRIVATE
        $<$<CONFIG:Debug>:
        _DEBUG
        >
)

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#clang - format : run automatically before compiling
#Only format when a source is newer than the stamp (i.e. changed), avoiding reformatting on every build,
#which would touch source mtimes and trigger endless recompiles
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
find_program(CLANG_FORMAT_EXECUTABLE clang-format)

if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE PROJECT_FORMAT_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/app_config/*.cpp*
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

    # Format all project targets before compiling
    foreach(_tgt utility thread_pool app_config vulkan_lib gltf_loader vulkan_render)
        add_dependencies(${_tgt} clang-format)
    endforeach()

    # Check only, do not modify (useful for CI)
    add_custom_target(clang-format-check
            COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run -Werror --style=file ${PROJECT_FORMAT_SOURCES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Checking sources with clang-format (dry-run)"
            VERBATIM)
else()
    message(WARNING "clang-format not found, skipping auto-format before build")
endif()