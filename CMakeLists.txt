cmake_minimum_required(VERSION 4.3)
project(vulkan_render)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_FLAGS "-std=c++23")
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

# GLFW (window + input): linked STATICALLY so the released exe carries no glfw3.dll.
# The MSYS2 clang64 package's own CMake config exports a SHARED import (libglfw3.dll.a),
# so we do not use find_package(glfw3): a plain INTERFACE target linking the bare name
# resolves to the static archive once the executable links with -static (see below), and
# crucially does NOT define GLFW_DLL (the dynamic-import macro static builds must not see).
# Headers live on the compiler's implicit include path; an explicit dir is added when found.
add_library(glfw INTERFACE)
if(WIN32)
    target_link_libraries(glfw INTERFACE glfw3 gdi32) # glfw3.pc: Libs.private = -lgdi32
else()
    find_package(Threads REQUIRED)
    find_library(GLFW_LIBRARY NAMES glfw3 REQUIRED)
    target_link_libraries(glfw INTERFACE ${GLFW_LIBRARY} Threads::Threads)
endif()

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
        utility/frame_clock/frame_clock.cppm
        utility/frame_stats/frame_stats.cppm
        utility/thread_pool/thread_pool.cppm
        PRIVATE
        utility/utility.cpp
        utility/BVH/bvh.cpp
        utility/better_pmr/better_pmr.cpp
        utility/thread_pool/thread_pool.cpp
)

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

# The engine core library (vulkancorekit): Vulkan object management + math + the runtime
# facade + scene tree + the debug GUI + the format-neutral animation controller
# (vulkan.animation: plays keyframe clips / skins / morphs onto the runtime scene; it never
# imports a loader - gltf::scenes satisfies its structural source concept at the call site).
# This library does NOT depend on gltf_loader (pure CPU, used by the demo layer only).
add_library(vulkancorekit STATIC)

target_sources(vulkancorekit
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        vulkan/core/core.cppm
        vulkan/core/filter/filter.cppm
        vulkan/core/vma/vma.cppm
        vulkan/core/vma/handles/vma_handles.cppm
        vulkan/core/handles/handles.cppm
        vulkan/core/init_utils/init_utils.cppm
        vulkan/core/pipeline/pipeline.cppm
        vulkan/math/math.cppm
        vulkan/runtime.cppm
        vulkan/core/pipeline/spirv_parser.cppm
        vulkan/runtime/scene_tree/scene_tree.cppm
        vulkan/render_environment/render_environment.cppm
        vulkan/gui/gui.cppm
        vulkan/animation/animation_controller.cppm
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
        vulkan/animation/animation_controller.cpp
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

# fastgltf + simdjson, vendored under third_party/ (the sources ship in the repo, so no
# network FetchContent is needed and both Windows/MSYS2 and Linux build self-contained).
# fastgltf's headers deliberately do NOT include simdjson (it is hidden inside the .cpp
# files), so only this library's own sources need the simdjson include; consumers of
# gltf_loader only see fastgltf's headers. simdjson is compiled from its amalgamated
# single-header/single-source pair (simdjson.h + simdjson.cpp), and the three fastgltf
# translation units are compiled here as plain C++ (vendored code does not follow the
# project's -Werror set, mirroring the imgui/xxhash policy).
add_library(fastgltf_vendored STATIC
        third_party/fastgltf/src/fastgltf.cpp
        third_party/fastgltf/src/base64.cpp
        third_party/fastgltf/src/io.cpp
        third_party/simdjson/singleheader/simdjson.cpp
)

target_include_directories(fastgltf_vendored
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastgltf/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson/singleheader
)

# stb_image.h (vendored): gltf_loader.cpp decodes embedded/external textures with it
# (#define STB_IMAGE_IMPLEMENTATION + #include <stb/stb_image.h>).
add_library(stb_vendored INTERFACE)

target_include_directories(stb_vendored
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
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

# xxHash (vendored under third_party/xxhash, see the LICENSE there): non-cryptographic
# 64-bit content hashing (XXH3_64bits) used by utility::xxh3_64bits for image/texture
# dedup. Compiled as its own plain-C target, mirroring the imgui policy above (vendored
# code does not follow the project's -Werror set). The utility module links it.
add_library(xxhash STATIC
        third_party/xxhash/xxhash.c
)

target_include_directories(xxhash
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxhash
)

# mimalloc (vendored under third_party/mimalloc, v3.4.4, MIT — see the LICENSE there):
# allocator backing utility.better_pmr, which routes every std::pmr allocation through it
# (utility.cppm -> utility.better_pmr -> mi_aligned_alloc / mi_free_aligned). Compiled as
# its own plain-C static target from the upstream mi_sources set, mirroring the imgui /
# xxhash policy (vendored code does not follow the project's -Werror set; headers are
# SYSTEM so the -Werror consumers that include <mimalloc.h> do not see upstream warnings).
add_library(mimalloc_vendored STATIC
        third_party/mimalloc/src/alloc.c
        third_party/mimalloc/src/alloc-aligned.c
        third_party/mimalloc/src/alloc-posix.c
        third_party/mimalloc/src/arena.c
        third_party/mimalloc/src/arena-meta.c
        third_party/mimalloc/src/bitmap.c
        third_party/mimalloc/src/heap.c
        third_party/mimalloc/src/init.c
        third_party/mimalloc/src/libc.c
        third_party/mimalloc/src/options.c
        third_party/mimalloc/src/os.c
        third_party/mimalloc/src/page.c
        third_party/mimalloc/src/page-map.c
        third_party/mimalloc/src/random.c
        third_party/mimalloc/src/stats.c
        third_party/mimalloc/src/theap.c
        third_party/mimalloc/src/threadlocal.c
        third_party/mimalloc/src/prim/prim.c
)

target_include_directories(mimalloc_vendored
    SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mimalloc/include
)

if(WIN32)
    # upstream links psapi/shell32/user32/advapi32/bcrypt on Windows (see mimalloc's
    # CMakeLists.txt mi_libraries); _WIN32_WINNT=0x600 matches the upstream MINGW define
    target_compile_definitions(mimalloc_vendored PRIVATE _WIN32_WINNT=0x600)
    target_link_libraries(mimalloc_vendored PUBLIC psapi shell32 user32 advapi32 bcrypt)
endif()

target_link_libraries(gltf_loader
        PRIVATE
        fastgltf_vendored
        stb_vendored
        std_lib
        utility
)

# gltf_loader is deliberately absent here: the engine core is loader-agnostic (see the
# library comment above). The executable links gltf_loader itself (main.cpp is its only
# consumer) and passes the parsed gltf::scenes down to the runtime at the call site.
target_link_libraries(vulkancorekit
        Vulkan::Vulkan
        glfw
        utility
        glm::glm
        ${CMAKE_DL_LIBS}
        std_lib
        imgui
)

# Vendored VMA (third_party/vma/vk_mem_alloc.h): vma.cppm includes <vma/vk_mem_alloc.h>,
# resolved against this third_party include root. The header also ships in the Vulkan SDK's
# Include/vma, but Linux distro Vulkan packages do not provide it, so the vendored copy is
# preferred (it precedes Vulkan::Vulkan's include dirs). Marked SYSTEM so upstream VMA's
# warnings are suppressed (the SDK copy is an isystem include for the same reason - VMA code
# trips -Werror on nullability annotations otherwise).
target_include_directories(vulkancorekit
    SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

add_executable(vulkan_render main.cpp)

# Release builds link as a Windows GUI-subsystem executable: no console window appears when
# running. On the MSYS2 clang64 / MinGW-w64 toolchain WIN32_EXECUTABLE maps to -mwindows; the CRT
# still calls main(), only the subsystem changes. Debug keeps the console for terminal log output.
set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE $<$<CONFIG:Release>:ON>)

# utility.cpp implements the xxHash-based hash functions (utility::xxh3_64bits) on top of
# the vendored xxhash library, so the dependency is declared here, on the library that
# actually uses the XXH3_64bits symbol. It propagates to consumers (vulkancorekit / gltf_loader
# / vulkan_render) at link time, so the executable does not need to link xxhash itself.
# better_pmr (utility.better_pmr) routes std::pmr allocations through the vendored mimalloc
# (third_party/mimalloc), so the executable no longer needs MSYS2's libmimalloc.dll.
target_link_libraries(utility
        std_lib
        xxhash
        mimalloc_vendored
)

target_link_libraries(app_config
        utility
        std_lib
)

# Demo bootstrap chores (root-level module, next to main.cpp): main's startup-config analysis
# (analyse_config: config-file + argv merge, shaders/model location) and shader-loading /
# pipeline helpers, kept out of main.cpp. Split like the other modules: chores.cppm
# (export import std pattern) declares, chores.cpp implements. Demo/application layer only -
# these helpers know about app_config (settings), the runtime (vulkan.runtime) and utility
# log/panic, never about glTF.
add_library(chores STATIC)

target_sources(chores
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        chores.cppm

    PRIVATE
        chores.cpp
)

target_link_libraries(chores
        app_config
        vulkancorekit
        utility
        std_lib
)

target_link_libraries(${PROJECT_NAME}
        utility
        vulkancorekit
        chores
        gltf_loader
        app_config
        std_lib
        glm::glm
        imgui
)

# Release on Windows (MSYS2 clang64) links fully static (-static): libc++.a / libc++abi.a /
# libunwind.a and libglfw3.a are resolved by the bare names above, so the shipped exe is a
# single file (only the OS's own DLLs - kernel32, the UCRT, the Vulkan loader - remain
# dynamic). Debug stays dynamic for faster dev iteration; POSIX builds stay dynamic (system
# distro packages are shared objects; static glibc would need extra dev packages).
target_link_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<PLATFORM_ID:Windows>>:-static>
)

target_compile_options(std_lib PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(vulkan_render PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(vulkancorekit PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(utility PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(gltf_loader PRIVATE
        ${COMPILE_OPTIONS}
        # vendored fastgltf headers are upstream code: relax one warning their templates trip
        # under -fno-exceptions (raise() leaves its arguments unused on the abort() path)
        -Wno-unused-parameter
)

target_compile_options(app_config PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_options(chores PRIVATE
        ${COMPILE_OPTIONS}
)

target_compile_definitions(vulkancorekit PRIVATE
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

    # Format all project targets before compiling (chores is deliberately NOT in this list: a
    # root-level module source feeding both the format stamp and a target that depends on other
    # format-listed targets makes ninja see a cycle; format chores.cppm manually with
    # 'clang-format -i chores.cppm' - same style, just not automatic)
    foreach(_tgt utility app_config vulkancorekit gltf_loader vulkan_render)
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