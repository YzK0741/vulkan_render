cmake_minimum_required(VERSION 4.3)
project(vulkan_render VERSION 0.3.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
# NOTE: do NOT add "-std=c++23" to CMAKE_CXX_FLAGS here - CMAKE_CXX_STANDARD already injects it,
# and doing both duplicates the flag on every TU (same for -O3 / -g, which the default per-config
# flags already provide - see COMPILE_OPTIONS below).
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

# Project-wide compiler flags, applied to every first-party C++ target below via a single
# foreach (see "compile options" further down). Per-config additions ride on the DEFAULT
# per-config flags (Release already adds -O3 -DNDEBUG, Debug -g), so only the extras that the
# defaults do not cover are listed here - keeping the flags duplicated here AND in the defaults
# would put -O3 / -g on the command line twice.
# Tuning target for Release. -march=native is right for a build on your own machine and wrong for an
# artifact someone else downloads: a binary tuned for the building CPU can die with SIGILL on an older
# one, which is the opposite of what the single-file Release build is for. CI publishes the executable
# (see .github/workflows/ci.yml), so it configures with -DVR_NATIVE_ARCH=OFF to get a baseline.
option(VR_NATIVE_ARCH "Tune Release builds for the building CPU (-march=native)" ON)
set(VR_RELEASE_MARCH "$<IF:$<BOOL:${VR_NATIVE_ARCH}>,-march=native,-march=x86-64-v3>")

set(COMPILE_OPTIONS
        $<$<CONFIG:Release>:
        -fno-rtti
        -ffunction-sections
        -fdata-sections
        -flto
        ${VR_RELEASE_MARCH}
        -fno-unwind-tables
        -fno-asynchronous-unwind-tables
        -fmerge-all-constants
        >
        $<$<CONFIG:Debug>:
        -fno-omit-frame-pointer
        >
        -fno-exceptions
        -Wall
        -Wextra
        -Werror
        -Wno-reserved-module-identifier
)

add_library(vstd_lib STATIC)

# The project's STL module is `vstd` (vstd/vstd.cppm): modified from libc++'s
# (LLVM) generated std-module output and trimmed to the headers the project
# uses (see vstd/README.md).
target_sources(vstd_lib
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        vstd/vstd.cppm
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
)

# Startup configuration: parses config.toml (+ --config / argv merging) into app_settings.
# Uses toml++ (header-only, system package mingw-w64-clang-x86_64-tomlplusplus; its headers
# live under the clang64 include root, no extra include dirs needed). Pure CPU, no Vulkan.
add_library(app_config STATIC)

target_sources(app_config
    PUBLIC
        FILE_SET CXX_MODULES
        FILES
        application_configuration/application_configuration.cppm

    PRIVATE
        application_configuration/application_configuration.cpp
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
    vulkan/core/core.declarations.cppm
    vulkan/core/core.constructor.cppm
        vulkan/core/filter/filters.cppm
        vulkan/core/vma/vma.cppm
        vulkan/core/descriptor_heap/descriptor_heap.cppm
        vulkan/core/vma/handles/vma_handles.cppm
        vulkan/core/handles/handles.cppm
        vulkan/core/init_utils/init_utils.cppm
        vulkan/init_utils/init_utils.cppm
        vulkan/core/pipeline/pipeline.cppm
        vulkan/math/math.cppm
        vulkan/constant_init/constant_init.cppm
        vulkan/runtime/runtime.cppm
    vulkan/runtime/runtime.declarations.cppm
    vulkan/runtime/runtime.constructor.cppm
    vulkan/runtime/runtime.probes.cppm
    vulkan/runtime/runtime.frames.cppm
    vulkan/runtime/runtime.readback.cppm
        vulkan/bindings/bindings.cppm
        vulkan/pipelines/pipelines.cppm
        vulkan/render_resource/render_resource.cppm
        vulkan/render_resource/shared.cppm
        vulkan/frame_constants/frame_constants.cppm
        vulkan/pass/pass.cppm
        vulkan/pass/taa.cppm
        vulkan/pass/scene.cppm
        vulkan/pass/transparent.cppm
        vulkan/pass/megalights_trace.cppm
        vulkan/pass/megalights_temporal.cppm
        vulkan/pass/ray_traced_shadow.cppm
        vulkan/pass/mask_bake.cppm
        vulkan/pass/compute_skin.cppm
        vulkan/pass/cluster.cppm
        vulkan/pass/deferred.cppm
        vulkan/pass/post.cppm
        vulkan/pass/fxaa.cppm
        vulkan/pass/geometry_buffer_debug.cppm
        vulkan/pass/shadow.cppm
        vulkan/pass/chain.cppm
        vulkan/render_start_demo/render_start_demo.cppm
        vulkan/profiling/profiling.cppm
        vulkan/shadow_fit/shadow_fit.cppm
        vulkan/readback/readback.cppm
        vulkan/acceleration_structure/acceleration_structure.cppm
        vulkan/ray_tracing/ray_tracing.cppm
        vulkan/core/pipeline/spirv_parser.cppm
        vulkan/scene_tree/scene_tree.cppm
        vulkan/meshlet/meshlet.cppm
        vulkan/primitive/primitive.cppm
        vulkan/render_environment/render_environment.cppm
        vulkan/graphical_user_interface/graphical_user_interface.cppm
        vulkan/animation/controller.cppm
    PRIVATE
        vulkan/core/core.cpp
        vulkan/core/filter/filters.cpp
        vulkan/core/handles/handles.cpp
        vulkan/init_utils/init_utils.cpp
        vulkan/core/pipeline/pipeline.cpp
        vulkan/math/math.cpp
        vulkan/runtime/runtime.cpp
        vulkan/pass/taa.cpp
        vulkan/pass/scene.cpp
        vulkan/pass/transparent.cpp
        vulkan/pass/megalights_trace.cpp
        vulkan/pass/megalights_temporal.cpp
        vulkan/pass/ray_traced_shadow.cpp
        vulkan/pass/mask_bake.cpp
        vulkan/pass/compute_skin.cpp
        vulkan/pass/cluster.cpp
        vulkan/pass/deferred.cpp
        vulkan/pass/post.cpp
        vulkan/pass/fxaa.cpp
        vulkan/pass/geometry_buffer_debug.cpp
        vulkan/pass/shadow.cpp
        vulkan/render_start_demo/render_start_demo.cpp
        vulkan/shadow_fit/shadow_fit.cpp
        vulkan/readback/readback.cpp
        vulkan/acceleration_structure/acceleration_structure.cpp
        vulkan/ray_tracing/ray_tracing.cpp
        vulkan/scene_tree/scene_tree.cpp
        vulkan/primitive/primitive.cpp
        vulkan/graphical_user_interface/graphical_user_interface.cpp
        vulkan/animation/controller.cpp
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
# content hashing (XXH3_128bits) used by utility::xxh3_128bits for image/texture dedup.
# Compiled as its own plain-C target, mirroring the imgui policy above (vendored
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
        vstd_lib
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
        vstd_lib
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

# utility.cpp implements the xxHash-based hash functions (utility::xxh3_128bits) on top of
# the vendored xxhash library, so the dependency is declared here, on the library that
# actually uses the XXH3_128bits symbol. It propagates to consumers (vulkancorekit / gltf_loader
# / vulkan_render) at link time, so the executable does not need to link xxhash itself.
# better_pmr (utility.better_pmr) routes std::pmr allocations through the vendored mimalloc
# (third_party/mimalloc), so the executable no longer needs MSYS2's libmimalloc.dll.
target_link_libraries(utility
        vstd_lib
        xxhash
        mimalloc_vendored
)
# The file dialog (utility/platform_dialog.cpp, for the config's `model = "ask"`) is GetOpenFileNameW,
# which lives in the common dialog library. Windows-only, like the backend itself.
if(WIN32)
    target_link_libraries(utility comdlg32)
endif()

target_link_libraries(app_config
        utility
        vstd_lib
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
        vstd_lib
)

target_link_libraries(${PROJECT_NAME}
        utility
        vulkancorekit
        chores
        gltf_loader
        app_config
        vstd_lib
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
        # drop unused function/data sections at link (first-party code is compiled with
        # -ffunction-sections -fdata-sections; see the vendored-lib gc test below)
        $<$<CONFIG:Release>:-Wl,--gc-sections>
)

# Release exe: strip all symbols at link time (smaller single-file build; llvm-strip saves
# ~0.8 MB here). Debug keeps its symbols for debugging. Crash reports on a shipped build are
# symbol-less by design - keep the unstripped exe around if you need to map addresses.
target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:-s>
)

# ---------------------------------------------------------------------------------------------
# Headless unit tests (no GPU, no external test framework - see tests/vk_test.h): each test is
# an ordinary module consumer that imports the module under test and links its library, so the
# tests compile with the exact interfaces the rest of the code uses. CTest runs them in a
# per-build scratch directory (module logging writes debug.log there, not into the source
# tree); tests that open fixtures/models receive the source root via VR_TEST_SOURCE_DIR.
# ---------------------------------------------------------------------------------------------
option(VR_BUILD_TESTS "Build the headless unit tests" ON)

if(VR_BUILD_TESTS)
    enable_testing()

    set(VR_TEST_TARGETS test_app_config test_utility test_math test_animation test_gltf_loader test_shadow_fit test_render_resources test_pass test_shader_sources test_meshlet test_docs)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-run")

    foreach(test_target IN ITEMS ${VR_TEST_TARGETS})
        add_executable(${test_target} tests/${test_target}.cpp)
        add_test(NAME ${test_target} COMMAND $<TARGET_FILE:${test_target}>)
        set_tests_properties(${test_target} PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-run"
                TIMEOUT 120)
    endforeach()

    # source root for fixture/model paths (tests no longer assume the repo root is the cwd)
    target_compile_definitions(test_app_config PRIVATE VR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(test_gltf_loader PRIVATE VR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # test_render_resources reads shaders/heap_slots.glsl and vulkan/core/core.declarations.cppm to compare the slot grid's two
    # sources of truth (see the last block of that test): both are text, neither is generated from the other.
    target_compile_definitions(test_render_resources PRIVATE VR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # test_shader_sources parses the FOUR places a .spv is named (VR_SLANG_SOURCES, the two compile scripts, and
    # chores.cpp's load_shader calls) and compares them: one shader built by the build and not by the scripts is
    # the drift that test exists for, and it is the mesh path's CI gate (see the test's header).
    target_compile_definitions(test_shader_sources PRIVATE VR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # test_docs parses the WRITTEN artifacts the manual and CI are built from - Doxyfile's INPUT, the
    # documents themselves, CMakeLists' VR_TEST_TARGETS and the workflow that mirrors it - so it needs
    # the source root and no library at all (see the test's header for the invariants).
    target_compile_definitions(test_docs PRIVATE VR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # vulkan.meshlet is pure CPU alongside vulkan.math: the split's coverage/bounds/determinism checks need no
    # device, which is the only way they can run in CI (the capture gate's references are machine-specific).
    target_link_libraries(test_meshlet
            vulkancorekit
            vstd_lib
    )

    target_link_libraries(test_app_config
            app_config
            vstd_lib
    )
    target_link_libraries(test_utility
            utility
            glm::glm
            vstd_lib
    )
    # vulkan.math lives in vulkancorekit; the module itself is pure CPU (see math.cppm), the
    # link chain just rides the library's PUBLIC dependencies (Vulkan/glfw are not pulled in
    # unless the math code actually references them).
    target_link_libraries(test_math
            vulkancorekit
            vstd_lib
    )
    # vulkan.render_resource is pure CPU on purpose (its own enums, no Vulkan type): the invariants it exists
    # for - usage is a subset of the schema, kind and access fit, a pass's own bindings are contiguous, a pool
    # count is derivable - are checkable WITHOUT a device, which is the only way they can be checked in CI.
    target_link_libraries(test_render_resources
            vulkancorekit
            vstd_lib
    )
    # vulkan.pass is the framework: its base class and behaviour kinds name Vulkan handle types but never call
    # Vulkan, so its contract - resolve, apply the behaviour, record, mark, recreate - is testable with a fake
    # host and no device at all.
    target_link_libraries(test_pass
            vulkancorekit
            vstd_lib
    )
    # vulkan.animation is format-neutral / runtime-agnostic; the sampling tests only touch the
    # pure functions, so nothing Vulkan is pulled in.
    target_link_libraries(test_animation
            vulkancorekit
            glm::glm
            vstd_lib
    )
    # gltf_loader is pure CPU (fastgltf + stb vendored, linked by the library itself); the
    # test loads the repo's gltf_model/DamagedHelmet sample from the working directory.
    target_link_libraries(test_gltf_loader
            gltf_loader
            glm::glm
            vstd_lib
    )
    # vulkan.shadow_fit is pure CPU arithmetic (no Vulkan object is touched), and the module lives in
    # vulkancorekit next to vulkan.math; the link chain brings Vulkan/glfw in, which the tests do not
    # use. This is the first test of the shadow system, which had none: the fit had no device
    # dependency, it just lived where nothing could reach it.
    target_link_libraries(test_shadow_fit
            vulkancorekit
            glm::glm
            vstd_lib
    )
endif()

# ---- compile options: one shared flag set on every first-party C++ target (vendored code is
# deliberately excluded - imgui / fastgltf / simdjson / xxhash / mimalloc do not follow the
# project's -Werror set, see their own targets above).
foreach(compile_target IN ITEMS vstd_lib vulkan_render vulkancorekit utility gltf_loader app_config chores ${VR_TEST_TARGETS})
    target_compile_options(${compile_target} PRIVATE ${COMPILE_OPTIONS})
endforeach()
# vendored fastgltf headers are upstream code: relax one warning their templates trip under
# -fno-exceptions (raise() leaves its arguments unused on the abort() path)
target_compile_options(gltf_loader PRIVATE -Wno-unused-parameter)

# toml++ is a SYSTEM header-only package, so its code is compiled into OUR translation unit and a
# sanitizer sees it as ours. app_config is the only TU that includes it (it is the only toml++ user
# in the project), which is what makes a per-target suppression the honest scope rather than a
# project-wide one.
#
# The finding it silences: toml++'s formatter computes a node's printed width with
# `static_cast<size_t>(log10(val))`, and for a float in (0, 1) - e.g. the DEFAULT
# `clear_color = [0.02, 0.02, 0.03]` - log10 is negative, so the cast is UB
# ("-1.69897 is outside the range of representable values of type 'unsigned long long'",
# toml++/impl/toml_formatter.inl:94). It is reachable from application_configuration.cpp:29 alone, which dumps every
# key of the loaded file into the log. Upstream's bug, not this project's - and it fires from a
# default config value, so it is not a corner case.
#
# Only float-cast-overflow is dropped: every other UBSan check, and all of ASan, stay active for this
# target, so a real float->int bug elsewhere here is still reported. A __attribute__((no_sanitize))
# on the dump function does NOT work for this (measured): UBSan attributes the diagnostic to the
# template instantiation, not to the function it was instantiated from.
target_compile_options(app_config PRIVATE -fno-sanitize=float-cast-overflow)

# Vendored libs get function/data sections AND -flto in Release too (plain C/C++ flags, no
# -Werror set involved): without sections --gc-sections can only drop whole objects, so unused
# imgui / fastgltf / xxhash / mimalloc functions would survive into the exe; without -flto the
# linker cannot see into their translation units, so cross-TU dead code stays. Both cost build
# time only - runtime speed is unaffected (LTO is at worst neutral).
foreach(vendored_target IN ITEMS imgui fastgltf_vendored xxhash mimalloc_vendored)
    target_compile_options(${vendored_target} PRIVATE
            $<$<CONFIG:Release>:-flto;-ffunction-sections;-fdata-sections>
    )
endforeach()

target_compile_definitions(vulkancorekit PRIVATE
        $<$<CONFIG:Debug>:
        _DEBUG
        >
)

# -- version: single source of truth is project(VERSION) above. Injected into the C++ targets
#    that surface it: the Vulkan app_info (vulkan/core/core.cpp, applicationVersion /
#    engineVersion), the --version flag and the startup log banner (main.cpp). Bump the number
#    in project(VERSION) and keep README/docs' "current version" lines in sync.
foreach(version_target IN ITEMS vulkancorekit vulkan_render)
    target_compile_definitions(${version_target} PRIVATE
            VULKAN_RENDER_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            VULKAN_RENDER_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            VULKAN_RENDER_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    )
endforeach()

# ---------------------------------------------------------------------------------------------
# clang-format: run automatically before compiling. Only formats when a source is newer than
# the stamp (i.e. changed), avoiding reformatting on every build, which would touch source
# mtimes and trigger endless recompiles.
# ---------------------------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXECUTABLE clang-format)

if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE PROJECT_FORMAT_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/application_configuration/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/gltf_loader/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.cpp*
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp*)

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

# The platform halves of utility live in plain (non-module) TUs so that <windows.h> stays out of a
# module's global module fragment - including it there made libc++'s operator new ambiguous in every
# TU that imported `utility`. One file per platform facility; the module side only sees their C
# entry points (see utility/platform_sleep.cpp, platform_path.cpp and platform_dialog.cpp).
target_sources(utility PRIVATE
        utility/platform_sleep.cpp
        utility/platform_path.cpp
        utility/platform_dialog.cpp
)
# ---- SLANG -> SPIR-V: the build owns the step (docs/slang_migration.md) ----
# Editing a .glsl without recompiling used to leave you running stale SPIR-V ("my shader change does
# nothing"), so the compiled binaries are no longer committed and are never hand-synced: every build
# regenerates them from the sources (and the shared includes they pull in). SLANGC IS REQUIRED rather
# than optional - a configuration without the compiler is exactly the one where a stale binary could
# be used without anyone noticing, which is the failure this removes.
#
# Output goes to the BUILD tree (<build>/shaders), not next to the .glsl:
#  - the source tree keeps exactly the files a human edits, so .spv is never something to commit,
#    gitignore by hand, or forget to regenerate;
#  - the runtime finds it without any search-order guesswork, because that directory is a sibling of
#    the executable and locate_shaders_dir() prefers a sibling of the running binary;
#  - -I still points at the SOURCE directory, so #include "surface.glsl" resolves against the real
#    sources rather than against copies.
# A mirror/ POST_BUILD step was tried first and is worse: a POST_BUILD command only runs when the
# executable is RELINKED, so a shader-only edit recompiled the .spv and left the copy stale (measured).
# Note this assumes a single-configuration generator, which is what this project configures - the
# executable lands in the build directory, so build/shaders is its sibling. A multi-config generator
# would put the binary in a per-config subdirectory and the runtime would fall back to its cwd walk.
#
# shaders/compile_shaders.ps1 / .sh remain as a manual escape hatch for a machine without CMake; they
# compile in place, which is what a reader of the source tree would expect from a manual tool.
set(VR_SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(VR_SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
find_program(VR_SLANGC_EXECUTABLE NAMES slangc slangc.exe)
if(NOT VR_SLANGC_EXECUTABLE)
    message(FATAL_ERROR
        "slangc not found. It ships with a Slang release (the Vulkan SDK's Bin directory carries one too), "
        "and every shader in this project is built from a .slang source - the GLSL FALLBACK THIS RULE USED "
        "TO HAVE IS GONE, along with the GLSL stage sources themselves (shaders/glsl.old/) - so a build "
        "without it would be the one that silently compiles modules the project no longer uses. Install "
        "Slang or the Vulkan SDK, or add slangc to PATH.")
endif()
# the shared includes: every generated shader lists them all as dependencies. The chain is three
# files and glslc takes milliseconds, so recompiling an unaffected shader is cheaper than the
# dependency bookkeeping that would avoid it.
set(VR_SHADER_INCLUDES
    "${VR_SHADER_SRC_DIR}/surface.glsl"
    "${VR_SHADER_SRC_DIR}/shading.glsl"
    "${VR_SHADER_SRC_DIR}/sky.glsl"
    # ibl_specular.glsl is included by shading.glsl, so it belongs here for the same reason the three above
    # do: editing it alone must not leave stale SPIR-V behind.
    "${VR_SHADER_SRC_DIR}/ibl_specular.glsl"
    # heap_slot_constants.glsl is the SLOT GRID's shader-side source of truth (core.declarations.cppm is the
    # host's): a shader that includes it and is not rebuilt after it changes would index the heap by stale
    # numbers, which validation cannot see. The C++ side is compared against it by a test (see
    # test_render_resources). It is its own file rather than part of heap_slots.glsl because the Slang
    # migration needs the same numbers - see docs/slang_migration.md - and heap_slots.glsl (the GLSL heap
    # arrays and the fetch helper) includes it.
    "${VR_SHADER_SRC_DIR}/heap_slot_constants.glsl"
    "${VR_SHADER_SRC_DIR}/heap_slots.glsl"
    # heap_access.slang and mesh_geometry.slang are the two OTHER includes every leaf pulls in - the shim's heap
    # accessors, and the device-address geometry fetch a mesh stage uses instead of the input assembler. Both were
    # missing from this list, and the mesh one cost a measurement to find: the build tree's pbr.vert.spv and
    # shadow.frag.spv were STALE (built before the geometry lanes existed) while the mesh entries sharing their
    # source had been rebuilt, so the two halves of one pipeline declared different blocks - visible only as a
    # byte difference against the escape-hatch scripts' output, never as a wrong picture.
    # `test_shader_sources` now compares this list against every #include in shaders/, so the next include cannot
    # be forgotten the same way.
    "${VR_SHADER_SRC_DIR}/heap_access.slang"
    "${VR_SHADER_SRC_DIR}/mesh_geometry.slang"
)
# (THE GLSL STAGE LIST USED TO LIVE HERE: twenty-four names, glslc's input, kept in step with the sources by
# hand. Every one of them is in shaders/glsl.old/ now and built from Slang instead, so the list is gone -
# VR_SLANG_SOURCES below IS the list of what this build compiles.)
set(VR_SHADER_OUTPUTS "")
# ---- SLANG SOURCES: the GLSL -> Slang migration, one shader at a time (docs/slang_migration.md) --------
#
# Each entry is "<source>:<entry>:<stage>:<output .spv name>", and the OUTPUT NAME IS THE SAME .spv THE GLSL
# VERSION PRODUCED: the runtime loads shaders by file name, so a port needs no host change at all, and the
# acceptance for every port is the capture gate staying byte-identical.
#
# THE FLAGS ARE THE BINDING CONTRACT, and every one of them was measured rather than guessed:
#   -allow-glsl                 ... the shared slot constants are a plain `const uint` file (valid GLSL);
#                                   Slang rejects a non-static global const without this (E31224).
#   -capability spvDescriptorHeapEXT ... without it Slang emits the descriptor-set model instead of the heap.
#   -spirv-resource-heap-stride 64 / -spirv-sampler-heap-stride 32 ... this renderer's grid is a flat 64-byte
#                                   slot grid plus a 32-byte sampler heap, not Slang's per-type default.
#   -fvk-use-gl-layout          ... std430, which is what the CPU structs and the GLSL blocks already use.
#   -matrix-layout-column-major ... see the note on `row_major` below; kept even though this build appears to
#                                   ignore it (measured: all four flag combinations emit the same decoration).
#
# A MATRIX MEMBER MUST BE DECLARED `row_major` TO GET SPIR-V `ColMajor`, which is what glslang emits for a GLSL
# `mat4` and therefore what matches the host's column-major glm::mat4. MEASURED, and the HLSL keyword reads
# backwards here: `column_major float4x4` emits RowMajor, `row_major float4x4` emits ColMajor.
set(VR_SLANG_SOURCES
    unlit.slang:main:fragment:unlit.frag.spv
    fxaa.slang:main:fragment:fxaa.frag.spv
    post.slang:main:vertex:post.vert.spv
    # post.frag is the post chain's stage for ALL THREE modes (bright pass / downsample / composite), so every
    # scenario records it - the widest verification a single port can have.
    post.slang:frag_main:fragment:post.frag.spv
    gbuffer_debug.slang:main:fragment:gbuffer_debug.frag.spv
    taa.slang:main:fragment:taa.frag.spv
    # gbuffer.slang IS the first stage that reads a shared body AND a heap UBO, and it cost two bugs, both
    # documented in docs/slang_migration.md section 9 because neither is visible in the SPIR-V's shape:
    # `mul(M, v)` on a `row_major float4x4` emits the TRANSPOSED product (the GLSL spelling `M * v` does
    # not), and a heap descriptor's TYPE has to match the storage class the shader reads it through - a
    # UNIFORM descriptor read through Slang's StorageBuffer pointer returns zeros with NO validation
    # finding. That is also why the camera and the light are written as STORAGE descriptors.
    gbuffer.slang:main:fragment:gbuffer.frag.spv
    # pbr.slang is the FORWARD path's shader PAIR: the fragment stage is where the sun/shadow/punctual/
    # cluster/IBL fetches are proven under Slang, and its GEOMETRY entry is what the G-BUFFER pass loads for
    # every opaque draw - so it is verified by all ten scenarios, and by the transparent pass's
    # forward default too. THAT ENTRY IS THE MESH ONE SINCE STEP 4 (docs/mesh_shaders.md): `pbr.vert.spv`
    # was removed with the rest of the vertex geometry path, so there is no vertex entry to list here.
    pbr.slang:main:fragment:pbr.frag.spv
    # ... the same file's MESH entry (docs/mesh_shaders.md step 2): the G-buffer/forward geometry as
    # dispatches, which is what the scene's opaque, forward and transparent leaves are drawn with.
    pbr.slang:mesh_main:mesh:pbr.mesh.spv
    # ... and the MESHLET entry (docs/mesh_shaders.md step 3): one workgroup per meshlet, camera-culled.
    pbr.slang:meshlet_main:mesh:pbr.meshlet.spv
    # shadow.slang shares no body: the material table and the albedo fetch through the shim's accessors
    # (`material_at`, `heap_sample`), and the entry returns void because a depth-only stage has no output.
    # Its MESH entry reuses the same matrix-array accessors pbr.slang does, which is the point of the pair:
    # both stages must place the geometry identically or the shadow silhouette detaches from the surface.
    # Its vertex entry went the same way as pbr's in step 4.
    shadow.slang:main:fragment:shadow.frag.spv
    # ... and the SAME file's MESH entry (docs/mesh_shaders.md step 1): the depth pass drawn as dispatches
    # instead of indexed draws. It REPLACES the vertex entry rather than joining it (one pipeline is MESH +
    # fragment), and the host keeps both - a device without VK_EXT_mesh_shader has no mesh stage at all.
    shadow.slang:mesh_main:mesh:shadow.mesh.spv
    # ... and the MESHLET entry (docs/mesh_shaders.md step 3): one workgroup per meshlet, window read from the table.
    shadow.slang:meshlet_main:mesh:shadow.meshlet.spv
    # light_cluster.slang is the FIRST compute stage: the clustered-light sort, dispatched once per frame by
    # default (clustered_lights = true), so every scenario records it. Its writable buffers are the heap slots
    # the GLSL declares as `writeonly buffer`.
    light_cluster.slang:main:compute:light_cluster.comp.spv
    # heap_probe.slang is the graphics half of the heap-native probe. No gate scenario renders it, but it is
    # the one group that VERIFIES ITSELF: the runtime dispatches it at startup, reads a pixel back and logs it
    # against the value it knows, so this port is accepted by comparing those log lines (see the file).
    heap_probe.slang:main:vertex:heap_probe.vert.spv
    heap_probe.slang:frag_main:fragment:heap_probe.frag.spv
    # The MESH half of the same probe (docs/mesh_shaders.md step 0): the same triangle, the same fragment
    # stage, the same 4x4 readback, emitted by a mesh pipeline instead of a vertex one. A mesh stage is a
    # different stage type, so this .spv name is new - the host learns it explicitly.
    heap_probe.slang:mesh_main:mesh:heap_probe.mesh.spv
    # The COMPUTE half is its own file because it has its own push layout (one push block per stage).
    heap_probe_comp.slang:comp_main:compute:heap_probe.comp.spv
    # compute_skin.slang feeds the acceleration structures the deformed vertex positions; its A/B needs the
    # skinned fixture with rt_shadow on (see docs/slang_migration.md).
    compute_skin.slang:comp_main:compute:compute_skin.comp.spv
    # mask_bake.slang is the alphaMode MASK bake (off by default; [render] rt_mask_bake turns it on, which is
    # also how its A/B is run).
    mask_bake.slang:comp_main:compute:mask_bake.comp.spv
    # megalights_temporal.slang is the stochastic lighting's denoise (off by default; [render] megalights = true
    # turns the path on, which is also how its A/B is run).
    megalights_temporal.slang:comp_main:compute:megalights_temporal.comp.spv
    # megalights_trace.slang is the stochastic punctual lighting tracer the temporal pass denoises.
    megalights_trace.slang:comp_main:compute:megalights_trace.comp.spv
    # rt_shadow.slang is the ray-traced shadow's shader file, growing one stage at a time; it starts with the
    # closest-hit stage, whose whole job is the payload contract (see the file). No gate scenario enables
    # rt_shadows, so this port is shape-verified and belongs to the A/B capture route in docs/slang_migration.md.
    rt_shadow.slang:chit_main:closesthit:rt_shadow.rchit.spv
    rt_shadow.slang:miss_main:miss:rt_shadow.rmiss.spv
    rt_shadow.slang:rgen_main:raygeneration:rt_shadow.rgen.spv
    rt_shadow.slang:ahit_main:anyhit:rt_shadow.rahit.spv

    # deferred.slang is the deferred path's lighting stage: the shared shading code again (so the forward and
    # the deferred image are comparable rather than merely similar), plus the screen-space AO it owns. It is
    # on every scenario's path, and it is the ONE stage whose frames differ from the GLSL build - by ONE pixel
    # of 1036800, a front-end codegen difference rather than a porting mistake (both modules call the same
    # GLSL.std.450 instructions; their SEQUENCES differ, and the driver contracts the multiply-adds
    # differently). It was accepted by an explicit re-baseline with the reason written down: see
    # docs/slang_migration.md section 9, "the last stage, and the one-pixel re-baseline".
    deferred.slang:main:fragment:deferred.frag.spv
)
set(VR_SLANG_OUTPUTS "")
foreach(entry IN LISTS VR_SLANG_SOURCES)
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 slang_source)
    list(GET parts 1 slang_entry)
    list(GET parts 2 slang_stage)
    list(GET parts 3 slang_output)
    add_custom_command(
        OUTPUT "${VR_SHADER_OUT_DIR}/${slang_output}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${VR_SHADER_OUT_DIR}"
        COMMAND "${VR_SLANGC_EXECUTABLE}" "${VR_SHADER_SRC_DIR}/${slang_source}" -I "${VR_SHADER_SRC_DIR}" -allow-glsl -DVR_SLANG
                -target spirv -profile spirv_1_6 -capability spvDescriptorHeapEXT
                -spirv-resource-heap-stride 64 -spirv-sampler-heap-stride 32
                -fvk-use-gl-layout -matrix-layout-column-major
                # NO -fp-mode HERE, and that is a measurement rather than an omission: `-fp-mode precise`
                # was tried against the byte-identical standard and made NINE scenarios differ, including
                # two that had been identical for eleven stages - so Slang's DEFAULT mode is the one whose
                # codegen matches glslang's, and the alternative is strictly worse. The difference the
                # default mode leaves is the one re-baselined for deferred.frag; see
                # docs/slang_migration.md section 9, "the last stage, and the one-pixel re-baseline".
                -entry ${slang_entry} -stage ${slang_stage} -o "${VR_SHADER_OUT_DIR}/${slang_output}"
        DEPENDS "${VR_SHADER_SRC_DIR}/${slang_source}" ${VR_SHADER_INCLUDES}
        COMMENT "slangc ${slang_source} -> ${slang_output}"
        VERBATIM
    )
    list(APPEND VR_SLANG_OUTPUTS "${VR_SHADER_OUT_DIR}/${slang_output}")
endforeach()
message(STATUS "slangc: ${VR_SLANGC_EXECUTABLE} - ${VR_SLANG_SOURCES}")
list(APPEND VR_SHADER_OUTPUTS ${VR_SLANG_OUTPUTS})
# THERE IS NO glslc RULE ANY MORE, and no VR_GLSL_SOURCES to compute: every stage is built from a .slang
# source above, so the split ("which sources does glslc still have to build?") that this section used to
# compute while the migration ran has no members left. The retired GLSL stage sources are archived in
# shaders/glsl.old/ and are NOT part of the build - they exist to be read next to their ports.
# the runtime's log names the SOURCE file it is loading the SPIR-V for, and it looks for that name in
# the same directory: copy the sources there so the log points at a file a reader of the build tree
# can actually open. Not a correctness requirement - the .spv beside it is what gets loaded. The list is
# TAKEN FROM VR_SLANG_SOURCES rather than written out again: it used to be a hand-maintained GLSL list, which
# is exactly the kind of duplication that goes stale (this one did, twice).
set(VR_SHADER_SOURCE_COPIES "")
foreach(entry IN LISTS VR_SLANG_SOURCES)
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 slang_source)
    list(APPEND VR_SHADER_SOURCE_COPIES "${VR_SHADER_SRC_DIR}/${slang_source}")
endforeach()
add_custom_command(
    OUTPUT "${VR_SHADER_OUT_DIR}/.sources-copied"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${VR_SHADER_SRC_DIR}" "${VR_SHADER_OUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E touch "${VR_SHADER_OUT_DIR}/.sources-copied"
    DEPENDS ${VR_SHADER_OUTPUTS} ${VR_SHADER_SOURCE_COPIES} ${VR_SHADER_INCLUDES}
    COMMENT "copying the GLSL sources beside the compiled shaders"
    VERBATIM
)
add_custom_target(shaders ALL DEPENDS ${VR_SHADER_OUTPUTS} "${VR_SHADER_OUT_DIR}/.sources-copied")
add_dependencies(vulkan_render shaders)
message(STATUS "slangc: ${VR_SLANGC_EXECUTABLE} - every build compiles shaders/*.spv into ${VR_SHADER_OUT_DIR}")