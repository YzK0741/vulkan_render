#!/bin/sh
# Recompile every shader in shaders/ to SPIR-V binaries, in place.
#
# The BUILD already does this (see the SHADERS section of CMakeLists.txt): `cmake --build` recompiles every
# .spv from its source, and the resulting directory is mirrored next to the executable, which is the copy the
# runtime loads. This script is the escape hatch for a machine without CMake - it produces the same binaries
# in the same place.
#
# THE SHADERS ARE MOSTLY SLANG NOW. The canonical list of what builds which .spv, with which entry point and
# stage, is VR_SLANG_SOURCES in CMakeLists.txt; the table below MIRRORS it, and every flag matches the CMake
# rule exactly. Keep the two in step: the migration's endgame removes the GLSL sources, and this table is what
# has to survive it. POSIX-sh twin of compile_shaders.ps1, so the two tables must stay identical.
#
# Usage:
#     sh shaders/compile_shaders.sh      (from the project root, or anywhere)
#     ./shaders/compile_shaders.sh       (when the file is executable)

set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)

# ---- locate the two compilers: PATH first, then VULKAN_SDK ----
find_tool() {
    name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return
    fi
    if [ -n "${VULKAN_SDK:-}" ]; then
        for candidate in "$VULKAN_SDK/Bin/$name.exe" "$VULKAN_SDK/bin/$name" "$VULKAN_SDK/Bin/$name"; do
            if [ -x "$candidate" ]; then
                echo "$candidate"
                return
            fi
        done
    fi
    echo ""
}

slangc_path=$(find_tool slangc)
glslc_path=$(find_tool glslc)
if [ -z "$slangc_path" ]; then
    echo "slangc not found. Install the Vulkan SDK or add slangc to PATH." >&2
    exit 1
fi
if [ -z "$glslc_path" ]; then
    echo "glslc not found. Install the Vulkan SDK or add glslc to PATH." >&2
    exit 1
fi

# ---- the SLANG stages: source:entry:stage:output, the shape VR_SLANG_SOURCES uses ----
#
# EXACTLY the CMake flags, including the deliberate absence of -fp-mode: the DEFAULT mode is the one whose
# codegen matches glslang's ('-fp-mode precise' made nine gate scenarios differ, measured).
slang_count=0
while IFS=: read -r src entry_point stage dst; do
    [ -n "$src" ] || continue
    "$slangc_path" "$script_dir/$src" -I "$script_dir" -allow-glsl -DVR_SLANG \
        -target spirv -profile spirv_1_6 -capability spvDescriptorHeapEXT \
        -spirv-resource-heap-stride 64 -spirv-sampler-heap-stride 32 \
        -fvk-use-gl-layout -matrix-layout-column-major \
        -entry "$entry_point" -stage "$stage" -o "$script_dir/$dst"
    echo "compiled: $src ($entry_point/$stage) -> $dst"
    slang_count=$((slang_count + 1))
done <<'SLANG_SOURCES'
unlit.slang:main:fragment:unlit.frag.spv
fxaa.slang:main:fragment:fxaa.frag.spv
post.slang:main:vertex:post.vert.spv
post.slang:frag_main:fragment:post.frag.spv
gbuffer_debug.slang:main:fragment:gbuffer_debug.frag.spv
taa.slang:main:fragment:taa.frag.spv
gbuffer.slang:main:fragment:gbuffer.frag.spv
pbr.slang:main:fragment:pbr.frag.spv
pbr.slang:vertex_main:vertex:pbr.vert.spv
shadow.slang:main:fragment:shadow.frag.spv
shadow.slang:vertex_main:vertex:shadow.vert.spv
light_cluster.slang:main:compute:light_cluster.comp.spv
heap_probe.slang:main:vertex:heap_probe.vert.spv
heap_probe.slang:frag_main:fragment:heap_probe.frag.spv
heap_probe_comp.slang:comp_main:compute:heap_probe.comp.spv
rt_shadow.slang:chit_main:closesthit:rt_shadow.rchit.spv
rt_shadow.slang:miss_main:miss:rt_shadow.rmiss.spv
rt_shadow.slang:rgen_main:raygeneration:rt_shadow.rgen.spv
rt_shadow.slang:ahit_main:anyhit:rt_shadow.rahit.spv
compute_skin.slang:comp_main:compute:compute_skin.comp.spv
mask_bake.slang:comp_main:compute:mask_bake.comp.spv
megalights_temporal.slang:comp_main:compute:megalights_temporal.comp.spv
megalights_trace.slang:comp_main:compute:megalights_trace.comp.spv
SLANG_SOURCES

# ---- the GLSL stages that are NOT ported yet - the SLANG-LESS remainder ----
#
# `deferred.frag` is the last one, and the reason it is not in the list above is in docs/slang_migration.md
# (one pixel of 1036800, a codegen difference rather than a porting mistake).
glsl_count=0
for src in deferred.frag; do
    # Same as CMake's glslc rule: -I for the shared includes and --target-env=vulkan1.3 (glslc defaults to
    # vulkan1.0, and the heap extensions need 1.3).
    "$glslc_path" -I "$script_dir" --target-env=vulkan1.3 "$script_dir/$src"
    echo "compiled: $src -> $src.spv"
    glsl_count=$((glsl_count + 1))
done

echo "all shaders compiled successfully ($slang_count from Slang, $glsl_count still GLSL)."
