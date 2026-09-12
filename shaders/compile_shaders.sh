#!/bin/sh
# Recompile the GLSL shaders in shaders/ to SPIR-V binaries, in place.
#
# The BUILD already does this (see the GLSL -> SPIR-V section of CMakeLists.txt): `cmake --build`
# recompiles every .spv from its source, and the resulting directory is mirrored next to the
# executable, which is the copy the runtime loads. This script is the escape hatch for a machine
# without CMake - it produces the same binaries in the same place.
#
# POSIX-sh companion to compile_shaders.ps1: run it under git-bash / MSYS2 /
# WSL / any Linux shell. Usage:
#     sh shaders/compile_shaders.sh      (from the project root, or anywhere)
#     ./shaders/compile_shaders.sh       (when the file is executable)

set -eu

# ---- locate glslc: PATH first, then VULKAN_SDK (Windows layout Bin\glslc.exe) ----
glslc_path=""
if command -v glslc >/dev/null 2>&1; then
    glslc_path=$(command -v glslc)
elif [ -n "${VULKAN_SDK:-}" ]; then
    for candidate in "$VULKAN_SDK/Bin/glslc.exe" "$VULKAN_SDK/bin/glslc" "$VULKAN_SDK/Bin/glslc"; do
        if [ -x "$candidate" ]; then
            glslc_path=$candidate
            break
        fi
    done
fi
if [ -z "$glslc_path" ]; then
    echo "glslc not found. Install the Vulkan SDK or add glslc to PATH." >&2
    exit 1
fi

# directory of this script (shaders/), resolved wherever it is invoked from
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

# compile <source> <output-spv>: abort the whole run on the first failure.
# -I: shaders/surface.glsl is #included by pbr.frag / gbuffer.frag (the shared material-surface
# gather); glslc resolves includes against the given directories.
compile() {
    src=$1
    dst=$2
    "$glslc_path" -I "$script_dir" "$script_dir/$src" -o "$script_dir/$dst"
    echo "compiled: $src -> $dst"
}

compile pbr.vert pbr.vert.spv
compile pbr.frag pbr.frag.spv
compile unlit.frag unlit.frag.spv
compile gbuffer.frag gbuffer.frag.spv
compile gbuffer_debug.frag gbuffer_debug.frag.spv
compile deferred.frag deferred.frag.spv
compile taa.frag taa.frag.spv
compile light_cluster.comp light_cluster.comp.spv
compile skybox.vert skybox.vert.spv
compile skybox.frag skybox.frag.spv
compile shadow.vert shadow.vert.spv
compile shadow.frag shadow.frag.spv
compile post.vert post.vert.spv
compile post.frag post.frag.spv
compile fxaa.frag fxaa.frag.spv

echo "all shaders compiled successfully."
