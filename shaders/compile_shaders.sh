#!/bin/sh
# Recompile the GLSL shaders in shaders/ to SPIR-V binaries.
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

# compile <source> <output-spv>: abort the whole run on the first failure
compile() {
    src=$1
    dst=$2
    "$glslc_path" "$script_dir/$src" -o "$script_dir/$dst"
    echo "compiled: $src -> $dst"
}

compile triangle.vert triangle.vert.spv
compile triangle.frag triangle.frag.spv
compile pbr.vert pbr.vert.spv
compile pbr.frag pbr.frag.spv
compile skybox.vert skybox.vert.spv
compile skybox.frag skybox.frag.spv
compile shadow.vert shadow.vert.spv
compile shadow.frag shadow.frag.spv

echo "all shaders compiled successfully."
