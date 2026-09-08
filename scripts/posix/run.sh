#!/bin/sh
# Run vulkan_render on POSIX (Linux / WSL / macOS). Runs from the repo root so
# shaders/ and gltf_model/ are auto-located. Extra args are forwarded to the
# executable (model path, grid side).
# Usage:
#     sh scripts/posix/run.sh
#     sh scripts/posix/run.sh path/to/model.glb
#     sh scripts/posix/run.sh --config my.toml

set -eu

root=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
cd "$root"

config="${VULKAN_RENDER_CONFIG:-Release}"
exe="build-$(echo "$config" | tr '[:upper:]' '[:lower:]')/vulkan_render"

if [ ! -x "$exe" ]; then
    echo "error: not built: $exe" >&2
    echo "       run 'sh scripts/posix/build.sh' first (or set VULKAN_RENDER_CONFIG=Debug)." >&2
    exit 1
fi

echo "== $exe $* =="
exec "$exe" "$@"
