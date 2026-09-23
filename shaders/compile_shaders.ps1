# Recompile every shader in shaders/ to SPIR-V binaries, in place.
#
# The BUILD already does this (see the SHADERS section of CMakeLists.txt): `cmake --build` recompiles every
# .spv from its source, and the resulting directory is mirrored next to the executable, which is the copy the
# runtime loads. This script is the escape hatch for a machine without CMake - it produces the same binaries
# in the same place.
#
# THE SHADERS ARE MOSTLY SLANG NOW. The canonical list of what builds which .spv, and with which entry point
# and stage, is VR_SLANG_SOURCES in CMakeLists.txt; the table below MIRRORS it, and every flag matches the
# CMake rule exactly (verified by running both and comparing the binaries byte for byte). Keep the two in
# step: the migration's endgame removes the GLSL sources, and this script is the list that has to survive it.
#
# Usage: powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1

$ErrorActionPreference = "Stop"

function Find-Tool([string]$name, [string]$sdkRelative) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if ($env:VULKAN_SDK) {
        $candidate = Join-Path $env:VULKAN_SDK $sdkRelative
        if (Test-Path $candidate) { return $candidate }
    }
    return ""
}

$slangcPath = Find-Tool "slangc" "Bin\slangc.exe"
$glslcPath = Find-Tool "glslc" "Bin\glslc.exe"
if (-not $slangcPath) {
    Write-Error "slangc not found. Install the Vulkan SDK or add slangc to PATH."
    exit 1
}
if (-not $glslcPath) {
    Write-Error "glslc not found. Install the Vulkan SDK or add glslc to PATH."
    exit 1
}

$shaderDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# source:entry:stage:output - the same shape CMakeLists.txt's VR_SLANG_SOURCES uses.
$slangSources = @(
    "unlit.slang:main:fragment:unlit.frag.spv",
    "fxaa.slang:main:fragment:fxaa.frag.spv",
    "post.slang:main:vertex:post.vert.spv",
    "post.slang:frag_main:fragment:post.frag.spv",
    "gbuffer_debug.slang:main:fragment:gbuffer_debug.frag.spv",
    "taa.slang:main:fragment:taa.frag.spv",
    "gbuffer.slang:main:fragment:gbuffer.frag.spv",
    "pbr.slang:main:fragment:pbr.frag.spv",
    "pbr.slang:vertex_main:vertex:pbr.vert.spv",
    "shadow.slang:main:fragment:shadow.frag.spv",
    "shadow.slang:vertex_main:vertex:shadow.vert.spv",
    "light_cluster.slang:main:compute:light_cluster.comp.spv",
    "heap_probe.slang:main:vertex:heap_probe.vert.spv",
    "heap_probe.slang:frag_main:fragment:heap_probe.frag.spv",
    "heap_probe_comp.slang:comp_main:compute:heap_probe.comp.spv",
    "rt_shadow.slang:chit_main:closesthit:rt_shadow.rchit.spv",
    "rt_shadow.slang:miss_main:miss:rt_shadow.rmiss.spv",
    "rt_shadow.slang:rgen_main:raygeneration:rt_shadow.rgen.spv",
    "rt_shadow.slang:ahit_main:anyhit:rt_shadow.rahit.spv",
    "compute_skin.slang:comp_main:compute:compute_skin.comp.spv",
    "mask_bake.slang:comp_main:compute:mask_bake.comp.spv",
    "megalights_temporal.slang:comp_main:compute:megalights_temporal.comp.spv",
    "megalights_trace.slang:comp_main:compute:megalights_trace.comp.spv"
)

# The GLSL stages that are NOT ported yet - the SLANG-LESS remainder, which shrinks to nothing as the
# migration finishes. `deferred.frag` is the last one, and the reason it is not in the list above is in
# docs/slang_migration.md (one pixel of 1036800, a codegen difference rather than a porting mistake).
$glslSources = @(
    "deferred.frag"
)

foreach ($entry in $slangSources) {
    $parts = $entry.Split(":")
    $src = Join-Path $shaderDir $parts[0]
    $dst = Join-Path $shaderDir $parts[3]
    # EXACTLY the CMake flags: -allow-glsl for the shared constants, -DVR_SLANG for the guarded declarations,
    # the two heap strides for the descriptor heap, -fvk-use-gl-layout for the buffer layout, and the
    # column-major default. Deliberately NO -fp-mode: the default is the mode whose codegen matches glslang's
    # (measured - '-fp-mode precise' made nine gate scenarios differ).
    & $slangcPath $src -I $shaderDir -allow-glsl -DVR_SLANG `
        -target spirv -profile spirv_1_6 -capability spvDescriptorHeapEXT `
        -spirv-resource-heap-stride 64 -spirv-sampler-heap-stride 32 `
        -fvk-use-gl-layout -matrix-layout-column-major `
        -entry $parts[1] -stage $parts[2] -o $dst
    if ($LASTEXITCODE -ne 0) {
        Write-Error "failed to compile $src (entry $($parts[1]))"
        exit $LASTEXITCODE
    }
    Write-Host "compiled: $src ($($parts[1])/$($parts[2])) -> $dst"
}

foreach ($src in $glslSources) {
    $dst = Join-Path $shaderDir "$src.spv"
    # Same as CMake's glslc rule: -I for the shared includes and --target-env=vulkan1.3 (glslc defaults to
    # vulkan1.0, and the heap extensions need 1.3).
    & $glslcPath -I $shaderDir --target-env=vulkan1.3 (Join-Path $shaderDir $src)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "failed to compile $src"
        exit $LASTEXITCODE
    }
    Write-Host "compiled: $src -> $dst"
}

Write-Host "all shaders compiled successfully ($($slangSources.Count) from Slang, $($glslSources.Count) still GLSL)."
