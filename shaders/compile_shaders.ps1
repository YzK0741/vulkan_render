# Recompile the GLSL shaders in shaders/ to SPIR-V binaries
# Usage: powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1

$ErrorActionPreference = "Stop"

# Locate glslc: PATH first, then VULKAN_SDK
$glslcPath = ""
$cmd = Get-Command glslc -ErrorAction SilentlyContinue
if ($cmd) {
    $glslcPath = $cmd.Source
}
if (-not $glslcPath -and $env:VULKAN_SDK) {
    $candidate = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
    if (Test-Path $candidate) {
        $glslcPath = $candidate
    }
}
if (-not $glslcPath) {
    Write-Error "glslc not found. Install the Vulkan SDK or add glslc to PATH."
    exit 1
}

$shaderDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$pairs = @(
    @("triangle.vert", "triangle.vert.spv"),
    @("triangle.frag", "triangle.frag.spv"),
    @("pbr.vert", "pbr.vert.spv"),
    @("pbr.frag", "pbr.frag.spv"),
    @("skybox.vert", "skybox.vert.spv"),
    @("skybox.frag", "skybox.frag.spv")
)

foreach ($pair in $pairs) {
    $src = Join-Path $shaderDir $pair[0]
    $dst = Join-Path $shaderDir $pair[1]
    & $glslcPath $src -o $dst
    if ($LASTEXITCODE -ne 0) {
        Write-Error "failed to compile $src"
        exit $LASTEXITCODE
    }
    Write-Host "compiled: $src -> $dst"
}

Write-Host "all shaders compiled successfully."
