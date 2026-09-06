# Run vulkan_render (Windows). Usage:
#     powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1
#     powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1 -Model path/to/model.glb
#     powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1 -Model ... -Demo gui
#     powershell -ExecutionPolicy Bypass -File scripts/windows/run.ps1 -Config my.toml -Type Debug
#
# Runs from the repo root so shaders/ and gltf_model/ are located by the
# auto-discovery (or read config.toml). Extra positional args after -- are
# forwarded to the executable as-is (model, demo, grid side).

param(
    [string]$Model = '',
    [string]$Demo = '',
    [ValidateSet('Debug', 'Release')]
    [string]$Type = 'Release',
    [string]$Config = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Forward
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot | Split-Path -Parent | Split-Path -Parent
$exe = Join-Path $root "build-$($Type.ToLowerInvariant())-clang64\vulkan_render.exe"
if (-not (Test-Path $exe)) {
    Write-Error "not built: $exe`nRun scripts/windows/build.ps1 first."
}

$args = [System.Collections.Generic.List[string]]::new()
if ($Config) { $args.Add("--config"); $args.Add($Config) }
if ($Model)  { $args.Add($Model) }
if ($Demo)   { $args.Add($Demo) }
foreach ($a in $Forward) { $args.Add($a) }

Write-Host "== $exe $($args -join ' ') =="
Push-Location $root
& $exe @args
$code = $LASTEXITCODE
Pop-Location
exit $code
