# Local screenshot regression check for the renderer.
#
# WHAT IT IS: it renders a fixed set of scenarios (fixed config, fixed camera, fixed frame count),
# screenshots each one, and compares it to a reference captured earlier ON THIS MACHINE. A difference
# means the frame CHANGED - not that it got better or worse. An intended visual change is accepted by
# re-running with -Update, which is the explicit step that makes the change visible in review.
#
# WHAT IT CATCHES: logic and wiring regressions, which is what the bugs found so far have all been -
# the deferred path's emissive term silently not being written, and the debug overlay opening a second
# rendering instance that CLEARED the composite it was supposed to sit on top of. Both changed the
# whole image.
# WHAT IT DOES NOT CATCH: a slightly noisier shadow, a 2% colour shift, or anything about performance.
#
# IT IS DELIBERATELY NOT A CI TEST. CI has no GPU, so this cannot run there at all; and the references
# are tied to this machine's GPU + driver, so a shared baseline would be red for everyone else.
#
# TWO PROPERTIES IT DEPENDS ON, both measured rather than assumed:
#  1. [gui] show = false. With the overlay on, the frame is NOT deterministic - it prints a live fps
#     counter, and three consecutive runs of one binary produced three different hashes. With it off,
#     three runs produced one hash. Every scenario therefore forces the overlay off and passes its own
#     config file, so a developer's own config.toml cannot influence it.
#  2. A fixed frame count: TAA accumulates over frames, so frame 10 and frame 40 differ.
# Check mode runs every scenario TWICE and requires the two runs to agree before comparing against the
# reference, so "flaky" is reported as flaky instead of as a regression.
#
# Usage:
#   pwsh -File scripts/windows/check_render.ps1                 # compare against the references
#   pwsh -File scripts/windows/check_render.ps1 -Update         # accept the current output as reference
#   pwsh -File scripts/windows/check_render.ps1 -Only deferred_taa_fxaa
#   pwsh -File scripts/windows/check_render.ps1 -List
#
# References live OUTSIDE the repository (they are machine-specific; committing them would be red for
# everyone else and would tie every accepted change to a multi-megabyte commit). Override the location
# with VR_RENDER_BASELINE_DIR.

param(
    [switch]$Update,
    [switch]$List,
    [string]$Only = "",
    [string]$BuildDir = "",
    [string]$Model = "C:\Users\23530\Desktop\yzk\glTF-Sample-Assets\Models\DamagedHelmet\glTF\DamagedHelmet.gltf",
    [int]$Frames = 40,
    [string]$Camera = "35,20,7,0,-1.6,0",
    [int]$Width = 1080,
    [int]$Height = 960
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $BuildDir) { $BuildDir = Join-Path $repo "build-release-clang64" }
$exe = Join-Path $BuildDir "vulkan_render.exe"
if (-not (Test-Path $exe)) { Write-Error "no executable at $exe - build first (-BuildDir to point elsewhere)"; exit 1 }

$baseDir = if ($env:VR_RENDER_BASELINE_DIR) { $env:VR_RENDER_BASELINE_DIR } else { Join-Path $env:LOCALAPPDATA "vulkan_render\baseline" }
$workDir = Join-Path $BuildDir "render-check"

# ---------------------------------------------------------------------------------------------
# The scenarios. Each is a full config (only the keys that differ from the defaults are listed) plus
# the frame count; the camera, model and extent are shared above so a scenario only varies what it
# means to - a scenario may override `model` / `camera` when it has to (see transparent_blend).
# Coverage is deliberately small to start - the G-buffer path with its optional stages, the
# shadow-cascade count and the transparent pass, because those are the paths whose wiring has broken.
# ---------------------------------------------------------------------------------------------
$scenarios = @(
    @{ name = "deferred";          desc = "deferred G-buffer + lighting, no AA stage";  extra = @{ msaa = "1" } }
    @{ name = "deferred_taa_fxaa"; desc = "deferred + TAA + FXAA (the AA path)";        extra = @{ msaa = "1"; taa = "true"; fxaa = "true" } }
    @{ name = "deferred_ssao_off"; desc = "deferred with SSAO disabled";               extra = @{ msaa = "1"; ssao = "false" } }
    @{ name = "shadow_single";     desc = "one cascade, i.e. the historic shadow path"; extra = @{ shadow_cascades = "1"; msaa = "1" } }
    @{ name = "unlit";             desc = "flat base colour, no shading";              extra = @{ unlit = "true"; msaa = "1" } }
    # The one scenario that uses a different model, and it has to: alphaMode BLEND geometry is drawn
    # by a pass of its own, so no model without a BLEND material can exercise it - DamagedHelmet has
    # only OPAQUE/MASK. AlphaBlendModeTest carries one of each alphaMode (OPAQUE / MASK at two cutoffs
    # / BLEND) plus a decal, so this also covers the G-buffer's MASK discard path.
    @{ name = "transparent_blend"; desc = "deferred + an alphaMode BLEND material";    extra = @{ msaa = "1" }
       model = "C:\Users\23530\Desktop\yzk\glTF-Sample-Assets\Models\AlphaBlendModeTest\glTF\AlphaBlendModeTest.gltf"
       camera = "0,5,12.4,0,-4.511,0" }
)

if ($List) {
    Write-Host "scenarios ($($scenarios.Count)):"
    foreach ($s in $scenarios) { "  {0,-20} {1}" -f $s.name, $s.desc }
    Write-Host "`nreferences: $baseDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $workDir, $baseDir | Out-Null

function Write-ScenarioConfig {
    param([hashtable]$Scenario, [string]$Path)
    # Every scenario pins the settings that decide what the frame looks like OR whether it is
    # deterministic at all. [gui] show = false is the determinism prerequisite (see the header), and
    # max_fps/vsync are pinned so two runs of one scenario pace identically.
    #
    # `msaa` is only a default here: a scenario that overrides it must not get both lines, because TOML
    # forbids a duplicate key and toml++ rejects the whole file (which reads as "the scenario silently
    # fell back to the compiled-in defaults"). Hence the -notcontains filter below.
    $defaults = [ordered]@{
        "window_width"      = "$Width"
        "window_height"     = "$Height"
        "vsync"             = "false"
        "max_fps"           = "240"
        "msaa"              = "8"
        "shadow"            = "true"
        "validation_layers" = "true"
    }
    $scenarioModel = if ($Scenario.ContainsKey('model')) { $Scenario.model } else { $Model }
    $lines = @(
        "model = '$($scenarioModel.Replace('\','\\'))'",
        "grid_side = 0",
        "",
        "[paths]",
        "shaders_dir = ''",
        "model_dir = ''",
        "screenshot_dir = '$($workDir.Replace('\','\\'))'",
        "",
        "[render]"
    )
    foreach ($k in $defaults.Keys) {
        if ($Scenario.extra.ContainsKey($k)) { continue } # the scenario's value wins (see above)
        $lines += "$k = $($defaults[$k])"
    }
    foreach ($k in $Scenario.extra.Keys) { $lines += "$k = $($Scenario.extra[$k])" }
    $lines += @(
        "",
        "[gui]",
        "show = false",
        "",
        "[lighting]",
        "env_size = 256",
        "env_mip_count = 5",
        "irr_size = 32",
        "lut_size = 256"
    )
    Set-Content -Path $Path -Value ($lines -join "`n") -NoNewline
}

function Invoke-Scenario {
    param([hashtable]$Scenario)
    $cfg = Join-Path $workDir "$($Scenario.name).toml"
    Write-ScenarioConfig -Scenario $Scenario -Path $cfg
    Get-ChildItem $workDir -Filter "screenshot_*.png" -ErrorAction SilentlyContinue | Remove-Item -Force
    $log = Join-Path $workDir "debug.log"
    Remove-Item $log -Force -ErrorAction SilentlyContinue

    # NOT $args: that is PowerShell's automatic argument array, and assigning to it is the kind of
    # thing that works until it does not.
    $scenarioCamera = if ($Scenario.ContainsKey('camera')) { $Scenario.camera } else { $Camera }
    $launch = @("--config", $cfg, "--capture-frames", "$Frames", "--capture-camera=$scenarioCamera")
    $p = Start-Process -FilePath $exe -ArgumentList $launch -WorkingDirectory $workDir -PassThru -WindowStyle Hidden
    # the capture exits on its own; the timeout is a safety net, not the expected path
    if (-not $p.WaitForExit(180000)) { $p.Kill(); return @{ ok = $false; why = "timed out" } }
    if ($p.ExitCode -ne 0) { return @{ ok = $false; why = "exit code $($p.ExitCode)" } }

    $shot = Get-ChildItem $workDir -Filter "screenshot_*.png" | Select-Object -First 1
    if (-not $shot) { return @{ ok = $false; why = "no screenshot produced" } }

    # Keep this run's image under a name that survives the NEXT run: the app writes a timestamped
    # screenshot_*.png and every scenario deletes those before it starts, so returning $shot.FullName
    # handed the caller a path that the second determinism run had already deleted - which is exactly
    # when the caller wants it (to save the diff of a CHANGED scenario).
    $kept = Join-Path $workDir "$($Scenario.name).last.png"
    Copy-Item $shot.FullName $kept -Force
    $shot = Get-Item $kept

    # the run has to have been VALIDATION clean: a difference in validation output is a finding on its
    # own, and a scenario that silently stopped being validation-clean should not be accepted as a
    # baseline either
    if (Test-Path $log) {
        $bad = Select-String -Path $log -Pattern "VUID-|Validation Error|\[ERROR\]|panic|recorded out of order"
        if ($bad) { return @{ ok = $false; why = "validation/log problems: $($bad[0].Line.Trim())" } }
    }
    return @{ ok = $true; path = $shot.FullName; hash = (Get-FileHash $shot.FullName -Algorithm SHA256).Hash }
}

$pass = 0; $fail = 0; $missing = 0; $flaky = 0; $skipped = 0
foreach ($s in $scenarios) {
    if ($Only -and $s.name -ne $Only) { $skipped++; continue }
    $ref = Join-Path $baseDir "$($s.name).png"
    Write-Host ("`n=== {0} ({1})" -f $s.name, $s.desc) -ForegroundColor Cyan

    $a = Invoke-Scenario -Scenario $s
    if (-not $a.ok) { Write-Host "  FAIL: $($a.why)" -ForegroundColor Red; $fail++; continue }
    if ($Update) {
        Copy-Item $a.path $ref -Force
        Write-Host "  reference updated ($($a.hash.Substring(0,16)))"
        $pass++
        continue
    }
    if (-not (Test-Path $ref)) {
        Write-Host "  NOT SEEDED: no reference at $ref" -ForegroundColor Yellow
        Write-Host "  run with -Update once to accept the current output as the reference"
        $missing++
        continue
    }

    # determinism first: two runs of the SAME binary must agree, or a mismatch below would be noise
    $b = Invoke-Scenario -Scenario $s
    if (-not $b.ok) { Write-Host "  FAIL (second run): $($b.why)" -ForegroundColor Red; $fail++; continue }
    if ($a.hash -ne $b.hash) {
        Write-Host "  FLAKY: two runs of this build differ ($($a.hash.Substring(0,16)) vs $($b.hash.Substring(0,16)))" -ForegroundColor Yellow
        Write-Host "  this scenario cannot be a regression check until it is deterministic"
        $flaky++
        continue
    }

    $refHash = (Get-FileHash $ref -Algorithm SHA256).Hash
    if ($a.hash -eq $refHash) {
        Write-Host "  ok  ($($a.hash.Substring(0,16)))" -ForegroundColor Green
        $pass++
    } else {
        Write-Host "  CHANGED: $($a.hash.Substring(0,16)) vs reference $($refHash.Substring(0,16))" -ForegroundColor Red
        $diff = Join-Path $workDir "$($s.name).actual.png"
        Copy-Item $a.path $diff -Force
        Write-Host "  current output kept at $diff"
        Write-Host "  if the change is intended: re-run with -Update"
        $fail++
    }
}

Write-Host "`n---------------- summary ----------------"
Write-Host "  passed   : $pass"
Write-Host "  changed  : $fail"
Write-Host "  flaky    : $flaky"
Write-Host "  unseeded : $missing"
if ($skipped) { Write-Host "  skipped  : $skipped" }
Write-Host "  references: $baseDir"
if ($fail -gt 0 -or $flaky -gt 0) { exit 1 }
exit 0
