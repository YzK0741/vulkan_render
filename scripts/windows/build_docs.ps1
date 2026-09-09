# Build the project documentation: Doxygen HTML (docs/html) + the LaTeX manual
# compiled to PDF (docs/latex/refman.pdf).
# PowerShell (Windows): run from anywhere (the script resolves the repo root from its own dir):
#     powershell -ExecutionPolicy Bypass -File scripts/windows/build_docs.ps1
# Requires: doxygen on PATH (or the standard Windows install dir) and a TeX
# toolchain — make + pdflatex/makeindex, latexmk, or bare pdflatex (MiKTeX's
# per-user install under %LOCALAPPDATA% is found automatically).
#
# The LaTeX steps run SILENTLY: the (very chatty) pdflatex/make/makeindex
# stdout+stderr goes to a throwaway log file instead of the console. On
# failure the tail of that log is printed so a broken build still says why;
# pdflatex additionally keeps its full transcript in docs/latex/refman.log.

$ErrorActionPreference = 'Stop'

# project root = two levels up from scripts/<platform>/ holding this script
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $root) {
    $root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
}
Push-Location $root

# ---------- 1. doxygen: HTML + LaTeX sources ----------
$doxygen = Get-Command doxygen -ErrorAction SilentlyContinue
if (-not $doxygen) {
    $candidate = 'C:\Program Files\doxygen\bin\doxygen.exe'
    if (Test-Path $candidate) {
        $doxygen = Get-Item $candidate
    }
}
if (-not $doxygen) {
    Write-Error "doxygen not found on PATH (install Doxygen or add its bin dir)."
    exit 1
}

Write-Host "== doxygen: $($doxygen.Source) =="
& $doxygen.Source 'Doxyfile'
if ($LASTEXITCODE -ne 0) {
    Write-Error "doxygen failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}
Write-Host 'html written to docs/html/index.html'

# ---------- 2. LaTeX manual -> refman.pdf ----------
# put a TeX toolchain on PATH when it lives at a standard location
function Add-TexDir([string]$dir) {
    if (-not $dir -or -not (Test-Path $dir)) {
        return
    }
    if ($env:PATH -split ';' -notcontains $dir) {
        $env:PATH = "$dir;$env:PATH"
    }
}

Add-TexDir (Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64')
Add-TexDir (Join-Path $env:ProgramFiles 'MiKTeX\miktex\bin\x64')

# Run a native build step with its stdout+stderr captured into $LogFile instead
# of the console; return the exit code. $ErrorActionPreference is relaxed around
# the call so native stderr (redirected as error records on PowerShell 5.1) does
# not trip the script-wide 'Stop'.
function Invoke-BuildStep {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$LogFile
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $FilePath @ArgumentList *> $LogFile
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

# print the tail of a captured log (failure diagnostics); no-op when absent
function Show-LogTail {
    param([string]$Path, [int]$Lines = 30)
    if (Test-Path $Path) {
        Get-Content $Path -Tail $Lines | ForEach-Object { Write-Host $_ }
    }
}

# manual rerun loop replicating the generated Makefile: pdflatex, makeindex,
# repeat pdflatex while the log asks for another pass, makeindex, pdflatex
function Invoke-Pdflatex {
    $code = Invoke-BuildStep 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', 'refman.tex') 'latex_pass.log'
    if ($code -ne 0) {
        Write-Error 'pdflatex failed (see docs/latex/refman.log for the full transcript):'
        Show-LogTail 'latex_pass.log'
        exit $code
    }
}

function Invoke-Makeindex {
    if ((Test-Path 'refman.idx') -and (Get-Command makeindex -ErrorAction SilentlyContinue)) {
        $code = Invoke-BuildStep 'makeindex' @('refman.idx') 'latex_makeindex.log'
        if ($code -ne 0) {
            Write-Error 'makeindex failed:'
            Show-LogTail 'latex_makeindex.log'
            exit $code
        }
        Remove-Item 'latex_makeindex.log' -ErrorAction SilentlyContinue
    }
}

function Invoke-LatexManual {
    Invoke-Pdflatex
    Invoke-Makeindex
    $count = 0
    while ((Select-String -Path 'refman.log' -Pattern 'Rerun' -Quiet -ErrorAction SilentlyContinue) -and ($count -lt 8)) {
        Invoke-Pdflatex
        $count++
    }
    Invoke-Makeindex
    Invoke-Pdflatex
    Remove-Item 'latex_pass.log' -ErrorAction SilentlyContinue
}

Set-Location 'docs\latex'

if (Get-Command make -ErrorAction SilentlyContinue) {
    # doxygen generates docs/latex/Makefile with 'all' -> refman.pdf
    Write-Host '== latex via make (docs/latex/Makefile, output suppressed) =='
    $make = Get-Command make -ErrorAction SilentlyContinue
    $code = Invoke-BuildStep $make.Source @() 'latex_make.log'
    if ($code -ne 0) {
        Write-Error "make failed (exit $code), tail of docs/latex/latex_make.log:"
        Show-LogTail 'latex_make.log'
        exit $code
    }
    Remove-Item 'latex_make.log' -ErrorAction SilentlyContinue
} elseif (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host '== latex via pdflatex (no make found, output suppressed) =='
    Invoke-LatexManual
} else {
    Write-Error 'no LaTeX toolchain found. Install MiKTeX/TeX Live (or make), then rerun.'
    exit 1
}

Pop-Location

if (-not (Test-Path 'docs\latex\refman.pdf')) {
    Write-Error 'refman.pdf was not produced (see docs/latex/refman.log).'
    exit 1
}
Write-Host ''
Write-Host 'documentation built:'
Write-Host '  html: docs/html/index.html'
Write-Host '  pdf:  docs/latex/refman.pdf'
exit 0
