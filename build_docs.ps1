# Build the project documentation: Doxygen HTML (docs/html) + the LaTeX manual
# compiled to PDF (docs/latex/refman.pdf).
# PowerShell: run from anywhere (the script resolves its own dir):
#     powershell -ExecutionPolicy Bypass -File build_docs.ps1
# Requires: doxygen on PATH (or the standard Windows install dir) and a TeX
# toolchain — make + pdflatex/makeindex, latexmk, or bare pdflatex (MiKTeX's
# per-user install under %LOCALAPPDATA% is found automatically).

$ErrorActionPreference = 'Stop'

# project root = directory of this script
$root = $PSScriptRoot
if (-not $root) {
    $root = Split-Path -Parent $MyInvocation.MyCommand.Path
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

# manual rerun loop replicating the generated Makefile: pdflatex, makeindex,
# repeat pdflatex while the log asks for another pass, makeindex, pdflatex
function Invoke-Pdflatex {
    & pdflatex -interaction=nonstopmode -halt-on-error refman.tex
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'pdflatex failed (see docs/latex/refman.log).'
        exit $LASTEXITCODE
    }
}

function Invoke-Makeindex {
    if ((Test-Path 'refman.idx') -and (Get-Command makeindex -ErrorAction SilentlyContinue)) {
        & makeindex refman.idx
        if ($LASTEXITCODE -ne 0) {
            Write-Error 'makeindex failed.'
            exit $LASTEXITCODE
        }
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
}

Set-Location 'docs\latex'

if (Get-Command make -ErrorAction SilentlyContinue) {
    # doxygen generates docs/latex/Makefile with 'all' -> refman.pdf
    Write-Host '== latex via make (docs/latex/Makefile) =='
    & make
    if ($LASTEXITCODE -ne 0) {
        Write-Error "make failed (exit $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
} elseif (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Write-Host '== latex via pdflatex (no make found) =='
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
