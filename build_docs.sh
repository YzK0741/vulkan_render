#!/bin/sh
# Build the project documentation: Doxygen HTML (docs/html) + the LaTeX manual
# compiled to PDF (docs/latex/refman.pdf).
# POSIX sh: run it under git-bash / MSYS2 / WSL / any Linux shell. Usage:
#     sh build_docs.sh          (from anywhere; the script resolves its own dir)
# Requires: doxygen on PATH (or the standard Windows/Unix install dirs) and a
# TeX toolchain — make + pdflatex/makeindex, latexmk, or bare pdflatex
# (MiKTeX's per-user install under %LOCALAPPDATA% is found automatically).

set -eu

# project root = directory of this script
root=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$root"

# normalize a (possibly Windows-style) path for POSIX tests
to_posix() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$1" 2>/dev/null || printf '%s' "$1" | tr '\\' '/'
    else
        printf '%s' "$1" | tr '\\' '/'
    fi
}

# ---------- 1. doxygen: HTML + LaTeX sources ----------
doxygen_cmd=""
if command -v doxygen >/dev/null 2>&1; then
    doxygen_cmd=$(command -v doxygen)
else
    for candidate in "/c/Program Files/doxygen/bin/doxygen.exe" "/usr/bin/doxygen" "/opt/homebrew/bin/doxygen"; do
        if [ -x "$candidate" ]; then
            doxygen_cmd=$candidate
            break
        fi
    done
fi
if [ -z "$doxygen_cmd" ]; then
    echo "error: doxygen not found on PATH (install Doxygen or add its bin dir)." >&2
    exit 1
fi

echo "== doxygen: $doxygen_cmd =="
"$doxygen_cmd" Doxyfile
echo "html written to docs/html/index.html"

# ---------- 2. LaTeX manual -> refman.pdf ----------
# ---------- 2. LaTeX manual -> refman.pdf ----------
# put a TeX toolchain on PATH when it lives at a standard location
add_tex_to_path() {
    [ -n "$1" ] || return 0
    d=$(to_posix "$1")
    [ -d "$d" ] || return 0
    case ":$PATH:" in
        *":$d:"*) ;;
        *) PATH="$d:$PATH" ;;
    esac
}
# manual rerun loop replicating the generated Makefile: pdflatex, makeindex,
# repeat pdflatex while the log asks for another pass, makeindex, pdflatex
run_pdflatex() {
    pdflatex -interaction=nonstopmode -halt-on-error refman.tex
}
run_makeindex() {
    if [ -f refman.idx ] && command -v makeindex >/dev/null 2>&1; then
        makeindex refman.idx
    fi
}
compile_latex_manually() {
    run_pdflatex
    run_makeindex
    count=0
    while grep -qs "Rerun" refman.log && [ "$count" -lt 8 ]; do
        run_pdflatex
        count=$((count + 1))
    done
    run_makeindex
    run_pdflatex
}

add_tex_to_path "${LOCALAPPDATA:-}/Programs/MiKTeX/miktex/bin/x64"
add_tex_to_path "${PROGRAMFILES:-}/MiKTeX/miktex/bin/x64"
# TeX Live installs versioned bin dirs: <root>/<year>/bin/<arch>
for texlive_root in "/c/texlive" "/usr/local/texlive" "/opt/texlive"; do
    for d in "$texlive_root"/*/bin/*; do
        [ -d "$d" ] && add_tex_to_path "$d"
    done
done

cd docs/latex

if command -v make >/dev/null 2>&1; then
    # doxygen generates docs/latex/Makefile with 'all' -> refman.pdf
    echo "== latex via make (docs/latex/Makefile) =="
    make
elif command -v pdflatex >/dev/null 2>&1; then
    echo "== latex via pdflatex (no make found) =="
    compile_latex_manually
else
    echo "error: no LaTeX toolchain found. Install MiKTeX/TeX Live (or make), then rerun." >&2
    exit 1
fi

echo
echo "documentation built:"
echo "  html: docs/html/index.html"
echo "  pdf:  docs/latex/refman.pdf"
