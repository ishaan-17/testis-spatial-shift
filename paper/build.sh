#!/usr/bin/env bash
# Compile the manuscript. Tables are \input from ../results/, figures from figs/ (both produced by code/04_analyze.py).
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex | grep -iE "^!|undefined" || true
pdfinfo main.pdf | grep Pages
