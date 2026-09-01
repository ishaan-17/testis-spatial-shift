#!/usr/bin/env bash
# Download the processed Slide-seqV2 testis pucks from Chen et al. 2021 (Cell Reports 37:109915).
# Links are the ones published in https://github.com/thechenlab/Testis_Slide-seq (README).
# Produces:
#   data/raw/mouse/Data/WT Slide-seq data/{WT1,WT2,WT3}/...
#   data/raw/mouse/Data/Diabetes Slide-seq data/{Diabetes_1,Diabetes_2,Diabetes_3}/...
#   data/raw/human/Human/...
# ~310 MB compressed, ~14 GB extracted (dense CSVs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

MOUSE_URL="https://www.dropbox.com/s/ygzpj0d0oh67br0/Testis_Slideseq_Data.zip?dl=1"
HUMAN_URL="https://www.dropbox.com/s/q5djhy006dq1yhw/Human.7z?dl=1"

if [ ! -d "$RAW/mouse/Data" ]; then
  echo "downloading mouse pucks (WT + ob/ob) ..."
  curl -L --retry 3 -o "$RAW/mouse.zip" "$MOUSE_URL"
  unzip -q -o "$RAW/mouse.zip" -d "$RAW/mouse"
fi

if [ ! -d "$RAW/human/Human" ]; then
  echo "downloading human pucks ..."
  curl -L --retry 3 -o "$RAW/human.7z" "$HUMAN_URL"
  python3 -c "import py7zr; py7zr.SevenZipFile('$RAW/human.7z').extractall('$RAW/human')"
fi

echo "done:"
find "$RAW" -name "MappedDGE*" | sort
