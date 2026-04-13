#!/usr/bin/env bash
set -euo pipefail

# compile.sh — build starfield.cpp (OpenCV + OpenMP)
# Usage:
#   chmod +x compile.sh
#   ./compile.sh

SRC="starfield.cpp"
OUT="starfield_openmp"

if ! command -v g++ >/dev/null 2>&1; then
  echo "ERROR: g++ not found. Install: sudo apt install g++"
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "ERROR: pkg-config not found. Install: sudo apt install pkg-config"
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "ERROR: opencv4 pkg-config not found."
  echo "Install: sudo apt install libopencv-dev"
  exit 1
fi

echo "[build] SRC=$SRC"
echo "[build] OUT=$OUT"
echo "[build] OpenCV: $(pkg-config --modversion opencv4)"
echo "[build] OpenMP: enabled"

g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 \
  "$SRC" \
  $(pkg-config --cflags --libs opencv4) \
  -o "$OUT"

echo "[ok] Built: ./$OUT"
echo "Run: ./$OUT"

