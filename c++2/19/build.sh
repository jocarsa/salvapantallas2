#!/usr/bin/env bash
# build.sh — compile vortex_growth_imagedata_fade_4k_cuda_mp4.cu

set -euo pipefail

SRC="vortex_growth_imagedata_fade_4k_cuda_mp4.cu"
OUT="vortex_growth_imagedata_fade_4k_cuda_mp4"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Install CUDA Toolkit and ensure nvcc is in PATH."
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "ERROR: pkg-config not found. Install it (e.g., sudo apt install pkg-config)."
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "ERROR: opencv4 pkg-config not found. Install OpenCV dev package (e.g., sudo apt install libopencv-dev)."
  exit 1
fi

echo "Compiling: $SRC -> $OUT"

# Set SM to match your GPU if you want:
#   export SM=75   # Turing
#   export SM=80   # Ampere A100
#   export SM=86   # Ampere RTX 30
#   export SM=89   # Ada RTX 40
#   export SM=90   # Hopper
SM="${SM:-86}"

nvcc -O3 -std=c++17 \
  -arch="sm_${SM}" \
  "$SRC" -o "$OUT" \
  $(pkg-config --cflags --libs opencv4)

echo "Done."
echo "Run examples:"
echo "  ./$OUT out.mp4 10 nvenc"
echo "  ./$OUT out.mp4 10 libx265"
