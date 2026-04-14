#!/usr/bin/env bash
# build.sh — compile random_walk_particles_4k_cuda_mp4.cu

set -euo pipefail

SRC="random_walk_particles_4k_cuda_mp4.cu"
OUT="random_walk_particles_4k_cuda_mp4"

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

# You can change SM to match your GPU if you want (e.g. 75, 80, 86, 89, 90).
# Leaving it generic is also ok:
#   -arch=native is NOT valid for nvcc. Use -arch=sm_XX or -gencode.
SM="${SM:-86}"

nvcc -O3 -std=c++17 \
  -arch="sm_${SM}" \
  "$SRC" -o "$OUT" \
  $(pkg-config --cflags --libs opencv4)

echo "Done."
echo "Run example:"
echo "  ./$OUT out.mp4 10 100 nvenc"
echo "  ./$OUT out.mp4 10 100 libx265"
