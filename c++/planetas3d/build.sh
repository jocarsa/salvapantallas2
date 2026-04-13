#!/usr/bin/env bash
set -euo pipefail

# build.sh
# Builds the CUDA + OpenCV renderer with collisions/merge/split + live preview.

SRC="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview.cu"
OUT="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview"

# If you want a specific GPU arch, export CUDA_ARCH, e.g.:
#   export CUDA_ARCH=89
CUDA_ARCH="${CUDA_ARCH:-native}"

# Extra flags (optional), e.g.:
#   export EXTRA_CXXFLAGS="-g -G"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# OpenCV flags
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"

echo "[build] nvcc -O3 -std=c++17 ${SRC} -> ${OUT}"
nvcc -O3 -std=c++17 ${EXTRA_CXXFLAGS} \
  -arch="${CUDA_ARCH}" \
  "${SRC}" -o "${OUT}" \
  ${OPENCV_FLAGS}

echo "[build] done: ./${OUT}"
