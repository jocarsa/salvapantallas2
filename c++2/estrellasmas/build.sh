#!/usr/bin/env bash
set -euo pipefail

SRC="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview.cu"
OUT="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview"

EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"

echo "[build] checking toolchain..."
nvcc --version
nvidia-smi

echo "[build] building with forward-compatible PTX (Blackwell fallback)"

nvcc -O3 -std=c++17 ${EXTRA_CXXFLAGS} \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  "${SRC}" -o "${OUT}" \
  ${OPENCV_FLAGS}

echo "[build] done: ./${OUT}"
