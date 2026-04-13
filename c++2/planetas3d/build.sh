#!/usr/bin/env bash
set -euo pipefail

SRC="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview.cu"
OUT="vortex_lines_whitefade_4k_cuda_mp4_collide_merge_split_preview"

EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"

echo "[build] nvcc -O3 -std=c++17 ${SRC} -> ${OUT}"

nvcc -O3 --use_fast_math -std=c++17 ${EXTRA_CXXFLAGS} \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  "${SRC}" -o "${OUT}" \
  ${OPENCV_FLAGS}

echo "[build] done: ./${OUT}"
