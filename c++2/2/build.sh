#!/usr/bin/env bash
set -e

rm -f cells_cuda_render_mp4_grid

nvcc -O3 -std=c++17 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_61,code=compute_61 \
  cells_cuda_render_mp4_grid.cu \
  -o cells_cuda_render_mp4_grid \
  $(pkg-config --cflags --libs opencv4)

echo "Build OK"
