#!/usr/bin/env bash
set -e
nvcc -O3 -std=c++17 -arch=native \
  cells_cuda_render_mp4_grid.cu -o cells_cuda_render_mp4_grid \
  $(pkg-config --cflags --libs opencv4)

echo "Examples:"
echo "  ./cells_cuda_render_mp4_grid out.mp4 10"
echo "  ./cells_cuda_render_mp4_grid out.mp4 10 8000"
echo "  ./cells_cuda_render_mp4_grid out.mp4 10 8000 libx265"
