#!/usr/bin/env bash
set -e
nvcc -O3 -std=c++17 -arch=native vortex_particles_4k_cuda_mp4.cu -o vortex_particles_4k_cuda_mp4 \
  $(pkg-config --cflags --libs opencv4)
echo "Run:"
echo "  ./vortex_particles_4k_cuda_mp4 out.mp4 10 10000 nvenc"
echo "  ./vortex_particles_4k_cuda_mp4 out.mp4 10 10000 libx265"
