#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++17 --use_fast_math iso_cuda_pathtraced_simulation.cu $(pkg-config --cflags --libs opencv4) -lX11 -o iso_cuda_pathtraced_sim

echo "Compilation successful: ./iso_cuda_pathtraced_sim"
