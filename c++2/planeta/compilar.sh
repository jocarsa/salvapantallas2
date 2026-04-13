#!/bin/bash
set -e

# Nombre del archivo fuente CUDA
SOURCE="terrain_cuda.cu"
OUTPUT="terrain_cuda"

echo "🔧 Compilando $SOURCE con CUDA + OpenMP + OpenCV..."

# Compilación con NVCC (compila .cu y enlaza OpenCV; pasa OpenMP al compilador host)
nvcc -O3 -std=c++17 "$SOURCE" -o "$OUTPUT" \
  -Xcompiler -fopenmp \
  $(pkg-config --cflags --libs opencv4)

echo "✅ Compilación completada. Ejecuta con: ./$OUTPUT"

