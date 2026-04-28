#!/bin/bash

set -e

nvcc tunel_cuda.cu \
    -o tunel_cuda \
    -O3 \
    -std=c++17 \
    $(pkg-config --cflags --libs opencv4)

echo "Compilado correctamente:"
echo "./tunel_cuda"
