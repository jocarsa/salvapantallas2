#!/usr/bin/env bash
set -euo pipefail

# Adjust this if you want a specific SM:
# e.g. for RTX 30xx: -gencode arch=compute_86,code=sm_86
# e.g. for RTX 40xx: -gencode arch=compute_89,code=sm_89
GENCODE="${GENCODE:--gencode arch=compute_75,code=sm_75}"

OUT="${OUT:-starfield_cuda}"

nvcc -O3 -std=c++17 \
  ${GENCODE} \
  -lineinfo \
  starfield_cuda.cu \
  -o "${OUT}" \
  $(pkg-config --cflags --libs opencv4) \
  -lcurand

echo "Built: ./${OUT}"

