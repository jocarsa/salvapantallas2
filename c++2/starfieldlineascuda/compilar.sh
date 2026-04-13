#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-starfield_cuda}"
GENCODE="${GENCODE:--gencode arch=compute_120,code=sm_120}"

echo "Using nvcc: $(which nvcc)"
nvcc --version

nvcc -O3 -std=c++17 \
  ${GENCODE} \
  -lineinfo \
  starfield_cuda.cu \
  -o "${OUT}" \
  $(pkg-config --cflags --libs opencv4) \
  -lcurand

echo "Built: ./${OUT}"
