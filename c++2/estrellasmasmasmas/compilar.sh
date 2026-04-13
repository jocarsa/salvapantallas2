#!/usr/bin/env bash
set -euo pipefail

# build.sh for RTX 5060 Ti with old nvcc (12.0) + new driver
# Uses PTX forward compatibility instead of native sm_120,
# because nvcc 12.0 cannot compile compute_120.

OUT="${OUT:-starfield_cuda}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# For your current setup, this is the best fallback:
GENCODE="${GENCODE:--gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90}"

echo "[build] checking toolchain..."
nvcc --version
nvidia-smi

echo "[build] building ${OUT}"
nvcc -O3 -std=c++17 \
  ${EXTRA_CXXFLAGS} \
  ${GENCODE} \
  -lineinfo \
  starfield_cuda.cu \
  -o "${OUT}" \
  $(pkg-config --cflags --libs opencv4) \
  -lcurand

echo "[build] done: ./${OUT}"
