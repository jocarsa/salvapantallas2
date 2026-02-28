#!/usr/bin/env bash
# build.sh — compile vortex_lines_whitefade_4k_cuda_mp4.cu

set -euo pipefail

SRC="vortex_lines_whitefade_4k_cuda_mp4.cu"
OUT="vortex_lines_whitefade_4k_cuda_mp4"

echo "=== CUDA Build Script ==="

# ---- checks ----
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found."
  echo "Install CUDA Toolkit and ensure nvcc is in PATH."
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "ERROR: pkg-config not installed."
  echo "sudo apt install pkg-config"
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "ERROR: OpenCV dev not found."
  echo "sudo apt install libopencv-dev"
  exit 1
fi

# ---- GPU architecture ----
# Override manually if needed:
#   export SM=89   (RTX 40)
#   export SM=86   (RTX 30)
#   export SM=75   (RTX 20)
#   export SM=90   (Hopper)
SM="${SM:-86}"

echo "Source : $SRC"
echo "Output : $OUT"
echo "SM     : sm_${SM}"
echo

# ---- compile ----
nvcc -O3 \
  -std=c++17 \
  -arch=sm_${SM} \
  "$SRC" \
  -o "$OUT" \
  $(pkg-config --cflags --libs opencv4)

echo
echo "✅ Build finished."
echo
echo "Run examples:"
echo "  ./$OUT out.mp4 10"
echo "  ./$OUT out.mp4 20 nvenc"
echo "  ./$OUT out.mp4 20 libx265"
