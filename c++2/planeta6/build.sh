#!/usr/bin/env bash
set -e

nvcc -O3 -std=c++17 \
  src/main.cu \
  src/config.cpp src/biome.cpp src/host_utils.cpp src/ffmpeg_writer.cpp src/preview.cpp \
  -Iinclude \
  `pkg-config --cflags --libs opencv4` \
  -o terrain_cuda
  
./terrain_cuda
