# terrain_cuda_project

Refactor of the original monolithic `terrain_cuda.cu` into a more maintainable structure.

## Structure

- `src/main.cu` → orchestration only
- `include/config.hpp` / `src/config.cpp` → global configuration and defaults
- `include/types.hpp` → shared plain structs
- `include/biome.hpp` / `src/biome.cpp` → host-side biome style generation and blending
- `include/host_utils.hpp` / `src/host_utils.cpp` → host-side math, rotation, glow, motion blur, permutation upload
- `include/preview.hpp` / `src/preview.cpp` → preview window
- `include/ffmpeg_writer.hpp` / `src/ffmpeg_writer.cpp` → ffmpeg pipe output
- `include/cuda_common.cuh` → low-level CUDA helpers and constants
- `include/noise.cuh` → Perlin/fractal noise
- `include/curvature.cuh` → local planet curvature helpers
- `include/sampling.cuh` → inverse-perspective row/column sampling
- `include/projection.cuh` → GPU projection
- `include/color.cuh` → terrain/cloud/sky appearance logic
- `include/water.cuh` → water reflection subsystem
- `include/render_kernels.cuh` → framebuffer packing and render kernels

## Build

```bash
nvcc -O3 -std=c++17 \
  src/main.cu \
  src/config.cpp src/biome.cpp src/host_utils.cpp src/ffmpeg_writer.cpp src/preview.cpp \
  -Iinclude \
  `pkg-config --cflags --libs opencv4` \
  -o terrain_cuda
```

## Notes

This split preserves the original behavior as closely as possible while making the project easier to evolve.
