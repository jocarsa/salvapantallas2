#pragma once

#include <algorithm>
#include <cmath>

#include "config.hpp"
#include "cuda_common.cuh"

// Host-side helper used by src/main.cu
inline int compute_inverse_perspective_row_count(float /*z_near*/, float /*z_far*/, int subdivisions){
    subdivisions = std::max(8, subdivisions);
    return std::max(2, subdivisions);
}

// =========================
// Consistent-density sampling helpers
// =========================

__device__ __forceinline__ float visible_world_width_at_depth(float depth){
    return ((float)cfg::width * depth) / cfg::focal_length;
}

__device__ __forceinline__ float sample_depth_inverse_perspective(int row, int total_rows, float z_near, float z_far){
    if (total_rows <= 1) return z_near;

    float t = (float)row / (float)(total_rows - 1);
    float inv_near = 1.0f / z_near;
    float inv_far  = 1.0f / z_far;
    float inv_z    = lerpf(inv_near, inv_far, t);
    return 1.0f / inv_z;
}

__device__ __forceinline__ float sample_x_for_column(int col, int total_cols, float world_width){
    if (total_cols <= 1) return 0.0f;
    float t = (float)col / (float)(total_cols - 1);
    return -world_width * 0.5f + t * world_width;
}

__device__ __forceinline__ float row_stagger_offset(int row, int total_cols, float world_width){
    if (total_cols <= 1) return 0.0f;
    float cell = world_width / (float)(total_cols - 1);
    return (row & 1) ? (0.5f * cell) : 0.0f;
}

__device__ __forceinline__ float cell_width_for_cols(int total_cols, float world_width){
    if (total_cols <= 1) return world_width;
    return world_width / (float)(total_cols - 1);
}

__device__ __forceinline__ float depth_band_size_inverse_perspective(int row, int total_rows, float z_near, float z_far){
    if (total_rows <= 1) return z_far - z_near;
    float z0 = sample_depth_inverse_perspective(row, total_rows, z_near, z_far);
    int row1 = min(row + 1, total_rows - 1);
    float z1 = sample_depth_inverse_perspective(row1, total_rows, z_near, z_far);
    float dz = fabsf(z1 - z0);
    return (dz > 1e-6f) ? dz : 1e-6f;
}

__device__ __forceinline__ int compute_cols_for_row(float row_world_width, float base_world_width, int max_cols){
    float ratio = clampf(row_world_width / fmaxf(base_world_width, 1e-6f), 0.12f, 1.0f);
    int cols = (int)(ratio * (float)max_cols + 0.5f);
    cols = max(cols, 24);
    cols = min(cols, max_cols);
    return cols;
}

__device__ __forceinline__ int compute_radius_gpu(float rel_z, float radius_mult){
    float z = (rel_z > 1.f) ? rel_z : 1.f;
    float normalized_z = z / cfg::plane_depth;
    float perspective_scale = 1.f / normalized_z;
    float t = clampf(perspective_scale, 0.f, 1.f);
    float base = ((float)cfg::min_circle_radius + t * (float)(cfg::max_circle_radius - cfg::min_circle_radius));
    float scaled = base * radius_mult;
    if (scaled < 1.f) scaled = 1.f;
    if (scaled > 48.f) scaled = 48.f;
    return (int)lrintf(scaled);
}
