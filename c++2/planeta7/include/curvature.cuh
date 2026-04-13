#pragma once
#include "config.hpp"
#include "cuda_common.cuh"

__device__ __forceinline__ float curvature_drop_local(float local_x, float local_z){
    if (!cfg::enable_planet_curvature) return 0.0f;
    return (local_x * local_x + local_z * local_z) / (2.0f * cfg::planet_radius);
}

__device__ __forceinline__ float curved_surface_y(float base_y, float local_x, float local_z){
    return base_y - curvature_drop_local(local_x, local_z);
}

__device__ __forceinline__ float3 curvature_normal_local(float local_x, float local_z){
    if (!cfg::enable_planet_curvature) return make_float3(0.0f, 1.0f, 0.0f);
    return normalize3(make_float3(local_x / cfg::planet_radius, 1.0f, local_z / cfg::planet_radius));
}
