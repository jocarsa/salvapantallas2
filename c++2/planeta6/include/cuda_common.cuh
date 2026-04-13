#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

__device__ __forceinline__ float clampf(float v, float lo, float hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float fade(float t){
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

__device__ __forceinline__ float lerpf(float a, float b, float t){
    return a + t * (b - a);
}

__device__ __forceinline__ float smooth01(float t){
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float3 add3(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float s){
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b){
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float3 normalize3(float3 v){
    float len2 = dot3(v, v);
    if (len2 <= 1e-20f) return make_float3(0.f, 1.f, 0.f);
    float inv = 1.0f / sqrtf(len2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ __forceinline__ float3 reflect3(float3 i, float3 n){
    return sub3(i, mul3(n, 2.0f * dot3(i, n)));
}

// Define them here once for this project layout.
// They are used by the device helpers included into src/main.cu.
__constant__ int d_perm[512];
__constant__ float d_terrain_height_multiplier;
