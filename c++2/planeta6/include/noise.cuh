#pragma once

#include "cuda_common.cuh"

// =========================
// Perlin helpers
// =========================

__device__ __forceinline__ float grad2(int hash, float x, float y){
    int h = hash & 3;
    float u = (h < 2) ? x : y;
    float v = (h < 2) ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.f * v : 2.f * v);
}

__device__ float perlin2(float x, float y){
    int xi = ((int)floorf(x)) & 255;
    int yi = ((int)floorf(y)) & 255;

    float xf = x - floorf(x);
    float yf = y - floorf(y);

    float u = fade(xf);
    float v = fade(yf);

    int aa = d_perm[d_perm[xi] + yi];
    int ab = d_perm[d_perm[xi] + yi + 1];
    int ba = d_perm[d_perm[xi + 1] + yi];
    int bb = d_perm[d_perm[xi + 1] + yi + 1];

    float x1 = lerpf(
        grad2(aa, xf, yf),
        grad2(ba, xf - 1.f, yf),
        u
    );

    float x2 = lerpf(
        grad2(ab, xf, yf - 1.f),
        grad2(bb, xf - 1.f, yf - 1.f),
        u
    );

    return lerpf(x1, x2, v);
}

__device__ float fractal_noise2(float x, float y){
    float total = 0.f;
    float freq  = 1.f;
    float amp   = 1.f;
    float maxa  = 0.f;

    #pragma unroll
    for (int i = 0; i < 5; i++){ // matches octaves
        total += perlin2(x * freq, y * freq) * amp;
        maxa  += amp;
        amp   *= 0.58f;  // persistence
        freq  *= 2.15f;  // lacunarity
    }

    return total / maxa;
}
