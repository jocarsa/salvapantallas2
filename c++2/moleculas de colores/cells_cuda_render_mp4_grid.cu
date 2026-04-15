// cells_cuda_render_mp4_grid_molecules.cu
//
// CUDA offline renderer -> FFmpeg MP4 (H.265), with:
// - Uniform grid neighbor search
// - Per-particle radius
// - Bright particles with white core
// - 1920x1080 @ 60fps -> mp4 h265
// - Live framebuffer preview via OpenCV
// - Stable position-based collision solver with substeps
// - One kinematic "chaos ball" that moves through the scene and perturbs particles
// - Molecular behavior:
//      * fixed palette: red, green, blue, cyan, magenta, yellow
//      * same-color particles can bond when close
//      * bonds act like molecule links
//      * bonds can break if chaos ball hits hard enough
//
// Compile:
//   nvcc -O3 -std=c++17 cells_cuda_render_mp4_grid_molecules.cu `pkg-config --cflags --libs opencv4` -o cells_cuda_render_mp4_grid_molecules
//
// Usage:
//   ./cells_cuda_render_mp4_grid_molecules
//   ./cells_cuda_render_mp4_grid_molecules out.mp4
//   ./cells_cuda_render_mp4_grid_molecules out.mp4 3600
//   ./cells_cuda_render_mp4_grid_molecules out.mp4 3600 12000
//   ./cells_cuda_render_mp4_grid_molecules out.mp4 3600 12000 nvenc
//   ./cells_cuda_render_mp4_grid_molecules out.mp4 3600 12000 libx265
//   ./cells_cuda_render_mp4_grid_molecules out.mp4 3600 12000 nvenc 1

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#include <ctime>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);         \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

struct Particle {
    float x, y;
    float px, py;
    float vx, vy;
    float r, g, b;
    float rad;
    int   colorId;
    float hit;      // accumulates recent chaos-ball hit force
};

// ---------------- Device helpers ----------------
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(b, fmaxf(a, v));
}

__device__ __forceinline__ float smoothstep(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ int cellIndex(int cx, int cy, int gridW) {
    return cy * gridW + cx;
}

__device__ __forceinline__ float len2(float x, float y) {
    return x * x + y * y;
}

__device__ __forceinline__ bool try_insert_one_side(
    int* bonds, float* rest,
    int base, int maxBonds,
    int other, float restLen
) {
    for (int k = 0; k < maxBonds; k++) {
        int idx = base + k;
        int old = atomicCAS(&bonds[idx], -1, other);
        if (old == -1) {
            rest[idx] = restLen;
            return true;
        }
        if (old == other) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ void remove_one_side(
    int* bonds, float* rest,
    int base, int maxBonds,
    int other
) {
    for (int k = 0; k < maxBonds; k++) {
        int idx = base + k;
        if (bonds[idx] == other) {
            bonds[idx] = -1;
            rest[idx]  = 0.0f;
        }
    }
}

__device__ __forceinline__ bool already_bonded(
    const int* bonds, int base, int maxBonds, int other
) {
    for (int k = 0; k < maxBonds; k++) {
        if (bonds[base + k] == other) return true;
    }
    return false;
}

// ---------------- Grid kernels ----------------
__global__ void k_clear_cells(int* cellHead, int numCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCells) return;
    cellHead[i] = -1;
}

__global__ void k_build_grid(
    const Particle* p, int n,
    int* cellHead, int* next,
    float cellSize,
    int gridW, int gridH
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = p[i].px;
    float y = p[i].py;

    int cx = (int)floorf(x / cellSize);
    int cy = (int)floorf(y / cellSize);

    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    int c = cellIndex(cx, cy, gridW);

    int old = atomicExch(&cellHead[c], i);
    next[i] = old;
}

// ---------------- Bond init / decay ----------------
__global__ void k_init_bonds(int* bonds, float* rest, int totalSlots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalSlots) return;
    bonds[i] = -1;
    rest[i]  = 0.0f;
}

__global__ void k_decay_hit(Particle* p, int n, float keep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].hit *= keep;
}

// ---------------- Physics kernels ----------------
__global__ void k_predict_particles(
    Particle* p, int n,
    int W, int H,
    float dt,
    float swirlStrength,
    float damping
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    float cx0 = 0.5f * W;
    float cy0 = 0.5f * H;

    float dxC = me.x - cx0;
    float dyC = me.y - cy0;
    float invLenC = rsqrtf(dxC * dxC + dyC * dyC + 1e-6f);

    float ax = (-dyC * invLenC) * swirlStrength;
    float ay = ( dxC * invLenC) * swirlStrength;

    me.vx = (me.vx + ax * dt) * damping;
    me.vy = (me.vy + ay * dt) * damping;

    me.px = me.x + me.vx * dt;
    me.py = me.y + me.vy * dt;

    p[i] = me;
}

__global__ void k_try_create_bonds(
    Particle* p, int n,
    const int* cellHead, const int* next,
    float cellSize, int gridW, int gridH,
    int* bonds, float* rest,
    int maxBondsPerParticle,
    float createDist,
    float maxRelSpeed,
    float maxHitForBonding
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];
    if (me.hit > maxHitForBonding) return;

    int baseI = i * maxBondsPerParticle;

    int cx = (int)floorf(me.px / cellSize);
    int cy = (int)floorf(me.py / cellSize);
    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    float createDist2 = createDist * createDist;

    for (int oy = -1; oy <= 1; oy++) {
        int ny = cy + oy;
        if (ny < 0 || ny >= gridH) continue;

        for (int ox = -1; ox <= 1; ox++) {
            int nx = cx + ox;
            if (nx < 0 || nx >= gridW) continue;

            int c = cellIndex(nx, ny, gridW);
            int j = cellHead[c];

            while (j != -1) {
                if (j > i) {
                    Particle pj = p[j];
                    if (pj.colorId == me.colorId && pj.hit <= maxHitForBonding) {
                        float dx = pj.px - me.px;
                        float dy = pj.py - me.py;
                        float d2 = dx * dx + dy * dy;

                        if (d2 < createDist2) {
                            float rvx = pj.vx - me.vx;
                            float rvy = pj.vy - me.vy;
                            float relV2 = rvx * rvx + rvy * rvy;

                            if (relV2 <= maxRelSpeed * maxRelSpeed) {
                                if (!already_bonded(bonds, baseI, maxBondsPerParticle, j)) {
                                    int baseJ = j * maxBondsPerParticle;

                                    float d = sqrtf(d2 + 1e-8f);
                                    float restLen = fmaxf(d, me.rad + pj.rad + 1.0f);

                                    bool okI = try_insert_one_side(
                                        bonds, rest, baseI, maxBondsPerParticle, j, restLen
                                    );
                                    bool okJ = try_insert_one_side(
                                        bonds, rest, baseJ, maxBondsPerParticle, i, restLen
                                    );

                                    if (!(okI && okJ)) {
                                        remove_one_side(bonds, rest, baseI, maxBondsPerParticle, j);
                                        remove_one_side(bonds, rest, baseJ, maxBondsPerParticle, i);
                                    }
                                }
                            }
                        }
                    }
                }
                j = next[j];
            }
        }
    }
}

__global__ void k_solve_bonds(
    Particle* p, int n,
    const int* bonds, const float* rest,
    int maxBondsPerParticle,
    float bondStiffness,
    float maxBondCorrection
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];
    int base = i * maxBondsPerParticle;

    float corrX = 0.0f;
    float corrY = 0.0f;

    for (int k = 0; k < maxBondsPerParticle; k++) {
        int j = bonds[base + k];
        if (j < 0) continue;

        Particle pj = p[j];
        float restLen = rest[base + k];

        float dx = pj.px - me.px;
        float dy = pj.py - me.py;
        float d2 = dx * dx + dy * dy;
        float d = sqrtf(d2 + 1e-8f);

        if (d < 1e-6f) continue;

        float nx = dx / d;
        float ny = dy / d;
        float delta = d - restLen;

        float push = 0.5f * bondStiffness * delta;
        corrX += nx * push;
        corrY += ny * push;
    }

    float c2 = corrX * corrX + corrY * corrY;
    float m2 = maxBondCorrection * maxBondCorrection;
    if (c2 > m2) {
        float invL = rsqrtf(c2 + 1e-8f);
        corrX *= maxBondCorrection * invL;
        corrY *= maxBondCorrection * invL;
    }

    me.px += corrX;
    me.py += corrY;
    p[i] = me;
}

__global__ void k_solve_collisions_pbd(
    Particle* p, int n,
    const int* cellHead, const int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH,
    float maxPushPerIter,
    float stiffness,
    float chaosX,
    float chaosY,
    float chaosR,
    float chaosVX,
    float chaosVY,
    float hitForceScale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    int cx = (int)floorf(me.px / cellSize);
    int cy = (int)floorf(me.py / cellSize);
    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    float corrX = 0.0f;
    float corrY = 0.0f;

    // particle-particle collisions
    for (int oy = -1; oy <= 1; oy++) {
        int ny = cy + oy;
        if (ny < 0 || ny >= gridH) continue;

        for (int ox = -1; ox <= 1; ox++) {
            int nx = cx + ox;
            if (nx < 0 || nx >= gridW) continue;

            int c = cellIndex(nx, ny, gridW);
            int j = cellHead[c];

            while (j != -1) {
                if (j != i) {
                    Particle pj = p[j];

                    float dx = me.px - pj.px;
                    float dy = me.py - pj.py;
                    float d2 = dx * dx + dy * dy;

                    float minDist = me.rad + pj.rad;
                    float minDist2 = minDist * minDist;

                    if (d2 < minDist2) {
                        float d = sqrtf(d2 + 1e-8f);

                        float nxn, nyn;
                        if (d > 1e-6f) {
                            nxn = dx / d;
                            nyn = dy / d;
                        } else {
                            nxn = 1.0f;
                            nyn = 0.0f;
                            d = minDist;
                        }

                        float overlap = minDist - d;
                        float push = 0.5f * stiffness * overlap;

                        corrX += nxn * push;
                        corrY += nyn * push;
                    }
                }
                j = next[j];
            }
        }
    }

    // particle-chaosBall collision + hit accumulation
    {
        float dx = me.px - chaosX;
        float dy = me.py - chaosY;
        float d2 = dx * dx + dy * dy;

        float minDist = me.rad + chaosR;
        float minDist2 = minDist * minDist;

        if (d2 < minDist2) {
            float d = sqrtf(d2 + 1e-8f);

            float nxn, nyn;
            if (d > 1e-6f) {
                nxn = dx / d;
                nyn = dy / d;
            } else {
                nxn = 1.0f;
                nyn = 0.0f;
                d = minDist;
            }

            float overlap = minDist - d;
            float push = 1.2f * stiffness * overlap;

            corrX += nxn * push;
            corrY += nyn * push;

            float relVX = me.vx - chaosVX;
            float relVY = me.vy - chaosVY;
            float relAlongN = fabsf(relVX * nxn + relVY * nyn);

            float hitAdd = hitForceScale * (overlap * 0.9f + relAlongN * 0.12f);
            me.hit = fminf(me.hit + hitAdd, 1000.0f);
        }
    }

    float corrLen2 = corrX * corrX + corrY * corrY;
    float maxPush2 = maxPushPerIter * maxPushPerIter;
    if (corrLen2 > maxPush2) {
        float invLen = rsqrtf(corrLen2 + 1e-8f);
        corrX *= maxPushPerIter * invLen;
        corrY *= maxPushPerIter * invLen;
    }

    me.px += corrX;
    me.py += corrY;

    float pad = me.rad + 2.0f;
    if (me.px < pad)     me.px = pad;
    if (me.px > W - pad) me.px = W - pad;
    if (me.py < pad)     me.py = pad;
    if (me.py > H - pad) me.py = H - pad;

    p[i] = me;
}

__global__ void k_break_bonds(
    Particle* p, int n,
    int* bonds, float* rest,
    int maxBondsPerParticle,
    float breakStretch,
    float breakHitThreshold,
    float breakRelativeSpeed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];
    int baseI = i * maxBondsPerParticle;

    for (int k = 0; k < maxBondsPerParticle; k++) {
        int idxI = baseI + k;
        int j = bonds[idxI];
        if (j < 0) continue;
        if (j < i) continue; // process pair only once

        Particle pj = p[j];
        float restLen = rest[idxI];

        float dx = pj.px - me.px;
        float dy = pj.py - me.py;
        float d = sqrtf(dx * dx + dy * dy + 1e-8f);

        float stretchRatio = (restLen > 1e-6f) ? d / restLen : 1.0f;

        float rvx = pj.vx - me.vx;
        float rvy = pj.vy - me.vy;
        float relSpeed = sqrtf(rvx * rvx + rvy * rvy + 1e-8f);

        bool breakIt = false;
        if (stretchRatio > breakStretch) breakIt = true;
        if (relSpeed > breakRelativeSpeed) breakIt = true;
        if (me.hit > breakHitThreshold || pj.hit > breakHitThreshold) breakIt = true;

        if (breakIt) {
            int baseJ = j * maxBondsPerParticle;
            bonds[idxI] = -1;
            rest[idxI] = 0.0f;
            remove_one_side(bonds, rest, baseJ, maxBondsPerParticle, i);
        }
    }
}

__global__ void k_finalize_particles(
    Particle* p, int n,
    float dt,
    float velocityDamping,
    float maxSpeed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    me.vx = ((me.px - me.x) / dt) * velocityDamping;
    me.vy = ((me.py - me.y) / dt) * velocityDamping;

    float speed2 = me.vx * me.vx + me.vy * me.vy;
    float maxSpeed2 = maxSpeed * maxSpeed;
    if (speed2 > maxSpeed2) {
        float invLen = rsqrtf(speed2 + 1e-8f);
        me.vx *= maxSpeed * invLen;
        me.vy *= maxSpeed * invLen;
    }

    me.x = me.px;
    me.y = me.py;

    p[i] = me;
}

// ---------------- Rendering ----------------
__global__ void k_clear_accum(float4* accum, int Npix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npix) return;
    accum[i] = make_float4(0, 0, 0, 0);
}

__global__ void k_draw_particles(
    const Particle* p, int n,
    float4* accum, int W, int H,
    float glowRadiusFactor,
    float glowIntensity,
    float coreEdgeSoftness,
    float whiteCoreBoost
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    float rad = me.rad;
    float R = rad * glowRadiusFactor;
    R = fminf(R, 80.0f);

    int x0 = (int)floorf(me.x - R);
    int x1 = (int)ceilf (me.x + R);
    int y0 = (int)floorf(me.y - R);
    int y1 = (int)ceilf (me.y + R);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W - 1, x1); y1 = min(H - 1, y1);

    float sigma = 0.33f * R;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    float coreIn  = rad * (1.0f - coreEdgeSoftness);
    float coreOut = rad * (1.0f + coreEdgeSoftness);

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx * dx + dy * dy;
            if (d2 > R * R) continue;

            float d = sqrtf(d2 + 1e-6f);

            float core = 1.0f - smoothstep(coreIn, coreOut, d);
            core = clampf(core, 0.0f, 1.0f);

            float glow = expf(-d2 * inv2s2);
            float w = glowIntensity * (0.95f * glow + 3.1f * core);

            int idx = y * W + x;

            atomicAdd(&accum[idx].x, me.r * w);
            atomicAdd(&accum[idx].y, me.g * w);
            atomicAdd(&accum[idx].z, me.b * w);

            float wWhite = whiteCoreBoost * core * glowIntensity * 1.25f;
            atomicAdd(&accum[idx].x, 1.0f * wWhite);
            atomicAdd(&accum[idx].y, 1.0f * wWhite);
            atomicAdd(&accum[idx].z, 1.0f * wWhite);

            atomicAdd(&accum[idx].w, w + wWhite);
        }
    }
}

__global__ void k_draw_chaos_ball(
    float4* accum, int W, int H,
    float cx, float cy, float rad,
    float rr, float gg, float bb,
    float glowRadiusFactor,
    float glowIntensity,
    float coreEdgeSoftness,
    float whiteCoreBoost
) {
    float R = rad * glowRadiusFactor;
    R = fminf(R, 120.0f);

    int x0 = (int)floorf(cx - R);
    int x1 = (int)ceilf (cx + R);
    int y0 = (int)floorf(cy - R);
    int y1 = (int)ceilf (cy + R);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W - 1, x1); y1 = min(H - 1, y1);

    float sigma = 0.33f * R;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    float coreIn  = rad * (1.0f - coreEdgeSoftness);
    float coreOut = rad * (1.0f + coreEdgeSoftness);

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = (x + 0.5f) - cx;
            float dy = (y + 0.5f) - cy;
            float d2 = dx * dx + dy * dy;
            if (d2 > R * R) continue;

            float d = sqrtf(d2 + 1e-6f);

            float core = 1.0f - smoothstep(coreIn, coreOut, d);
            core = clampf(core, 0.0f, 1.0f);

            float glow = expf(-d2 * inv2s2);
            float w = glowIntensity * (0.90f * glow + 3.0f * core);

            int idx = y * W + x;

            atomicAdd(&accum[idx].x, rr * w);
            atomicAdd(&accum[idx].y, gg * w);
            atomicAdd(&accum[idx].z, bb * w);

            float wWhite = whiteCoreBoost * core * glowIntensity * 1.35f;
            atomicAdd(&accum[idx].x, 1.0f * wWhite);
            atomicAdd(&accum[idx].y, 1.0f * wWhite);
            atomicAdd(&accum[idx].z, 1.0f * wWhite);

            atomicAdd(&accum[idx].w, w + wWhite);
        }
    }
}

__global__ void k_tonemap_to_bgr(
    const float4* accum,
    unsigned char* outBGR,
    int Npix,
    float exposure,
    float lift,
    float gammaInv,
    float saturationBoost
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npix) return;

    float4 a = accum[i];

    float r = 1.0f - expf(-(a.x + lift) * exposure);
    float g = 1.0f - expf(-(a.y + lift) * exposure);
    float b = 1.0f - expf(-(a.z + lift) * exposure);

    float avg = (r + g + b) * (1.0f / 3.0f);
    r = avg + (r - avg) * saturationBoost;
    g = avg + (g - avg) * saturationBoost;
    b = avg + (b - avg) * saturationBoost;

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3 * i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3 * i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3 * i + 2] = (unsigned char)lrintf(255.0f * r);
}

// ---------------- Chaos ball motion ----------------
static void computeChaosBall(
    float t, int W, int H,
    float& x, float& y, float& rad
) {
    rad = 64.0f;

    const float cx = 0.5f * W;
    const float cy = 0.5f * H;

    const float orbitRx = 0.30f * W;
    const float orbitRy = 0.25f * H;

    const float baseAng  = 0.38f * t;
    const float wobble1  = 0.65f * sinf(0.91f * t);
    const float wobble2  = 0.45f * cosf(1.37f * t);
    const float ang      = baseAng + wobble1 + wobble2;

    const float rxMod = orbitRx * (0.82f + 0.18f * sinf(0.53f * t));
    const float ryMod = orbitRy * (0.82f + 0.18f * cosf(0.71f * t));

    const float smallX = 120.0f * cosf(2.17f * t + 0.7f);
    const float smallY =  95.0f * sinf(1.83f * t + 1.2f);

    x = cx + cosf(ang) * rxMod + smallX;
    y = cy + sinf(ang * 1.07f) * ryMod + smallY;

    float pad = rad + 8.0f;
    if (x < pad) x = pad;
    if (x > W - pad) x = W - pad;
    if (y < pad) y = pad;
    if (y > H - pad) y = H - pad;
}

// ---------------- Datetime output filename ----------------
static std::string makeDefaultOutputFilename() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
#if defined(_WIN32)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif

    char datetimeBuf[64];
    std::strftime(datetimeBuf, sizeof(datetimeBuf), "%Y%m%d_%H%M%S", &tm_now);

    return std::string("out_") + datetimeBuf + ".mp4";
}

// ---------------- FFmpeg pipe ----------------
static FILE* open_ffmpeg_pipe(
    const std::string& outPath, int W, int H, int fps,
    const std::string& encoder
) {
    std::string cmd;

    if (encoder == "libx265") {
        cmd =
            "ffmpeg -y "
            "-f rawvideo -pix_fmt bgr24 "
            "-s " + std::to_string(W) + "x" + std::to_string(H) + " "
            "-r " + std::to_string(fps) + " "
            "-i - "
            "-an "
            "-c:v libx265 -preset ultrafast "
            "-x265-params log-level=error:repeat-headers=1 "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    } else {
        cmd =
            "ffmpeg -y "
            "-f rawvideo -pix_fmt bgr24 "
            "-s " + std::to_string(W) + "x" + std::to_string(H) + " "
            "-r " + std::to_string(fps) + " "
            "-i - "
            "-an "
            "-c:v hevc_nvenc -preset p1 -tune ll "
            "-rc vbr -cq 28 -b:v 0 "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        fprintf(stderr, "Failed to start ffmpeg. Command:\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

int main(int argc, char** argv) {
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;

    std::string outPath = makeDefaultOutputFilename();
    int seconds = 36000;
    int N =6000;
    std::string encoder = "nvenc";
    int preview = 1;

    if (argc >= 2) outPath = argv[1];
    if (argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) N = std::max(128, std::atoi(argv[3]));
    if (argc >= 5) encoder = argv[4];
    if (argc >= 6) preview = std::atoi(argv[5]) ? 1 : 0;

    const int totalFrames = seconds * fps;
    const float dt = 1.0f / (float)fps;

    // physics
    const int substeps = 4;
    const int solverIters = 3;
    const float subDt = dt / (float)substeps;

    const float swirlStrength = 18.0f;
    const float damping = 0.9994f;
    const float velocityDamping = 0.998f;
    const float maxSpeed = 520.0f;

    const float minRadius = 2.5f;
    const float maxRadius = 8.0f;
    const float chaosBallRadius = 64.0f;

    const float neighborRadius = 2.0f * fmaxf(maxRadius, chaosBallRadius) + 28.0f;
    const float cellSize = neighborRadius;

    const float solverMaxPushPerIter = 3.2f;
    const float solverStiffness = 0.86f;

    // molecular system
    const int   MAX_BONDS_PER_PARTICLE = 6;
    const float bondCreateDist = 17.0f;
    const float bondMaxRelSpeed = 42.0f;
    const float bondStiffness = 0.42f;
    const float maxBondCorrection = 2.4f;
    const float maxHitForBonding = 26.0f;
    const float breakStretch = 1.70f;
    const float breakHitThreshold = 20.0f;
    const float breakRelativeSpeed = 150.0f;
    const float hitForceScale = 3.0f;
    const float hitDecayKeep = 0.92f;

    // rendering
    const float glowRadiusFactor = 3.2f;
    const float glowIntensity = 0.040f;
    const float coreEdgeSoftness = 0.035f;
    const float whiteCoreBoost = 1.0f;

    const float exposure = 2.95f;
    const float lift = 0.010f;
    const float gammaInv = 1.0f / 1.9f;
    const float saturationBoost = 1.35f;

    const float chaosR = 1.00f;
    const float chaosG = 0.95f;
    const float chaosB = 0.35f;

    const int gridW = (int)ceilf(W / cellSize);
    const int gridH = (int)ceilf(H / cellSize);
    const int numCells = gridW * gridH;
    const int Npix = W * H;
    const int totalBondSlots = N * MAX_BONDS_PER_PARTICLE;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Seconds: %d | FPS: %d | Frames: %d\n", seconds, fps, totalFrames);
    fprintf(stderr, "Particles: %d\n", N);
    fprintf(stderr, "Grid: %dx%d cells (cellSize=%.1f)\n", gridW, gridH, cellSize);
    fprintf(stderr, "Preview: %s\n", preview ? "ON" : "OFF");
    fprintf(stderr, "Substeps: %d | Solver iterations: %d\n", substeps, solverIters);
    fprintf(stderr, "Chaos ball radius: %.1f\n", chaosBallRadius);
    fprintf(stderr, "Molecular bonds per particle: %d\n", MAX_BONDS_PER_PARTICLE);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> uv(-140.0f, 140.0f);
    std::uniform_real_distribution<float> urad(minRadius, maxRadius);
    std::uniform_int_distribution<int> ucol(0, 5);

    std::vector<Particle> hP(N);

    // fixed 6-color palette
    const float palette[6][3] = {
        {1.35f, 0.10f, 0.10f}, // red
        {0.10f, 1.35f, 0.10f}, // green
        {0.10f, 0.45f, 1.35f}, // blue
        {0.10f, 1.25f, 1.25f}, // cyan
        {1.25f, 0.10f, 1.25f}, // magenta
        {1.35f, 1.25f, 0.10f}  // yellow
    };

    for (int i = 0; i < N; i++) {
        hP[i].x = ux(rng);
        hP[i].y = uy(rng);
        hP[i].px = hP[i].x;
        hP[i].py = hP[i].y;
        hP[i].vx = uv(rng);
        hP[i].vy = uv(rng);
        hP[i].rad = urad(rng);
        hP[i].hit = 0.0f;

        hP[i].colorId = ucol(rng);
        hP[i].r = palette[hP[i].colorId][0];
        hP[i].g = palette[hP[i].colorId][1];
        hP[i].b = palette[hP[i].colorId][2];
    }

    // device buffers
    Particle* dP = nullptr;
    float4* dAccum = nullptr;
    unsigned char* dBGR = nullptr;
    int* dCellHead = nullptr;
    int* dNext = nullptr;
    int* dBonds = nullptr;
    float* dBondRest = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dAccum, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));
    CUDA_CHECK(cudaMalloc(&dCellHead, sizeof(int) * (size_t)numCells));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dBonds, sizeof(int) * (size_t)totalBondSlots));
    CUDA_CHECK(cudaMalloc(&dBondRest, sizeof(float) * (size_t)totalBondSlots));

    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle) * (size_t)N, cudaMemcpyHostToDevice));

    {
        int block = 256;
        int grid = (totalBondSlots + block - 1) / block;
        k_init_bonds<<<grid, block>>>(dBonds, dBondRest, totalBondSlots);
        CUDA_CHECK(cudaGetLastError());
    }

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if (!ff) {
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        return 1;
    }

    cv::Mat previewFrame;
    if (preview) {
        previewFrame = cv::Mat(H, W, CV_8UC3, hFramePinned);
        cv::namedWindow("framebuffer", cv::WINDOW_NORMAL);
        cv::resizeWindow("framebuffer", 1280, 720);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < totalFrames; f++) {
        for (int s = 0; s < substeps; s++) {
            float simTime = ((float)f + ((float)s + 1.0f) / (float)substeps) / (float)fps;
            float prevTime = simTime - subDt;

            float chaosX, chaosY, chaosRad;
            float chaosPrevX, chaosPrevY, chaosPrevRad;
            computeChaosBall(simTime, W, H, chaosX, chaosY, chaosRad);
            computeChaosBall(prevTime, W, H, chaosPrevX, chaosPrevY, chaosPrevRad);

            float chaosVX = (chaosX - chaosPrevX) / subDt;
            float chaosVY = (chaosY - chaosPrevY) / subDt;

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_predict_particles<<<grid, block, 0, stream>>>(
                    dP, N, W, H,
                    subDt,
                    swirlStrength,
                    damping
                );
            }

            // build grid once for bond creation
            {
                int block = 256;
                int grid = (numCells + block - 1) / block;
                k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
            }
            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_build_grid<<<grid, block, 0, stream>>>(
                    dP, N,
                    dCellHead, dNext,
                    cellSize,
                    gridW, gridH
                );
            }
            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_try_create_bonds<<<grid, block, 0, stream>>>(
                    dP, N,
                    dCellHead, dNext,
                    cellSize, gridW, gridH,
                    dBonds, dBondRest,
                    MAX_BONDS_PER_PARTICLE,
                    bondCreateDist,
                    bondMaxRelSpeed,
                    maxHitForBonding
                );
            }

            for (int iter = 0; iter < solverIters; iter++) {
                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_solve_bonds<<<grid, block, 0, stream>>>(
                        dP, N,
                        dBonds, dBondRest,
                        MAX_BONDS_PER_PARTICLE,
                        bondStiffness,
                        maxBondCorrection
                    );
                }

                {
                    int block = 256;
                    int grid = (numCells + block - 1) / block;
                    k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
                }

                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_build_grid<<<grid, block, 0, stream>>>(
                        dP, N,
                        dCellHead, dNext,
                        cellSize,
                        gridW, gridH
                    );
                }

                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_solve_collisions_pbd<<<grid, block, 0, stream>>>(
                        dP, N,
                        dCellHead, dNext,
                        W, H,
                        cellSize,
                        gridW, gridH,
                        solverMaxPushPerIter,
                        solverStiffness,
                        chaosX,
                        chaosY,
                        chaosRad,
                        chaosVX,
                        chaosVY,
                        hitForceScale
                    );
                }
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_break_bonds<<<grid, block, 0, stream>>>(
                    dP, N,
                    dBonds, dBondRest,
                    MAX_BONDS_PER_PARTICLE,
                    breakStretch,
                    breakHitThreshold,
                    breakRelativeSpeed
                );
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_finalize_particles<<<grid, block, 0, stream>>>(
                    dP, N,
                    subDt,
                    velocityDamping,
                    maxSpeed
                );
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_decay_hit<<<grid, block, 0, stream>>>(
                    dP, N, hitDecayKeep
                );
            }
        }

        float frameTime = (float)f / (float)fps;
        float chaosX, chaosY, chaosRad;
        computeChaosBall(frameTime, W, H, chaosX, chaosY, chaosRad);

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_clear_accum<<<grid, block, 0, stream>>>(dAccum, Npix);
        }

        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_draw_particles<<<grid, block, 0, stream>>>(
                dP, N,
                dAccum, W, H,
                glowRadiusFactor,
                glowIntensity,
                coreEdgeSoftness,
                whiteCoreBoost
            );
        }

        {
            k_draw_chaos_ball<<<1, 1, 0, stream>>>(
                dAccum, W, H,
                chaosX, chaosY, chaosRad,
                chaosR, chaosG, chaosB,
                glowRadiusFactor,
                glowIntensity * 1.45f,
                coreEdgeSoftness,
                whiteCoreBoost * 1.15f
            );
        }

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_tonemap_to_bgr<<<grid, block, 0, stream>>>(
                dAccum, dBGR, Npix,
                exposure, lift, gammaInv, saturationBoost
            );
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            hFramePinned, dBGR, (size_t)Npix * 3,
            cudaMemcpyDeviceToHost, stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (preview) {
            cv::imshow("framebuffer", previewFrame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                fprintf(stderr, "\nInterrupted by user from preview window.\n");
                break;
            }
        }

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix * 3, ff);
        if (written != (size_t)Npix * 3) {
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", f);
            break;
        }

        if ((f % 60) == 0) {
            auto tNow = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(tNow - t0).count();
            double effFps = (elapsed > 0.0) ? (double)(f + 1) / elapsed : 0.0;

            fprintf(stderr,
                    "\rFrame %d / %d | %.2f%% | %.2f fps effective",
                    f, totalFrames,
                    100.0 * (double)f / (double)totalFrames,
                    effFps);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\nFinalizing encode...\n");
    fflush(ff);
    pclose(ff);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "Done. Time: %.2f s for %d frames (%.2f fps effective)\n",
            sec, totalFrames, totalFrames / std::max(sec, 1e-9));

    if (preview) {
        cv::destroyAllWindows();
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));

    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dAccum));
    CUDA_CHECK(cudaFree(dBGR));
    CUDA_CHECK(cudaFree(dCellHead));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dBonds));
    CUDA_CHECK(cudaFree(dBondRest));

    return 0;
}
