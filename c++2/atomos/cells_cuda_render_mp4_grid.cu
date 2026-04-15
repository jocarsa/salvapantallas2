// chemistry_atoms_2d_cuda_extended.cu
//
// Simplified 2D chemical atom simulation in CUDA
//
// New in this version:
// 1. More chemical elements with conventional display colors
// 2. Per-element max bonds and preferred angles
// 3. Global SIZE_MUL that scales:
//      - atom radii
//      - ideal bond distances
//      - collision spacing
//      - bond thickness
//      - glow radius
//
// Notes:
// - This is still a visual toy chemistry simulation, not exact chemistry.
// - Preferred angles are adapted to 2D.
// - For multi-valence elements, the code chooses a 2D-friendly angle model.
//
// Compile:
//   nvcc -O3 -std=c++17 -diag-suppress=611 chemistry_atoms_2d_cuda_extended.cu \
//     $(pkg-config --cflags --libs opencv4) -o chemistry_atoms_2d_cuda_extended
//
// Usage:
//   ./chemistry_atoms_2d_cuda_extended
//   ./chemistry_atoms_2d_cuda_extended out.mp4
//   ./chemistry_atoms_2d_cuda_extended out.mp4 30
//   ./chemistry_atoms_2d_cuda_extended out.mp4 30 320
//   ./chemistry_atoms_2d_cuda_extended out.mp4 30 320 nvenc
//   ./chemistry_atoms_2d_cuda_extended out.mp4 30 320 libx265
//   ./chemistry_atoms_2d_cuda_extended out.mp4 30 320 nvenc 1

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------------------------------------------------
// Elements
// ------------------------------------------------------------
enum ElementType {
    ELEM_H  = 0,
    ELEM_C  = 1,
    ELEM_N  = 2,
    ELEM_O  = 3,
    ELEM_F  = 4,
    ELEM_P  = 5,
    ELEM_S  = 6,
    ELEM_CL = 7,
    ELEM_BR = 8,
    ELEM_I  = 9,
    ELEM_NA = 10,
    ELEM_K  = 11,
    ELEM_COUNT = 12
};

struct Atom {
    float x, y;
    float px, py;
    float vx, vy;

    float r, g, b;
    float rad;

    int element;
    int maxBonds;

    float idealBondLen;
    float preferredAngleDeg;
};

struct BondDraw {
    int a, b;
    int valid;
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
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

__device__ __forceinline__ float deg2rad(float d) {
    return d * (float)(M_PI / 180.0);
}

__device__ __forceinline__ int bond_count(const int* bonds, int base, int maxSlots) {
    int c = 0;
    for (int k = 0; k < maxSlots; k++) {
        if (bonds[base + k] >= 0) c++;
    }
    return c;
}

__device__ __forceinline__ bool already_bonded(
    const int* bonds, int base, int maxSlots, int other
) {
    for (int k = 0; k < maxSlots; k++) {
        if (bonds[base + k] == other) return true;
    }
    return false;
}

__device__ __forceinline__ bool try_insert_one_side(
    int* bonds, float* rest,
    int base, int maxSlots,
    int other, float restLen
) {
    for (int k = 0; k < maxSlots; k++) {
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
    int base, int maxSlots,
    int other
) {
    for (int k = 0; k < maxSlots; k++) {
        int idx = base + k;
        if (bonds[idx] == other) {
            bonds[idx] = -1;
            rest[idx] = 0.0f;
        }
    }
}

// ------------------------------------------------------------
// Grid
// ------------------------------------------------------------
__global__ void k_clear_cells(int* cellHead, int numCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) cellHead[i] = -1;
}

__global__ void k_build_grid(
    const Atom* a, int n,
    int* cellHead, int* next,
    float cellSize,
    int gridW, int gridH
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cx = (int)floorf(a[i].px / cellSize);
    int cy = (int)floorf(a[i].py / cellSize);

    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    int c = cellIndex(cx, cy, gridW);
    int old = atomicExch(&cellHead[c], i);
    next[i] = old;
}

// ------------------------------------------------------------
// Bonds
// ------------------------------------------------------------
__global__ void k_init_bonds(int* bonds, float* rest, int totalSlots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalSlots) {
        bonds[i] = -1;
        rest[i] = 0.0f;
    }
}

__global__ void k_try_create_bonds(
    Atom* a, int n,
    const int* cellHead, const int* next,
    float cellSize, int gridW, int gridH,
    int* bonds, float* rest,
    int maxSlots,
    float createFactor,
    float maxRelSpeed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];
    int baseI = i * maxSlots;

    int countI = bond_count(bonds, baseI, maxSlots);
    if (countI >= me.maxBonds) return;

    int cx = (int)floorf(me.px / cellSize);
    int cy = (int)floorf(me.py / cellSize);
    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

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
                    Atom aj = a[j];
                    int baseJ = j * maxSlots;

                    if (!already_bonded(bonds, baseI, maxSlots, j)) {
                        int countJ = bond_count(bonds, baseJ, maxSlots);

                        if (countJ < aj.maxBonds) {
                            // simplified rule:
                            // allow bonding if at least one of the atoms has normal covalent behavior
                            // alkali metals (Na, K) kept monovalent but still allowed visually
                            float dx = aj.px - me.px;
                            float dy = aj.py - me.py;
                            float d2 = dx * dx + dy * dy;

                            float pairRest = 0.5f * (me.idealBondLen + aj.idealBondLen);
                            float createDist = pairRest * createFactor;

                            if (d2 < createDist * createDist) {
                                float rvx = aj.vx - me.vx;
                                float rvy = aj.vy - me.vy;
                                float relV2 = rvx * rvx + rvy * rvy;

                                if (relV2 <= maxRelSpeed * maxRelSpeed) {
                                    bool okI = try_insert_one_side(
                                        bonds, rest, baseI, maxSlots, j, pairRest
                                    );
                                    bool okJ = try_insert_one_side(
                                        bonds, rest, baseJ, maxSlots, i, pairRest
                                    );

                                    if (!(okI && okJ)) {
                                        remove_one_side(bonds, rest, baseI, maxSlots, j);
                                        remove_one_side(bonds, rest, baseJ, maxSlots, i);
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
    Atom* a, int n,
    const int* bonds, const float* rest,
    int maxSlots,
    float stiffness,
    float maxCorrection
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];
    int base = i * maxSlots;

    float corrX = 0.0f;
    float corrY = 0.0f;

    for (int k = 0; k < maxSlots; k++) {
        int j = bonds[base + k];
        if (j < 0) continue;

        Atom aj = a[j];
        float restLen = rest[base + k];

        float dx = aj.px - me.px;
        float dy = aj.py - me.py;
        float d = sqrtf(dx * dx + dy * dy + 1e-8f);
        if (d < 1e-6f) continue;

        float nx = dx / d;
        float ny = dy / d;
        float delta = d - restLen;

        float push = 0.5f * stiffness * delta;
        corrX += nx * push;
        corrY += ny * push;
    }

    float c2 = corrX * corrX + corrY * corrY;
    float m2 = maxCorrection * maxCorrection;
    if (c2 > m2) {
        float invL = rsqrtf(c2 + 1e-8f);
        corrX *= maxCorrection * invL;
        corrY *= maxCorrection * invL;
    }

    me.px += corrX;
    me.py += corrY;
    a[i] = me;
}

__device__ __forceinline__ float preferred_angle_for_center(const Atom& center, int m) {
    // 2D-friendly local geometries
    switch (center.element) {
        case ELEM_H:
            return 180.0f;
        case ELEM_F:
        case ELEM_CL:
        case ELEM_BR:
        case ELEM_I:
            return 180.0f;
        case ELEM_O:
            return (m >= 2) ? 104.5f : 180.0f;
        case ELEM_S:
            if (m == 2) return 104.5f;
            if (m == 3) return 120.0f;
            return 90.0f;
        case ELEM_N:
            if (m == 2) return 120.0f;
            if (m >= 3) return 107.0f;
            return 180.0f;
        case ELEM_P:
            if (m == 2) return 120.0f;
            if (m >= 3) return 107.0f;
            return 180.0f;
        case ELEM_C:
            if (m == 2) return 180.0f;   // sp-like in 2D
            if (m == 3) return 120.0f;   // sp2-like
            if (m >= 4) return 90.0f;    // 2D simplification of tetrahedral
            return 180.0f;
        case ELEM_NA:
        case ELEM_K:
            return 180.0f;
        default:
            return center.preferredAngleDeg;
    }
}

__global__ void k_solve_angles(
    Atom* a, int n,
    const int* bonds,
    int maxSlots,
    float angleStrength,
    float maxAngleCorrection
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom center = a[i];
    int base = i * maxSlots;

    int neigh[8];
    int m = 0;
    for (int k = 0; k < maxSlots; k++) {
        int j = bonds[base + k];
        if (j >= 0 && m < 8) neigh[m++] = j;
    }

    if (m < 2) return;

    float preferredDeg = preferred_angle_for_center(center, m);
    float preferred = deg2rad(preferredDeg);

    float corrX = 0.0f;
    float corrY = 0.0f;

    for (int u = 0; u < m; u++) {
        for (int v = u + 1; v < m; v++) {
            Atom a1 = a[neigh[u]];
            Atom a2 = a[neigh[v]];

            float v1x = a1.px - center.px;
            float v1y = a1.py - center.py;
            float v2x = a2.px - center.px;
            float v2y = a2.py - center.py;

            float l1 = sqrtf(v1x * v1x + v1y * v1y + 1e-8f);
            float l2 = sqrtf(v2x * v2x + v2y * v2y + 1e-8f);
            if (l1 < 1e-5f || l2 < 1e-5f) continue;

            float n1x = v1x / l1;
            float n1y = v1y / l1;
            float n2x = v2x / l2;
            float n2y = v2y / l2;

            float c = clampf(n1x * n2x + n1y * n2y, -1.0f, 1.0f);
            float ang = acosf(c);

            float err = ang - preferred;

            float t1x = -n1y;
            float t1y =  n1x;
            float t2x =  n2y;
            float t2y = -n2x;

            float s = angleStrength * err * 0.5f;
            corrX += s * (t1x + t2x);
            corrY += s * (t1y + t2y);
        }
    }

    float c2 = corrX * corrX + corrY * corrY;
    float m2 = maxAngleCorrection * maxAngleCorrection;
    if (c2 > m2) {
        float invL = rsqrtf(c2 + 1e-8f);
        corrX *= maxAngleCorrection * invL;
        corrY *= maxAngleCorrection * invL;
    }

    center.px += corrX;
    center.py += corrY;
    a[i] = center;
}

__global__ void k_break_bonds(
    Atom* a, int n,
    int* bonds, float* rest,
    int maxSlots,
    float breakStretch,
    float breakRelativeSpeed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];
    int baseI = i * maxSlots;

    for (int k = 0; k < maxSlots; k++) {
        int idxI = baseI + k;
        int j = bonds[idxI];
        if (j < 0) continue;
        if (j < i) continue;

        Atom aj = a[j];
        float restLen = rest[idxI];

        float dx = aj.px - me.px;
        float dy = aj.py - me.py;
        float d = sqrtf(dx * dx + dy * dy + 1e-8f);

        float stretch = (restLen > 1e-6f) ? d / restLen : 1.0f;

        float rvx = aj.vx - me.vx;
        float rvy = aj.vy - me.vy;
        float relSpeed = sqrtf(rvx * rvx + rvy * rvy + 1e-8f);

        if (stretch > breakStretch || relSpeed > breakRelativeSpeed) {
            int baseJ = j * maxSlots;
            bonds[idxI] = -1;
            rest[idxI] = 0.0f;
            remove_one_side(bonds, rest, baseJ, maxSlots, i);
        }
    }
}

// ------------------------------------------------------------
// Physics
// ------------------------------------------------------------
__global__ void k_predict_atoms(
    Atom* a, int n,
    int W, int H,
    float dt,
    float damping,
    float centerPull,
    float temperature
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];

    float cx = 0.5f * W;
    float cy = 0.5f * H;

    float dx = cx - me.x;
    float dy = cy - me.y;

    float ax = dx * centerPull;
    float ay = dy * centerPull;

    float t = 0.003f * me.x + 0.004f * me.y + 0.07f * i;
    ax += temperature * sinf(0.8f * t + 0.2f * me.y);
    ay += temperature * cosf(0.7f * t + 0.2f * me.x);

    me.vx = (me.vx + ax * dt) * damping;
    me.vy = (me.vy + ay * dt) * damping;

    me.px = me.x + me.vx * dt;
    me.py = me.y + me.vy * dt;

    a[i] = me;
}

__global__ void k_solve_collisions(
    Atom* a, int n,
    const int* cellHead, const int* next,
    int W, int H,
    float cellSize, int gridW, int gridH,
    float stiffness,
    float maxPush,
    float extraPad
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];

    int cx = (int)floorf(me.px / cellSize);
    int cy = (int)floorf(me.py / cellSize);
    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    float corrX = 0.0f;
    float corrY = 0.0f;

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
                    Atom aj = a[j];

                    float dx = me.px - aj.px;
                    float dy = me.py - aj.py;
                    float d2 = dx * dx + dy * dy;

                    float minDist = me.rad + aj.rad + extraPad;
                    float minD2 = minDist * minDist;

                    if (d2 < minD2) {
                        float d = sqrtf(d2 + 1e-8f);

                        float nxn = 1.0f;
                        float nyn = 0.0f;
                        if (d > 1e-6f) {
                            nxn = dx / d;
                            nyn = dy / d;
                        } else {
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

    float c2 = corrX * corrX + corrY * corrY;
    float maxP2 = maxPush * maxPush;
    if (c2 > maxP2) {
        float invL = rsqrtf(c2 + 1e-8f);
        corrX *= maxPush * invL;
        corrY *= maxPush * invL;
    }

    me.px += corrX;
    me.py += corrY;

    float pad = me.rad + 4.0f + extraPad;
    if (me.px < pad) me.px = pad;
    if (me.py < pad) me.py = pad;
    if (me.px > W - pad) me.px = W - pad;
    if (me.py > H - pad) me.py = H - pad;

    a[i] = me;
}

__global__ void k_finalize_atoms(
    Atom* a, int n,
    float dt,
    float velocityDamping,
    float maxSpeed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom me = a[i];

    me.vx = ((me.px - me.x) / dt) * velocityDamping;
    me.vy = ((me.py - me.y) / dt) * velocityDamping;

    float s2 = me.vx * me.vx + me.vy * me.vy;
    float ms2 = maxSpeed * maxSpeed;
    if (s2 > ms2) {
        float invL = rsqrtf(s2 + 1e-8f);
        me.vx *= maxSpeed * invL;
        me.vy *= maxSpeed * invL;
    }

    me.x = me.px;
    me.y = me.py;

    a[i] = me;
}

// ------------------------------------------------------------
// Draw bond list
// ------------------------------------------------------------
__global__ void k_build_draw_bonds(
    const int* bonds, int n, int maxSlots,
    BondDraw* out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int base = i * maxSlots;
    for (int k = 0; k < maxSlots; k++) {
        int idx = base + k;
        int j = bonds[idx];
        if (j >= 0 && j > i) {
            out[idx].a = i;
            out[idx].b = j;
            out[idx].valid = 1;
        } else {
            out[idx].a = -1;
            out[idx].b = -1;
            out[idx].valid = 0;
        }
    }
}

// ------------------------------------------------------------
// Rendering
// ------------------------------------------------------------
__global__ void k_clear_accum(float4* accum, int Npix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Npix) accum[i] = make_float4(0, 0, 0, 0);
}

__device__ __forceinline__ float point_segment_distance(
    float px, float py,
    float ax, float ay,
    float bx, float by
) {
    float abx = bx - ax;
    float aby = by - ay;
    float apx = px - ax;
    float apy = py - ay;

    float ab2 = abx * abx + aby * aby + 1e-8f;
    float t = clampf((apx * abx + apy * aby) / ab2, 0.0f, 1.0f);

    float qx = ax + t * abx;
    float qy = ay + t * aby;

    float dx = px - qx;
    float dy = py - qy;
    return sqrtf(dx * dx + dy * dy + 1e-8f);
}

__global__ void k_draw_bonds(
    const Atom* a,
    const BondDraw* bd,
    int totalSlots,
    float4* accum,
    int W, int H,
    float thickness,
    float intensity
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalSlots) return;
    if (!bd[i].valid) return;

    Atom A = a[bd[i].a];
    Atom B = a[bd[i].b];

    float x0 = A.x;
    float y0 = A.y;
    float x1 = B.x;
    float y1 = B.y;

    float minx = fminf(x0, x1) - thickness - 2.0f;
    float maxx = fmaxf(x0, x1) + thickness + 2.0f;
    float miny = fminf(y0, y1) - thickness - 2.0f;
    float maxy = fmaxf(y0, y1) + thickness + 2.0f;

    int ix0 = max(0, (int)floorf(minx));
    int ix1 = min(W - 1, (int)ceilf(maxx));
    int iy0 = max(0, (int)floorf(miny));
    int iy1 = min(H - 1, (int)ceilf(maxy));

    float cr = 0.55f * (A.r + B.r) * 0.5f + 0.75f;
    float cg = 0.55f * (A.g + B.g) * 0.5f + 0.75f;
    float cb = 0.55f * (A.b + B.b) * 0.5f + 0.75f;

    for (int y = iy0; y <= iy1; y++) {
        for (int x = ix0; x <= ix1; x++) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            float d = point_segment_distance(px, py, x0, y0, x1, y1);
            if (d > thickness * 2.2f) continue;

            float core = 1.0f - smoothstep(0.0f, thickness, d);
            float glow = expf(-(d * d) / (2.0f * thickness * thickness + 1e-6f));
            float w = intensity * (1.25f * core + 0.75f * glow);

            int idx = y * W + x;
            atomicAdd(&accum[idx].x, cr * w);
            atomicAdd(&accum[idx].y, cg * w);
            atomicAdd(&accum[idx].z, cb * w);
            atomicAdd(&accum[idx].w, w);
        }
    }
}

__global__ void k_draw_atoms(
    const Atom* a, int n,
    float4* accum,
    int W, int H,
    float glowFactor,
    float glowIntensity,
    float whiteCoreBoost
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Atom at = a[i];
    float rad = at.rad;
    float R = rad * glowFactor;

    int x0 = max(0, (int)floorf(at.x - R));
    int x1 = min(W - 1, (int)ceilf (at.x + R));
    int y0 = max(0, (int)floorf(at.y - R));
    int y1 = min(H - 1, (int)ceilf (at.y + R));

    float sigma = 0.33f * R;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = (x + 0.5f) - at.x;
            float dy = (y + 0.5f) - at.y;
            float d2 = dx * dx + dy * dy;
            if (d2 > R * R) continue;

            float d = sqrtf(d2 + 1e-8f);
            float core = 1.0f - smoothstep(rad * 0.70f, rad * 1.02f, d);
            float glow = expf(-d2 * inv2s2);

            float w = glowIntensity * (3.0f * core + 1.1f * glow);

            int idx = y * W + x;
            atomicAdd(&accum[idx].x, at.r * w);
            atomicAdd(&accum[idx].y, at.g * w);
            atomicAdd(&accum[idx].z, at.b * w);

            float ww = whiteCoreBoost * core * glowIntensity * 1.15f;
            atomicAdd(&accum[idx].x, ww);
            atomicAdd(&accum[idx].y, ww);
            atomicAdd(&accum[idx].z, ww);
            atomicAdd(&accum[idx].w, w + ww);
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
    float saturation
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npix) return;

    float4 a = accum[i];

    float r = 1.0f - expf(-(a.x + lift) * exposure);
    float g = 1.0f - expf(-(a.y + lift) * exposure);
    float b = 1.0f - expf(-(a.z + lift) * exposure);

    float avg = (r + g + b) * (1.0f / 3.0f);
    r = avg + (r - avg) * saturation;
    g = avg + (g - avg) * saturation;
    b = avg + (b - avg) * saturation;

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3 * i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3 * i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3 * i + 2] = (unsigned char)lrintf(255.0f * r);
}

// ------------------------------------------------------------
// Filename / ffmpeg
// ------------------------------------------------------------
static std::string makeDefaultOutputFilename() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
#if defined(_WIN32)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif

    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_now);
    return std::string("chemistry_extended_") + buf + ".mp4";
}

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
            "-rc vbr -cq 24 -b:v 0 "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        fprintf(stderr, "Failed to start ffmpeg.\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

// ------------------------------------------------------------
// Element setup
// ------------------------------------------------------------
static void setup_element(Atom& a, int elem, float sizeMul) {
    a.element = elem;

    switch (elem) {
        case ELEM_H:
            // white
            a.r = 1.30f; a.g = 1.30f; a.b = 1.30f;
            a.rad = 8.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 26.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_C:
            // black/dark gray
            a.r = 0.18f; a.g = 0.18f; a.b = 0.18f;
            a.rad = 16.0f * sizeMul;
            a.maxBonds = 4;
            a.idealBondLen = 38.0f * sizeMul;
            a.preferredAngleDeg = 109.5f;
            break;

        case ELEM_N:
            // blue
            a.r = 0.18f; a.g = 0.40f; a.b = 1.15f;
            a.rad = 13.0f * sizeMul;
            a.maxBonds = 3;
            a.idealBondLen = 36.0f * sizeMul;
            a.preferredAngleDeg = 107.0f;
            break;

        case ELEM_O:
            // red
            a.r = 1.00f; a.g = 0.18f; a.b = 0.18f;
            a.rad = 12.0f * sizeMul;
            a.maxBonds = 2;
            a.idealBondLen = 34.0f * sizeMul;
            a.preferredAngleDeg = 104.5f;
            break;

        case ELEM_F:
            // light green
            a.r = 0.55f; a.g = 1.20f; a.b = 0.55f;
            a.rad = 11.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 33.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_P:
            // orange
            a.r = 1.25f; a.g = 0.55f; a.b = 0.10f;
            a.rad = 18.0f * sizeMul;
            a.maxBonds = 3;
            a.idealBondLen = 42.0f * sizeMul;
            a.preferredAngleDeg = 107.0f;
            break;

        case ELEM_S:
            // yellow
            a.r = 1.30f; a.g = 1.20f; a.b = 0.18f;
            a.rad = 17.0f * sizeMul;
            a.maxBonds = 2;
            a.idealBondLen = 41.0f * sizeMul;
            a.preferredAngleDeg = 104.5f;
            break;

        case ELEM_CL:
            // green
            a.r = 0.10f; a.g = 0.95f; a.b = 0.10f;
            a.rad = 14.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 38.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_BR:
            // dark red/brown
            a.r = 0.60f; a.g = 0.15f; a.b = 0.10f;
            a.rad = 16.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 40.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_I:
            // violet
            a.r = 0.58f; a.g = 0.12f; a.b = 0.75f;
            a.rad = 18.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 43.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_NA:
            // violet/blue-ish conventional CPK-like style
            a.r = 0.40f; a.g = 0.35f; a.b = 1.20f;
            a.rad = 20.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 48.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        case ELEM_K:
            // purple
            a.r = 0.55f; a.g = 0.25f; a.b = 1.15f;
            a.rad = 22.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 52.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;

        default:
            a.r = 1.0f; a.g = 1.0f; a.b = 1.0f;
            a.rad = 12.0f * sizeMul;
            a.maxBonds = 1;
            a.idealBondLen = 32.0f * sizeMul;
            a.preferredAngleDeg = 180.0f;
            break;
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;

    std::string outPath = "chemistry_extended.mp4";
    int seconds = 36000;
    int N = 1320;
    std::string encoder = "nvenc";
    int preview = 1;

    if (argc >= 2) outPath = argv[1];
    if (argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) N = std::max(40, std::atoi(argv[3]));
    if (argc >= 5) encoder = argv[4];
    if (argc >= 6) preview = std::atoi(argv[5]) ? 1 : 0;

    // --------------------------------------------------------
    // Global size multiplier
    // --------------------------------------------------------
    const float SIZE_MUL = 0.5f;
    // examples:
    // 0.75f -> smaller chemistry
    // 1.25f -> larger chemistry
    // 1.50f -> quite large atoms and longer bonds

    const int totalFrames = seconds * fps;
    const float dt = 1.0f / (float)fps;

    const int substeps = 4;
    const int solverIters = 3;
    const float subDt = dt / (float)substeps;

    const float damping = 0.9992f;
    const float velocityDamping = 0.9985f;
    const float maxSpeed = 110.0f * SIZE_MUL;
    const float centerPull = 0.0035f;
    const float temperature = 10.0f * SIZE_MUL;

    const int MAX_BONDS_PER_ATOM = 4;

    const float bondCreateFactor = 1.45f;
    const float bondMaxRelSpeed = 65.0f * SIZE_MUL;
    const float bondStiffness = 0.60f;
    const float maxBondCorrection = 2.8f * SIZE_MUL;

    const float angleStrength = 0.008f;
    const float maxAngleCorrection = 0.8f * SIZE_MUL;

    const float breakStretch = 1.95f;
    const float breakRelativeSpeed = 180.0f * SIZE_MUL;

    const float maxRadiusForGrid = 22.0f * SIZE_MUL;
    const float cellSize = (maxRadiusForGrid * 2.0f) + (34.0f * SIZE_MUL);

    const float collisionStiffness = 0.72f;
    const float collisionMaxPush = 2.2f * SIZE_MUL;
    const float collisionExtraPad = 0.6f * SIZE_MUL;

    const float bondThickness = 5.0f * SIZE_MUL;
    const float bondIntensity = 0.055f;

    const float atomGlowFactor = 2.6f;
    const float atomGlowIntensity = 0.055f;
    const float atomWhiteCoreBoost = 1.0f;

    const float exposure = 2.65f;
    const float lift = 0.01f;
    const float gammaInv = 1.0f / 1.9f;
    const float saturation = 1.22f;

    const int gridW = (int)ceilf(W / cellSize);
    const int gridH = (int)ceilf(H / cellSize);
    const int numCells = gridW * gridH;
    const int Npix = W * H;
    const int totalBondSlots = N * MAX_BONDS_PER_ATOM;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Seconds: %d | FPS: %d | Frames: %d\n", seconds, fps, totalFrames);
    fprintf(stderr, "Atoms: %d\n", N);
    fprintf(stderr, "Size multiplier: %.2f\n", SIZE_MUL);
    fprintf(stderr, "Grid: %dx%d cells (cellSize=%.1f)\n", gridW, gridH, cellSize);
    fprintf(stderr, "Preview: %s\n", preview ? "ON" : "OFF");

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::uniform_real_distribution<float> ux(W * 0.32f, W * 0.68f);
    std::uniform_real_distribution<float> uy(H * 0.32f, H * 0.68f);
    std::uniform_real_distribution<float> uv(-24.0f * SIZE_MUL, 24.0f * SIZE_MUL);
    std::uniform_real_distribution<float> ur(0.0f, 1.0f);

    std::vector<Atom> hA(N);

    for (int i = 0; i < N; i++) {
        hA[i].x = ux(rng);
        hA[i].y = uy(rng);
        hA[i].px = hA[i].x;
        hA[i].py = hA[i].y;
        hA[i].vx = uv(rng);
        hA[i].vy = uv(rng);

        float rr = ur(rng);
        int elem = ELEM_C;

        // weighted distribution
        if      (rr < 0.26f) elem = ELEM_H;
        else if (rr < 0.44f) elem = ELEM_C;
        else if (rr < 0.56f) elem = ELEM_O;
        else if (rr < 0.65f) elem = ELEM_N;
        else if (rr < 0.72f) elem = ELEM_S;
        else if (rr < 0.78f) elem = ELEM_P;
        else if (rr < 0.84f) elem = ELEM_CL;
        else if (rr < 0.88f) elem = ELEM_F;
        else if (rr < 0.92f) elem = ELEM_BR;
        else if (rr < 0.95f) elem = ELEM_I;
        else if (rr < 0.975f) elem = ELEM_NA;
        else                  elem = ELEM_K;

        setup_element(hA[i], elem, SIZE_MUL);
    }

    Atom* dA = nullptr;
    float4* dAccum = nullptr;
    unsigned char* dBGR = nullptr;
    int* dCellHead = nullptr;
    int* dNext = nullptr;
    int* dBonds = nullptr;
    float* dBondRest = nullptr;
    BondDraw* dBondDraw = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(Atom) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dAccum, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));
    CUDA_CHECK(cudaMalloc(&dCellHead, sizeof(int) * (size_t)numCells));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dBonds, sizeof(int) * (size_t)totalBondSlots));
    CUDA_CHECK(cudaMalloc(&dBondRest, sizeof(float) * (size_t)totalBondSlots));
    CUDA_CHECK(cudaMalloc(&dBondDraw, sizeof(BondDraw) * (size_t)totalBondSlots));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(Atom) * (size_t)N, cudaMemcpyHostToDevice));

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
    if (!ff) return 1;

    cv::Mat previewFrame;
    if (preview) {
        previewFrame = cv::Mat(H, W, CV_8UC3, hFramePinned);
        cv::namedWindow("framebuffer", cv::WINDOW_NORMAL);
        cv::resizeWindow("framebuffer", 1280, 720);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int renderedFrames = 0;

    for (int f = 0; f < totalFrames; f++) {
        for (int s = 0; s < substeps; s++) {
            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_predict_atoms<<<grid, block, 0, stream>>>(
                    dA, N, W, H,
                    subDt,
                    damping,
                    centerPull,
                    temperature
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
                    dA, N,
                    dCellHead, dNext,
                    cellSize,
                    gridW, gridH
                );
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_try_create_bonds<<<grid, block, 0, stream>>>(
                    dA, N,
                    dCellHead, dNext,
                    cellSize, gridW, gridH,
                    dBonds, dBondRest,
                    MAX_BONDS_PER_ATOM,
                    bondCreateFactor,
                    bondMaxRelSpeed
                );
            }

            for (int iter = 0; iter < solverIters; iter++) {
                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_solve_bonds<<<grid, block, 0, stream>>>(
                        dA, N,
                        dBonds, dBondRest,
                        MAX_BONDS_PER_ATOM,
                        bondStiffness,
                        maxBondCorrection
                    );
                }

                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_solve_angles<<<grid, block, 0, stream>>>(
                        dA, N,
                        dBonds,
                        MAX_BONDS_PER_ATOM,
                        angleStrength,
                        maxAngleCorrection
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
                        dA, N,
                        dCellHead, dNext,
                        cellSize,
                        gridW, gridH
                    );
                }

                {
                    int block = 256;
                    int grid = (N + block - 1) / block;
                    k_solve_collisions<<<grid, block, 0, stream>>>(
                        dA, N,
                        dCellHead, dNext,
                        W, H,
                        cellSize, gridW, gridH,
                        collisionStiffness,
                        collisionMaxPush,
                        collisionExtraPad
                    );
                }
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_break_bonds<<<grid, block, 0, stream>>>(
                    dA, N,
                    dBonds, dBondRest,
                    MAX_BONDS_PER_ATOM,
                    breakStretch,
                    breakRelativeSpeed
                );
            }

            {
                int block = 256;
                int grid = (N + block - 1) / block;
                k_finalize_atoms<<<grid, block, 0, stream>>>(
                    dA, N,
                    subDt,
                    velocityDamping,
                    maxSpeed
                );
            }
        }

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_clear_accum<<<grid, block, 0, stream>>>(dAccum, Npix);
        }

        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_build_draw_bonds<<<grid, block, 0, stream>>>(
                dBonds, N, MAX_BONDS_PER_ATOM, dBondDraw
            );
        }

        {
            int block = 128;
            int grid = (totalBondSlots + block - 1) / block;
            k_draw_bonds<<<grid, block, 0, stream>>>(
                dA, dBondDraw, totalBondSlots,
                dAccum, W, H,
                bondThickness,
                bondIntensity
            );
        }

        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_draw_atoms<<<grid, block, 0, stream>>>(
                dA, N,
                dAccum, W, H,
                atomGlowFactor,
                atomGlowIntensity,
                atomWhiteCoreBoost
            );
        }

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_tonemap_to_bgr<<<grid, block, 0, stream>>>(
                dAccum, dBGR, Npix,
                exposure, lift, gammaInv, saturation
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
                fprintf(stderr, "\nInterrupted by user.\n");
                break;
            }
        }

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix * 3, ff);
        if (written != (size_t)Npix * 3) {
            fprintf(stderr, "FFmpeg write failed at frame %d\n", f);
            break;
        }

        renderedFrames = f + 1;

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
            sec, renderedFrames, renderedFrames / std::max(sec, 1e-9));

    if (preview) cv::destroyAllWindows();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dAccum));
    CUDA_CHECK(cudaFree(dBGR));
    CUDA_CHECK(cudaFree(dCellHead));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dBonds));
    CUDA_CHECK(cudaFree(dBondRest));
    CUDA_CHECK(cudaFree(dBondDraw));

    return 0;
}
