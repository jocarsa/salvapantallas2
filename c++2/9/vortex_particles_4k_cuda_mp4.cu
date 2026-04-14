// vortex_particles_realistic_star_system_stable.cu
//
// Stable-tuned CUDA star-system style particle simulation.
// This version is intentionally calmer than the previous one:
//
// - CUDA + OpenCV preview + FFmpeg MP4 output
// - particles have conserved mass and radius derived from 2D density
// - collision outcomes:
//      * ricochet
//      * merge
//      * fragmentation (DISABLED by default for stability)
// - merge conserves mass and linear momentum
// - orbital initialization around a weak central attractor
// - reduced gravity, stronger damping, lower elasticity
//
// Build:
//   nvcc -O3 -std=c++17 vortex_particles_realistic_star_system_stable.cu -o vortex_particles_realistic_star_system_stable \
//     $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_particles_realistic_star_system_stable
//   ./vortex_particles_realistic_star_system_stable out.mp4
//   ./vortex_particles_realistic_star_system_stable out.mp4 10
//   ./vortex_particles_realistic_star_system_stable out.mp4 10 2500
//   ./vortex_particles_realistic_star_system_stable out.mp4 10 2500 nvenc
//   ./vortex_particles_realistic_star_system_stable out.mp4 10 2500 libx265

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
#include <iostream>

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

struct Particle {
    float x, y;
    float vx, vy;
    float r, g, b;
    float m;
    float radius;
    int active;
};

__host__ __device__ __forceinline__ float clampf(float v, float a, float b){
    return v < a ? a : (v > b ? b : v);
}

__host__ __device__ __forceinline__ int clampi(int v, int a, int b){
    return v < a ? a : (v > b ? b : v);
}

__device__ __forceinline__ uint32_t hash_u32(uint32_t x){
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float hash_to_01(uint32_t x){
    return (x & 0x00FFFFFFU) * (1.0f / 16777215.0f);
}

// 2D density model: area proportional to mass
__host__ __device__ __forceinline__ float mass_to_radius(float m){
    const float density2D = 2.2f;
    return sqrtf(fmaxf(m, 0.001f) / (density2D * 3.14159265358979323846f));
}

__host__ __device__ __forceinline__ void star_color_from_mass(float m, float& r, float& g, float& b){
    float t = clampf((log2f(fmaxf(m, 1.0f)) - 2.0f) / 5.0f, 0.0f, 1.0f);

    if (t < 0.35f) {
        float u = t / 0.35f;
        r = 1.00f;
        g = 0.45f + 0.45f * u;
        b = 0.20f + 0.60f * u;
    } else if (t < 0.75f) {
        float u = (t - 0.35f) / 0.40f;
        r = 1.00f;
        g = 0.90f + 0.10f * u;
        b = 0.80f + 0.18f * u;
    } else {
        float u = (t - 0.75f) / 0.25f;
        r = 1.00f - 0.18f * u;
        g = 1.00f - 0.08f * u;
        b = 0.98f + 0.02f * u;
    }

    r = clampf(r, 0.0f, 1.0f);
    g = clampf(g, 0.0f, 1.0f);
    b = clampf(b, 0.0f, 1.0f);
}

// ----------------------- Host helpers -----------------------
static std::string now_timestamp_string() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return std::string(buf);
}

static std::string add_timestamp_to_filename(const std::string& inputPath) {
    const std::string stamp = now_timestamp_string();

    if (inputPath.empty()) {
        return "vortex_realistic_stable_" + stamp + ".mp4";
    }

    std::string path = inputPath;
    size_t slashPos = path.find_last_of("/\\");
    std::string dir  = (slashPos == std::string::npos) ? "" : path.substr(0, slashPos + 1);
    std::string file = (slashPos == std::string::npos) ? path : path.substr(slashPos + 1);

    size_t dotPos = file.find_last_of('.');
    if (dotPos == std::string::npos) {
        return dir + file + "_" + stamp + ".mp4";
    }

    std::string base = file.substr(0, dotPos);
    std::string ext  = file.substr(dotPos);
    return dir + base + "_" + stamp + ext;
}

static FILE* open_ffmpeg_pipe(const std::string& outPath, int W, int H, int fps, const std::string& encoder){
    std::string cmd;
    if(encoder == "libx265"){
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
            "-rc vbr -cq 27 -b:v 0 "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if(!pipe){
        fprintf(stderr, "Failed to start ffmpeg.\n");
        return nullptr;
    }
    return pipe;
}

// ----------------------- Grid helpers -----------------------
__device__ __forceinline__ int cellIndex(int cx, int cy, int gridW){
    return cy * gridW + cx;
}

__global__ void k_clear_cells(int* cellHead, int numCells){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numCells) cellHead[i] = -1;
}

__global__ void k_clear_ints(int* arr, int n, int value){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) arr[i] = value;
}

__global__ void k_build_grid(
    const Particle* p, int n,
    int* cellHead, int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!p[i].active) return;

    int cx = clampi((int)floorf(p[i].x / cellSize), 0, gridW - 1);
    int cy = clampi((int)floorf(p[i].y / cellSize), 0, gridH - 1);

    int c = cellIndex(cx, cy, gridW);
    int old = atomicExch(&cellHead[c], i);
    next[i] = old;
}

// ----------------------- Rendering buffers -----------------------
__global__ void k_fade_buffers(
    float4* buf1, float4* buf2,
    int Npix,
    float buf1MulRGB,
    float buf1MulA,
    float buf2Mul
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float4 a = buf1[i];
    a.x *= buf1MulRGB;
    a.y *= buf1MulRGB;
    a.z *= buf1MulRGB;
    a.w *= buf1MulA;
    buf1[i] = a;

    float4 b = buf2[i];
    b.x *= buf2Mul;
    b.y *= buf2Mul;
    b.z *= buf2Mul;
    b.w *= buf2Mul;
    buf2[i] = b;
}

__global__ void k_deposit_pixels_buf1(
    const Particle* p, int n,
    float4* buf1,
    int W, int H,
    float ink
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!p[i].active) return;

    int x = (int)lrintf(p[i].x);
    int y = (int)lrintf(p[i].y);
    if(x < 0 || x >= W || y < 0 || y >= H) return;

    int idx = y * W + x;
    float luminosity = 0.55f + 0.015f * powf(fmaxf(p[i].m, 1.0f), 0.78f);
    float s = ink * luminosity;

    atomicAdd(&buf1[idx].x, p[i].r * s);
    atomicAdd(&buf1[idx].y, p[i].g * s);
    atomicAdd(&buf1[idx].z, p[i].b * s);
    atomicAdd(&buf1[idx].w, s);
}

__global__ void k_draw_blobs_buf2(
    const Particle* p, int n,
    float4* buf2,
    int W, int H,
    float blobIntensity
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!p[i].active) return;

    Particle me = p[i];

    float rad = fmaxf(1.2f, me.radius * 2.2f);
    int x0 = max(0, (int)floorf(me.x - rad));
    int x1 = min(W - 1, (int)ceilf(me.x + rad));
    int y0 = max(0, (int)floorf(me.y - rad));
    int y1 = min(H - 1, (int)ceilf(me.y + rad));

    float sigma = 0.36f * rad;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    float luminosity = 0.65f + 0.018f * powf(fmaxf(me.m, 1.0f), 0.80f);
    float glow = blobIntensity * luminosity;

    for(int y=y0; y<=y1; y++){
        for(int x=x0; x<=x1; x++){
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx*dx + dy*dy;
            if(d2 > rad * rad) continue;

            float w = glow * expf(-d2 * inv2s2);
            int idx = y * W + x;

            atomicAdd(&buf2[idx].x, me.r * w);
            atomicAdd(&buf2[idx].y, me.g * w);
            atomicAdd(&buf2[idx].z, me.b * w);
            atomicAdd(&buf2[idx].w, w);
        }
    }
}

// ----------------------- Motion -----------------------
__global__ void k_step_particles_grid(
    Particle* p, int n,
    const int* cellHead, const int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH,
    float dt,
    float gravityG,
    float centralGM,
    float softening,
    int frameIndex
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!p[i].active) return;

    Particle me = p[i];

    float ax = 0.0f;
    float ay = 0.0f;

    int cx = clampi((int)floorf(me.x / cellSize), 0, gridW - 1);
    int cy = clampi((int)floorf(me.y / cellSize), 0, gridH - 1);

    for(int oy=-1; oy<=1; oy++){
        int ny = cy + oy;
        if(ny < 0 || ny >= gridH) continue;
        for(int ox=-1; ox<=1; ox++){
            int nx = cx + ox;
            if(nx < 0 || nx >= gridW) continue;

            int c = cellIndex(nx, ny, gridW);
            int j = cellHead[c];

            while(j != -1){
                if(j != i && p[j].active){
                    Particle pj = p[j];
                    float dx = pj.x - me.x;
                    float dy = pj.y - me.y;
                    float d2 = dx*dx + dy*dy + softening * softening;
                    float invD = rsqrtf(d2);
                    float invD3 = invD * invD * invD;

                    float s = gravityG * pj.m * invD3;
                    ax += dx * s;
                    ay += dy * s;
                }
                j = next[j];
            }
        }
    }

    float dx0 = (W * 0.5f) - me.x;
    float dy0 = (H * 0.5f) - me.y;
    float d20 = dx0*dx0 + dy0*dy0 + 1600.0f;
    float invD0 = rsqrtf(d20);
    float invD03 = invD0 * invD0 * invD0;
    ax += dx0 * (centralGM * invD03);
    ay += dy0 * (centralGM * invD03);

    me.vx += ax * dt;
    me.vy += ay * dt;

    float drag = 0.0065f;
    me.vx *= (1.0f - drag);
    me.vy *= (1.0f - drag);

    float vmax = 320.0f;
    me.vx = clampf(me.vx, -vmax, vmax);
    me.vy = clampf(me.vy, -vmax, vmax);

    me.x += me.vx * dt;
    me.y += me.vy * dt;

    if(me.x < -180.0f || me.x >= W + 180.0f || me.y < -180.0f || me.y >= H + 180.0f){
        uint32_t seed0 = hash_u32((uint32_t)i ^ (uint32_t)(frameIndex * 9781 + 0x9e3779b9u));
        uint32_t seed1 = hash_u32(seed0 ^ 0x85ebca6bu);
        uint32_t seed2 = hash_u32(seed1 ^ 0xc2b2ae35u);

        float ring = 180.0f + 220.0f * hash_to_01(seed0);
        float ang  = 6.2831853f * hash_to_01(seed1);

        me.x = W * 0.5f + cosf(ang) * ring;
        me.y = H * 0.5f + sinf(ang) * ring;

        float tx = -sinf(ang);
        float ty =  cosf(ang);
        float speed = 55.0f + 70.0f * hash_to_01(seed2);

        me.vx = tx * speed;
        me.vy = ty * speed;
    }

    p[i] = me;
}

// ----------------------- Collisions -----------------------
__global__ void k_resolve_collisions(
    Particle* p, int n,
    const int* cellHead, const int* next,
    int* locks,
    float cellSize,
    int gridW, int gridH,
    int frameIndex
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    if(!p[i].active) return;

    Particle me = p[i];
    int cx = clampi((int)floorf(me.x / cellSize), 0, gridW - 1);
    int cy = clampi((int)floorf(me.y / cellSize), 0, gridH - 1);

    for(int oy=-1; oy<=1; oy++){
        int ny = cy + oy;
        if(ny < 0 || ny >= gridH) continue;
        for(int ox=-1; ox<=1; ox++){
            int nx = cx + ox;
            if(nx < 0 || nx >= gridW) continue;

            int c = cellIndex(nx, ny, gridW);
            int j = cellHead[c];

            while(j != -1){
                if(j > i && p[j].active){
                    Particle a = p[i];
                    Particle b = p[j];

                    float dx = b.x - a.x;
                    float dy = b.y - a.y;
                    float d2 = dx*dx + dy*dy;
                    float rr = a.radius + b.radius;

                    if(d2 < rr * rr){
                        if(atomicCAS(&locks[i], 0, 1) == 0){
                            if(atomicCAS(&locks[j], 0, 1) == 0){

                                a = p[i];
                                b = p[j];

                                if(a.active && b.active){
                                    dx = b.x - a.x;
                                    dy = b.y - a.y;
                                    d2 = dx*dx + dy*dy;
                                    rr = a.radius + b.radius;

                                    if(d2 < rr * rr){
                                        float dist = sqrtf(d2 + 1e-8f);
                                        float nxn = dx / dist;
                                        float nyn = dy / dist;
                                        float tx = -nyn;
                                        float ty =  nxn;

                                        float rvx = b.vx - a.vx;
                                        float rvy = b.vy - a.vy;
                                        float vn = rvx * nxn + rvy * nyn;
                                        float vt = rvx * tx + rvy * ty;

                                        float ma = fmaxf(a.m, 0.001f);
                                        float mb = fmaxf(b.m, 0.001f);
                                        float reducedMass = (ma * mb) / (ma + mb + 1e-6f);

                                        float kineticNormal = 0.5f * reducedMass * vn * vn;
                                        float bindingScale  = 10.0f * (ma + mb) / fmaxf(rr, 0.5f);
                                        float overlap = rr - dist;
                                        float sizeRatio = fminf(ma, mb) / fmaxf(ma, mb);

                                        uint32_t h = hash_u32((uint32_t)(i * 73856093u) ^
                                                              (uint32_t)(j * 19349663u) ^
                                                              (uint32_t)(frameIndex * 83492791u));
                                        float rnd = hash_to_01(h);

                                        bool approaching = (vn < 0.0f);

                                        bool mergeEvent =
                                            approaching &&
                                            kineticNormal < bindingScale * 1.8f &&
                                            sizeRatio > 0.08f &&
                                            overlap > 0.05f * rr &&
                                            rnd < 0.92f;

                                        // Disabled by default for stability.
                                        bool splitEvent = false;

                                        if(mergeEvent){
                                            float newM = ma + mb;
                                            float invM = 1.0f / fmaxf(newM, 1e-6f);

                                            Particle out{};
                                            out.active = 1;
                                            out.m = newM;
                                            out.radius = mass_to_radius(out.m);
                                            out.vx = (a.vx * ma + b.vx * mb) * invM;
                                            out.vy = (a.vy * ma + b.vy * mb) * invM;
                                            out.x = (a.x * ma + b.x * mb) * invM;
                                            out.y = (a.y * ma + b.y * mb) * invM;
                                            star_color_from_mass(out.m, out.r, out.g, out.b);

                                            p[i] = out;
                                            p[j].active = 0;
                                        }
                                        else if(splitEvent){
                                            // Intentionally disabled in this stable version.
                                        }
                                        else{
                                            float invA = 1.0f / ma;
                                            float invB = 1.0f / mb;

                                            float e = (kineticNormal > 28.0f) ? 0.45f : 0.28f;
                                            float jimp = 0.0f;

                                            if(vn < 0.0f){
                                                jimp = -(1.0f + e) * vn / (invA + invB + 1e-6f);
                                            }

                                            float impX = jimp * nxn;
                                            float impY = jimp * nyn;

                                            a.vx -= impX * invA;
                                            a.vy -= impY * invA;
                                            b.vx += impX * invB;
                                            b.vy += impY * invB;

                                            float tangentialDamp = 0.22f;
                                            a.vx += tx * vt * tangentialDamp * invA * reducedMass;
                                            a.vy += ty * vt * tangentialDamp * invA * reducedMass;
                                            b.vx -= tx * vt * tangentialDamp * invB * reducedMass;
                                            b.vy -= ty * vt * tangentialDamp * invB * reducedMass;

                                            float corr = fmaxf(overlap, 0.0f) * 0.70f;
                                            float wa = invA / (invA + invB);
                                            float wb = invB / (invA + invB);

                                            a.x -= nxn * corr * wa;
                                            a.y -= nyn * corr * wa;
                                            b.x += nxn * corr * wb;
                                            b.y += nyn * corr * wb;

                                            p[i] = a;
                                            p[j] = b;
                                        }
                                    }
                                }

                                atomicExch(&locks[j], 0);
                            }
                            atomicExch(&locks[i], 0);
                        }
                    }
                }
                j = next[j];
            }
        }
    }
}

// ----------------------- Final compose -----------------------
__global__ void k_compose_to_bgr(
    const float4* buf1, const float4* buf2,
    unsigned char* outBGR,
    int Npix,
    float exposure,
    float lift,
    float gammaInv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float r = buf1[i].x + buf2[i].x;
    float g = buf1[i].y + buf2[i].y;
    float b = buf1[i].z + buf2[i].z;

    r = fminf(r, 90.0f);
    g = fminf(g, 90.0f);
    b = fminf(b, 90.0f);

    r = 1.0f - expf(-(r + lift) * exposure);
    g = 1.0f - expf(-(g + lift) * exposure);
    b = 1.0f - expf(-(b + lift) * exposure);

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3*i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3*i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3*i + 2] = (unsigned char)lrintf(255.0f * r);
}

int main(int argc, char** argv){
    const int W = 1280;
    const int H = 720;
    const int fps = 60;

    std::string baseOutPath = "vortex_realistic_stable.mp4";
    int seconds = 36000;
    int initialN = 2500;
    std::string encoder = "nvenc";

    if(argc >= 2) baseOutPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) initialN = std::max(256, std::atoi(argv[3]));
    if(argc >= 5) encoder = argv[4];

    std::string outPath = add_timestamp_to_filename(baseOutPath);
    const int totalFrames = seconds * fps;
    const float dt = 1.0f / 60.0f;
    const int capacity = std::max(initialN * 2, initialN + 4000);

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Resolution: %dx%d | FPS: %d | Seconds: %d | Frames: %d\n", W, H, fps, seconds, totalFrames);
    fprintf(stderr, "Initial particles: %d | Capacity: %d\n", initialN, capacity);

    const float cellSize = 42.0f;
    const int gridW = (int)ceilf(W / cellSize);
    const int gridH = (int)ceilf(H / cellSize);
    const int numCells = gridW * gridH;

    const float gravityG  = 10.0f;
    const float centralGM = 42000.0f;
    const float softening = 14.0f;

    const float buf1MulRGB = 0.989f;
    const float buf1MulA   = 0.985f;
    const float buf2Mul    = 0.925f;

    const float ink = 0.42f;
    const float blobIntensity = 0.030f;

    const float exposure = 1.60f;
    const float lift = 0.008f;
    const float gammaInv = 1.0f / 2.0f;

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> um(3.0f, 110.0f);

    std::vector<Particle> hP(capacity);
    for(int i=0; i<capacity; i++){
        hP[i].active = 0;
        hP[i].x = hP[i].y = 0.0f;
        hP[i].vx = hP[i].vy = 0.0f;
        hP[i].m = 1.0f;
        hP[i].radius = mass_to_radius(hP[i].m);
        star_color_from_mass(hP[i].m, hP[i].r, hP[i].g, hP[i].b);
    }

    for(int i=0; i<initialN; i++){
        float ring = 140.0f + 360.0f * u01(rng);
        float ang = 6.2831853f * u01(rng);

        float x = W * 0.5f + cosf(ang) * ring;
        float y = H * 0.5f + sinf(ang) * ring;

        float tx = -sinf(ang);
        float ty =  cosf(ang);

        float m = um(rng);
        float r = mass_to_radius(m);

        float v = sqrtf(centralGM / fmaxf(ring, 40.0f));
        v *= 0.70f + 0.20f * u01(rng);

        hP[i].x = x;
        hP[i].y = y;
        hP[i].vx = tx * v;
        hP[i].vy = ty * v;
        hP[i].m = m;
        hP[i].radius = r;
        star_color_from_mass(hP[i].m, hP[i].r, hP[i].g, hP[i].b);
        hP[i].active = 1;
    }

    Particle* dP = nullptr;
    int* dCellHead = nullptr;
    int* dNext = nullptr;
    int* dLocks = nullptr;

    float4* dBuf1 = nullptr;
    float4* dBuf2 = nullptr;
    unsigned char* dBGR = nullptr;

    const int Npix = W * H;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)capacity));
    CUDA_CHECK(cudaMalloc(&dCellHead, sizeof(int) * (size_t)numCells));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * (size_t)capacity));
    CUDA_CHECK(cudaMalloc(&dLocks, sizeof(int) * (size_t)capacity));

    CUDA_CHECK(cudaMalloc(&dBuf1, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBuf2, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle) * (size_t)capacity, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBuf1, 0, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMemset(dBuf2, 0, sizeof(float4) * (size_t)Npix));

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if(!ff){
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        return 1;
    }

    cv::namedWindow("Realistic Star System Stable", cv::WINDOW_NORMAL);
    cv::resizeWindow("Realistic Star System Stable", 1600, 900);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool stopRequested = false;
    int framesDone = 0;

    for(int f=0; f<totalFrames; f++){
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_buffers<<<grid, block, 0, stream>>>(dBuf1, dBuf2, Npix, buf1MulRGB, buf1MulA, buf2Mul);
        }

        {
            int block = 256;
            int grid = (numCells + block - 1) / block;
            k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_clear_ints<<<grid, block, 0, stream>>>(dLocks, capacity, 0);
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_build_grid<<<grid, block, 0, stream>>>(dP, capacity, dCellHead, dNext, W, H, cellSize, gridW, gridH);
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_step_particles_grid<<<grid, block, 0, stream>>>(
                dP, capacity,
                dCellHead, dNext,
                W, H,
                cellSize, gridW, gridH,
                dt,
                gravityG,
                centralGM,
                softening,
                f
            );
        }

        {
            int block = 256;
            int grid = (numCells + block - 1) / block;
            k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_build_grid<<<grid, block, 0, stream>>>(dP, capacity, dCellHead, dNext, W, H, cellSize, gridW, gridH);
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_resolve_collisions<<<grid, block, 0, stream>>>(
                dP, capacity,
                dCellHead, dNext,
                dLocks,
                cellSize, gridW, gridH,
                f
            );
        }

        {
            int block = 256;
            int grid = (capacity + block - 1) / block;
            k_deposit_pixels_buf1<<<grid, block, 0, stream>>>(dP, capacity, dBuf1, W, H, ink);
        }

        {
            int block = 128;
            int grid = (capacity + block - 1) / block;
            k_draw_blobs_buf2<<<grid, block, 0, stream>>>(dP, capacity, dBuf2, W, H, blobIntensity);
        }

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf1, dBuf2, dBGR, Npix, exposure, lift, gammaInv);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix * 3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cv::Mat frame(H, W, CV_8UC3, hFramePinned);
        cv::imshow("Realistic Star System Stable", frame);

        int key = cv::waitKey(1);
        if(key == 27){
            fprintf(stderr, "\nESC pressed. Stopping early.\n");
            stopRequested = true;
        }

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix * 3, ff);
        if(written != (size_t)Npix * 3){
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", f);
            break;
        }

        framesDone = f + 1;

        if((f % 60) == 0){
            fprintf(stderr, "\rFrame %d / %d", f, totalFrames);
            fflush(stderr);
        }

        if(stopRequested) break;
    }

    fprintf(stderr, "\nFinalizing encode...\n");
    fflush(ff);
    pclose(ff);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "Done. Time: %.2f s (effective %.2f fps)\n",
            sec, (sec > 0.0 ? (double)framesDone / sec : 0.0));

    cv::destroyAllWindows();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));

    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dCellHead));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dLocks));
    CUDA_CHECK(cudaFree(dBuf1));
    CUDA_CHECK(cudaFree(dBuf2));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
