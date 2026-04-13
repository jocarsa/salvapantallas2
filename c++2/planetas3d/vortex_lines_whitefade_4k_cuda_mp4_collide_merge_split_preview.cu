// vortex_lines_blackfade_4k_cuda_mp4_collide_merge_split_preview_energycolor_reset_nvencfix_3d.cu
//
// 3D version:
// - Adds real Z dimension to physics and collisions
// - Orthographic render: screen uses X,Y only
// - Depth visibility: nearest fragments win using a per-frame z-buffer
// - Endless segments: each new video starts when simulation resets
// - Output filename: "particle simulation creating planet [datetime].mp4"
//
// Build:
//   nvcc -O3 -std=c++17 vortex_lines_blackfade_4k_cuda_mp4_collide_merge_split_preview_energycolor_reset_nvencfix_3d.cu \
//     -o vortex_energycolor_reset_3d $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_energycolor_reset_3d
//   ./vortex_energycolor_reset_3d /path/to/outdir
//   ./vortex_energycolor_reset_3d /path/to/outdir libx265

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);         \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}

struct Particle {
    float x, y, z;
    float x2, y2, z2;
    float vx, vy, vz;

    // kept for compatibility
    uint8_t r, g, b;

    float m;      // 0..10 => mass = m+1

    // Energy channels:
    float heat;   // collision/thermal (CPU accumulates, decays)
    float press;  // pressure proxy (GPU)
    float ke;     // kinetic proxy (GPU)
};

static inline float frand(std::mt19937& rng, float a, float b){
    std::uniform_real_distribution<float> d(a, b);
    return d(rng);
}
static inline int irand(std::mt19937& rng, int a, int b){
    std::uniform_int_distribution<int> d(a, b);
    return d(rng);
}

// ----------------------- Filename helper -----------------------
static std::string current_datetime_string(){
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tmv, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

static std::string join_path(const std::string& dir, const std::string& file){
    if(dir.empty()) return file;
    char last = dir.back();
    if(last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

static std::string make_segment_outpath(const std::string& outDir){
    std::string dt = current_datetime_string();
    std::string filename = "particle simulation creating planet " + dt + ".mp4";
    return join_path(outDir, filename);
}

// ----------------------- FFmpeg pipe -----------------------
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
            "-pix_fmt yuv420p "
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
            "-vf format=nv12 "
            "-c:v hevc_nvenc "
            "-preset p1 "
            "-tune ll "
            "-rc vbr -cq 28 -b:v 0 "
            "-bf 0 "
            "-rc-lookahead 0 "
            "-spatial_aq 0 -temporal_aq 0 "
            "-surfaces 4 "
            "-pix_fmt yuv420p "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if(!pipe){
        fprintf(stderr, "Failed to start ffmpeg. Command:\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

// ----------------------- Constants for 3D render -----------------------
__device__ __constant__ float KE_W;
__device__ __constant__ float P_W;
__device__ __constant__ float H_W;
__device__ __constant__ float KE_NORM;
__device__ __constant__ float P_NORM;
__device__ __constant__ float H_NORM;

// Z range used by z-buffer normalization.
// More positive z = nearer to camera.
// More negative z = farther from camera.
__device__ __constant__ float Z_NEAR_WORLD;
__device__ __constant__ float Z_FAR_WORLD;

__device__ __forceinline__ uint32_t depth_to_key(float z){
    float f = (z - Z_FAR_WORLD) / (Z_NEAR_WORLD - Z_FAR_WORLD);
    f = clampf(f, 0.0f, 1.0f);
    return (uint32_t)(f * 4294967295.0f);
}

// ----------------------- Kernels -----------------------
__global__ void k_clear_black(float3* buf, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    buf[i] = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void k_clear_depth(uint32_t* depthBuf, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    depthBuf[i] = 0u;
}

__global__ void k_fade_to_black(float3* buf, int Npix, float mul){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    float3 c = buf[i];
    c.x *= mul;
    c.y *= mul;
    c.z *= mul;
    buf[i] = c;
}

__device__ __forceinline__ float3 hsv_to_rgb(float h, float s, float v){
    h = h - floorf(h);
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;

    float3 rgb;
    float hh = h * 6.0f;
    if      (hh < 1.0f) rgb = make_float3(c, x, 0.0f);
    else if (hh < 2.0f) rgb = make_float3(x, c, 0.0f);
    else if (hh < 3.0f) rgb = make_float3(0.0f, c, x);
    else if (hh < 4.0f) rgb = make_float3(0.0f, x, c);
    else if (hh < 5.0f) rgb = make_float3(x, 0.0f, c);
    else                rgb = make_float3(c, 0.0f, x);

    rgb.x += m; rgb.y += m; rgb.z += m;
    return rgb;
}

__global__ void k_step_particles(Particle* p, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    me.x2 = me.x;
    me.y2 = me.y;
    me.z2 = me.z;

    me.x += me.vx;
    me.y += me.vy;
    me.z += me.vz;

    float dvx = 0.0f, dvy = 0.0f, dvz = 0.0f;
    float press = 0.0f;

    for(int j=0; j<n; j++){
        if(j == i) continue;
        Particle pj = p[j];

        float dx = pj.x - me.x;
        float dy = pj.y - me.y;
        float dz = pj.z - me.z;

        if(fabsf(dx) < 3500.0f && fabsf(dy) < 3500.0f && fabsf(dz) < 3500.0f){
            float d2 = dx*dx + dy*dy + dz*dz;
            float inv = 1.0f / (d2 + 4.0f);
            float f = 0.002f * (pj.m + 1.0f) * inv;
            dvx += dx * f;
            dvy += dy * f;
            dvz += dz * f;

            press += (pj.m + 1.0f) * inv;
        }
    }

    me.vx += dvx;
    me.vy += dvy;
    me.vz += dvz;

    float v2 = me.vx*me.vx + me.vy*me.vy + me.vz*me.vz;
    me.ke = v2;
    me.press = press;

    p[i] = me;
}

__device__ __forceinline__ float radius_from_m(float m){
    return 1.0f + 0.5f * m;
}

__device__ __forceinline__ void stamp_disc_depth(
    float3* buf,
    uint32_t* depthBuf,
    int W, int H,
    int cx, int cy,
    int rad,
    float z,
    float3 col
){
    int x0 = max(0, cx - rad);
    int x1 = min(W - 1, cx + rad);
    int y0 = max(0, cy - rad);
    int y1 = min(H - 1, cy + rad);

    int r2 = rad * rad;
    uint32_t zKey = depth_to_key(z);

    for(int y=y0; y<=y1; y++){
        int dy = y - cy;
        for(int x=x0; x<=x1; x++){
            int dx = x - cx;
            if(dx*dx + dy*dy > r2) continue;

            int idx = y * W + x;

            uint32_t old = atomicMax(&depthBuf[idx], zKey);
            if(zKey >= old){
                buf[idx] = col;
            }
        }
    }
}

__global__ void k_draw_lines_energy_ortho_depth(
    const Particle* p, int n,
    float3* buf,
    uint32_t* depthBuf,
    int W, int H
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    float keN = clampf(me.ke    * KE_NORM, 0.0f, 1.0f);
    float prN = clampf(me.press * P_NORM,  0.0f, 1.0f);
    float htN = clampf(me.heat  * H_NORM,  0.0f, 1.0f);

    float t = clampf(KE_W * keN + P_W * prN + H_W * htN, 0.0f, 1.0f);

    float hue = (270.0f / 360.0f) * t;
    float3 col = hsv_to_rgb(hue, 1.0f, 1.0f);

    float x0 = me.x2;
    float y0 = me.y2;
    float z0 = me.z2;

    float x1 = me.x;
    float y1 = me.y;
    float z1 = me.z;

    float radf = radius_from_m(me.m);
    int rad = (int)ceilf(radf);

    float dx = x1 - x0;
    float dy = y1 - y0;
    float dz = z1 - z0;

    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    int steps = (int)ceilf(len);
    steps = max(1, min(steps, 4096));

    float inv = 1.0f / (float)steps;
    for(int s=0; s<=steps; s++){
        float tt = s * inv;

        int cx = (int)lrintf(x0 + dx * tt);
        int cy = (int)lrintf(y0 + dy * tt);
        float cz = z0 + dz * tt;

        if(cx < 0 || cx >= W || cy < 0 || cy >= H) continue;

        stamp_disc_depth(buf, depthBuf, W, H, cx, cy, rad, cz, col);
    }
}

__global__ void k_compose_to_bgr(
    const float3* buf,
    unsigned char* outBGR,
    int Npix,
    float gammaInv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float r = clampf(buf[i].x, 0.0f, 1.0f);
    float g = clampf(buf[i].y, 0.0f, 1.0f);
    float b = clampf(buf[i].z, 0.0f, 1.0f);

    r = powf(r, gammaInv);
    g = powf(g, gammaInv);
    b = powf(b, gammaInv);

    outBGR[3*i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3*i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3*i + 2] = (unsigned char)lrintf(255.0f * r);
}

// ----------------------- Host particle constructor -----------------------
static Particle make_particle(std::mt19937& rng, int W, int H, float D){
    Particle p{};

    p.x = frand(rng, 0.0f, (float)W);
    p.y = frand(rng, 0.0f, (float)H);
    p.z = frand(rng, -0.5f * D, 0.5f * D);

    p.x2 = p.x;
    p.y2 = p.y;
    p.z2 = p.z;

    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;
    float cz = 0.0f;

    float dx = cx - p.x;
    float dy = cy - p.y;
    float dz = cz - p.z;

    float dlen = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
    dx /= dlen; dy /= dlen; dz /= dlen;

    // 3D swirl around the center.
    // Tangential velocity in XY plus a little Z motion.
    float tangx = -dy;
    float tangy =  dx;
    float tangz =  frand(rng, -0.35f, 0.35f);

    float radial = frand(rng, -0.20f, 0.20f);
    tangx += radial * dx;
    tangy += radial * dy;
    tangz += radial * dz;

    float slen = sqrtf(tangx*tangx + tangy*tangy + tangz*tangz) + 1e-6f;
    tangx /= slen; tangy /= slen; tangz /= slen;

    float sp = frand(rng, 0.15f, 0.85f);
    p.vx = tangx * sp;
    p.vy = tangy * sp;
    p.vz = tangz * sp;

    p.r = (uint8_t)irand(rng, 64, 255);
    p.g = (uint8_t)irand(rng, 64, 255);
    p.b = (uint8_t)irand(rng, 64, 255);

    p.m = frand(rng, 0.0f, 10.0f);

    p.heat  = 0.0f;
    p.press = 0.0f;
    p.ke    = 0.0f;

    return p;
}

static inline void set_spawn_velocity(Particle& p, std::mt19937& rng, int W, int H, float D, float baseScale){
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;
    float cz = 0.0f;

    float dx = cx - p.x;
    float dy = cy - p.y;
    float dz = cz - p.z;

    float dlen = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
    dx /= dlen; dy /= dlen; dz /= dlen;

    float tangx = -dy;
    float tangy =  dx;
    float tangz =  frand(rng, -0.50f, 0.50f);

    tangx += frand(rng, -0.50f, 0.50f);
    tangy += frand(rng, -0.50f, 0.50f);
    tangz += frand(rng, -0.50f, 0.50f);

    float radial = frand(rng, -0.35f, 0.35f);
    tangx += radial * dx;
    tangy += radial * dy;
    tangz += radial * dz;

    float slen = sqrtf(tangx*tangx + tangy*tangy + tangz*tangz) + 1e-6f;
    tangx /= slen; tangy /= slen; tangz /= slen;

    float sp = frand(rng, 0.20f, 1.10f) * baseScale;

    float mix = frand(rng, 0.0f, 0.35f);
    p.vx = (1.0f - mix) * (tangx * sp) + mix * p.vx;
    p.vy = (1.0f - mix) * (tangy * sp) + mix * p.vy;
    p.vz = (1.0f - mix) * (tangz * sp) + mix * p.vz;
}

// ----------------------- CPU collision / merge / split -----------------------
static inline float radius_from_m_cpu(float m){ return 1.0f + 0.5f * m; }
static inline float mass_from_m(float m){ return m + 1.0f; }

static void collide_merge_split_cpu(std::vector<Particle>& p, int W, int H, float D, std::mt19937& rng, int maxParticles){
    const float restitution = 0.75f;
    const float mergeSpeed  = 0.25f;
    const float splitSpeed  = 1.20f;
    const float maxMass     = 30.0f;
    const float minSplitMass= 6.0f;
    const float mergeProb   = 0.65f;
    const float splitProb   = 0.40f;

    const float heatGain = 0.65f;
    const float heatOverlapGain = 0.15f;
    const float heatClamp = 20.0f;
    const float heatDecay = 0.985f;

    const int n = (int)p.size();
    if(n < 2){
        for(auto &Q : p) Q.heat *= heatDecay;
        return;
    }

    std::vector<uint8_t> dead((size_t)n, 0);

    for(int i=0;i<n;i++){
        if(dead[i]) continue;
        for(int j=i+1;j<n;j++){
            if(dead[j]) continue;

            Particle &A = p[i];
            Particle &B = p[j];

            float rA = radius_from_m_cpu(A.m);
            float rB = radius_from_m_cpu(B.m);

            float dx = B.x - A.x;
            float dy = B.y - A.y;
            float dz = B.z - A.z;

            float dist2 = dx*dx + dy*dy + dz*dz;
            float minDist = rA + rB;

            if(dist2 > minDist*minDist) continue;

            float dist = sqrtf(std::max(dist2, 1e-8f));
            float nx = dx / dist;
            float ny = dy / dist;
            float nz = dz / dist;

            float rvx = B.vx - A.vx;
            float rvy = B.vy - A.vy;
            float rvz = B.vz - A.vz;
            float relSpeed = sqrtf(rvx*rvx + rvy*rvy + rvz*rvz);

            float mA = mass_from_m(A.m);
            float mB = mass_from_m(B.m);

            float overlap = (minDist - dist);
            if(overlap > 0.0f){
                float invSum = 1.0f / (mA + mB);
                A.x -= nx * overlap * (mB * invSum);
                A.y -= ny * overlap * (mB * invSum);
                A.z -= nz * overlap * (mB * invSum);

                B.x += nx * overlap * (mA * invSum);
                B.y += ny * overlap * (mA * invSum);
                B.z += nz * overlap * (mA * invSum);
            }

            {
                float add = heatGain * relSpeed + heatOverlapGain * fmaxf(overlap, 0.0f);
                float wA = mB / (mA + mB);
                float wB = mA / (mA + mB);
                A.heat = std::min(heatClamp, A.heat + add * wA);
                B.heat = std::min(heatClamp, B.heat + add * wB);
            }

            float u = frand(rng, 0.0f, 1.0f);

            if(relSpeed < mergeSpeed && (mA + mB) <= maxMass && u < mergeProb){
                Particle C{};
                float mC = mA + mB;
                float wA = mA / mC;
                float wB = mB / mC;

                C.x  = A.x * wA + B.x * wB;
                C.y  = A.y * wA + B.y * wB;
                C.z  = A.z * wA + B.z * wB;

                C.x2 = C.x;
                C.y2 = C.y;
                C.z2 = C.z;

                C.vx = A.vx * wA + B.vx * wB;
                C.vy = A.vy * wA + B.vy * wB;
                C.vz = A.vz * wA + B.vz * wB;

                C.r = (uint8_t)std::clamp((int)lrintf(A.r * wA + B.r * wB), 0, 255);
                C.g = (uint8_t)std::clamp((int)lrintf(A.g * wA + B.g * wB), 0, 255);
                C.b = (uint8_t)std::clamp((int)lrintf(A.b * wA + B.b * wB), 0, 255);

                C.m = std::clamp(mC - 1.0f, 0.0f, 30.0f);

                C.heat  = std::min(heatClamp, A.heat * wA + B.heat * wB);
                C.press = 0.0f;
                C.ke    = 0.0f;

                A = C;
                dead[j] = 1;
                continue;
            }

            if(relSpeed > splitSpeed && (int)p.size() < maxParticles){
                int heavy = (mA >= mB) ? i : j;
                Particle &Hh = p[heavy];
                float mH = mass_from_m(Hh.m);

                float uu = frand(rng, 0.0f, 1.0f);
                if(mH >= minSplitMass && uu < splitProb && (int)p.size() < maxParticles){
                    Particle S = Hh;

                    float m1 = 0.55f * mH;
                    float m2 = mH - m1;

                    Hh.m = std::clamp(m1 - 1.0f, 0.0f, 30.0f);
                    S.m  = std::clamp(m2 - 1.0f, 0.0f, 30.0f);

                    // A perpendicular-ish kick in 3D.
                    float ax = frand(rng, -1.0f, 1.0f);
                    float ay = frand(rng, -1.0f, 1.0f);
                    float az = frand(rng, -1.0f, 1.0f);

                    float px = ny * az - nz * ay;
                    float py = nz * ax - nx * az;
                    float pz = nx * ay - ny * ax;

                    float plen = sqrtf(px*px + py*py + pz*pz);
                    if(plen < 1e-6f){
                        px = -ny; py = nx; pz = 0.0f;
                        plen = sqrtf(px*px + py*py + pz*pz) + 1e-6f;
                    }
                    px /= plen; py /= plen; pz /= plen;

                    float kick = 0.35f * relSpeed;
                    Hh.vx += px * kick;
                    Hh.vy += py * kick;
                    Hh.vz += pz * kick;

                    S.vx  -= px * kick;
                    S.vy  -= py * kick;
                    S.vz  -= pz * kick;

                    float sep = radius_from_m_cpu(Hh.m) + radius_from_m_cpu(S.m) + 1.0f;
                    S.x += nx * sep;
                    S.y += ny * sep;
                    S.z += nz * sep;
                    S.x2 = S.x;
                    S.y2 = S.y;
                    S.z2 = S.z;

                    float hh = Hh.heat;
                    Hh.heat = std::min(heatClamp, hh * 0.55f);
                    S.heat  = std::min(heatClamp, hh * 0.45f);

                    p.push_back(S);
                }
            }

            float vn = rvx*nx + rvy*ny + rvz*nz;
            if(vn > 0.0f){
                continue;
            }

            float invA = 1.0f / std::max(mA, 1e-6f);
            float invB = 1.0f / std::max(mB, 1e-6f);

            float jImpulse = -(1.0f + restitution) * vn / (invA + invB);
            float impX = jImpulse * nx;
            float impY = jImpulse * ny;
            float impZ = jImpulse * nz;

            A.vx -= impX * invA;
            A.vy -= impY * invA;
            A.vz -= impZ * invA;

            B.vx += impX * invB;
            B.vy += impY * invB;
            B.vz += impZ * invB;

            A.vx *= 0.995f; A.vy *= 0.995f; A.vz *= 0.995f;
            B.vx *= 0.995f; B.vy *= 0.995f; B.vz *= 0.995f;

            auto clampVel = [](Particle& Q){
                Q.vx = std::clamp(Q.vx, -8.0f, 8.0f);
                Q.vy = std::clamp(Q.vy, -8.0f, 8.0f);
                Q.vz = std::clamp(Q.vz, -8.0f, 8.0f);
            };
            clampVel(A);
            clampVel(B);
        }
    }

    if(std::any_of(dead.begin(), dead.end(), [](uint8_t v){return v!=0;})){
        std::vector<Particle> out;
        out.reserve(p.size());
        for(size_t i=0;i<p.size();i++){
            if(i < dead.size() && dead[i]) continue;
            out.push_back(p[i]);
        }
        p.swap(out);
    }

    for(auto &Q : p){
        Q.heat *= heatDecay;
        if(!std::isfinite(Q.x) || !std::isfinite(Q.y) || !std::isfinite(Q.z) ||
           !std::isfinite(Q.vx) || !std::isfinite(Q.vy) || !std::isfinite(Q.vz)){
            Q = make_particle(rng, W, H, D);
        }
    }
}

// ----------------------- Reset helper -----------------------
static void reset_simulation(
    std::vector<Particle>& hP,
    int startParticles,
    int maxParticles,
    int W, int H, float D,
    std::mt19937& rng,
    float3* dBuf,
    int Npix,
    cudaStream_t stream
){
    hP.clear();
    hP.reserve(maxParticles + 256);
    for(int i=0; i<startParticles; i++){
        hP.push_back(make_particle(rng, W, H, D));
    }

    int block = 256;
    int grid  = (Npix + block - 1) / block;
    k_clear_black<<<grid, block, 0, stream>>>(dBuf, Npix);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------- Main -----------------------
int main(int argc, char** argv){
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;
    const float D = 2160.0f; // world depth span used for 3D simulation

    const int Npix = W * H;

    std::string outDir = "";
    std::string encoder = "nvenc";
    if(argc >= 2) outDir = argv[1];
    if(argc >= 3) encoder = argv[2];

    fprintf(stderr, "Res: %dx%d | FPS: %d\n", W, H, fps);
    fprintf(stderr, "Depth world: %.2f\n", D);
    fprintf(stderr, "Encoder: %s\n", encoder.c_str());
    if(!outDir.empty()) fprintf(stderr, "Output dir: %s\n", outDir.c_str());
    fprintf(stderr, "Segments: NEW MP4 each time simulation resets.\n");
    fprintf(stderr, "Render: 3D physics + orthographic XY + z-buffer visibility.\n");
    fprintf(stderr, "Press ESC to stop.\n");

    const int startParticles = 100;
    const int maxParticles   = 3000;

    const float fadeAlpha = 0.02f;
    const float fadeMul   = 1.0f - fadeAlpha;

    const float gammaInv = 1.0f / 2.2f;

    const float h_KE_W    = 0.50f;
    const float h_P_W     = 0.30f;
    const float h_H_W     = 0.20f;

    const float h_KE_NORM = 0.20f;
    const float h_P_NORM  = 1.50f;
    const float h_H_NORM  = 0.12f;

    const float h_Z_NEAR_WORLD =  D * 0.5f;
    const float h_Z_FAR_WORLD  = -D * 0.5f;

    CUDA_CHECK(cudaMemcpyToSymbol(KE_W,        &h_KE_W,         sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(P_W,         &h_P_W,          sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(H_W,         &h_H_W,          sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(KE_NORM,     &h_KE_NORM,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(P_NORM,      &h_P_NORM,       sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(H_NORM,      &h_H_NORM,       sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(Z_NEAR_WORLD,&h_Z_NEAR_WORLD, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(Z_FAR_WORLD, &h_Z_FAR_WORLD,  sizeof(float)));

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<Particle> hP;
    hP.reserve(maxParticles + 256);
    for(int i=0; i<startParticles; i++) hP.push_back(make_particle(rng, W, H, D));

    Particle* dP = nullptr;
    float3* dBuf = nullptr;
    unsigned char* dBGR = nullptr;
    uint32_t* dDepth = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxParticles));
    CUDA_CHECK(cudaMalloc(&dBuf, sizeof(float3) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));
    CUDA_CHECK(cudaMalloc(&dDepth, sizeof(uint32_t) * (size_t)Npix));

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::namedWindow("preview", cv::WINDOW_NORMAL);
    cv::resizeWindow("preview", 1280, 720);
    cv::Mat preview(H, W, CV_8UC3, hFramePinned);

    bool stopAll = false;
    long long segmentIndex = 0;

    while(!stopAll){
        std::string outPath = make_segment_outpath(outDir);

        fprintf(stderr, "\n============================================================\n");
        fprintf(stderr, "Starting segment #%lld\n", segmentIndex);
        fprintf(stderr, "Output: %s\n", outPath.c_str());
        fprintf(stderr, "============================================================\n");

        reset_simulation(hP, startParticles, maxParticles, W, H, D, rng, dBuf, Npix, stream);
        int simFrame = 0;

        FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
        if(!ff){
            fprintf(stderr, "Could not open FFmpeg pipe.\n");
            break;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        long long frame = 0;

        while(true){
            bool needResetSegment = false;

            if(simFrame > 0 && (simFrame % 100) == 0){
                if((int)hP.size() >= maxParticles){
                    needResetSegment = true;
                } else {
                    int tempIndex = (int)hP.size();
                    hP.push_back(make_particle(rng, W, H, D));
                    Particle parent = hP[tempIndex];
                    set_spawn_velocity(hP[tempIndex], rng, W, H, D, 1.0f);

                    for(int k=0; k<20; k++){
                        if((int)hP.size() >= maxParticles){
                            needResetSegment = true;
                            break;
                        }
                        Particle c = make_particle(rng, W, H, D);

                        float randx = (frand(rng, -0.5f, 0.5f)) * 20.0f;
                        float randy = (frand(rng, -0.5f, 0.5f)) * 20.0f;
                        float randz = (frand(rng, -0.5f, 0.5f)) * 20.0f;

                        c.x  = parent.x  + randx;
                        c.y  = parent.y  + randy;
                        c.z  = parent.z  + randz;

                        c.x2 = c.x;
                        c.y2 = c.y;
                        c.z2 = c.z;

                        c.vx = parent.vx;
                        c.vy = parent.vy;
                        c.vz = parent.vz;
                        set_spawn_velocity(c, rng, W, H, D, 1.15f);

                        c.heat  = 0.0f;
                        c.press = 0.0f;
                        c.ke    = 0.0f;

                        hP.push_back(c);
                    }
                }
            }

            int N = (int)hP.size();
            if(N <= 0){
                needResetSegment = true;
                N = 0;
            }
            if(N > maxParticles){
                needResetSegment = true;
                N = std::min(N, maxParticles);
            }

            if(needResetSegment){
                break;
            }

            CUDA_CHECK(cudaMemcpyAsync(dP, hP.data(), sizeof(Particle)*(size_t)N, cudaMemcpyHostToDevice, stream));

            {
                int block = 256;
                int grid = (Npix + block - 1) / block;
                k_fade_to_black<<<grid, block, 0, stream>>>(dBuf, Npix, fadeMul);
                k_clear_depth<<<grid, block, 0, stream>>>(dDepth, Npix);
            }

            {
                int block = 128;
                int grid = (N + block - 1) / block;
                k_step_particles<<<grid, block, 0, stream>>>(dP, N);
            }

            {
                int block = 64;
                int grid = (N + block - 1) / block;
                k_draw_lines_energy_ortho_depth<<<grid, block, 0, stream>>>(dP, N, dBuf, dDepth, W, H);
            }

            {
                int block = 256;
                int grid = (Npix + block - 1) / block;
                k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf, dBGR, Npix, gammaInv);
            }

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(hP.data(), dP, sizeof(Particle)*(size_t)N, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cv::imshow("preview", preview);
            int key = cv::waitKey(1);
            if(key == 27){
                stopAll = true;
            }

            collide_merge_split_cpu(hP, W, H, D, rng, maxParticles);

            hP.erase(
                std::remove_if(hP.begin(), hP.end(), [&](const Particle& q){
                    return (q.x < -0.5f * W) || (q.x > 1.5f * W) ||
                           (q.y < -0.5f * H) || (q.y > 1.5f * H) ||
                           (q.z < -1.0f * D) || (q.z >  1.0f * D);
                }),
                hP.end()
            );

            if((int)hP.size() >= maxParticles){
                needResetSegment = true;
            } else {
                simFrame++;
            }

            size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
            if(written != (size_t)Npix*3){
                fprintf(stderr, "\nFFmpeg write failed at frame %lld.\n", frame);
                stopAll = true;
                break;
            }

            if((frame % 60) == 0){
                fprintf(stderr, "\rSegment #%lld | Frame %lld | Particles: %d | SimFrame: %d   ",
                        segmentIndex, frame, (int)hP.size(), simFrame);
                fflush(stderr);
            }

            frame++;

            if(stopAll) break;
            if(needResetSegment) break;
        }

        fprintf(stderr, "\nFinalizing encode...\n");
        fflush(ff);
        pclose(ff);

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double effFps = (sec > 1e-9) ? ((double)frame / sec) : 0.0;
        fprintf(stderr, "Segment #%lld done. Frames: %lld | Time: %.2f s | %.2f fps effective\n",
                segmentIndex, frame, sec, effFps);

        segmentIndex++;

        if(stopAll){
            fprintf(stderr, "ESC received. Stopping after segment finalize.\n");
            break;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));

    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dBGR));
    CUDA_CHECK(cudaFree(dDepth));

    return 0;
}
