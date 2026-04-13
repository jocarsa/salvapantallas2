// vortex_lines_blackfade_4k_cuda_mp4_collide_merge_split_preview_energycolor_reset_nvencfix.cu
//
// CHANGES requested:
// 1) Output filename is ALWAYS: "particle simulation creating planet [datetime].mp4"
// 2) [datetime] is replaced with the actual current datetime at the moment each video starts
// 3) Program runs in an endless loop, generating 1-hour videos back-to-back (until you press ESC)
//
// Build:
//   nvcc -O3 -std=c++17 vortex_lines_blackfade_4k_cuda_mp4_collide_merge_split_preview_energycolor_reset_nvencfix.cu \
//     -o vortex_energycolor_reset $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_energycolor_reset                 (endless 1h segments, nvenc, current folder)
//   ./vortex_energycolor_reset /path/to/outdir (endless 1h segments, nvenc, output directory)
//   ./vortex_energycolor_reset /path/to/outdir libx265 (endless 1h segments, encoder = libx265 or nvenc)
//
// Notes:
// - ESC stops after finishing the current frame loop iteration (it will finalize the current mp4 and exit).
// - Each segment begins with a clean simulation reset (particles + framebuffer + timeline).

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
                cudaGetErrorString(err), __FILE__, __LINE__);        \
        std::exit(1);                                                \
    }                                                                 \
} while (0)

__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}

struct Particle {
    float x, y;
    float x2, y2;
    float vx, vy;

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
    // Format: YYYY-MM-DD_HH-MM-SS (safe for filenames)
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
// IMPORTANT: NVENC branch uses LOW-MEM settings to avoid "CreateInputBuffer failed: out of memory".
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
        // Low-memory NVENC settings:
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

// ----------------------- Kernels -----------------------

__global__ void k_clear_black(float3* buf, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    buf[i] = make_float3(0.0f, 0.0f, 0.0f);
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

// HSV -> RGB (h in [0,1), s,v in [0,1])
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

    me.x += me.vx;
    me.y += me.vy;

    float dvx = 0.0f, dvy = 0.0f;
    float press = 0.0f;

    for(int j=0; j<n; j++){
        if(j == i) continue;
        Particle pj = p[j];

        float dx = pj.x - me.x;
        float dy = pj.y - me.y;

        if(fabsf(dx) < 3500.0f && fabsf(dy) < 3500.0f){
            float d2 = dx*dx + dy*dy;
            float inv = 1.0f / (d2 + 4.0f);
            float f = 0.002f * (pj.m + 1.0f) * inv;
            dvx += dx * f;
            dvy += dy * f;

            press += (pj.m + 1.0f) * inv;
        }
    }

    me.vx += dvx;
    me.vy += dvy;

    float v2 = me.vx*me.vx + me.vy*me.vy;
    me.ke = v2;
    me.press = press;

    p[i] = me;
}

__device__ __forceinline__ void stamp_disc(float3* buf, int W, int H, int cx, int cy, int rad, float3 col){
    int x0 = max(0, cx - rad);
    int x1 = min(W - 1, cx + rad);
    int y0 = max(0, cy - rad);
    int y1 = min(H - 1, cy + rad);

    int r2 = rad * rad;

    for(int y=y0; y<=y1; y++){
        int dy = y - cy;
        for(int x=x0; x<=x1; x++){
            int dx = x - cx;
            if(dx*dx + dy*dy > r2) continue;
            buf[y * W + x] = col; // overwrite
        }
    }
}

__device__ __forceinline__ float radius_from_m(float m){
    return 1.0f + 0.5f * m;
}

__device__ __constant__ float KE_W;
__device__ __constant__ float P_W;
__device__ __constant__ float H_W;
__device__ __constant__ float KE_NORM;
__device__ __constant__ float P_NORM;
__device__ __constant__ float H_NORM;

__global__ void k_draw_lines_energy(
    const Particle* p, int n,
    float3* buf,
    int W, int H
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    float keN = clampf(me.ke    * KE_NORM, 0.0f, 1.0f);
    float prN = clampf(me.press * P_NORM,  0.0f, 1.0f);
    float htN = clampf(me.heat  * H_NORM,  0.0f, 1.0f);

    float t = clampf(KE_W * keN + P_W * prN + H_W * htN, 0.0f, 1.0f);

    float hue = (270.0f / 360.0f) * t; // 0..0.75
    float3 col = hsv_to_rgb(hue, 1.0f, 1.0f);

    float x0 = me.x2;
    float y0 = me.y2;
    float x1 = me.x;
    float y1 = me.y;

    float radf = radius_from_m(me.m);
    int rad = (int)ceilf(radf);

    float dx = x1 - x0;
    float dy = y1 - y0;

    float len = sqrtf(dx*dx + dy*dy);
    int steps = (int)ceilf(len);
    steps = max(1, min(steps, 4096));

    float inv = 1.0f / (float)steps;
    for(int s=0; s<=steps; s++){
        float tt = s * inv;
        int cx = (int)lrintf(x0 + dx * tt);
        int cy = (int)lrintf(y0 + dy * tt);
        if(cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
        stamp_disc(buf, W, H, cx, cy, rad, col);
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
static Particle make_particle(std::mt19937& rng, int W, int H){
    Particle p{};
    p.x = frand(rng, 0.0f, (float)W);
    p.y = frand(rng, 0.0f, (float)H);
    p.x2 = p.x;
    p.y2 = p.y;

    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;

    float toC  = atan2f(cy - p.y, cx - p.x);
    float tang = toC + 3.1415926535f * 0.5f;

    tang += frand(rng, -0.65f, 0.65f);

    float sp = frand(rng, 0.15f, 0.85f);
    float radial = frand(rng, -0.20f, 0.20f);

    float dirx = cosf(tang) + radial * cosf(toC);
    float diry = sinf(tang) + radial * sinf(toC);

    float dlen = sqrtf(dirx*dirx + diry*diry) + 1e-6f;
    dirx /= dlen; diry /= dlen;

    p.vx = dirx * sp;
    p.vy = diry * sp;

    p.r = (uint8_t)irand(rng, 64, 255);
    p.g = (uint8_t)irand(rng, 64, 255);
    p.b = (uint8_t)irand(rng, 64, 255);

    p.m = frand(rng, 0.0f, 10.0f);

    p.heat  = 0.0f;
    p.press = 0.0f;
    p.ke    = 0.0f;

    return p;
}

static inline void set_spawn_velocity(Particle& p, std::mt19937& rng, int W, int H, float baseScale){
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;

    float toC  = atan2f(cy - p.y, cx - p.x);
    float tang = toC + 3.1415926535f * 0.5f;

    tang += frand(rng, -0.85f, 0.85f);
    float sp = frand(rng, 0.20f, 1.10f) * baseScale;

    float radial = frand(rng, -0.35f, 0.35f);
    float dirx = cosf(tang) + radial * cosf(toC);
    float diry = sinf(tang) + radial * sinf(toC);

    float dlen = sqrtf(dirx*dirx + diry*diry) + 1e-6f;
    dirx /= dlen; diry /= dlen;

    float mix = frand(rng, 0.0f, 0.35f);
    p.vx = (1.0f - mix) * (dirx * sp) + mix * p.vx;
    p.vy = (1.0f - mix) * (diry * sp) + mix * p.vy;
}

// ----------------------- CPU collision / merge / split -----------------------
static inline float radius_from_m_cpu(float m){ return 1.0f + 0.5f * m; }
static inline float mass_from_m(float m){ return m + 1.0f; }

static void collide_merge_split_cpu(std::vector<Particle>& p, int W, int H, std::mt19937& rng, int maxParticles){
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
            float dist2 = dx*dx + dy*dy;
            float minDist = rA + rB;

            if(dist2 > minDist*minDist) continue;

            float dist = sqrtf(std::max(dist2, 1e-8f));
            float nx = dx / dist;
            float ny = dy / dist;

            float rvx = B.vx - A.vx;
            float rvy = B.vy - A.vy;
            float relSpeed = sqrtf(rvx*rvx + rvy*rvy);

            float mA = mass_from_m(A.m);
            float mB = mass_from_m(B.m);

            float overlap = (minDist - dist);
            if(overlap > 0.0f){
                float invSum = 1.0f / (mA + mB);
                A.x -= nx * overlap * (mB * invSum);
                A.y -= ny * overlap * (mB * invSum);
                B.x += nx * overlap * (mA * invSum);
                B.y += ny * overlap * (mA * invSum);
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
                C.x2 = C.x;
                C.y2 = C.y;

                C.vx = A.vx * wA + B.vx * wB;
                C.vy = A.vy * wA + B.vy * wB;

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

                    float px = -ny;
                    float py = nx;

                    float kick = 0.35f * relSpeed;
                    Hh.vx += px * kick;
                    Hh.vy += py * kick;
                    S.vx  -= px * kick;
                    S.vy  -= py * kick;

                    float sep = radius_from_m_cpu(Hh.m) + radius_from_m_cpu(S.m) + 1.0f;
                    S.x += nx * sep;
                    S.y += ny * sep;
                    S.x2 = S.x;
                    S.y2 = S.y;

                    float hh = Hh.heat;
                    Hh.heat = std::min(heatClamp, hh * 0.55f);
                    S.heat  = std::min(heatClamp, hh * 0.45f);

                    p.push_back(S);
                }
            }

            float vn = rvx*nx + rvy*ny;
            if(vn > 0.0f){
                continue;
            }

            float invA = 1.0f / std::max(mA, 1e-6f);
            float invB = 1.0f / std::max(mB, 1e-6f);

            float jImpulse = -(1.0f + restitution) * vn / (invA + invB);
            float impX = jImpulse * nx;
            float impY = jImpulse * ny;

            A.vx -= impX * invA;
            A.vy -= impY * invA;
            B.vx += impX * invB;
            B.vy += impY * invB;

            A.vx *= 0.995f; A.vy *= 0.995f;
            B.vx *= 0.995f; B.vy *= 0.995f;

            auto clampVel = [](Particle& Q){
                Q.vx = std::clamp(Q.vx, -8.0f, 8.0f);
                Q.vy = std::clamp(Q.vy, -8.0f, 8.0f);
            };
            clampVel(A); clampVel(B);
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
        if(!std::isfinite(Q.x) || !std::isfinite(Q.y) || !std::isfinite(Q.vx) || !std::isfinite(Q.vy)){
            Q = make_particle(rng, W, H);
        }
    }
}

// ----------------------- Reset helper -----------------------
static void reset_simulation(
    std::vector<Particle>& hP,
    int startParticles,
    int maxParticles,
    int W, int H,
    std::mt19937& rng,
    float3* dBuf,
    int Npix,
    cudaStream_t stream
){
    hP.clear();
    hP.reserve(maxParticles + 256);
    for(int i=0; i<startParticles; i++){
        hP.push_back(make_particle(rng, W, H));
    }

    int block = 256;
    int grid  = (Npix + block - 1) / block;
    k_clear_black<<<grid, block, 0, stream>>>(dBuf, Npix);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------- Main -----------------------
int main(int argc, char** argv){
    const int W = 3840;
    const int H = 2160;
    const int fps = 60;

    // Endless loop, always 1-hour segments:
    const int secondsPerVideo = 36000;
    const int framesPerVideo  = secondsPerVideo * fps;
    const int Npix = W * H;

    // Optional args:
    // argv[1] = output directory (optional)
    // argv[2] = encoder: "nvenc" (default) or "libx265"
    std::string outDir = "";
    std::string encoder = "nvenc";
    if(argc >= 2) outDir = argv[1];
    if(argc >= 3) encoder = argv[2];

    fprintf(stderr, "Res: %dx%d | FPS: %d | Segment: %d seconds (%d frames)\n",
            W, H, fps, secondsPerVideo, framesPerVideo);
    fprintf(stderr, "Encoder: %s\n", encoder.c_str());
    if(!outDir.empty()) fprintf(stderr, "Output dir: %s\n", outDir.c_str());
    fprintf(stderr, "Press ESC to stop.\n");

    const int startParticles = 100;
    const int maxParticles   = 1000;

    const float fadeAlpha = 0.02f;
    const float fadeMul   = 1.0f - fadeAlpha;

    const float gammaInv = 1.0f / 2.2f;

    const float h_KE_W    = 0.50f;
    const float h_P_W     = 0.30f;
    const float h_H_W     = 0.20f;

    const float h_KE_NORM = 0.20f;
    const float h_P_NORM  = 1.50f;
    const float h_H_NORM  = 0.12f;

    CUDA_CHECK(cudaMemcpyToSymbol(KE_W,    &h_KE_W,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(P_W,     &h_P_W,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(H_W,     &h_H_W,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(KE_NORM, &h_KE_NORM, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(P_NORM,  &h_P_NORM,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(H_NORM,  &h_H_NORM,  sizeof(float)));

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<Particle> hP;
    hP.reserve(maxParticles + 256);
    for(int i=0; i<startParticles; i++) hP.push_back(make_particle(rng, W, H));

    Particle* dP = nullptr;
    float3* dBuf = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxParticles));
    CUDA_CHECK(cudaMalloc(&dBuf, sizeof(float3) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // OpenCV preview (single window reused across segments)
    cv::namedWindow("preview", cv::WINDOW_NORMAL);
    cv::resizeWindow("preview", 1280, 720);
    cv::Mat preview(H, W, CV_8UC3, hFramePinned);

    bool stopAll = false;
    long long segmentIndex = 0;

    while(!stopAll){
        // Each segment gets a fresh datetime-based filename
        std::string outPath = make_segment_outpath(outDir);

        fprintf(stderr, "\n============================================================\n");
        fprintf(stderr, "Starting segment #%lld\n", segmentIndex);
        fprintf(stderr, "Output: %s\n", outPath.c_str());
        fprintf(stderr, "============================================================\n");

        // Hard reset at the start of each 1-hour video (clean timeline)
        reset_simulation(hP, startParticles, maxParticles, W, H, rng, dBuf, Npix, stream);
        int simFrame = 0;

        FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
        if(!ff){
            fprintf(stderr, "Could not open FFmpeg pipe.\n");
            break;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        for(int frame=0; frame<framesPerVideo; frame++){
            if((int)hP.size() >= maxParticles){
                reset_simulation(hP, startParticles, maxParticles, W, H, rng, dBuf, Npix, stream);
                simFrame = 0;
            }

            if(simFrame > 0 && (simFrame % 100) == 0 && (int)hP.size() < maxParticles){
                int tempIndex = (int)hP.size();
                hP.push_back(make_particle(rng, W, H));
                Particle parent = hP[tempIndex];
                set_spawn_velocity(hP[tempIndex], rng, W, H, 1.0f);

                for(int k=0; k<20 && (int)hP.size() < maxParticles; k++){
                    Particle c = make_particle(rng, W, H);

                    float randx = (frand(rng, -0.5f, 0.5f)) * 20.0f;
                    float randy = (frand(rng, -0.5f, 0.5f)) * 20.0f;

                    c.x  = parent.x  + randx;
                    c.y  = parent.y  + randy;
                    c.x2 = c.x;
                    c.y2 = c.y;

                    c.vx = parent.vx;
                    c.vy = parent.vy;
                    set_spawn_velocity(c, rng, W, H, 1.15f);

                    c.heat  = 0.0f;
                    c.press = 0.0f;
                    c.ke    = 0.0f;

                    hP.push_back(c);
                }
            }

            int N = (int)hP.size();
            if(N > maxParticles){
                reset_simulation(hP, startParticles, maxParticles, W, H, rng, dBuf, Npix, stream);
                simFrame = 0;
                N = (int)hP.size();
            }

            CUDA_CHECK(cudaMemcpyAsync(dP, hP.data(), sizeof(Particle)*(size_t)N, cudaMemcpyHostToDevice, stream));

            // 1) fade to black
            {
                int block = 256;
                int grid = (Npix + block - 1) / block;
                k_fade_to_black<<<grid, block, 0, stream>>>(dBuf, Npix, fadeMul);
            }

            // 2) step physics
            {
                int block = 128;
                int grid = (N + block - 1) / block;
                k_step_particles<<<grid, block, 0, stream>>>(dP, N);
            }

            // 3) draw
            {
                int block = 64;
                int grid = (N + block - 1) / block;
                k_draw_lines_energy<<<grid, block, 0, stream>>>(dP, N, dBuf, W, H);
            }

            // 4) compose
            {
                int block = 256;
                int grid = (Npix + block - 1) / block;
                k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf, dBGR, Npix, gammaInv);
            }

            CUDA_CHECK(cudaGetLastError());

            // download frame + particles
            CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(hP.data(), dP, sizeof(Particle)*(size_t)N, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cv::imshow("preview", preview);
            int key = cv::waitKey(1);
            if(key == 27){
                stopAll = true; // finish this segment loop and then exit after finalize
            }

            collide_merge_split_cpu(hP, W, H, rng, maxParticles);

            hP.erase(
                std::remove_if(hP.begin(), hP.end(), [&](const Particle& q){
                    return (q.x < -0.5f * W) || (q.x > 1.5f * W) || (q.y < -0.5f * H) || (q.y > 1.5f * H);
                }),
                hP.end()
            );

            if((int)hP.size() >= maxParticles){
                reset_simulation(hP, startParticles, maxParticles, W, H, rng, dBuf, Npix, stream);
                simFrame = 0;
            } else {
                simFrame++;
            }

            size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
            if(written != (size_t)Npix*3){
                fprintf(stderr, "\nFFmpeg write failed at frame %d.\n", frame);
                stopAll = true;
                break;
            }

            if((frame % 60) == 0){
                fprintf(stderr, "\rSegment #%lld | Frame %d / %d | Particles: %d | SimFrame: %d   ",
                        segmentIndex, frame, framesPerVideo, (int)hP.size(), simFrame);
                fflush(stderr);
            }

            if(stopAll) break;
        }

        fprintf(stderr, "\nFinalizing encode...\n");
        fflush(ff);
        pclose(ff);

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "Segment #%lld done. Time: %.2f s for %d frames (%.2f fps effective)\n",
                segmentIndex, sec, framesPerVideo, framesPerVideo / std::max(sec, 1e-9));

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

    return 0;
}
