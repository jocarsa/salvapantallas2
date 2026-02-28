// cells_cuda_network_mp4_v2.cu
// Fixes:
//  1) Connections: stronger + thicker + clearer (lines now very visible)
//  2) No vibration: replace per-frame random jitter with SMOOTH angular velocity (damped),
//     and make attachment offset adjustments springy + damped (no jittery push/pull)
//
// Output: 1920x1080 @ 60fps -> MP4 H.265 (NVENC if available)
//
// Usage:
//   ./cells_cuda_network_mp4_v2 out.mp4 10
//   ./cells_cuda_network_mp4_v2 out.mp4 10 600 nvenc
//   ./cells_cuda_network_mp4_v2 out.mp4 10 600 libx265
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native cells_cuda_network_mp4_v2.cu -o cells_cuda_network_mp4_v2 \
//     $(pkg-config --cflags --libs opencv4)

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

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);        \
        std::exit(1);                                                \
    }                                                                 \
} while (0)

struct Particle {
    float x, y;
    float dir;       // direction angle
    float dirVel;    // angular velocity (smooth, damped)
    float r, g, b;   // 0..1
    int   parent;    // -1 if free
    float offx, offy;// offset from parent
    float offvx, offvy; // offset velocity (smooth attachments)
};

__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}
__device__ __forceinline__ float smoothstep(float e0, float e1, float x){
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// cheap hash -> [0,1)
__device__ __forceinline__ float hash01_u32(uint32_t x){
    x ^= x >> 16; x *= 0x7feb352d;
    x ^= x >> 15; x *= 0x846ca68b;
    x ^= x >> 16;
    return (x & 0x00FFFFFF) / 16777216.0f;
}

// smooth "noise" from time using sin; deterministic, no per-frame randomness
__device__ __forceinline__ float smoothNoise(uint32_t id, float t){
    float a = 6.2831853f * (hash01_u32(id * 2654435761u) + 0.07f * t);
    float b = 6.2831853f * (hash01_u32(id * 2246822519u) + 0.11f * t);
    // sum of sines gives smooth variation in [-2..2]
    return 0.5f * (sinf(a) + 0.7f * sinf(b));
}

// ---------------- FFmpeg pipe ----------------
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
    }else{
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
    if(!pipe){
        fprintf(stderr, "Failed to start ffmpeg. Command:\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

// ---------------- Simulation ----------------
//
// This keeps your logic idea, but makes motion smooth:
//  - free particles: dirVel is driven by smoothNoise(t) then damped -> no vibration
//  - attached particles: offx/offy is a spring with offvx/offvy damping -> no jitter
//
// NOTE: Pairwise scan (O(n^2)) is kept for exact "connect within radius" feel.
// If you want 10k+ particles, ask for the grid-accelerated neighbor version.

__global__ void k_step_network_smooth(
    Particle* p, int n,
    int W, int H,
    float t, float dt,
    float speed,            // movement speed
    float linkDist,         // connection dist (~50)
    float linkBox,          // precheck (~109)
    float collideDist,      // hard collision (~10)
    float attachedNear,     // near (~40)
    float attachedFar,      // far (~100)
    float pushNear,         // strength
    float pullFar,          // strength
    float dirVelDrive,      // how much noise drives angular velocity
    float dirVelDamp,       // damping for angular velocity
    float offSpringK,       // spring constant for offset adjustments
    float offDamp           // damping for offset velocity
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];
    uint32_t id = (uint32_t)i;

    if(me.parent == -1){
        // Smooth angular velocity drive (no random vibration)
        float n1 = smoothNoise(id, t);
        me.dirVel += (n1 * dirVelDrive) * dt;
        me.dirVel *= expf(-dirVelDamp * dt);
        me.dir += me.dirVel * dt;

        // Move
        me.x += cosf(me.dir) * speed;
        me.y += sinf(me.dir) * speed;

        // Bounce
        if(me.x > W || me.x < 0 || me.y > H || me.y < 0){
            me.dir += 3.14159265f;
            me.dirVel = -me.dirVel * 0.6f;
            me.x = clampf(me.x, 0.0f, (float)W);
            me.y = clampf(me.y, 0.0f, (float)H);
        }

        // Pair interactions
        for(int j=0;j<n;j++){
            if(j==i) continue;

            Particle pj = p[j];

            float dx = me.x - pj.x;
            float dy = me.y - pj.y;

            if(fabsf(dx) < linkBox && fabsf(dy) < linkBox){
                float d2 = dx*dx + dy*dy;
                float d  = sqrtf(d2 + 1e-6f);

                // Collision (c < 10) -> bounce away gently (no jitter)
                if(d < collideDist){
                    float ang = atan2f(me.y - pj.y, me.x - pj.x);
                    me.x += cosf(ang) * 2.0f;
                    me.y += sinf(ang) * 2.0f;
                    me.dir += 0.9f; // small turn
                    me.dirVel *= 0.7f;
                }

                // Link (c < 50): deterministic attachment to avoid races
                if(d < linkDist){
                    // attach higher index to lower index when lower is "more stable"
                    if(i > j){
                        // Prefer attaching to a free parent, else still allow chaining
                        me.parent = j;
                        me.offx = me.x - pj.x;
                        me.offy = me.y - pj.y;
                        me.offvx = 0.0f;
                        me.offvy = 0.0f;
                    }
                }
            }
        }
    }else{
        int pid = me.parent;
        if(pid >= 0 && pid < n){
            Particle pp = p[pid];

            // Follow parent + offset
            float tx = pp.x + me.offx;
            float ty = pp.y + me.offy;

            me.x = tx;
            me.y = ty;

            // Offset adjustments: springy, damped (no vibration)
            // We accumulate a "desired offset change" force.
            float fx = 0.0f, fy = 0.0f;

            for(int j=0;j<n;j++){
                if(j==i) continue;

                Particle pj = p[j];
                float dx = me.x - pj.x;
                float dy = me.y - pj.y;

                if(fabsf(dx) < linkBox && fabsf(dy) < linkBox){
                    float d2 = dx*dx + dy*dy;
                    float d  = sqrtf(d2 + 1e-6f);
                    float ang = atan2f(pj.y - me.y, pj.x - me.x);

                    if(d < attachedNear){
                        fx -= cosf(ang) * pushNear;
                        fy -= sinf(ang) * pushNear;
                    }
                    if(d > attachedFar){
                        fx += cosf(ang) * pullFar;
                        fy += sinf(ang) * pullFar;
                    }
                }
            }

            // Apply spring + damping in "offset space"
            me.offvx = (me.offvx + fx * offSpringK * dt) * expf(-offDamp * dt);
            me.offvy = (me.offvy + fy * offSpringK * dt) * expf(-offDamp * dt);

            me.offx += me.offvx;
            me.offy += me.offvy;

            // Keep offsets bounded to avoid runaway
            float maxOff = 180.0f;
            me.offx = clampf(me.offx, -maxOff, maxOff);
            me.offy = clampf(me.offy, -maxOff, maxOff);
        }else{
            me.parent = -1;
        }
    }

    p[i] = me;
}

// ---------------- Rendering ----------------
__global__ void k_clear_accum(float4* accum, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    accum[i] = make_float4(0,0,0,0);
}

// Very visible links: thicker + brighter + slight glow
__global__ void k_draw_links_strong(
    const Particle* p, int n,
    float4* accum, int W, int H,
    float linkDist,
    float lineRadius,
    float lineIntensity
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle a = p[i];

    for(int j=i+1; j<n; j++){
        Particle b = p[j];

        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float d2 = dx*dx + dy*dy;
        if(d2 >= linkDist*linkDist) continue;

        float d = sqrtf(d2 + 1e-6f);

        // More samples for smoother/thicker line
        int steps = (int)ceilf(d / 4.0f);
        steps = max(12, min(96, steps));
        float invSteps = 1.0f / (float)steps;

        // Line color: strong, near-white tinted by average of endpoints
        float lr = 0.75f + 0.25f * (a.r + b.r) * 0.5f;
        float lg = 0.75f + 0.25f * (a.g + b.g) * 0.5f;
        float lb = 0.75f + 0.25f * (a.b + b.b) * 0.5f;

        float sigma = 0.45f * lineRadius;
        float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

        for(int s=0; s<=steps; s++){
            float t = s * invSteps;
            float x = a.x + dx * t;
            float y = a.y + dy * t;

            int x0 = (int)floorf(x - lineRadius);
            int x1 = (int)ceilf (x + lineRadius);
            int y0 = (int)floorf(y - lineRadius);
            int y1 = (int)ceilf (y + lineRadius);

            x0 = max(0, x0); y0 = max(0, y0);
            x1 = min(W-1, x1); y1 = min(H-1, y1);

            for(int yy=y0; yy<=y1; yy++){
                for(int xx=x0; xx<=x1; xx++){
                    float px = (xx + 0.5f) - x;
                    float py = (yy + 0.5f) - y;
                    float dd2 = px*px + py*py;
                    if(dd2 > lineRadius*lineRadius) continue;

                    // gaussian weight
                    float w = lineIntensity * expf(-dd2 * inv2s2);

                    // also fade a bit by distance (short links brighter)
                    float fade = 1.0f - (d / (linkDist + 1e-6f));
                    fade = clampf(fade, 0.15f, 1.0f);
                    w *= fade;

                    int idx = yy * W + xx;
                    atomicAdd(&accum[idx].x, lr * w);
                    atomicAdd(&accum[idx].y, lg * w);
                    atomicAdd(&accum[idx].z, lb * w);
                    atomicAdd(&accum[idx].w, w);
                }
            }
        }
    }
}

// Bright discs + glow + strong white core (clear particles)
__global__ void k_draw_particles(
    const Particle* p, int n,
    float4* accum, int W, int H,
    float outerR,
    float innerR,
    float glowR,
    float glowIntensity,
    float whiteCoreBoost
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // stable per-particle scale (no jitter)
    float s = 0.85f + 0.35f * hash01_u32((uint32_t)i * 2654435761u);
    float Rb = outerR * s;
    float Rc = innerR * s;
    float Rg = glowR  * s;

    int x0 = (int)floorf(me.x - Rg);
    int x1 = (int)ceilf (me.x + Rg);
    int y0 = (int)floorf(me.y - Rg);
    int y1 = (int)ceilf (me.y + Rg);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W-1, x1); y1 = min(H-1, y1);

    float sigma = 0.33f * Rg;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    for(int y=y0;y<=y1;y++){
        for(int x=x0;x<=x1;x++){
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx*dx + dy*dy;
            if(d2 > Rg*Rg) continue;

            float d = sqrtf(d2 + 1e-6f);

            float insideBorder = 1.0f - smoothstep(Rb*0.98f, Rb*1.02f, d);
            insideBorder = clampf(insideBorder, 0.0f, 1.0f);

            float insideColor = 1.0f - smoothstep(Rc*0.98f, Rc*1.02f, d);
            insideColor = clampf(insideColor, 0.0f, 1.0f);

            float glow = expf(-d2 * inv2s2);

            float wGlow = glowIntensity * (0.95f * glow);
            float wCore = glowIntensity * (2.9f  * insideColor);

            int idx = y * W + x;

            // Border ring: darker by adding less light there (subtle but visible)
            float ring = insideBorder * (1.0f - insideColor);
            if(ring > 0.0f){
                float wRing = glowIntensity * 0.06f * ring;
                atomicAdd(&accum[idx].x, 0.01f * wRing);
                atomicAdd(&accum[idx].y, 0.01f * wRing);
                atomicAdd(&accum[idx].z, 0.01f * wRing);
                atomicAdd(&accum[idx].w, wRing);
            }

            atomicAdd(&accum[idx].x, me.r * (wGlow + wCore));
            atomicAdd(&accum[idx].y, me.g * (wGlow + wCore));
            atomicAdd(&accum[idx].z, me.b * (wGlow + wCore));
            atomicAdd(&accum[idx].w, (wGlow + wCore));

            // White core highlight for clarity
            float wWhite = whiteCoreBoost * insideColor * glowIntensity * 3.1f;
            atomicAdd(&accum[idx].x, 1.0f * wWhite);
            atomicAdd(&accum[idx].y, 1.0f * wWhite);
            atomicAdd(&accum[idx].z, 1.0f * wWhite);
            atomicAdd(&accum[idx].w, wWhite);
        }
    }
}

__global__ void k_tonemap_to_bgr(
    const float4* accum,
    unsigned char* outBGR,
    int Npix,
    float exposure,
    float lift,
    float gammaInv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float4 a = accum[i];

    float r = 1.0f - expf(-(a.x + lift) * exposure);
    float g = 1.0f - expf(-(a.y + lift) * exposure);
    float b = 1.0f - expf(-(a.z + lift) * exposure);

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3*i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3*i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3*i + 2] = (unsigned char)lrintf(255.0f * r);
}

// ---------------- Host HSV init ----------------
static void hsv2rgb_host(float h, float s, float v, float& r, float& g, float& b){
    h = fmodf(h, 1.0f);
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rp=0,gp=0,bp=0;
    int seg = (int)floorf(h * 6.0f);
    switch(seg){
        case 0: rp=c; gp=x; bp=0; break;
        case 1: rp=x; gp=c; bp=0; break;
        case 2: rp=0; gp=c; bp=x; break;
        case 3: rp=0; gp=x; bp=c; break;
        case 4: rp=x; gp=0; bp=c; break;
        default: rp=c; gp=0; bp=x; break;
    }
    r = rp + m; g = gp + m; b = bp + m;
}

int main(int argc, char** argv){
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    int N0 = 400;                // start count
    std::string encoder = "nvenc";

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) N0 = std::max(50, std::atoi(argv[3]));
    if(argc >= 5) encoder = argv[4];

    // Like your JS tiempo%10, but per FRAME
    const int SPAWN_EVERY_FRAMES = 10;
    const int totalFrames = seconds * fps;
    const int maxN = N0 + (totalFrames / SPAWN_EVERY_FRAMES) + 32;

    // Distances from your JS
    const float linkBox      = 109.0f;
    const float linkDist     = 50.0f;
    const float collideDist  = 10.0f;

    const float attachedNear = 40.0f;
    const float attachedFar  = 100.0f;

    // Smooth motion + no vibration
    const float dt = 1.0f / (float)fps;
    const float speed = 1.85f;        // movement step
    const float dirVelDrive = 2.8f;   // how strongly noise drives turning
    const float dirVelDamp  = 2.2f;   // angular damping

    // Smooth attachment offsets
    const float pushNear  = 1.0f;     // reduced (JS used 2; this avoids jitter)
    const float pullFar   = 0.06f;    // reduced (JS used 0.1)
    const float offSpringK = 7.5f;    // converts force into off-velocity
    const float offDamp    = 3.0f;    // damping for offset velocity

    // Rendering (clear links + clear particles)
    const float outerR = 20.0f;
    const float innerR = 15.0f;
    const float glowR  = 56.0f;
    const float glowIntensity = 0.030f;
    const float whiteCoreBoost = 3.4f;

    // LINKS: stronger and thicker so they are obvious
    const float lineRadius = 9.0f;       // ~lineWidth 18px-ish look
    const float lineIntensity = 0.045f;  // stronger than before (this was your missing connections)

    const float exposure = 1.95f;
    const float lift     = 0.030f;
    const float gammaInv = 1.0f / 1.85f;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Seconds: %d | FPS: %d | Frames: %d\n", seconds, fps, totalFrames);
    fprintf(stderr, "Start particles: %d | Max particles: %d | Spawn every %d frames\n",
            N0, maxN, SPAWN_EVERY_FRAMES);

    // Init particles (host)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> uh(0.0f, 1.0f);

    std::vector<Particle> hP(maxN);
    for(int i=0;i<maxN;i++){
        hP[i].x = ux(rng);
        hP[i].y = uy(rng);
        hP[i].dir = uh(rng) * 6.2831853f;
        hP[i].dirVel = 0.0f;

        float h = uh(rng);
        float s = 0.92f;
        float v = 1.00f;
        hsv2rgb_host(h, s, v, hP[i].r, hP[i].g, hP[i].b);

        float mixW = 0.08f;
        hP[i].r = hP[i].r*(1.0f-mixW) + 1.0f*mixW;
        hP[i].g = hP[i].g*(1.0f-mixW) + 1.0f*mixW;
        hP[i].b = hP[i].b*(1.0f-mixW) + 1.0f*mixW;

        hP[i].parent = -1;
        hP[i].offx = 0.0f;
        hP[i].offy = 0.0f;
        hP[i].offvx = 0.0f;
        hP[i].offvy = 0.0f;
    }

    // Device buffers
    Particle* dP = nullptr;
    float4* dAccum = nullptr;
    unsigned char* dBGR = nullptr;

    const int Npix = W * H;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxN));
    CUDA_CHECK(cudaMalloc(&dAccum, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));
    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle)*(size_t)maxN, cudaMemcpyHostToDevice));

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if(!ff){
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        return 1;
    }

    int n = N0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for(int f=0; f<totalFrames; f++){
        float timeSec = f * dt;

        // Spawn
        if((f % SPAWN_EVERY_FRAMES) == 0 && n < maxN){
            Particle one = hP[n];
            one.parent = -1;
            one.dirVel = 0.0f;
            one.offvx = one.offvy = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(dP + n, &one, sizeof(Particle), cudaMemcpyHostToDevice, stream));
            n++;
        }

        // Step
        {
            int block = 256;
            int grid  = (n + block - 1) / block;
            k_step_network_smooth<<<grid, block, 0, stream>>>(
                dP, n, W, H,
                timeSec, dt,
                speed,
                linkDist, linkBox, collideDist,
                attachedNear, attachedFar,
                pushNear, pullFar,
                dirVelDrive, dirVelDamp,
                offSpringK, offDamp
            );
        }

        // Clear accum
        {
            int block = 256;
            int grid  = (Npix + block - 1) / block;
            k_clear_accum<<<grid, block, 0, stream>>>(dAccum, Npix);
        }

        // Draw links FIRST (then particles on top)
        {
            int block = 48;
            int grid  = (n + block - 1) / block;
            k_draw_links_strong<<<grid, block, 0, stream>>>(
                dP, n, dAccum, W, H,
                linkDist,
                lineRadius,
                lineIntensity
            );
        }

        // Draw particles
        {
            int block = 128;
            int grid  = (n + block - 1) / block;
            k_draw_particles<<<grid, block, 0, stream>>>(
                dP, n, dAccum, W, H,
                outerR, innerR, glowR,
                glowIntensity,
                whiteCoreBoost
            );
        }

        // Tonemap
        {
            int block = 256;
            int grid  = (Npix + block - 1) / block;
            k_tonemap_to_bgr<<<grid, block, 0, stream>>>(
                dAccum, dBGR, Npix,
                exposure, lift, gammaInv
            );
        }

        CUDA_CHECK(cudaGetLastError());

        // Copy + write
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
        if(written != (size_t)Npix*3){
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", f);
            break;
        }

        if((f % 60) == 0){
            fprintf(stderr, "\rFrame %d / %d | active=%d", f, totalFrames, n);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\nFinalizing encode...\n");
    fflush(ff);
    pclose(ff);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "Done. Time: %.2f s for %d frames (%.2f fps effective)\n",
            sec, totalFrames, totalFrames / sec);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dAccum));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
