// vortex_particles_4k_cuda_mp4_decay.cu
// Full code: CUDA version of your 2-canvas JS, with PROPER trail decay/fade.
//
// Features:
// - 3840x2160 (4K) offline render, 60fps, encode to MP4 H.265 (NVENC if available)
// - Two persistent buffers:
//    buf1: "pixel trails" (like imagen) with exponential decay of RGB+Alpha (no infinite trails)
//    buf2: "overlay blobs" (like lienzo2) with multiplicative fade (like fillRect alpha=0.1)
// - Particle motion:
//    initial tangential velocity around center (like JS)
//    neighbor steering within 200px window using accel ~ (m2+1)/dist
//    extreme-close interaction approximating velocity swapping (stable, race-safe-ish)
//    respawn if out of bounds
// - Bright vivid colors (avoid dark hues)
//
// Usage:
//   ./vortex_particles_4k_cuda_mp4_decay out.mp4 10
//   ./vortex_particles_4k_cuda_mp4_decay out.mp4 10 10000
//   ./vortex_particles_4k_cuda_mp4_decay out.mp4 10 20000 nvenc
//   ./vortex_particles_4k_cuda_mp4_decay out.mp4 10 20000 libx265
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native vortex_particles_4k_cuda_mp4_decay.cu -o vortex_particles_4k_cuda_mp4_decay \
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
    float vx, vy;
    float r, g, b;   // 0..1
    float m;         // mass-ish
};

__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}
__device__ __forceinline__ int clampi(int v, int a, int b){
    return v < a ? a : (v > b ? b : v);
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
    if(!pipe){
        fprintf(stderr, "Failed to start ffmpeg. Command:\n%s\n", cmd.c_str());
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
    if(i >= numCells) return;
    cellHead[i] = -1;
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

    int cx = (int)floorf(p[i].x / cellSize);
    int cy = (int)floorf(p[i].y / cellSize);
    cx = clampi(cx, 0, gridW - 1);
    cy = clampi(cy, 0, gridH - 1);

    int c = cellIndex(cx, cy, gridW);
    int old = atomicExch(&cellHead[c], i);
    next[i] = old;
}

// ----------------------- Persistent buffers (DECAY FIX) -----------------------
// buf1: trail buffer (pixel writes). We now do REAL exponential fade on RGB and A.
// buf2: overlay buffer fade multiply (like fillRect alpha 0.1 => ~0.9 retain)

__global__ void k_fade_buffers(
    float4* buf1, float4* buf2,
    int Npix,
    float buf1MulRGB,   // e.g. 0.985
    float buf1MulA,     // e.g. 0.980
    float buf2Mul       // e.g. 0.90
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

// Deposit trail pixels into buf1 (adds "ink", which then fades)
__global__ void k_deposit_pixels_buf1(
    const Particle* p, int n,
    float4* buf1,
    int W, int H,
    float ink
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    int x = (int)lrintf(p[i].x);
    int y = (int)lrintf(p[i].y);
    if(x < 0 || x >= W || y < 0 || y >= H) return;

    int idx = y * W + x;

    atomicAdd(&buf1[idx].x, p[i].r * ink);
    atomicAdd(&buf1[idx].y, p[i].g * ink);
    atomicAdd(&buf1[idx].z, p[i].b * ink);
    atomicAdd(&buf1[idx].w, ink);
}

// Draw soft blobs into buf2 (like lienzo2 arcs) with its own fade
__global__ void k_draw_blobs_buf2(
    const Particle* p, int n,
    float4* buf2,
    int W, int H,
    float blobIntensity
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];
    float rad = fmaxf(0.8f, me.m / 40.0f); // JS: m/40

    int x0 = (int)floorf(me.x - rad);
    int x1 = (int)ceilf (me.x + rad);
    int y0 = (int)floorf(me.y - rad);
    int y1 = (int)ceilf (me.y + rad);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W-1, x1); y1 = min(H-1, y1);

    float sigma = 0.45f * rad;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    for(int y=y0;y<=y1;y++){
        for(int x=x0;x<=x1;x++){
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx*dx + dy*dy;
            if(d2 > rad*rad) continue;

            float w = blobIntensity * expf(-d2 * inv2s2);
            int idx = y * W + x;

            atomicAdd(&buf2[idx].x, me.r * w);
            atomicAdd(&buf2[idx].y, me.g * w);
            atomicAdd(&buf2[idx].z, me.b * w);
            atomicAdd(&buf2[idx].w, w);
        }
    }
}

// ----------------------- Physics (grid neighbor) -----------------------
// accel ~ (m2+1)/dist within 200px window. Scaled for stability.

__global__ void k_step_particles_grid(
    Particle* p, int n,
    const int* cellHead, const int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH,
    float dt,
    float accelScale,
    float neighborBox,
    float swapDist,
    float speedScale
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // Integrate position (JS: /1250)
    me.x += me.vx * speedScale;
    me.y += me.vy * speedScale;

    int cx = (int)floorf(me.x / cellSize);
    int cy = (int)floorf(me.y / cellSize);
    cx = clampi(cx, 0, gridW - 1);
    cy = clampi(cy, 0, gridH - 1);

    float dvx = 0.0f, dvy = 0.0f;

    for(int oy=-1; oy<=1; oy++){
        int ny = cy + oy;
        if(ny < 0 || ny >= gridH) continue;
        for(int ox=-1; ox<=1; ox++){
            int nx = cx + ox;
            if(nx < 0 || nx >= gridW) continue;

            int c = cellIndex(nx, ny, gridW);
            int j = cellHead[c];
            while(j != -1){
                if(j != i){
                    Particle pj = p[j];
                    float dx = pj.x - me.x;
                    float dy = pj.y - me.y;

                    if(fabsf(dx) < neighborBox && fabsf(dy) < neighborBox){
                        float d2 = dx*dx + dy*dy;

                        // softening to avoid explosions
                        float invD = rsqrtf(d2 + 16.0f);
                        float d = 1.0f / invD;

                        float ux = dx * invD;
                        float uy = dy * invD;

                        float f = accelScale * (pj.m + 1.0f) * invD; // ~1/dist
                        dvx += ux * f;
                        dvy += uy * f;

                        // very close: approximate swap (race-safe-ish)
                        if(d < swapDist){
                            if(i < j){
                                float vx_i = me.vx, vy_i = me.vy;
                                float vx_j = pj.vx, vy_j = pj.vy;
                                me.vx = (0.25f * vx_i + 0.75f * vx_j) / 1.1f;
                                me.vy = (0.25f * vy_i + 0.75f * vy_j) / 1.1f;
                            }
                        }
                    }
                }
                j = next[j];
            }
        }
    }

    me.vx += dvx * dt;
    me.vy += dvy * dt;

    // Mild damping + clamp (stability)
    me.vx *= 0.9993f;
    me.vy *= 0.9993f;

    float vmax = 2200.0f;
    me.vx = clampf(me.vx, -vmax, vmax);
    me.vy = clampf(me.vy, -vmax, vmax);

    // Respawn if out of bounds
    if(me.x < 0 || me.x >= W || me.y < 0 || me.y >= H){
        uint32_t seed = (uint32_t)i * 747796405u ^ 0x9e3779b9u;
        float rx = (seed & 0xFFFF) / 65535.0f;
        float ry = ((seed >> 16) & 0xFFFF) / 65535.0f;

        me.x = rx * (W - 1);
        me.y = ry * (H - 1);

        float a = 6.2831853f * ((seed ^ 0x85ebca6bu) & 0xFFFF) / 65535.0f;
        me.vx = cosf(a) * 1000.0f;
        me.vy = sinf(a) * 1000.0f;
    }

    p[i] = me;
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

    // safety clamp to avoid long-term build-up
    r = fminf(r, 40.0f);
    g = fminf(g, 40.0f);
    b = fminf(b, 40.0f);

    // tonemap
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

// ----------------------- Host HSV init -----------------------
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
    // JS hard-sets 3840x2160
    const int W = 3840;
    const int H = 2160;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    int N = 10000;
    std::string encoder = "nvenc";

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) N = std::max(256, std::atoi(argv[3]));
    if(argc >= 5) encoder = argv[4];

    const int totalFrames = seconds * fps;
    const float dt = 1.0f / (float)fps;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Res: %dx%d | FPS: %d | Seconds: %d | Frames: %d\n", W, H, fps, seconds, totalFrames);
    fprintf(stderr, "Particles: %d\n", N);

    // Physics tuning
    const float neighborBox = 200.0f;
    const float swapDist = 1.2f;
    const float speedScale = 1.0f / 1250.0f;
    const float accelScale = 0.55f;

    // Grid (cellSize ~= neighborBox)
    const float cellSize = neighborBox;
    const int gridW = (int)ceilf(W / cellSize);
    const int gridH = (int)ceilf(H / cellSize);
    const int numCells = gridW * gridH;

    fprintf(stderr, "Grid: %dx%d cells (cellSize=%.1f)\n", gridW, gridH, cellSize);

    // ---- DECAY / FADE (THIS IS THE FIX) ----
    // Trails fade: lower -> longer trails, higher -> shorter trails
    const float buf1MulRGB = 0.985f; // try 0.992 longer, 0.975 shorter
    const float buf1MulA   = 0.980f; // alpha fades slightly faster

    // Overlay fade: like fillRect alpha=0.1 => multiply ~0.9
    const float buf2Mul    = 0.90f;

    // How strong each particle writes into trail buffer
    const float ink = 1.05f;

    // Overlay blobs intensity
    const float blobIntensity = 0.055f;

    // Tonemap
    const float exposure = 1.55f;
    const float lift = 0.012f;
    const float gammaInv = 1.0f / 2.0f;

    // Init particles
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> uh(0.0f, 1.0f);
    std::uniform_real_distribution<float> um(5.0f, 105.0f);

    std::vector<Particle> hP(N);
    for(int i=0;i<N;i++){
        float x = ux(rng);
        float y = uy(rng);
        hP[i].x = x;
        hP[i].y = y;

        // JS: angle to center + random*2pi, then tangential velocity *1000
        float ax = (W * 0.5f) - x;
        float ay = (H * 0.5f) - y;
        float base = atan2f(ay, ax) + uh(rng) * 6.2831853f;
        float ang = base + 1.5707963f; // +pi/2
        hP[i].vx = cosf(ang) * 1000.0f;
        hP[i].vy = sinf(ang) * 1000.0f;

        // Bright colors
        float h = uh(rng);
        float s = 0.92f;
        float v = 1.00f;
        hsv2rgb_host(h, s, v, hP[i].r, hP[i].g, hP[i].b);

        // mix with white to avoid muddy tones
        float mixW = 0.12f;
        hP[i].r = hP[i].r*(1.0f-mixW) + 1.0f*mixW;
        hP[i].g = hP[i].g*(1.0f-mixW) + 1.0f*mixW;
        hP[i].b = hP[i].b*(1.0f-mixW) + 1.0f*mixW;

        hP[i].m = um(rng);
    }

    // Device buffers
    Particle* dP = nullptr;
    int* dCellHead = nullptr;
    int* dNext = nullptr;

    float4* dBuf1 = nullptr;
    float4* dBuf2 = nullptr;
    unsigned char* dBGR = nullptr;

    const int Npix = W * H;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dCellHead, sizeof(int) * (size_t)numCells));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * (size_t)N));

    CUDA_CHECK(cudaMalloc(&dBuf1, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBuf2, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR,  sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle)*(size_t)N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBuf1, 0, sizeof(float4)*(size_t)Npix));
    CUDA_CHECK(cudaMemset(dBuf2, 0, sizeof(float4)*(size_t)Npix));

    // Pinned host output
    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if(!ff){
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for(int f=0; f<totalFrames; f++){
        // Fade buffers (THIS ensures trails disappear)
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_buffers<<<grid, block, 0, stream>>>(dBuf1, dBuf2, Npix, buf1MulRGB, buf1MulA, buf2Mul);
        }

        // Grid rebuild
        {
            int block = 256;
            int grid = (numCells + block - 1) / block;
            k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
        }
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_build_grid<<<grid, block, 0, stream>>>(dP, N, dCellHead, dNext, W, H, cellSize, gridW, gridH);
        }

        // Physics step
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_step_particles_grid<<<grid, block, 0, stream>>>(
                dP, N,
                dCellHead, dNext,
                W, H,
                cellSize, gridW, gridH,
                dt,
                accelScale,
                neighborBox,
                swapDist,
                speedScale
            );
        }

        // Deposit hard pixels into trail buffer
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_deposit_pixels_buf1<<<grid, block, 0, stream>>>(dP, N, dBuf1, W, H, ink);
        }

        // Draw overlay blobs
        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_draw_blobs_buf2<<<grid, block, 0, stream>>>(dP, N, dBuf2, W, H, blobIntensity);
        }

        // Compose to BGR
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf1, dBuf2, dBGR, Npix, exposure, lift, gammaInv);
        }

        CUDA_CHECK(cudaGetLastError());

        // Copy + write to ffmpeg
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
        if(written != (size_t)Npix*3){
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", f);
            break;
        }

        if((f % 60) == 0){
            fprintf(stderr, "\rFrame %d / %d", f, totalFrames);
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
    CUDA_CHECK(cudaFree(dCellHead));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dBuf1));
    CUDA_CHECK(cudaFree(dBuf2));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
