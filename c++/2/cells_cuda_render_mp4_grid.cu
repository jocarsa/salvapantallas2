// cells_cuda_render_mp4_grid.cu
// CUDA offline renderer -> FFmpeg MP4 (H.265), with:
// - Uniform grid neighbor search (scales to many particles)
// - Per-particle radius (random scales)
// - Much clearer / brighter particles (white core highlight)
// - 1920x1080 @ 60fps -> mp4 h265 (NVENC if available)
//
// Usage:
//   ./cells_cuda_render_mp4_grid out.mp4 10            # 10 seconds, default N=3000, nvenc
//   ./cells_cuda_render_mp4_grid out.mp4 10 8000       # more particles
//   ./cells_cuda_render_mp4_grid out.mp4 10 8000 libx265
//
// Notes:
// - Rendering cost grows with N * (glow area). If you push N high, reduce glowRadiusFactor.

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
    float rad;       // radius in pixels
};

// ---------------- Device helpers ----------------
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(b, fmaxf(a, v));
}
__device__ __forceinline__ float smoothstep(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ---------------- Grid indexing ----------------
__device__ __forceinline__ int cellIndex(int cx, int cy, int gridW) {
    return cy * gridW + cx;
}

// Kernel: clear cell heads
__global__ void k_clear_cells(int* cellHead, int numCells){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= numCells) return;
    cellHead[i] = -1;
}

// Kernel: build linked lists per cell
__global__ void k_build_grid(
    const Particle* p, int n,
    int* cellHead, int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    float x = p[i].x;
    float y = p[i].y;

    int cx = (int)floorf(x / cellSize);
    int cy = (int)floorf(y / cellSize);

    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    int c = cellIndex(cx, cy, gridW);

    // Push-front into cell list
    int old = atomicExch(&cellHead[c], i);
    next[i] = old;
}

// Kernel: update particles using neighbor cells (3x3)
__global__ void k_update_particles_grid(
    Particle* p, int n,
    const int* cellHead, const int* next,
    int W, int H,
    float cellSize,
    int gridW, int gridH,
    float dt,
    float neighborRadius,
    float repelStrength,
    float swirlStrength,
    float damping
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // Swirl around center
    float cx0 = 0.5f * W, cy0 = 0.5f * H;
    float dxC = me.x - cx0;
    float dyC = me.y - cy0;
    float invLenC = rsqrtf(dxC*dxC + dyC*dyC + 1e-6f);

    float ax = (-dyC * invLenC) * swirlStrength;
    float ay = ( dxC * invLenC) * swirlStrength;

    // Determine my cell
    int cx = (int)floorf(me.x / cellSize);
    int cy = (int)floorf(me.y / cellSize);
    cx = max(0, min(gridW - 1, cx));
    cy = max(0, min(gridH - 1, cy));

    float nr2 = neighborRadius * neighborRadius;

    // Search neighbor cells
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
                    float dx = me.x - pj.x;
                    float dy = me.y - pj.y;
                    float d2 = dx*dx + dy*dy;

                    if(d2 < nr2){
                        float invD = rsqrtf(d2 + 1e-6f);
                        float d = 1.0f / invD;

                        // variable-radius "no overlap" distance
                        float minDist = me.rad + pj.rad;
                        float overlap = fmaxf(0.0f, (minDist - d));
                        float wOverlap = overlap / (minDist + 1e-6f);

                        // mild separation inside neighbor radius
                        float wNear = 1.0f - (d / (neighborRadius + 1e-6f));
                        wNear = clampf(wNear, 0.0f, 1.0f);

                        float w = repelStrength * (0.06f*wNear + 2.40f*wOverlap);

                        ax += (dx * invD) * w;
                        ay += (dy * invD) * w;
                    }
                }
                j = next[j];
            }
        }
    }

    // Integrate
    me.vx = (me.vx + ax * dt) * damping;
    me.vy = (me.vy + ay * dt) * damping;

    me.x += me.vx * dt;
    me.y += me.vy * dt;

    // Boundary bounce
    float pad = me.rad + 2.0f;
    if(me.x < pad){ me.x = pad; me.vx = fabsf(me.vx); }
    if(me.x > W - pad){ me.x = W - pad; me.vx = -fabsf(me.vx); }
    if(me.y < pad){ me.y = pad; me.vy = fabsf(me.vy); }
    if(me.y > H - pad){ me.y = H - pad; me.vy = -fabsf(me.vy); }

    p[i] = me;
}

// ---------------- Rendering ----------------
__global__ void k_clear_accum(float4* accum, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    accum[i] = make_float4(0,0,0,0);
}

// Draw particles with glow + strong white core highlight for clarity
__global__ void k_draw_particles(
    const Particle* p, int n,
    float4* accum, int W, int H,
    float glowRadiusFactor,     // glowRadius = rad * glowRadiusFactor
    float glowIntensity,
    float coreEdgeSoftness,     // softness for core edge
    float whiteCoreBoost        // extra white added in core (clarity)
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    float rad = me.rad;
    float R = rad * glowRadiusFactor;
    R = fminf(R, 80.0f); // cap for performance

    int x0 = (int)floorf(me.x - R);
    int x1 = (int)ceilf (me.x + R);
    int y0 = (int)floorf(me.y - R);
    int y1 = (int)ceilf (me.y + R);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W-1, x1); y1 = min(H-1, y1);

    // Glow falloff
    float sigma = 0.33f * R;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    float coreIn  = rad * (1.0f - coreEdgeSoftness);
    float coreOut = rad * (1.0f + coreEdgeSoftness);

    for(int y=y0;y<=y1;y++){
        for(int x=x0;x<=x1;x++){
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx*dx + dy*dy;
            if(d2 > R*R) continue;

            float d = sqrtf(d2 + 1e-6f);

            // core mask (sharp disk)
            float core = 1.0f - smoothstep(coreIn, coreOut, d);
            core = clampf(core, 0.0f, 1.0f);

            // glow gaussian
            float glow = expf(-d2 * inv2s2);

            // final weight
            float w = glowIntensity * (0.85f*glow + 2.8f*core);

            int idx = y * W + x;

            // colored part
            atomicAdd(&accum[idx].x, me.r * w);
            atomicAdd(&accum[idx].y, me.g * w);
            atomicAdd(&accum[idx].z, me.b * w);

            // white highlight (makes them "clear", not dark)
            float wWhite = whiteCoreBoost * core * glowIntensity * 3.2f;
            atomicAdd(&accum[idx].x, 1.0f * wWhite);
            atomicAdd(&accum[idx].y, 1.0f * wWhite);
            atomicAdd(&accum[idx].z, 1.0f * wWhite);

            atomicAdd(&accum[idx].w, w + wWhite);
        }
    }
}

// Tonemap (brighter, lifted shadows)
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

// ---------------- HSV palette (host) ----------------
static void hsv2rgb(float h, float s, float v, float& r, float& g, float& b){
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

// ---------------- FFmpeg pipe ----------------
static FILE* open_ffmpeg_pipe(
    const std::string& outPath, int W, int H, int fps,
    const std::string& encoder // "nvenc" or "libx265"
){
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

int main(int argc, char** argv){
    // Fixed output requested
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    int N = 3000;                 // many more particles by default
    std::string encoder = "nvenc"; // or "libx265"

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) N = std::max(128, std::atoi(argv[3]));
    if(argc >= 5) encoder = argv[4];

    const int totalFrames = seconds * fps;
    const float dt = 1.0f / (float)fps;

    // ----- Physics params -----
    // neighborRadius should be >= ~2*maxRad + some margin
    const float neighborRadius = 120.0f;
    const float repelStrength  = 55.0f;
    const float swirlStrength  = 18.0f;
    const float damping        = 0.9965f;

    // ----- Rendering params (clarity) -----
    const float glowRadiusFactor = 3.4f;     // smaller => faster, sharper
    const float glowIntensity    = 0.028f;   // brightness
    const float coreEdgeSoftness = 0.035f;   // smaller => sharper disk edge
    const float whiteCoreBoost   = 2.8f;     // makes particles clearly visible

    const float exposure  = 1.85f;           // brighter
    const float lift      = 0.020f;          // lifts dark tones (fix "dark hues")
    const float gammaInv  = 1.0f / 1.9f;     // slightly brighter than 2.2

    // ----- Grid params -----
    // cellSize should be about neighborRadius for 3x3 search coverage
    const float cellSize = neighborRadius;
    const int gridW = (int)ceilf(W / cellSize);
    const int gridH = (int)ceilf(H / cellSize);
    const int numCells = gridW * gridH;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Seconds: %d | FPS: %d | Frames: %d\n", seconds, fps, totalFrames);
    fprintf(stderr, "Particles: %d\n", N);
    fprintf(stderr, "Grid: %dx%d cells (cellSize=%.1f)\n", gridW, gridH, cellSize);

    // Init particles (host)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> uv(-140.0f, 140.0f);
    std::uniform_real_distribution<float> uh(0.0f, 1.0f);
    std::uniform_real_distribution<float> urad(5.5f, 16.0f); // random scales

    std::vector<Particle> hP(N);
    for(int i=0;i<N;i++){
        hP[i].x = ux(rng);
        hP[i].y = uy(rng);
        hP[i].vx = uv(rng);
        hP[i].vy = uv(rng);
        hP[i].rad = urad(rng);

        // Bright, saturated colors
        float h = uh(rng);
        float s = 0.88f;
        float v = 1.00f;
        hsv2rgb(h, s, v, hP[i].r, hP[i].g, hP[i].b);

        // tiny mix with white to avoid any muddy tones
        float mixW = 0.10f;
        hP[i].r = hP[i].r*(1.0f-mixW) + 1.0f*mixW;
        hP[i].g = hP[i].g*(1.0f-mixW) + 1.0f*mixW;
        hP[i].b = hP[i].b*(1.0f-mixW) + 1.0f*mixW;
    }

    // Device buffers
    Particle* dP = nullptr;
    float4* dAccum = nullptr;
    unsigned char* dBGR = nullptr;

    int* dCellHead = nullptr;
    int* dNext = nullptr;

    const int Npix = W * H;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dAccum, sizeof(float4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMalloc(&dCellHead, sizeof(int) * (size_t)numCells));
    CUDA_CHECK(cudaMalloc(&dNext, sizeof(int) * (size_t)N));

    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle)*(size_t)N, cudaMemcpyHostToDevice));

    // Pinned host buffer for fast D->H copy
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
        // Clear grid heads
        {
            int block = 256;
            int grid  = (numCells + block - 1) / block;
            k_clear_cells<<<grid, block, 0, stream>>>(dCellHead, numCells);
        }

        // Build grid linked lists
        {
            int block = 256;
            int grid  = (N + block - 1) / block;
            k_build_grid<<<grid, block, 0, stream>>>(
                dP, N, dCellHead, dNext,
                W, H, cellSize, gridW, gridH
            );
        }

        // Update using neighbor cells
        {
            int block = 256;
            int grid  = (N + block - 1) / block;
            k_update_particles_grid<<<grid, block, 0, stream>>>(
                dP, N,
                dCellHead, dNext,
                W, H, cellSize, gridW, gridH,
                dt,
                neighborRadius,
                repelStrength,
                swirlStrength,
                damping
            );
        }

        // Clear framebuffer accum
        {
            int block = 256;
            int grid  = (Npix + block - 1) / block;
            k_clear_accum<<<grid, block, 0, stream>>>(dAccum, Npix);
        }

        // Draw
        {
            int block = 128;
            int grid  = (N + block - 1) / block;
            k_draw_particles<<<grid, block, 0, stream>>>(
                dP, N,
                dAccum, W, H,
                glowRadiusFactor,
                glowIntensity,
                coreEdgeSoftness,
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

        // Copy back + write to ffmpeg
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
    CUDA_CHECK(cudaFree(dAccum));
    CUDA_CHECK(cudaFree(dBGR));
    CUDA_CHECK(cudaFree(dCellHead));
    CUDA_CHECK(cudaFree(dNext));

    return 0;
}
