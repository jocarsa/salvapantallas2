// vortex_lines_whitefade_4k_cuda_mp4.cu
//
// CUDA conversion of the provided 2-canvas JS animation.
// Practical notes:
// - In the JS, lienzo2 is effectively unused (fillStyle set but no draw). The visible output is lienzo.
// - lienzo is white background and every frame does:
//     fillStyle = rgba(255,255,255,0.01); fillRect(0,0,W,H)
//   which is equivalent to:  buf = buf*(1-0.01) + white*(0.01)  => buf = buf*0.99 + 0.01
// - Then it draws colored strokes (alpha=1) from (x2,y2) -> (x,y), with lineWidth = m.
//
// This CUDA version:
// - 4K (3840x2160), 60fps, offline MP4 HEVC via ffmpeg pipe.
// - Persistent float3 buffer in linear RGB [0..1], initialized to white.
// - Each frame: fade toward white with mul/add (like JS fillRect alpha).
// - Particle physics: same structure (O(N^2), N up to 1000).
// - Line drawing: “thick segment” raster by stamping discs along the segment.
//   (This is visually close to Canvas stroke at these sizes.)
//
// Build:
//   nvcc -O3 -std=c++17 vortex_lines_whitefade_4k_cuda_mp4.cu -o vortex_lines_whitefade_4k_cuda_mp4 \
//     $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_lines_whitefade_4k_cuda_mp4 out.mp4 10
//   ./vortex_lines_whitefade_4k_cuda_mp4 out.mp4 10 nvenc
//   ./vortex_lines_whitefade_4k_cuda_mp4 out.mp4 10 libx265

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
    uint8_t r, g, b;
    float m;
};

static inline float frand(std::mt19937& rng, float a, float b){
    std::uniform_real_distribution<float> d(a, b);
    return d(rng);
}
static inline int irand(std::mt19937& rng, int a, int b){
    std::uniform_int_distribution<int> d(a, b);
    return d(rng);
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

// ----------------------- Kernels -----------------------

// Fade toward white: buf = buf*mul + add (component-wise)
__global__ void k_fade_to_white(float3* buf, int Npix, float mul, float add){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    float3 c = buf[i];
    c.x = c.x * mul + add;
    c.y = c.y * mul + add;
    c.z = c.z * mul + add;
    buf[i] = c;
}

// Step particles: x += vx; y += vy; neighbor accel within 3500 box
// JS: vx += (cos(angle)/dist*0.002)*(m+1)  => dx/dist^2 * 0.002*(m+1)
// Add softening to avoid extreme accelerations when very close.
__global__ void k_step_particles(Particle* p, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // move
    me.x += me.vx;
    me.y += me.vy;

    float dvx = 0.0f, dvy = 0.0f;

    for(int j=0; j<n; j++){
        if(j == i) continue;
        Particle pj = p[j];

        float dx = pj.x - me.x;
        float dy = pj.y - me.y;

        if(fabsf(dx) < 3500.0f && fabsf(dy) < 3500.0f){
            float d2 = dx*dx + dy*dy;
            float inv = 1.0f / (d2 + 4.0f); // softening
            float f = 0.002f * (pj.m + 1.0f) * inv;
            dvx += dx * f;
            dvy += dy * f;
        }
    }

    me.vx += dvx;
    me.vy += dvy;

    // keep previous for line
    me.x2 = me.x;
    me.y2 = me.y;

    p[i] = me;
}

// Draw thick line by stamping discs along the segment.
// Color is alpha=1 in JS (overwrite). Here we assign (not atomic); races are rare and acceptable visually.
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

__global__ void k_draw_lines(
    const Particle* p, int n,
    float3* buf,
    int W, int H
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // JS draws from current -> previous (but visually same segment)
    float x0 = me.x2;
    float y0 = me.y2;
    float x1 = me.x;
    float y1 = me.y;

    float3 col;
    col.x = me.r * (1.0f / 255.0f);
    col.y = me.g * (1.0f / 255.0f);
    col.z = me.b * (1.0f / 255.0f);

    // lineWidth = m/1, radius ~ lineWidth/2
    float lw = fmaxf(0.5f, me.m);
    int rad = (int)ceilf(0.5f * lw);

    float dx = x1 - x0;
    float dy = y1 - y0;

    // sample count proportional to length
    float len = sqrtf(dx*dx + dy*dy);
    int steps = (int)ceilf(len); // 1 stamp per pixel of length
    steps = max(1, min(steps, 4096)); // safety cap

    float inv = 1.0f / (float)steps;
    for(int s=0; s<=steps; s++){
        float t = s * inv;
        int cx = (int)lrintf(x0 + dx * t);
        int cy = (int)lrintf(y0 + dy * t);
        if(cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
        stamp_disc(buf, W, H, cx, cy, rad, col);
    }
}

// Compose to BGR over white background already baked in buf.
// Clamp and gamma (Canvas effectively displays sRGB-ish).
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

    float angle = atan2f((float)H * 0.5f - p.y, (float)W * 0.5f - p.x);
    // JS: tangential velocity * 0.1
    p.vx = cosf(angle + 3.1415926535f * 0.5f) * 0.1f;
    p.vy = sinf(angle + 3.1415926535f * 0.5f) * 0.1f;

    p.r = (uint8_t)irand(rng, 0, 255);
    p.g = (uint8_t)irand(rng, 0, 255);
    p.b = (uint8_t)irand(rng, 0, 255);

    p.m = frand(rng, 0.0f, 10.0f);
    return p;
}

int main(int argc, char** argv){
    const int W = 3840;
    const int H = 2160;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    std::string encoder = "nvenc";

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) encoder = argv[3];

    const int totalFrames = seconds * fps;
    const int Npix = W * H;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Res: %dx%d | FPS: %d | Seconds: %d | Frames: %d\n", W, H, fps, seconds, totalFrames);

    // JS: starts with 100, grows every 100 frames by 1 + 20 clones until 1000
    const int startParticles = 100;
    const int maxParticles   = 1000;

    // JS fade: fillStyle rgba(255,255,255,0.01) then fillRect => mul=0.99, add=0.01
    const float fadeAlpha = 0.01f;
    const float fadeMul = 1.0f - fadeAlpha; // 0.99
    const float fadeAdd = fadeAlpha;        // toward white

    // Display gamma
    const float gammaInv = 1.0f / 2.2f;

    // Host particles (dynamic)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::vector<Particle> hP;
    hP.reserve(maxParticles + 64);
    for(int i=0; i<startParticles; i++) hP.push_back(make_particle(rng, W, H));

    // Device
    Particle* dP = nullptr;
    float3* dBuf = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxParticles));
    CUDA_CHECK(cudaMalloc(&dBuf, sizeof(float3) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    // init buffer to white
    {
        std::vector<float3> init((size_t)Npix);
        for(size_t i=0;i<init.size();i++) init[i] = make_float3(1.0f,1.0f,1.0f);
        CUDA_CHECK(cudaMemcpy(dBuf, init.data(), sizeof(float3)*(size_t)Npix, cudaMemcpyHostToDevice));
    }

    // pinned host frame
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

    for(int frame=0; frame<totalFrames; frame++){
        // Growth rule (JS): if tiempo%100==0 and n<1000:
        // add 1 new particle; then 20 clones with offsets in [-10..10]*2? (JS: (rand-0.5)*20)
        if(frame > 0 && (frame % 100) == 0 && (int)hP.size() < maxParticles){
            int tempIndex = (int)hP.size();
            hP.push_back(make_particle(rng, W, H));
            Particle parent = hP[tempIndex];

            for(int k=0; k<20 && (int)hP.size() < maxParticles; k++){
                Particle c = make_particle(rng, W, H);
                float randx = (frand(rng, -0.5f, 0.5f)) * 20.0f;
                float randy = (frand(rng, -0.5f, 0.5f)) * 20.0f;
                c.x  = parent.x  + randx;
                c.y  = parent.y  + randy;
                c.x2 = parent.x2 + randx;
                c.y2 = parent.y2 + randy;
                hP.push_back(c);
            }
        }

        int N = (int)hP.size();

        // upload particles (small enough; simplest faithful behavior)
        CUDA_CHECK(cudaMemcpyAsync(dP, hP.data(), sizeof(Particle)*(size_t)N, cudaMemcpyHostToDevice, stream));

        // 1) fade toward white (like fillRect alpha=0.01 on white)
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_to_white<<<grid, block, 0, stream>>>(dBuf, Npix, fadeMul, fadeAdd);
        }

        // 2) step physics
        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_step_particles<<<grid, block, 0, stream>>>(dP, N);
        }

        // 3) draw lines into buffer (overwrites color)
        {
            int block = 64; // drawing is heavier; fewer threads can be friendlier
            int grid = (N + block - 1) / block;
            k_draw_lines<<<grid, block, 0, stream>>>(dP, N, dBuf, W, H);
        }

        // 4) compose to BGR
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf, dBGR, Npix, gammaInv);
        }

        CUDA_CHECK(cudaGetLastError());

        // download frame
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // write to ffmpeg
        size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
        if(written != (size_t)Npix*3){
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", frame);
            break;
        }

        // cull out-of-bounds like JS splice
        CUDA_CHECK(cudaMemcpyAsync(hP.data(), dP, sizeof(Particle)*(size_t)N, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        hP.erase(
            std::remove_if(hP.begin(), hP.end(), [&](const Particle& p){
                return (p.x < -0.5f * W) || (p.x > 1.5f * W) || (p.y < -0.5f * H) || (p.y > 1.5f * H);
            }),
            hP.end()
        );

        if((frame % 60) == 0){
            fprintf(stderr, "\rFrame %d / %d | Particles: %d", frame, totalFrames, (int)hP.size());
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
    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
