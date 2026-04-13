// cursor_glow_rotate_4k_cuda_mp4.cu
//
// FIXED to match Canvas transform semantics:
//
// JS does:
//   draw stuff
//   translate(cx,cy); rotate(0.001); translate(-cx,-cy);
// That transform affects ONLY FUTURE draws, NOT the already-drawn pixels.
//
// Therefore: we DO NOT rotate the framebuffer.
// We instead keep a global angle theta that increases each frame and we
// rotate the DRAW POSITION of each circle around the canvas center.
//
// Features:
// - 3840x2160 (4K), 60fps, offline encode to MP4 HEVC (NVENC default, libx265 optional)
// - 2000 cursors
// - soft circles with alpha=0.1 (additive into persistent HDR-ish buffer)
// - no clear (like JS), optional tonemap to avoid runaway saturation
//
// Build:
//   nvcc -O3 -std=c++17 cursor_glow_rotate_4k_cuda_mp4.cu -o cursor_glow_rotate_4k_cuda_mp4 \
//     $(pkg-config --cflags --libs opencv4)
//
// Run:
//   ./cursor_glow_rotate_4k_cuda_mp4 out.mp4 10
//   ./cursor_glow_rotate_4k_cuda_mp4 out.mp4 10 nvenc
//   ./cursor_glow_rotate_4k_cuda_mp4 out.mp4 10 libx265

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

struct Cursor {
    float x, y;
    float a;
    float r, g, b;   // 0..255
    float radius;    // 0..10
    uint32_t seed;   // per-cursor RNG seed
};

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

// ----------------------- RNG (xorshift32) -----------------------
__device__ __forceinline__ uint32_t xorshift32(uint32_t& s){
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
__device__ __forceinline__ float rand01(uint32_t& s){
    uint32_t v = xorshift32(s);
    return (v & 0x00FFFFFFu) / 16777215.0f; // 0..1
}
__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}

// ----------------------- Kernels -----------------------

// Move cursors (matches JS logic)
__global__ void k_step_cursors(Cursor* c, int n, int W, int H){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Cursor me = c[i];
    uint32_t s = me.seed;

    // position
    me.x += cosf(me.a) * 0.2f;
    me.y += sinf(me.a) * 0.2f;

    // color drift: (Math.random()-0.5)*5
    me.r += (rand01(s) - 0.5f) * 5.0f;
    me.g += (rand01(s) - 0.5f) * 5.0f;
    me.b += (rand01(s) - 0.5f) * 5.0f;

    me.r = clampf(me.r, 0.0f, 255.0f);
    me.g = clampf(me.g, 0.0f, 255.0f);
    me.b = clampf(me.b, 0.0f, 255.0f);

    // radius drift: += (Math.random()-0.5), clamp [0..10]
    me.radius += (rand01(s) - 0.5f);
    me.radius = clampf(me.radius, 0.0f, 10.0f);

    // bounce
    if(me.x < 0.0f || me.x > (float)W || me.y < 0.0f || me.y > (float)H){
        me.a += 3.14159265f;
    }

    me.seed = s;
    c[i] = me;
}

// Draw soft circles into persistent buffer (additive).
// IMPORTANT: we rotate the draw position by theta around the center (canvas-like transform).
__global__ void k_draw_circles(
    const Cursor* c, int n,
    float3* buf,
    int W, int H,
    float alphaBase, // 0.1
    float theta
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Cursor me = c[i];
    int rad = (int)ceilf(me.radius);
    if(rad <= 0) return;

    // rotate draw position around center
    float cx0 = W * 0.5f;
    float cy0 = H * 0.5f;

    float dx = me.x - cx0;
    float dy = me.y - cy0;

    float ca = cosf(theta);
    float sa = sinf(theta);

    float rx = dx * ca - dy * sa + cx0;
    float ry = dx * sa + dy * ca + cy0;

    int cx = (int)lrintf(rx);
    int cy = (int)lrintf(ry);

    float3 col;
    col.x = me.r / 255.0f;
    col.y = me.g / 255.0f;
    col.z = me.b / 255.0f;

    float r2 = (float)(rad * rad);
    float inv = 1.0f / (r2 + 1e-6f);

    // Gaussian-ish falloff
    for(int oy = -rad; oy <= rad; oy++){
        int py = cy + oy;
        if(py < 0 || py >= H) continue;

        for(int ox = -rad; ox <= rad; ox++){
            int px = cx + ox;
            if(px < 0 || px >= W) continue;

            float d2 = (float)(ox*ox + oy*oy);
            if(d2 > r2) continue;

            float a = alphaBase * expf(-d2 * inv); // <= alphaBase
            int idx = py * W + px;

            atomicAdd(&buf[idx].x, col.x * a);
            atomicAdd(&buf[idx].y, col.y * a);
            atomicAdd(&buf[idx].z, col.z * a);
        }
    }
}

// Tonemap + gamma to BGR
__global__ void k_compose(
    const float3* buf,
    unsigned char* out,
    int Npix,
    float exposure,
    float gammaInv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float r = buf[i].x;
    float g = buf[i].y;
    float b = buf[i].z;

    // prevent runaway
    r = fminf(r, 40.0f);
    g = fminf(g, 40.0f);
    b = fminf(b, 40.0f);

    // tonemap
    r = 1.0f - expf(-r * exposure);
    g = 1.0f - expf(-g * exposure);
    b = 1.0f - expf(-b * exposure);

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    out[3*i + 0] = (unsigned char)lrintf(b * 255.0f);
    out[3*i + 1] = (unsigned char)lrintf(g * 255.0f);
    out[3*i + 2] = (unsigned char)lrintf(r * 255.0f);
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
    const int N = 2000;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Res: %dx%d | FPS: %d | Seconds: %d | Frames: %d | Cursors: %d\n",
            W, H, fps, seconds, totalFrames, N);

    // init cursors (host)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> ua(0.0f, 6.2831853f);
    std::uniform_real_distribution<float> uc(0.0f, 255.0f);
    std::uniform_real_distribution<float> ur(0.0f, 10.0f);

    std::vector<Cursor> hC(N);
    for(int i=0;i<N;i++){
        hC[i].x = ux(rng);
        hC[i].y = uy(rng);
        hC[i].a = ua(rng);
        hC[i].r = uc(rng);
        hC[i].g = uc(rng);
        hC[i].b = uc(rng);
        hC[i].radius = ur(rng);

        uint32_t s = (uint32_t)(i * 747796405u) ^ 0x9E3779B9u;
        s ^= (uint32_t)rng();
        if(s == 0) s = 1;
        hC[i].seed = s;
    }

    // device allocations
    Cursor* dC = nullptr;
    float3* dBuf = nullptr;
    unsigned char* dOut = nullptr;

    CUDA_CHECK(cudaMalloc(&dC, sizeof(Cursor) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dBuf, sizeof(float3) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMemcpy(dC, hC.data(), sizeof(Cursor) * (size_t)N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBuf, 0, sizeof(float3) * (size_t)Npix));

    // pinned host frame
    unsigned char* hFrame = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFrame, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if(!ff){
        fprintf(stderr, "Failed to open ffmpeg pipe.\n");
        return 1;
    }

    // JS params
    const float rotatePerFrame = 0.001f; // canvas.rotate(0.001)
    const float circleAlpha = 0.1f;      // fill alpha
    const float exposure = 1.8f;         // tonemap exposure
    const float gammaInv = 1.0f / 2.2f;

    float theta = 0.0f;

    for(int f=0; f<totalFrames; f++){
        theta += rotatePerFrame;

        k_step_cursors<<<(N + 255)/256, 256, 0, stream>>>(dC, N, W, H);
        k_draw_circles<<<(N + 127)/128, 128, 0, stream>>>(dC, N, dBuf, W, H, circleAlpha, theta);
        k_compose<<<(Npix + 255)/256, 256, 0, stream>>>(dBuf, dOut, Npix, exposure, gammaInv);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(hFrame, dOut, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t written = fwrite(hFrame, 1, (size_t)Npix*3, ff);
        if(written != (size_t)Npix*3){
            fprintf(stderr, "ffmpeg write failed at frame %d.\n", f);
            break;
        }

        if((f % 60) == 0){
            fprintf(stderr, "\rFrame %d / %d", f, totalFrames);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\nFinalizing...\n");
    fflush(ff);
    pclose(ff);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFrame));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dOut));

    return 0;
}
