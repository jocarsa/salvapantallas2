// vortex_growth_imagedata_fade_4k_cuda_mp4.cu
//
// CUDA conversion of the provided JS ImageData-based animation.
// Key match:
// - Persistent RGBA8 buffer = ImageData
// - Each frame: alpha-- for every pixel (clamped) like JS
// - Stamp particle squares by overwriting RGBA = (r,g,b,255)
// - Particle motion + neighbor accel (N <= 100, O(N^2) is fine)
// - Every 10 frames, add 1 particle + 20 clones near it (until 100)
//
// Build:
//   nvcc -O3 -std=c++17 vortex_growth_imagedata_fade_4k_cuda_mp4.cu -o vortex_growth_imagedata_fade_4k_cuda_mp4 \
//     $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_growth_imagedata_fade_4k_cuda_mp4 out.mp4 10
//   ./vortex_growth_imagedata_fade_4k_cuda_mp4 out.mp4 10 nvenc
//   ./vortex_growth_imagedata_fade_4k_cuda_mp4 out.mp4 10 libx265

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

struct Particle {
    float x, y;
    float x2, y2;   // kept for parity; not used unless you add line drawing
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

// ----------------------- GPU kernels -----------------------

// JS: for every pixel: alpha -= 1 (clamp >=0)
__global__ void k_fade_alpha_1(uchar4* img, int Npix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    uchar4 c = img[i];
    c.w = (c.w > 0) ? (unsigned char)(c.w - 1) : 0;
    img[i] = c;
}

// Step particles with JS logic (O(N^2), N <= 100)
__global__ void k_step_particles(Particle* p, int n, int W, int H){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // Integrate position (JS: vx/1250, vy/1250)
    me.x += me.vx / 1250.0f;
    me.y += me.vy / 1250.0f;

    // Neighbor interaction within box 3500x3500 (almost whole screen at 4K)
    float dvx = 0.0f, dvy = 0.0f;

    for(int j=0; j<n; j++){
        if(j == i) continue;

        Particle pj = p[j];

        float dx = pj.x - me.x;
        float dy = pj.y - me.y;

        if(fabsf(dx) < 3500.0f && fabsf(dy) < 3500.0f){
            float d2 = dx*dx + dy*dy;

            // JS: distancia = sqrt(a*a + b*b); vx += (cos(angle)/dist)*(m+1)
            // which equals dx/dist^2 * (m+1). Add softening to avoid blow-ups.
            float inv = 1.0f / (d2 + 1.0f);   // softening ~ +1 px^2
            float f = (pj.m + 1.0f) * inv;

            dvx += dx * f;
            dvy += dy * f;
        }
    }

    me.vx += dvx;
    me.vy += dvy;

    // Save previous (parity)
    me.x2 = me.x;
    me.y2 = me.y;

    p[i] = me;
}

// Stamp square: overwrite RGBA exactly like JS sets imagen.data[...] = r,g,b,255
// anchuraParticula in JS = 4, loops x=-4..3, y=-4..3 => 8x8
__global__ void k_stamp_squares(
    const Particle* p, int n,
    uchar4* img,
    int W, int H,
    int halfSize // 4
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];
    int cx = (int)lrintf(me.x);
    int cy = (int)lrintf(me.y);

    // Overwrite pixels
    for(int oy = -halfSize; oy < halfSize; oy++){
        int y = cy + oy;
        if(y < 0 || y >= H) continue;
        for(int ox = -halfSize; ox < halfSize; ox++){
            int x = cx + ox;
            if(x < 0 || x >= W) continue;

            int idx = y * W + x;

            // Race if overlapping; JS is “last write wins” in loop order.
            // This is acceptable visually for N<=100.
            img[idx] = make_uchar4(me.r, me.g, me.b, (unsigned char)255);
        }
    }
}

// Convert RGBA buffer to BGR24 for ffmpeg.
// Canvas has black background; ImageData alpha affects visibility.
// We output: rgb * (alpha/255).
__global__ void k_rgba_to_bgr(
    const uchar4* img,
    unsigned char* outBGR,
    int Npix
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    uchar4 c = img[i];
    float a = c.w * (1.0f / 255.0f);

    unsigned char b = (unsigned char)lrintf((float)c.z * a);
    unsigned char g = (unsigned char)lrintf((float)c.y * a);
    unsigned char r = (unsigned char)lrintf((float)c.x * a);

    outBGR[3*i + 0] = b;
    outBGR[3*i + 1] = g;
    outBGR[3*i + 2] = r;
}

// ----------------------- Host particle constructor -----------------------
static Particle make_particle(std::mt19937& rng, int W, int H){
    Particle p{};
    p.x = frand(rng, 0.0f, (float)W);
    p.y = frand(rng, 0.0f, (float)H);
    p.x2 = p.x;
    p.y2 = p.y;

    float angle = atan2f((float)H * 0.5f - p.y, (float)W * 0.5f - p.x);
    p.vx = cosf(angle + 3.1415926535f * 0.5f) * 200.0f;
    p.vy = sinf(angle + 3.1415926535f * 0.5f) * 200.0f;

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

    // JS parameters
    const int anchuraParticula = 4; // half-size, JS uses [-4..3] so stamp is 8x8
    const int maxParticles = 100;

    // Host particle list (dynamic, matches JS splice/add logic)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<Particle> hP;
    hP.reserve(maxParticles + 32);

    // JS starts with 1 particle
    hP.push_back(make_particle(rng, W, H));

    // Device
    Particle* dP = nullptr;
    uchar4* dImg = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxParticles));
    CUDA_CHECK(cudaMalloc(&dImg, sizeof(uchar4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    // init image buffer to 0 (transparent black)
    CUDA_CHECK(cudaMemset(dImg, 0, sizeof(uchar4) * (size_t)Npix));

    // Pinned host frame
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
        // --- Growth rule (JS): every 10 frames, if n<100:
        // add 1 new particle; then 20 clones around it (rand +/-200)
        if(frame > 0 && (frame % 10) == 0 && (int)hP.size() < maxParticles){
            int tempIndex = (int)hP.size();
            hP.push_back(make_particle(rng, W, H)); // new base particle

            // spawn up to 20 clones around the just-added particle position
            Particle parent = hP[tempIndex];
            for(int k=0; k<20 && (int)hP.size() < maxParticles; k++){
                Particle c = make_particle(rng, W, H);
                float randx = (frand(rng, -0.5f, 0.5f)) * 200.0f;
                float randy = (frand(rng, -0.5f, 0.5f)) * 200.0f;
                c.x  = parent.x  + randx;
                c.y  = parent.y  + randy;
                c.x2 = parent.x2 + randx;
                c.y2 = parent.y2 + randy;
                hP.push_back(c);
            }
        }

        int N = (int)hP.size();

        // Upload particles (N is tiny; this keeps the dynamic behavior simple & faithful)
        CUDA_CHECK(cudaMemcpyAsync(dP, hP.data(), sizeof(Particle) * (size_t)N, cudaMemcpyHostToDevice, stream));

        // 1) Fade alpha (JS: alpha--)
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_alpha_1<<<grid, block, 0, stream>>>(dImg, Npix);
        }

        // 2) Step particles (physics)
        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_step_particles<<<grid, block, 0, stream>>>(dP, N, W, H);
        }

        // 3) Stamp squares into ImageData buffer
        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_stamp_squares<<<grid, block, 0, stream>>>(dP, N, dImg, W, H, anchuraParticula);
        }

        // 4) Convert RGBA->BGR (alpha over black)
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_rgba_to_bgr<<<grid, block, 0, stream>>>(dImg, dBGR, Npix);
        }

        CUDA_CHECK(cudaGetLastError());

        // Download BGR
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)Npix*3, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Write frame
        size_t written = fwrite(hFramePinned, 1, (size_t)Npix*3, ff);
        if(written != (size_t)Npix*3){
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", frame);
            break;
        }

        // Optional: cull out-of-bounds like JS splice
        // JS removes if x < -0.5W || x > 1.5W || y < -0.5H || y > 1.5H
        // We need updated particles from device, so pull just the small list.
        CUDA_CHECK(cudaMemcpyAsync(hP.data(), dP, sizeof(Particle) * (size_t)N, cudaMemcpyDeviceToHost, stream));
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
    CUDA_CHECK(cudaFree(dImg));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
