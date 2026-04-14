// vortex_growth_imagedata_fade_cuda_mp4_preview_centered.cu
//
// Full CUDA renderer with:
// - Persistent RGBA framebuffer on GPU
// - Alpha fade each frame
// - Particle square stamping
// - Live OpenCV preview while encoding
// - Default resolution: 1280x720
// - Default duration: 36000 seconds
// - Camera offset so image center = center of gravity of all particles
// - No zoom change
//
// Build:
//   nvcc -O3 -std=c++17 vortex_growth_imagedata_fade_cuda_mp4_preview_centered.cu -o vortex_growth_imagedata_fade_cuda_mp4_preview_centered \
//     $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered out.mp4
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered out.mp4 120
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered out.mp4 120 nvenc
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered out.mp4 120 nvenc 1920 1080
//   ./vortex_growth_imagedata_fade_cuda_mp4_preview_centered out.mp4 120 nvenc 1920 1080 nopreview

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
#include <iostream>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);         \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

struct Particle {
    float x, y;
    float x2, y2;
    float vx, vy;
    uint8_t r, g, b;
    float m;
};

static inline float frand(std::mt19937& rng, float a, float b) {
    std::uniform_real_distribution<float> d(a, b);
    return d(rng);
}

static inline int irand(std::mt19937& rng, int a, int b) {
    std::uniform_int_distribution<int> d(a, b);
    return d(rng);
}

static FILE* open_ffmpeg_pipe(const std::string& outPath, int W, int H, int fps, const std::string& encoder) {
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
            "-rc vbr -cq 28 -b:v 0 "
            "-tag:v hvc1 "
            "\"" + outPath + "\"";
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        fprintf(stderr, "Failed to start ffmpeg. Command:\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

__global__ void k_fade_alpha_1(uchar4* framebuffer, int Npix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npix) return;

    uchar4 c = framebuffer[i];
    c.w = (c.w > 0) ? (unsigned char)(c.w - 1) : 0;
    framebuffer[i] = c;
}

__global__ void k_step_particles(Particle* p, int n, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    me.x += me.vx / 1250.0f;
    me.y += me.vy / 1250.0f;

    float dvx = 0.0f;
    float dvy = 0.0f;

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        Particle pj = p[j];
        float dx = pj.x - me.x;
        float dy = pj.y - me.y;

        if (fabsf(dx) < 3500.0f && fabsf(dy) < 3500.0f) {
            float d2 = dx * dx + dy * dy;
            float inv = 1.0f / (d2 + 1.0f);
            float f = (pj.m + 1.0f) * inv;
            dvx += dx * f;
            dvy += dy * f;
        }
    }

    me.vx += dvx;
    me.vy += dvy;

    me.x2 = me.x;
    me.y2 = me.y;

    p[i] = me;
}

__global__ void k_stamp_squares(
    const Particle* p, int n,
    uchar4* framebuffer,
    int W, int H,
    int halfSize,
    float renderOffsetX,
    float renderOffsetY
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    int cx = (int)lrintf(me.x + renderOffsetX);
    int cy = (int)lrintf(me.y + renderOffsetY);

    for (int oy = -halfSize; oy < halfSize; oy++) {
        int y = cy + oy;
        if (y < 0 || y >= H) continue;

        for (int ox = -halfSize; ox < halfSize; ox++) {
            int x = cx + ox;
            if (x < 0 || x >= W) continue;

            int idx = y * W + x;
            framebuffer[idx] = make_uchar4(me.r, me.g, me.b, (unsigned char)255);
        }
    }
}

__global__ void k_rgba_to_bgr(
    const uchar4* framebuffer,
    unsigned char* outBGR,
    int Npix
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npix) return;

    uchar4 c = framebuffer[i];
    float a = c.w * (1.0f / 255.0f);

    unsigned char b = (unsigned char)lrintf((float)c.z * a);
    unsigned char g = (unsigned char)lrintf((float)c.y * a);
    unsigned char r = (unsigned char)lrintf((float)c.x * a);

    outBGR[3 * i + 0] = b;
    outBGR[3 * i + 1] = g;
    outBGR[3 * i + 2] = r;
}

static Particle make_particle(std::mt19937& rng, int W, int H) {
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

static void compute_center_of_gravity(
    const std::vector<Particle>& particles,
    float& cx,
    float& cy
) {
    if (particles.empty()) {
        cx = 0.0f;
        cy = 0.0f;
        return;
    }

    double sumWeightedX = 0.0;
    double sumWeightedY = 0.0;
    double sumMass = 0.0;

    for (const auto& p : particles) {
        double mass = std::max(0.0001, (double)p.m);
        sumWeightedX += (double)p.x * mass;
        sumWeightedY += (double)p.y * mass;
        sumMass += mass;
    }

    if (sumMass <= 0.0) {
        double sx = 0.0;
        double sy = 0.0;
        for (const auto& p : particles) {
            sx += p.x;
            sy += p.y;
        }
        cx = (float)(sx / particles.size());
        cy = (float)(sy / particles.size());
    } else {
        cx = (float)(sumWeightedX / sumMass);
        cy = (float)(sumWeightedY / sumMass);
    }
}

int main(int argc, char** argv) {
    int W = 1280;
    int H = 720;
    const int fps = 60;
    std::string outPath = "out.mp4";
    int seconds = 36000;
    std::string encoder = "nvenc";
    bool showPreview = true;

    if (argc >= 2) outPath = argv[1];
    if (argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) encoder = argv[3];
    if (argc >= 5) W = std::max(16, std::atoi(argv[4]));
    if (argc >= 6) H = std::max(16, std::atoi(argv[5]));
    if (argc >= 7) {
        std::string flag = argv[6];
        if (flag == "nopreview") showPreview = false;
    }

    const int totalFrames = seconds * fps;
    const int Npix = W * H;
    const int anchuraParticula = 4;
    const int maxParticles = 100;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Resolution: %dx%d\n", W, H);
    fprintf(stderr, "FPS: %d\n", fps);
    fprintf(stderr, "Duration: %d seconds\n", seconds);
    fprintf(stderr, "Frames: %d\n", totalFrames);
    fprintf(stderr, "Encoder: %s\n", encoder.c_str());
    fprintf(stderr, "Persistent framebuffer: ACTIVE\n");
    fprintf(stderr, "Preview window: %s\n", showPreview ? "ON" : "OFF");
    fprintf(stderr, "Camera mode: center image on center of gravity\n");

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::vector<Particle> hP;
    hP.reserve(maxParticles + 32);
    hP.push_back(make_particle(rng, W, H));

    Particle* dP = nullptr;
    uchar4* dFramebuffer = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(Particle) * (size_t)maxParticles));
    CUDA_CHECK(cudaMalloc(&dFramebuffer, sizeof(uchar4) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMemset(dFramebuffer, 0, sizeof(uchar4) * (size_t)Npix));

    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)Npix * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (showPreview) {
        cv::namedWindow("Framebuffer Preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("Framebuffer Preview", W, H);
    }

    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if (!ff) {
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        if (showPreview) cv::destroyAllWindows();
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFreeHost(hFramePinned));
        CUDA_CHECK(cudaFree(dP));
        CUDA_CHECK(cudaFree(dFramebuffer));
        CUDA_CHECK(cudaFree(dBGR));
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int framesWritten = 0;

    for (int frame = 0; frame < totalFrames; frame++) {
        if (frame > 0 && (frame % 10) == 0 && (int)hP.size() < maxParticles) {
            int tempIndex = (int)hP.size();
            hP.push_back(make_particle(rng, W, H));

            Particle parent = hP[tempIndex];
            for (int k = 0; k < 20 && (int)hP.size() < maxParticles; k++) {
                Particle c = make_particle(rng, W, H);
                float randx = frand(rng, -100.0f, 100.0f);
                float randy = frand(rng, -100.0f, 100.0f);
                c.x  = parent.x  + randx;
                c.y  = parent.y  + randy;
                c.x2 = parent.x2 + randx;
                c.y2 = parent.y2 + randy;
                hP.push_back(c);
            }
        }

        int N = (int)hP.size();

        CUDA_CHECK(cudaMemcpyAsync(
            dP,
            hP.data(),
            sizeof(Particle) * (size_t)N,
            cudaMemcpyHostToDevice,
            stream
        ));

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_alpha_1<<<grid, block, 0, stream>>>(dFramebuffer, Npix);
        }

        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_step_particles<<<grid, block, 0, stream>>>(dP, N, W, H);
        }

        CUDA_CHECK(cudaGetLastError());

        // Pull updated particle positions to host so we can compute the center of gravity.
        CUDA_CHECK(cudaMemcpyAsync(
            hP.data(),
            dP,
            sizeof(Particle) * (size_t)N,
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float cogX = 0.0f;
        float cogY = 0.0f;
        compute_center_of_gravity(hP, cogX, cogY);

        float renderOffsetX = (float)W * 0.5f - cogX;
        float renderOffsetY = (float)H * 0.5f - cogY;

        {
            int block = 128;
            int grid = (N + block - 1) / block;
            k_stamp_squares<<<grid, block, 0, stream>>>(
                dP, N, dFramebuffer, W, H, anchuraParticula, renderOffsetX, renderOffsetY
            );
        }

        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_rgba_to_bgr<<<grid, block, 0, stream>>>(dFramebuffer, dBGR, Npix);
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            hFramePinned,
            dBGR,
            (size_t)Npix * 3,
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (showPreview) {
            cv::Mat preview(H, W, CV_8UC3, hFramePinned);
            cv::imshow("Framebuffer Preview", preview);
            int key = cv::waitKey(1);
            if (key == 27) {
                fprintf(stderr, "\nStopped by user with ESC.\n");
                break;
            }
        }

        size_t written = fwrite(hFramePinned, 1, (size_t)Npix * 3, ff);
        if (written != (size_t)Npix * 3) {
            fprintf(stderr, "FFmpeg write failed at frame %d.\n", frame);
            break;
        }

        framesWritten++;

        hP.erase(
            std::remove_if(hP.begin(), hP.end(), [&](const Particle& p) {
                return (p.x < -0.5f * W) || (p.x > 1.5f * W) ||
                       (p.y < -0.5f * H) || (p.y > 1.5f * H);
            }),
            hP.end()
        );

        if ((frame % 60) == 0) {
            fprintf(stderr,
                    "\rFrame %d / %d | Particles: %d | COG: (%.1f, %.1f) | Offset: (%.1f, %.1f)",
                    frame, totalFrames, (int)hP.size(), cogX, cogY, renderOffsetX, renderOffsetY);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\nFinalizing encode...\n");
    fflush(ff);
    pclose(ff);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stderr, "Done. Time: %.2f s for %d written frames (%.2f fps effective)\n",
            sec, framesWritten, (sec > 0.0 ? (double)framesWritten / sec : 0.0));

    if (showPreview) {
        cv::destroyAllWindows();
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dFramebuffer));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
