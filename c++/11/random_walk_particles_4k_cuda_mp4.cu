// random_walk_particles_4k_cuda_mp4.cu
// CUDA adaptation of the provided JS canvas animation (random walk particles + color drift).
//
// JS reference behavior:
// - 100 particles, each frame:
//   - draw fillRect(x,y,3,3) with rgba(r,g,b,0.3) and NO clear
//   - bounce by flipping direction if out of bounds (d += PI)
//   - x += cos(d)*speed; y += sin(d)*speed; speed=3
//   - d += 0.1 + (Math.random()*-0.5)*0.2   -> ~ [0.0.08 .. 0.10]
//   - r,g,b drift by round((rand-0.5)*4) -> roughly [-2..+2]
//
// This CUDA version:
// - 4K (3840x2160), 60fps, offline MP4 HEVC encode via ffmpeg pipe.
// - Persistent float3 buffer with mild decay (prevents permanent burn-in at high res).
// - Each particle adds a 3x3 deposit with alpha ~ 0.3.
//
// Build:
//   nvcc -O3 -std=c++17 random_walk_particles_4k_cuda_mp4.cu -o random_walk_particles_4k_cuda_mp4 \
//     $(pkg-config --cflags --libs opencv4)
//
// Usage:
//   ./random_walk_particles_4k_cuda_mp4 out.mp4 10
//   ./random_walk_particles_4k_cuda_mp4 out.mp4 10 100
//   ./random_walk_particles_4k_cuda_mp4 out.mp4 10 100 nvenc
//   ./random_walk_particles_4k_cuda_mp4 out.mp4 10 100 libx265

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);        \
        std::exit(1);                                                \
    }                                                                 \
} while (0)

static inline float clampf_host(float v, float a, float b){
    return std::min(b, std::max(a, v));
}

__device__ __forceinline__ float clampf(float v, float a, float b){
    return fminf(b, fmaxf(a, v));
}

struct ParticleRW {
    float x, y;
    float d;        // direction angle (radians)
    float speed;    // pixels per frame (JS=3)
    float r, g, b;  // stored as 0..255-ish (drifts beyond; we emulate JS abs(mod 255))
    uint32_t seed;  // per-particle RNG seed
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

// ----------------------- Simple RNG (xorshift32) -----------------------
__device__ __forceinline__ uint32_t xorshift32(uint32_t& s){
    // xorshift32
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ __forceinline__ float rand01(uint32_t& s){
    // 0..1
    uint32_t v = xorshift32(s);
    return (v & 0x00FFFFFFu) / 16777215.0f;
}

__device__ __forceinline__ float wrap255_abs(float v){
    // JS draws: Math.abs(r % 255)
    // In JS, % is remainder that can be negative. We emulate:
    // remainder = fmod(v,255), abs(remainder).
    float rem = fmodf(v, 255.0f);
    return fabsf(rem);
}

// ----------------------- Buffer decay -----------------------
__global__ void k_fade_buf(float3* buf, int Npix, float mul){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;
    float3 c = buf[i];
    c.x *= mul;
    c.y *= mul;
    c.z *= mul;
    buf[i] = c;
}

// ----------------------- Particle step (JS logic) -----------------------
__global__ void k_step_particles(ParticleRW* p, int n, int W, int H){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    ParticleRW me = p[i];
    uint32_t s = me.seed;

    // Bounce if out of bounds (JS does it before moving)
    if(me.x < 0.0f || me.x > (float)W || me.y < 0.0f || me.y > (float)H){
        me.d += 3.1415926535f;
    }

    // Move
    me.x += cosf(me.d) * me.speed;
    me.y += sinf(me.d) * me.speed;

    // Direction update:
    // JS: d += 0.1 + (Math.random()*-0.5)*0.2  => 0.1 + rand * (-0.1) => [0.0 .. 0.1]
    // But the user comment suggests ~ [0.08..0.10]; their code actually gives [0.0..0.1].
    // We'll keep the literal math:
    float r0 = rand01(s);              // 0..1
    me.d += 0.1f + (r0 * -0.5f) * 0.2f; // 0.1 - 0.1*r0

    // Color drift:
    // JS: r += round((rand-0.5)*4) => approx [-2..+2]
    float rr = rand01(s) - 0.5f;
    float gg = rand01(s) - 0.5f;
    float bb = rand01(s) - 0.5f;

    me.r += nearbyintf(rr * 4.0f);
    me.g += nearbyintf(gg * 4.0f);
    me.b += nearbyintf(bb * 4.0f);

    me.seed = s;
    p[i] = me;
}

// ----------------------- Deposit 3x3 fillRect into buffer -----------------------
__global__ void k_deposit_3x3(
    const ParticleRW* p, int n,
    float3* buf,
    int W, int H,
    float alpha // ~0.3
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    ParticleRW me = p[i];

    int x = (int)lrintf(me.x);
    int y = (int)lrintf(me.y);

    // Convert to JS-like displayed RGB:
    float R = wrap255_abs(me.r) / 255.0f;
    float G = wrap255_abs(me.g) / 255.0f;
    float B = wrap255_abs(me.b) / 255.0f;

    // 3x3 fillRect at (x,y)
    // JS fillRect draws top-left aligned; here we approximate centered-ish.
    for(int oy = 0; oy < 3; oy++){
        for(int ox = 0; ox < 3; ox++){
            int xx = x + ox;
            int yy = y + oy;
            if(xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

            int idx = yy * W + xx;

            // Approximate alpha-blend on persistent buffer by additive deposit + global decay.
            // This matches the “paint accumulating” look reasonably well.
            atomicAdd(&buf[idx].x, R * alpha);
            atomicAdd(&buf[idx].y, G * alpha);
            atomicAdd(&buf[idx].z, B * alpha);
        }
    }
}

// ----------------------- Compose to BGR (tonemap) -----------------------
__global__ void k_compose_to_bgr(
    const float3* buf,
    unsigned char* outBGR,
    int Npix,
    float exposure,
    float gammaInv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= Npix) return;

    float r = buf[i].x;
    float g = buf[i].y;
    float b = buf[i].z;

    // clamp to avoid long-term blow-up
    r = fminf(r, 40.0f);
    g = fminf(g, 40.0f);
    b = fminf(b, 40.0f);

    // simple filmic-ish curve
    r = 1.0f - expf(-r * exposure);
    g = 1.0f - expf(-g * exposure);
    b = 1.0f - expf(-b * exposure);

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3*i + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3*i + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3*i + 2] = (unsigned char)lrintf(255.0f * r);
}

int main(int argc, char** argv){
    // Output settings (4K offline)
    const int W = 3840;
    const int H = 2160;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    int N = 100;                 // JS default
    std::string encoder = "nvenc";

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) N = std::max(1, std::atoi(argv[3]));
    if(argc >= 5) encoder = argv[4];

    const int totalFrames = seconds * fps;
    const int Npix = W * H;

    fprintf(stderr, "Output: %s\n", outPath.c_str());
    fprintf(stderr, "Res: %dx%d | FPS: %d | Seconds: %d | Frames: %d\n", W, H, fps, seconds, totalFrames);
    fprintf(stderr, "Particles: %d\n", N);

    // Rendering tuning
    // JS has no fade; at 4K, pure accumulation can saturate heavily. Use mild decay.
    // - 0.9995 => very long trails
    // - 0.995  => medium
    // - 0.990  => short
    const float decayMul = 0.999f;

    // JS fill alpha = 0.3
    const float alpha = 0.30f;

    // Tonemap
    const float exposure = 1.35f;
    const float gammaInv = 1.0f / 2.0f;

    // Init particles (host)
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> ua(0.0f, 6.2831853f);
    std::uniform_int_distribution<int>   uc(0, 255);

    // Scale speed so it feels similar despite 4K:
    // JS speed=3 at window size; at 4K it’s fine as-is, but you can tweak:
    float baseSpeed = 3.0f;

    std::vector<ParticleRW> hP(N);
    for(int i=0;i<N;i++){
        hP[i].x = ux(rng);
        hP[i].y = uy(rng);
        hP[i].d = ua(rng);
        hP[i].speed = baseSpeed;

        hP[i].r = (float)uc(rng);
        hP[i].g = (float)uc(rng);
        hP[i].b = (float)uc(rng);

        // unique-ish seed
        uint32_t s = (uint32_t)(i * 747796405u) ^ 0xA341316Cu;
        s ^= (uint32_t)rng();
        if(s == 0) s = 1;
        hP[i].seed = s;
    }

    // Device buffers
    ParticleRW* dP = nullptr;
    float3* dBuf = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP, sizeof(ParticleRW) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&dBuf, sizeof(float3) * (size_t)Npix));
    CUDA_CHECK(cudaMalloc(&dBGR, sizeof(unsigned char) * (size_t)Npix * 3));

    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(ParticleRW)*(size_t)N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBuf, 0, sizeof(float3)*(size_t)Npix));

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
        // Fade persistent buffer slightly (optional but recommended)
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_fade_buf<<<grid, block, 0, stream>>>(dBuf, Npix, decayMul);
        }

        // Step particles with JS logic
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_step_particles<<<grid, block, 0, stream>>>(dP, N, W, H);
        }

        // Deposit 3x3 rects
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            k_deposit_3x3<<<grid, block, 0, stream>>>(dP, N, dBuf, W, H, alpha);
        }

        // Compose to BGR
        {
            int block = 256;
            int grid = (Npix + block - 1) / block;
            k_compose_to_bgr<<<grid, block, 0, stream>>>(dBuf, dBGR, Npix, exposure, gammaInv);
        }

        CUDA_CHECK(cudaGetLastError());

        // Copy to host + write to ffmpeg
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
    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
