// cells_cuda_render_mp4.cu
// Offline CUDA particle renderer -> FFmpeg MP4 (H.265)
// Builds a 1920x1080 60fps video as fast as possible.
//
// Usage examples:
//   ./cells_cuda_render_mp4 out.mp4 10         # 10 seconds, NVENC if available (default)
//   ./cells_cuda_render_mp4 out.mp4 10 libx265 # force CPU x265 ultrafast
//
// Requirements:
//   - OpenCV (only for Mat convenience; not used for encoding)
//   - FFmpeg installed (ffmpeg in PATH)
//   - For fastest encode on NVIDIA: FFmpeg with hevc_nvenc enabled

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
    float r, g, b; // 0..1
};

__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(b, fmaxf(a, v));
}
__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__global__ void k_update_particles(
    Particle* p, int n, float dt,
    int W, int H,
    float cellRadius,
    float neighborRadius,
    float repelStrength,
    float swirlStrength,
    float damping
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    // Swirl around center (pretty motion)
    float cx = 0.5f * W, cy = 0.5f * H;
    float dxC = me.x - cx;
    float dyC = me.y - cy;
    float invLenC = rsqrtf(dxC*dxC + dyC*dyC + 1e-6f);

    float ax = (-dyC * invLenC) * swirlStrength;
    float ay = ( dxC * invLenC) * swirlStrength;

    // Soft collisions / repulsion
    float nr2 = neighborRadius * neighborRadius;
    float minDist = cellRadius * 2.0f;
    float minDist2 = minDist * minDist;

    for(int j=0;j<n;j++){
        if(j==i) continue;
        float dx = me.x - p[j].x;
        float dy = me.y - p[j].y;
        float d2 = dx*dx + dy*dy;
        if(d2 < nr2){
            float invD = rsqrtf(d2 + 1e-6f);
            float d = 1.0f / invD;

            float overlap = fmaxf(0.0f, (minDist - d));
            float wOverlap = overlap / (minDist + 1e-6f);

            float wNear = 1.0f - (d / (neighborRadius + 1e-6f));
            wNear = clampf(wNear, 0.0f, 1.0f);

            float w = repelStrength * (0.10f*wNear + 2.10f*wOverlap);
            ax += (dx * invD) * w;
            ay += (dy * invD) * w;
        }
    }

    me.vx = (me.vx + ax * dt) * damping;
    me.vy = (me.vy + ay * dt) * damping;

    me.x += me.vx * dt;
    me.y += me.vy * dt;

    // Boundary bounce
    float pad = cellRadius + 2.0f;
    if(me.x < pad){ me.x = pad; me.vx = fabsf(me.vx); }
    if(me.x > W - pad){ me.x = W - pad; me.vx = -fabsf(me.vx); }
    if(me.y < pad){ me.y = pad; me.vy = fabsf(me.vy); }
    if(me.y > H - pad){ me.y = H - pad; me.vy = -fabsf(me.vy); }

    p[i] = me;
}

__global__ void k_clear_accum(float4* accum, int W, int H){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = W * H;
    if(idx >= N) return;
    accum[idx] = make_float4(0,0,0,0);
}

__global__ void k_draw_glow(
    const Particle* p, int n,
    float4* accum, int W, int H,
    float cellRadius,
    float glowRadius,
    float glowIntensity,
    float coreBoost
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    Particle me = p[i];

    float R = glowRadius;
    int x0 = (int)floorf(me.x - R);
    int x1 = (int)ceilf (me.x + R);
    int y0 = (int)floorf(me.y - R);
    int y1 = (int)ceilf (me.y + R);

    x0 = max(0, x0); y0 = max(0, y0);
    x1 = min(W-1, x1); y1 = min(H-1, y1);

    // Glow falloff
    float sigma = 0.33f * R;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma + 1e-6f);

    for(int y=y0;y<=y1;y++){
        for(int x=x0;x<=x1;x++){
            float dx = (x + 0.5f) - me.x;
            float dy = (y + 0.5f) - me.y;
            float d2 = dx*dx + dy*dy;
            if(d2 > R*R) continue;

            float d = sqrtf(d2 + 1e-6f);

            // Sharper/brighter core so particles read clearly
            float core = 1.0f - smoothstep(cellRadius*0.95f, cellRadius*1.02f, d);
            core = clampf(core, 0.0f, 1.0f);

            float glow = expf(-d2 * inv2s2);

            float w = glowIntensity * (0.80f*glow + coreBoost*core);

            int idx = y * W + x;
            atomicAdd(&accum[idx].x, me.r * w);
            atomicAdd(&accum[idx].y, me.g * w);
            atomicAdd(&accum[idx].z, me.b * w);
            atomicAdd(&accum[idx].w, w);
        }
    }
}

__global__ void k_tonemap_to_bgr(
    const float4* accum,
    unsigned char* outBGR, int W, int H,
    float exposure,
    float gammaInv,
    float lift // raises dark tones (makes hues less "dark side")
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = W * H;
    if(idx >= N) return;

    float4 a = accum[idx];

    // Tonemap: lift + filmic-ish exp
    float r = 1.0f - expf(-(a.x + lift) * exposure);
    float g = 1.0f - expf(-(a.y + lift) * exposure);
    float b = 1.0f - expf(-(a.z + lift) * exposure);

    r = powf(clampf(r, 0.0f, 1.0f), gammaInv);
    g = powf(clampf(g, 0.0f, 1.0f), gammaInv);
    b = powf(clampf(b, 0.0f, 1.0f), gammaInv);

    outBGR[3*idx + 0] = (unsigned char)lrintf(255.0f * b);
    outBGR[3*idx + 1] = (unsigned char)lrintf(255.0f * g);
    outBGR[3*idx + 2] = (unsigned char)lrintf(255.0f * r);
}

// High saturation bright palette using HSV -> RGB
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

// Build an ffmpeg command and open a pipe for raw bgr24 frames
static FILE* open_ffmpeg_pipe(
    const std::string& outPath, int W, int H, int fps,
    const std::string& encoder // "nvenc" or "libx265"
){
    // As-fast-as-possible presets:
    // - NVENC: hevc_nvenc preset p1 (fastest) + low-latency tune
    // - libx265: ultrafast
    //
    // NOTE: If your ffmpeg doesn't support hevc_nvenc, use libx265.
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
        // default: NVENC
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

    // popen is POSIX; on Windows use _popen
    FILE* pipe = popen(cmd.c_str(), "w");
    if(!pipe){
        fprintf(stderr, "Failed to start ffmpeg. Command was:\n%s\n", cmd.c_str());
        return nullptr;
    }
    return pipe;
}

int main(int argc, char** argv){
    // Defaults requested
    const int W = 1920;
    const int H = 1080;
    const int fps = 60;

    std::string outPath = "out.mp4";
    int seconds = 10;
    std::string encoder = "nvenc"; // "nvenc" or "libx265"

    if(argc >= 2) outPath = argv[1];
    if(argc >= 3) seconds = std::max(1, std::atoi(argv[2]));
    if(argc >= 4) encoder = argv[3]; // "nvenc" or "libx265"

    const int N = 240;
    const int totalFrames = seconds * fps;
    const float dt = 1.0f / (float)fps;

    // Visual/physics tuned for clarity + brightness
    const float cellRadius     = 10.5f;
    const float neighborRadius = 110.0f;
    const float repelStrength  = 44.0f;
    const float swirlStrength  = 22.0f;
    const float damping        = 0.996f;

    const float glowRadius     = 46.0f;
    const float glowIntensity  = 0.030f;  // brighter overall
    const float coreBoost      = 2.60f;   // much clearer particle disks

    const float exposure       = 1.55f;   // brighter
    const float gammaInv       = 1.0f / 2.0f; // slightly less dark than 2.2
    const float lift           = 0.010f;  // lifts shadows a bit

    // Init particles
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ux(0.0f, (float)W);
    std::uniform_real_distribution<float> uy(0.0f, (float)H);
    std::uniform_real_distribution<float> uv(-120.0f, 120.0f);
    std::uniform_real_distribution<float> uh(0.0f, 1.0f);

    std::vector<Particle> hP(N);
    for(int i=0;i<N;i++){
        hP[i].x = ux(rng);
        hP[i].y = uy(rng);
        hP[i].vx = uv(rng);
        hP[i].vy = uv(rng);

        // Brighter, more saturated palette:
        float h = uh(rng);
        float s = 0.78f;  // saturation
        float v = 1.00f;  // value/brightness
        hsv2rgb(h, s, v, hP[i].r, hP[i].g, hP[i].b);

        // Small mix with white to avoid muddy dark hues
        float mixW = 0.18f;
        hP[i].r = hP[i].r*(1.0f-mixW) + 1.0f*mixW;
        hP[i].g = hP[i].g*(1.0f-mixW) + 1.0f*mixW;
        hP[i].b = hP[i].b*(1.0f-mixW) + 1.0f*mixW;
    }

    // Device buffers
    Particle* dP = nullptr;
    float4* dAccum = nullptr;
    unsigned char* dBGR = nullptr;

    CUDA_CHECK(cudaMalloc(&dP,     sizeof(Particle) * N));
    CUDA_CHECK(cudaMalloc(&dAccum, sizeof(float4) * (size_t)W * H));
    CUDA_CHECK(cudaMalloc(&dBGR,   sizeof(unsigned char) * (size_t)W * H * 3));
    CUDA_CHECK(cudaMemcpy(dP, hP.data(), sizeof(Particle)*N, cudaMemcpyHostToDevice));

    // Pinned host buffer for faster device->host copies
    unsigned char* hFramePinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hFramePinned, (size_t)W * H * 3));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Open ffmpeg pipe
    FILE* ff = open_ffmpeg_pipe(outPath, W, H, fps, encoder);
    if(!ff){
        fprintf(stderr, "Could not open FFmpeg pipe.\n");
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for(int f=0; f<totalFrames; f++){
        // Update
        {
            int block = 128;
            int grid  = (N + block - 1) / block;
            k_update_particles<<<grid, block, 0, stream>>>(
                dP, N, dt,
                W, H,
                cellRadius,
                neighborRadius,
                repelStrength,
                swirlStrength,
                damping
            );
        }

        // Clear accum
        {
            int total = W * H;
            int block = 256;
            int grid  = (total + block - 1) / block;
            k_clear_accum<<<grid, block, 0, stream>>>(dAccum, W, H);
        }

        // Draw
        {
            int block = 64;
            int grid  = (N + block - 1) / block;
            k_draw_glow<<<grid, block, 0, stream>>>(
                dP, N,
                dAccum, W, H,
                cellRadius,
                glowRadius,
                glowIntensity,
                coreBoost
            );
        }

        // Tonemap
        {
            int total = W * H;
            int block = 256;
            int grid  = (total + block - 1) / block;
            k_tonemap_to_bgr<<<grid, block, 0, stream>>>(
                dAccum, dBGR, W, H,
                exposure, gammaInv, lift
            );
        }

        CUDA_CHECK(cudaGetLastError());

        // Copy frame back (async) then write to ffmpeg
        CUDA_CHECK(cudaMemcpyAsync(hFramePinned, dBGR, (size_t)W*H*3,
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t written = fwrite(hFramePinned, 1, (size_t)W*H*3, ff);
        if(written != (size_t)W*H*3){
            fprintf(stderr, "FFmpeg pipe write failed at frame %d.\n", f);
            break;
        }

        // Optional: progress
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
    fprintf(stderr, "Done. Render+encode time: %.2f s for %d frames (%.2f fps effective)\n",
            sec, totalFrames, totalFrames / sec);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hFramePinned));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dAccum));
    CUDA_CHECK(cudaFree(dBGR));

    return 0;
}
