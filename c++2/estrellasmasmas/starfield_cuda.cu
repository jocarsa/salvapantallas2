// starfield_cuda_simple_random_count.cu
//
// Versión simplificada:
// - Fondo estelar fijo precalculado una sola vez
// - Partículas principales viniendo hacia la cámara
// - Sin líneas entre partículas
// - Cámara estática, solo roll
// - En cada ejecución, el número de partículas móviles es aleatorio
//   entre 20% y 200% del valor base
// - Codificación por pipe a ffmpeg
//
// Deps (Ubuntu):
//   sudo apt install ffmpeg libopencv-dev nvidia-cuda-toolkit
//
// Build:
//   nvcc -O3 -std=c++17 starfield_cuda_simple_random_count.cu -o starfield_simple `pkg-config --cflags --libs opencv4`
//
// Run:
//   ./starfield_simple
//

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// =========================
// CONFIG
// =========================
static constexpr int   W = 1920;
static constexpr int   H = 1080;
static constexpr int   FPS = 60;
static constexpr float DT = 1.0f / float(FPS);

// duración final
static constexpr int   VIDEO_SECONDS = 10 * 60 * 60; // 10 horas
static const char*     OUT_DIR = "videos_starfield";

// partículas principales (valor base)
static constexpr int   BASE_NUM_STARS = 28000;

// rango aleatorio por ejecución: 20%..200%
static constexpr float STAR_COUNT_MIN_FACTOR = 0.20f;
static constexpr float STAR_COUNT_MAX_FACTOR = 2.00f;

// fondo estelar fijo
static constexpr int   NUM_BG_STARS = 9000;

static constexpr float FOV_DEG = 70.0f;
static constexpr float NEAR_Z  = 0.20f;
static constexpr float FAR_Z   = 120.0f;

static constexpr float STAR_SPEED = 5.0f; // units/sec

// soma / partículas principales
static constexpr float RADIUS_SCALE = 120.0f;
static constexpr float RADIUS_MIN_F = 1.8f;
static constexpr float RADIUS_MAX_F = 16.0f;

static constexpr float BRIGHT_NEAR = 220.0f;
static constexpr float BRIGHT_FAR  = 35.0f;
static constexpr float BRIGHT_MIN  = 25.0f;
static constexpr float BRIGHT_MAX  = 255.0f;

// glow
static constexpr float CORE_FRACTION      = 0.80f;
static constexpr float GLOW_FRACTION      = 0.60f;

static constexpr float CORE_RADIUS_FACTOR = 0.72f;
static constexpr float CORE_R_MIN         = 1.20f;
static constexpr float CORE_R_MAX         = 6.50f;

static constexpr float GLOW_RADIUS_FACTOR = 2.10f;
static constexpr float GLOW_R_MIN         = 3.00f;
static constexpr float GLOW_R_MAX         = 22.0f;

// roll solamente
static constexpr float ROLL_AMP_DEG    = 12.0f;
static constexpr float ROLL_PERIOD_SEC = 45.0f;

// fondo fijo
static constexpr float BG_BRIGHT_MIN = 8.0f;
static constexpr float BG_BRIGHT_MAX = 42.0f;
static constexpr float BG_RADIUS_MIN = 0.7f;
static constexpr float BG_RADIUS_MAX = 2.0f;

// preview
static constexpr bool SHOW_PREVIEW = true;
static constexpr int  PREVIEW_EVERY_N_FRAMES = 2;
static constexpr int  PREVIEW_WAITKEY_MS = 1;

// encoder
static const char* HW_ENCODER = "h264_nvenc";
static const char* NVENC_PRESET = "p6";

static constexpr int NVENC_VBR_Mbps = 12;
static constexpr int NVENC_MAX_Mbps = 18;
static constexpr int NVENC_BUF_Mbps = 24;

// =========================
// Helpers
// =========================
static std::atomic<bool> g_stop{false};
static void handle_sigint(int) { g_stop.store(true, std::memory_order_relaxed); }

static std::string now_stamp() {
  auto t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

static std::string make_out_path_10h() {
  std::ostringstream oss;
  oss << OUT_DIR << "/starfield_simple_" << now_stamp() << "_10h.mp4";
  return oss.str();
}

static int choose_random_star_count() {
  std::random_device rd;
  std::mt19937 rng(rd());

  const int minStars = std::max(1, (int)std::lround(BASE_NUM_STARS * STAR_COUNT_MIN_FACTOR));
  const int maxStars = std::max(minStars, (int)std::lround(BASE_NUM_STARS * STAR_COUNT_MAX_FACTOR));

  std::uniform_int_distribution<int> dist(minStars, maxStars);
  return dist(rng);
}

// =========================
// FFmpeg pipe writer
// =========================
class FFmpegPipe {
public:
  FFmpegPipe(const std::string& out_path, int w, int h, int fps)
    : out_path_(out_path), w_(w), h_(h), fps_(fps) {}

  void open() {
    std::ostringstream cmd;

    cmd
      << "ffmpeg -y "
      << "-loglevel warning "
      << "-f rawvideo "
      << "-pix_fmt nv12 "
      << "-s " << w_ << "x" << h_ << " "
      << "-r " << fps_ << " "
      << "-i - "
      << "-an "
      << "-c:v h264_nvenc "
      << "-preset " << NVENC_PRESET << " "
      << "-tune hq "
      << "-rc vbr_hq "
      << "-b:v " << NVENC_VBR_Mbps << "M "
      << "-maxrate " << NVENC_MAX_Mbps << "M "
      << "-bufsize " << NVENC_BUF_Mbps << "M "
      << "-g " << (fps_ * 2) << " "
      << "-pix_fmt yuv420p "
      << "\"" << out_path_ << "\"";

    pipe_ = popen(cmd.str().c_str(), "w");
    if (!pipe_) throw std::runtime_error("Failed to open ffmpeg pipe.");
  }

  bool write_nv12(const uint8_t* nv12, size_t bytes) {
    if (!pipe_) return false;
    size_t written = std::fwrite(nv12, 1, bytes, pipe_);
    if (written != bytes) {
      std::cerr << "\nERROR: ffmpeg pipe write failed (written "
                << written << " / " << bytes << ").\n";
      return false;
    }
    return true;
  }

  int close() {
    if (!pipe_) return 0;
    std::fflush(pipe_);
    int rc = pclose(pipe_);
    pipe_ = nullptr;
    if (rc != 0) {
      std::cerr << "WARNING: ffmpeg exited with status: " << rc << "\n";
    }
    return rc;
  }

  ~FFmpegPipe() { close(); }

private:
  std::string out_path_;
  int w_, h_, fps_;
  FILE* pipe_{nullptr};
};

// =========================
// CPU NV12 pack
// =========================
static inline void gray_to_nv12_cpu(const cv::Mat& gray_u8, std::vector<uint8_t>& out_nv12) {
  const int w = gray_u8.cols, h = gray_u8.rows;
  const size_t y_bytes  = size_t(w) * size_t(h);
  const size_t uv_bytes = size_t(w) * size_t(h/2);
  out_nv12.resize(y_bytes + uv_bytes);
  std::memcpy(out_nv12.data(), gray_u8.data, y_bytes);
  std::memset(out_nv12.data() + y_bytes, 128, uv_bytes);
}

// =========================
// CUDA structs
// =========================
struct Star {
  float x, y, z;
};

struct DrawStar {
  float sx, sy;
  float b;
  float r;
  int valid;
};

struct BgStar {
  float sx, sy;
  float b;
  float r;
};

// =========================
// CUDA error check
// =========================
static inline void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(e) << "\n";
    std::exit(1);
  }
}

// =========================
// Roll matrix
// =========================
struct Mat2 {
  float a, b, c, d;
};

static inline Mat2 roll_matrix(float t_sec) {
  const float amp = ROLL_AMP_DEG * float(M_PI) / 180.0f;
  const float roll = amp * std::sin(2.0f * float(M_PI) * t_sec / ROLL_PERIOD_SEC);
  const float cr = std::cos(roll);
  const float sr = std::sin(roll);
  return Mat2{cr, -sr, sr, cr};
}

// =========================
// CUDA kernels
// =========================
__global__ void init_rng(curandState* st, unsigned long long seed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curand_init(seed, i, 0, &st[i]);
}

__global__ void init_stars(
  Star* stars, curandState* st, int n,
  float nearZ, float farZ, float tan_half, float aspect
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  curandState local = st[i];

  float z = nearZ + (farZ - nearZ) * curand_uniform(&local);
  float u = 2.0f * curand_uniform(&local) - 1.0f;
  float v = 2.0f * curand_uniform(&local) - 1.0f;

  float x = u * z * tan_half;
  float y = v * z * tan_half * aspect;

  stars[i] = Star{x, y, z};
  st[i] = local;
}

__global__ void update_stars(
  Star* stars, curandState* st, int n,
  float speed, float dt,
  float nearZ, float farZ,
  float tan_half, float aspect
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  Star s = stars[i];
  s.z -= speed * dt;

  if (s.z <= nearZ) {
    curandState local = st[i];

    float z = farZ * (0.85f + 0.15f * curand_uniform(&local));
    float u = 2.0f * curand_uniform(&local) - 1.0f;
    float v = 2.0f * curand_uniform(&local) - 1.0f;

    s.x = u * z * tan_half;
    s.y = v * z * tan_half * aspect;
    s.z = z;

    st[i] = local;
  }

  stars[i] = s;
}

__global__ void init_bg_stars(BgStar* bg, curandState* st, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  curandState local = st[i];

  float sx = curand_uniform(&local) * float(W - 1);
  float sy = curand_uniform(&local) * float(H - 1);

  float b = BG_BRIGHT_MIN + (BG_BRIGHT_MAX - BG_BRIGHT_MIN) * curand_uniform(&local);
  float r = BG_RADIUS_MIN + (BG_RADIUS_MAX - BG_RADIUS_MIN) * curand_uniform(&local);

  bg[i] = BgStar{sx, sy, b, r};
  st[i] = local;
}

__global__ void project_stars_roll_only(
  const Star* stars, DrawStar* out, int n,
  Mat2 R,
  float f, float cx, float cy,
  float nearZ, float farZ,
  float brightNear, float brightFar,
  float brightMin, float brightMax,
  float radiusScale, float rmin, float rmax
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  Star s = stars[i];
  DrawStar d{};
  d.valid = 0;

  if (s.z <= nearZ) {
    out[i] = d;
    return;
  }

  float invz = f / s.z;

  float px = s.x * invz;
  float py = s.y * invz;

  float rx = R.a * px + R.b * py;
  float ry = R.c * px + R.d * py;

  float sx = rx + cx;
  float sy = ry + cy;

  if (sx < -256.0f || sx > (W + 256.0f) || sy < -256.0f || sy > (H + 256.0f)) {
    out[i] = d;
    return;
  }

  float r = radiusScale / (s.z + 1e-6f);
  r = fminf(rmax, fmaxf(rmin, r));

  float t = 1.0f - (s.z - nearZ) / (farZ - nearZ);
  float b = brightNear * t + brightFar;
  b = fminf(brightMax, fmaxf(brightMin, b));

  d.sx = sx;
  d.sy = sy;
  d.b = b;
  d.r = r;
  d.valid = 1;

  out[i] = d;
}

__global__ void clear_fbuffer(float* fb, int n, float v = 0.0f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  fb[i] = v;
}

__device__ __forceinline__ void add_pixel(float* fb, int x, int y, float v) {
  if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H) {
    atomicAdd(&fb[y * W + x], v);
  }
}

__device__ __forceinline__ void add_disk(float* fb, int cx, int cy, float r, float b) {
  int ir = (int)ceilf(r);
  float rr = r * r;
  for (int y = cy - ir; y <= cy + ir; ++y) {
    float dy = float(y - cy);
    for (int x = cx - ir; x <= cx + ir; ++x) {
      float dx = float(x - cx);
      if (dx * dx + dy * dy <= rr) add_pixel(fb, x, y, b);
    }
  }
}

__global__ void render_background(float* fb, const BgStar* bg, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int cx = (int)lrintf(bg[i].sx);
  int cy = (int)lrintf(bg[i].sy);
  float r = bg[i].r;
  float b = bg[i].b;

  if (r < 1.05f) {
    add_pixel(fb, cx, cy, b);
  } else {
    add_disk(fb, cx, cy, r, b);
  }
}

__global__ void render_stars(float* fb, const DrawStar* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!d[i].valid) return;

  float sx = d[i].sx;
  float sy = d[i].sy;
  float b  = d[i].b;
  float r  = d[i].r;

  int cx = (int)lrintf(sx);
  int cy = (int)lrintf(sy);

  float core_r = fminf(CORE_R_MAX, fmaxf(CORE_R_MIN, r * CORE_RADIUS_FACTOR));
  float core_b = b * CORE_FRACTION;

  if (core_r < 1.05f) {
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - x0;
    float fy = sy - y0;
    float w00 = (1 - fx) * (1 - fy);
    float w10 = fx * (1 - fy);
    float w01 = (1 - fx) * fy;
    float w11 = fx * fy;
    add_pixel(fb, x0,   y0,   core_b * w00);
    add_pixel(fb, x0+1, y0,   core_b * w10);
    add_pixel(fb, x0,   y0+1, core_b * w01);
    add_pixel(fb, x0+1, y0+1, core_b * w11);
  } else {
    add_disk(fb, cx, cy, core_r, core_b);
  }

  float glow_r = fminf(GLOW_R_MAX, fmaxf(GLOW_R_MIN, r * GLOW_RADIUS_FACTOR));
  float glow_b = b * GLOW_FRACTION;

  float sigma = fmaxf(0.35f, glow_r / 2.0f);
  float inv2s2 = 1.0f / (2.0f * sigma * sigma);

  int half = (int)ceilf(3.0f * sigma);
  int x1 = cx - half, x2 = cx + half;
  int y1 = cy - half, y2 = cy + half;

  for (int yy = y1; yy <= y2; ++yy) {
    float dy = float(yy - cy);
    for (int xx = x1; xx <= x2; ++xx) {
      float dx = float(xx - cx);
      float w = expf(-(dx * dx + dy * dy) * inv2s2);
      add_pixel(fb, xx, yy, glow_b * w);
    }
  }
}

__global__ void fb_to_u8(const float* fb, unsigned char* out, int nPix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPix) return;
  float v = fb[i];
  v = fminf(255.0f, fmaxf(0.0f, v));
  out[i] = (unsigned char)(v + 0.5f);
}

// =========================
// MAIN
// =========================
int main() {
  std::signal(SIGINT, handle_sigint);

  std::string mkdir_cmd = std::string("mkdir -p \"") + OUT_DIR + "\"";
  std::system(mkdir_cmd.c_str());

  const int NUM_STARS = choose_random_star_count();

  std::cout << "Random moving star count selected: " << NUM_STARS
            << " (base=" << BASE_NUM_STARS << ", "
            << std::fixed << std::setprecision(1)
            << (100.0 * double(NUM_STARS) / double(BASE_NUM_STARS))
            << "%)\n";

  std::cout << "CUDA device init...\n";
  ck(cudaSetDevice(0), "cudaSetDevice");

  const float fov_rad = FOV_DEG * float(M_PI) / 180.0f;
  const float tan_half = std::tan(fov_rad * 0.5f);
  const float aspect = float(H) / float(W);
  const float f = (W * 0.5f) / tan_half;
  const float cx = W * 0.5f;
  const float cy = H * 0.5f;

  Star* d_stars = nullptr;
  DrawStar* d_draw = nullptr;
  BgStar* d_bg = nullptr;
  curandState* d_rng_main = nullptr;
  curandState* d_rng_bg = nullptr;
  float* d_fb = nullptr;
  unsigned char* d_u8 = nullptr;

  ck(cudaMalloc(&d_stars, NUM_STARS * sizeof(Star)), "malloc stars");
  ck(cudaMalloc(&d_draw,  NUM_STARS * sizeof(DrawStar)), "malloc draw");
  ck(cudaMalloc(&d_bg,    NUM_BG_STARS * sizeof(BgStar)), "malloc bg");
  ck(cudaMalloc(&d_rng_main, NUM_STARS * sizeof(curandState)), "malloc rng main");
  ck(cudaMalloc(&d_rng_bg,   NUM_BG_STARS * sizeof(curandState)), "malloc rng bg");
  ck(cudaMalloc(&d_fb, size_t(W) * size_t(H) * sizeof(float)), "malloc fb");
  ck(cudaMalloc(&d_u8, size_t(W) * size_t(H) * sizeof(unsigned char)), "malloc u8");

  dim3 B(256);
  dim3 Gs((NUM_STARS + B.x - 1) / B.x);
  dim3 Gbg((NUM_BG_STARS + B.x - 1) / B.x);
  dim3 Gpix(((int64_t)W * (int64_t)H + B.x - 1) / B.x);

  const unsigned long long seedBase =
      (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();

  init_rng<<<Gs, B>>>(d_rng_main, seedBase ^ 0x12345678ULL, NUM_STARS);
  init_stars<<<Gs, B>>>(d_stars, d_rng_main, NUM_STARS, NEAR_Z, FAR_Z, tan_half, aspect);

  init_rng<<<Gbg, B>>>(d_rng_bg, seedBase ^ 0x87654321ULL, NUM_BG_STARS);
  init_bg_stars<<<Gbg, B>>>(d_bg, d_rng_bg, NUM_BG_STARS);

  ck(cudaGetLastError(), "init kernels");
  ck(cudaDeviceSynchronize(), "sync init");

  cv::Mat frame_gray(H, W, CV_8UC1);
  std::vector<uint8_t> nv12;

  const std::string out_path = make_out_path_10h();
  FFmpegPipe writer(out_path, W, H, FPS);
  writer.open();

  std::cout << "Recording (10h): " << out_path << "\n";
  std::cout << "Background stars: " << NUM_BG_STARS << "\n";
  std::cout << "Moving stars: " << NUM_STARS << "\n";

  if (SHOW_PREVIEW) {
    try {
      cv::namedWindow("starfield_simple", cv::WINDOW_NORMAL);
      cv::resizeWindow("starfield_simple", 1280, 720);
    } catch (...) {
      std::cout << "Preview window failed to create (headless?).\n";
    }
  }

  const int64_t total_frames = int64_t(VIDEO_SECONDS) * int64_t(FPS);
  auto t0 = std::chrono::high_resolution_clock::now();
  float sim_time = 0.0f;

  for (int64_t frame = 0; frame < total_frames && !g_stop.load(std::memory_order_relaxed); ++frame) {
    update_stars<<<Gs, B>>>(
      d_stars, d_rng_main, NUM_STARS,
      STAR_SPEED, DT,
      NEAR_Z, FAR_Z,
      tan_half, aspect
    );

    Mat2 R = roll_matrix(sim_time);

    project_stars_roll_only<<<Gs, B>>>(
      d_stars, d_draw, NUM_STARS,
      R,
      f, cx, cy,
      NEAR_Z, FAR_Z,
      BRIGHT_NEAR, BRIGHT_FAR,
      BRIGHT_MIN, BRIGHT_MAX,
      RADIUS_SCALE, RADIUS_MIN_F, RADIUS_MAX_F
    );

    clear_fbuffer<<<Gpix, B>>>(d_fb, W * H, 0.0f);
    render_background<<<Gbg, B>>>(d_fb, d_bg, NUM_BG_STARS);
    render_stars<<<Gs, B>>>(d_fb, d_draw, NUM_STARS);
    fb_to_u8<<<Gpix, B>>>(d_fb, d_u8, W * H);

    ck(cudaMemcpy(frame_gray.data, d_u8, size_t(W) * size_t(H), cudaMemcpyDeviceToHost), "memcpy u8->host");

    if (SHOW_PREVIEW && (frame % PREVIEW_EVERY_N_FRAMES == 0)) {
      try {
        cv::imshow("starfield_simple", frame_gray);
        int key = cv::waitKey(PREVIEW_WAITKEY_MS);
        if (key == 27) g_stop.store(true, std::memory_order_relaxed);
      } catch (...) {}
    }

    gray_to_nv12_cpu(frame_gray, nv12);
    if (!writer.write_nv12(nv12.data(), nv12.size())) {
      g_stop.store(true, std::memory_order_relaxed);
    }

    sim_time += DT;

    if (frame % 600 == 0 && frame > 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double wall = std::chrono::duration<double>(t1 - t0).count();
      double vid  = double(frame) / double(FPS);
      double speed = (wall > 1e-9) ? (vid / wall) : 0.0;

      double remain_vid  = double(total_frames - frame) / double(FPS);
      double remain_wall = (speed > 1e-9) ? (remain_vid / speed) : 0.0;

      std::cout << "Frames: " << frame
                << " / " << total_frames
                << "  Video: " << std::fixed << std::setprecision(1) << vid << "s"
                << "  Speed: " << std::setprecision(2) << speed << "x realtime"
                << "  Est. remaining wall: " << std::setprecision(1) << remain_wall << "s\n";
    }
  }

  writer.close();
  std::cout << "Done. Wrote: " << out_path << "\n";

  cudaFree(d_u8);
  cudaFree(d_fb);
  cudaFree(d_rng_bg);
  cudaFree(d_rng_main);
  cudaFree(d_bg);
  cudaFree(d_draw);
  cudaFree(d_stars);

  return 0;
}
