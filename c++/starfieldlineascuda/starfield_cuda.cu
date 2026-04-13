// starfield_cuda.cu
//
// CUDA starfield + 3D-neighbor lines (camera-space) + offline encode via ffmpeg pipe.
// Output: single 10-hour MP4.
//
// Deps (Ubuntu):
//   sudo apt install ffmpeg libopencv-dev nvidia-cuda-toolkit
//
// Build with build.sh below.
//
// Notes:
// - Rendering uses atomicAdd into a float framebuffer.
// - Lines: built in 3D camera space via a uniform voxel grid (linked lists).
// - Lines are rasterized by sampling along the segment (in screen space) and splatting
//   a small disk for thickness. (Fast enough with caps.)
// - Stars: hard core disk + gaussian glow (analytic), both on GPU.
// - Preview: optional via OpenCV (downloads GRAY8).
//

#include <opencv2/opencv.hpp>

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

static constexpr int   NUM_STARS = 28000;

static constexpr float FOV_DEG = 70.0f;
static constexpr float NEAR_Z  = 0.20f;
static constexpr float FAR_Z   = 120.0f;

static constexpr float STAR_SPEED = 5.0f; // units/sec

// Medium star size
static constexpr float RADIUS_SCALE = 64.0f;
static constexpr float RADIUS_MIN_F = 1.60f;
static constexpr float RADIUS_MAX_F = 12.0f;

// brightness
static constexpr float BRIGHT_NEAR = 220.0f;
static constexpr float BRIGHT_FAR  = 35.0f;
static constexpr float BRIGHT_MIN  = 30.0f;
static constexpr float BRIGHT_MAX  = 255.0f;

// camera motion
static constexpr float AMP_DEG = 10.0f;
static constexpr float YAW_PERIOD_SEC   = 19.0f;
static constexpr float PITCH_PERIOD_SEC = 27.0f;
static constexpr float ROLL_PERIOD_SEC  = 41.0f;

// Final render duration (10 hours)
static constexpr int   VIDEO_SECONDS = 1 * 60 * 60; // 36000
static const char*     OUT_DIR = "videos_starfield";

// GPU encoder selection:
static const char* HW_ENCODER = "h264_nvenc"; // or "h264_qsv"

// NVENC knobs
static const char* NVENC_PRESET = "p1";
static constexpr int NVENC_QP = 18;

// QSV knobs
static constexpr int QSV_BITRATE_Mbps = 20;

// Core + Glow model
static constexpr float CORE_FRACTION      = 0.85f;
static constexpr float GLOW_FRACTION      = 0.55f;

static constexpr float CORE_RADIUS_FACTOR = 0.40f;
static constexpr float CORE_R_MIN         = 0.90f;
static constexpr float CORE_R_MAX         = 3.20f;

static constexpr float GLOW_RADIUS_FACTOR = 1.55f;
static constexpr float GLOW_R_MIN         = 2.20f;
static constexpr float GLOW_R_MAX         = 16.0f;

// =========================
// LINES (3D camera space)
// =========================
static constexpr float LINE_THRESH_3D = 10.0f;      // tune: 8..18
static constexpr float LINE_Z_MAX     = 60.0f;      // ignore far background for lines
static constexpr int   LINE_MAX_NEIGHBORS_PER_STAR = 6;

static constexpr float LINE_BRIGHT_FACTOR = 0.11f;
static constexpr float LINE_BRIGHT_MIN    = 4.0f;
static constexpr float LINE_BRIGHT_MAX    = 65.0f;

static constexpr float LINE_THICK_MIN = 1.0f;
static constexpr float LINE_THICK_MAX = 4.0f;

static constexpr float LINE_THICK_DIST_WEIGHT = 0.85f;
static constexpr float LINE_THICK_Z_WEIGHT    = 1.00f;

// For line rasterization sampling step (in pixels)
static constexpr float LINE_SAMPLE_STEP_PX = 1.0f;

// Cap total lines buffer:
static constexpr int   MAX_LINES = NUM_STARS * LINE_MAX_NEIGHBORS_PER_STAR;

// =========================
// LIVE PREVIEW
// =========================
static constexpr bool SHOW_PREVIEW = true;
static constexpr int  PREVIEW_EVERY_N_FRAMES = 2;
static constexpr int  PREVIEW_WAITKEY_MS = 1;

// =========================
// Helpers
// =========================
static std::atomic<bool> g_stop{false};
static void handle_sigint(int) { g_stop.store(true, std::memory_order_relaxed); }

static inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

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
  oss << OUT_DIR << "/starfield_" << now_stamp() << "_10h.mp4";
  return oss.str();
}

// =========================
// FFmpeg pipe writer (NV12 -> GPU encoder) [ROBUST]
// =========================
class FFmpegPipe {
public:
  FFmpegPipe(const std::string& out_path, int w, int h, int fps)
    : out_path_(out_path), w_(w), h_(h), fps_(fps) {}

  void open() {
    std::ostringstream cmd;

    // IMPORTANT: do NOT silence stderr while debugging.
    // You can change "-loglevel warning" to "error" later.
    if (std::string(HW_ENCODER) == "h264_nvenc") {
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
        << "-rc constqp -qp " << NVENC_QP << " "
        << "-pix_fmt yuv420p "
        << "\"" << out_path_ << "\"";
    } else if (std::string(HW_ENCODER) == "h264_qsv") {
      cmd
        << "ffmpeg -y "
        << "-loglevel warning "
        << "-f rawvideo "
        << "-pix_fmt nv12 "
        << "-s " << w_ << "x" << h_ << " "
        << "-r " << fps_ << " "
        << "-i - "
        << "-an "
        << "-c:v h264_qsv "
        << "-b:v " << QSV_BITRATE_Mbps << "M "
        << "-maxrate " << QSV_BITRATE_Mbps << "M "
        << "-bufsize " << (QSV_BITRATE_Mbps * 2) << "M "
        << "-pix_fmt yuv420p "
        << "\"" << out_path_ << "\"";
    } else {
      throw std::runtime_error("Unknown HW_ENCODER. Use h264_nvenc or h264_qsv.");
    }

    pipe_ = popen(cmd.str().c_str(), "w");
    if (!pipe_) throw std::runtime_error("Failed to open ffmpeg pipe.");
  }

  // returns false if write failed (broken pipe)
  bool write_nv12(const uint8_t* nv12, size_t bytes) {
    if (!pipe_) return false;
    size_t written = std::fwrite(nv12, 1, bytes, pipe_);
    if (written != bytes) {
      // Most likely ffmpeg exited; stop cleanly.
      std::cerr << "\nERROR: ffmpeg pipe write failed (written "
                << written << " / " << bytes << ").\n";
      return false;
    }
    return true;
  }

  // returns ffmpeg exit status (0 == ok)
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
// CPU NV12 pack (GRAY8 -> NV12)
// =========================
static inline void gray_to_nv12_cpu(const cv::Mat& gray_u8, std::vector<uint8_t>& out_nv12) {
  const int w = gray_u8.cols, h = gray_u8.rows;
  const size_t y_bytes = size_t(w) * size_t(h);
  const size_t uv_bytes = size_t(w) * size_t(h/2);
  out_nv12.resize(y_bytes + uv_bytes);
  std::memcpy(out_nv12.data(), gray_u8.data, y_bytes);
  std::memset(out_nv12.data() + y_bytes, 128, uv_bytes);
}

// =========================
// CUDA structs
// =========================
struct Star {
  float x,y,z; // world space in our simple model
};

struct DrawStar {
  float sx, sy;        // screen
  float b;             // brightness
  float r;             // base radius
  float camx, camy, camz; // camera space
  int   valid;         // 1 if on-screen + camz>near
};

struct LineSeg {
  float x1,y1,x2,y2;
  float b;
  int thickness;
};

struct Mat3 {
  float m00,m01,m02;
  float m10,m11,m12;
  float m20,m21,m22;
};

__host__ __device__ static inline Mat3 mat3_transpose(const Mat3& a) {
  Mat3 t;
  t.m00=a.m00; t.m01=a.m10; t.m02=a.m20;
  t.m10=a.m01; t.m11=a.m11; t.m12=a.m21;
  t.m20=a.m02; t.m21=a.m12; t.m22=a.m22;
  return t;
}

__host__ __device__ static inline void mat3_mul_vec(const Mat3& M, float x,float y,float z,
                                                    float& ox,float& oy,float& oz) {
  ox = M.m00*x + M.m01*y + M.m02*z;
  oy = M.m10*x + M.m11*y + M.m12*z;
  oz = M.m20*x + M.m21*y + M.m22*z;
}

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
// Camera rotation (CPU)
// =========================
static inline Mat3 rot_x(float a) {
  float ca=std::cos(a), sa=std::sin(a);
  return Mat3{1,0,0, 0,ca,-sa, 0,sa,ca};
}
static inline Mat3 rot_y(float a) {
  float ca=std::cos(a), sa=std::sin(a);
  return Mat3{ca,0,sa, 0,1,0, -sa,0,ca};
}
static inline Mat3 rot_z(float a) {
  float ca=std::cos(a), sa=std::sin(a);
  return Mat3{ca,-sa,0, sa,ca,0, 0,0,1};
}
static inline Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  C.m00=A.m00*B.m00 + A.m01*B.m10 + A.m02*B.m20;
  C.m01=A.m00*B.m01 + A.m01*B.m11 + A.m02*B.m21;
  C.m02=A.m00*B.m02 + A.m01*B.m12 + A.m02*B.m22;

  C.m10=A.m10*B.m00 + A.m11*B.m10 + A.m12*B.m20;
  C.m11=A.m10*B.m01 + A.m11*B.m11 + A.m12*B.m21;
  C.m12=A.m10*B.m02 + A.m11*B.m12 + A.m12*B.m22;

  C.m20=A.m20*B.m00 + A.m21*B.m10 + A.m22*B.m20;
  C.m21=A.m20*B.m01 + A.m21*B.m11 + A.m22*B.m21;
  C.m22=A.m20*B.m02 + A.m21*B.m12 + A.m22*B.m22;
  return C;
}
static inline Mat3 camera_rotation(float t_sec) {
  const float amp = (AMP_DEG * float(M_PI) / 180.0f);
  const float yaw   = amp * std::sin(2.0f * float(M_PI) * t_sec / YAW_PERIOD_SEC);
  const float pitch = amp * std::sin(2.0f * float(M_PI) * t_sec / PITCH_PERIOD_SEC);
  const float roll  = amp * std::sin(2.0f * float(M_PI) * t_sec / ROLL_PERIOD_SEC);
  // R = Rz * Ry * Rx
  return mat3_mul(rot_z(roll), mat3_mul(rot_y(yaw), rot_x(pitch)));
}

// =========================
// CUDA kernels
// =========================
__global__ void init_rng(curandState* st, unsigned long long seed, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  curand_init(seed, i, 0, &st[i]);
}

__global__ void init_stars(Star* stars, curandState* st, int n,
                           float nearZ, float farZ, float tan_half, float aspect) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandState local = st[i];

  float z = nearZ + (farZ - nearZ) * curand_uniform(&local); // (0,1]
  float u = 2.0f*curand_uniform(&local) - 1.0f;
  float v = 2.0f*curand_uniform(&local) - 1.0f;

  float x = u * z * tan_half;
  float y = v * z * tan_half * aspect;

  stars[i] = Star{x,y,z};
  st[i] = local;
}

__global__ void update_stars(Star* stars, curandState* st, int n,
                             float speed, float dt,
                             float nearZ, float farZ,
                             float tan_half, float aspect) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;

  Star s = stars[i];
  s.z -= speed * dt;

  if (s.z <= nearZ) {
    curandState local = st[i];
    float z = (farZ*0.8f) + (farZ - farZ*0.8f) * curand_uniform(&local);
    float u = 2.0f*curand_uniform(&local) - 1.0f;
    float v = 2.0f*curand_uniform(&local) - 1.0f;
    float x = u * z * tan_half;
    float y = v * z * tan_half * aspect;
    s = Star{x,y,z};
    st[i] = local;
  }

  stars[i] = s;
}

__global__ void project_stars(const Star* stars, DrawStar* out, int n,
                             Mat3 Rt,
                             float f, float cx, float cy,
                             float nearZ, float farZ,
                             float brightNear, float brightFar,
                             float brightMin, float brightMax,
                             float radiusScale, float rmin, float rmax) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;

  Star s = stars[i];

  float camx, camy, camz;
  mat3_mul_vec(Rt, s.x, s.y, s.z, camx, camy, camz);

  DrawStar d{};
  d.valid = 0;
  d.camx = camx; d.camy = camy; d.camz = camz;

  if (camz <= nearZ) { out[i]=d; return; }

  float invz = f / camz;
  float sx = camx * invz + cx;
  float sy = camy * invz + cy;

  // loose cull (some spill ok)
  if (sx < -128.0f || sx > (W + 128.0f) || sy < -128.0f || sy > (H + 128.0f)) {
    out[i]=d; return;
  }

  float r = radiusScale / (camz + 1e-6f);
  r = fminf(rmax, fmaxf(rmin, r));

  float t = 1.0f - (camz - nearZ) / (farZ - nearZ);
  float b = brightNear * t + brightFar;
  b = fminf(brightMax, fmaxf(brightMin, b));

  d.sx = sx; d.sy = sy; d.b = b; d.r = r; d.valid = 1;
  out[i] = d;
}

__global__ void clear_fbuffer(float* fb, int n, float v=0.0f) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  fb[i] = v;
}

// =========================
// 3D grid (linked list) for neighbor search
// =========================
struct Grid3 {
  int nx, ny, nz;
  float cell;        // cell size = LINE_THRESH_3D
  float minx, miny, minz;
  int* head;         // size nx*ny*nz, init -1
  int* next;         // size NUM_STARS
};

__device__ __forceinline__ int grid_index(const Grid3& g, int ix,int iy,int iz) {
  return (iz*g.ny + iy)*g.nx + ix;
}

__global__ void grid_reset(int* head, int cells) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=cells) return;
  head[i] = -1;
}

__global__ void grid_build(Grid3 g, const DrawStar* d, int n, float zMax) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  if (!d[i].valid) { g.next[i] = -1; return; }
  if (d[i].camz > zMax) { g.next[i] = -1; return; }

  int ix = (int)floorf((d[i].camx - g.minx)/g.cell);
  int iy = (int)floorf((d[i].camy - g.miny)/g.cell);
  int iz = (int)floorf((d[i].camz - g.minz)/g.cell);

  if (ix<0||iy<0||iz<0||ix>=g.nx||iy>=g.ny||iz>=g.nz) { g.next[i] = -1; return; }

  int idx = grid_index(g, ix,iy,iz);
  int old = atomicExch(&g.head[idx], i);
  g.next[i] = old;
}

// Lines are written into a fixed slot range per star: [i*K, i*K + count)
// (no atomic for output). Duplicates avoided by requiring j>i.
__global__ void gen_lines_3d(const Grid3 g, const DrawStar* d, int n,
                             LineSeg* outLines, int* outCounts,
                             float thresh, float nearZ, float zMax) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;

  outCounts[i] = 0;

  if (!d[i].valid) return;
  if (d[i].camz > zMax) return;

  float ax=d[i].camx, ay=d[i].camy, az=d[i].camz;

  int ix0 = (int)floorf((ax - g.minx)/g.cell);
  int iy0 = (int)floorf((ay - g.miny)/g.cell);
  int iz0 = (int)floorf((az - g.minz)/g.cell);

  if (ix0<0||iy0<0||iz0<0||ix0>=g.nx||iy0>=g.ny||iz0>=g.nz) return;

  float thresh2 = thresh*thresh;

  int count = 0;
  int base = i * LINE_MAX_NEIGHBORS_PER_STAR;

  for (int oz=-1; oz<=1 && count<LINE_MAX_NEIGHBORS_PER_STAR; ++oz) {
    int iz = iz0+oz; if (iz<0||iz>=g.nz) continue;
    for (int oy=-1; oy<=1 && count<LINE_MAX_NEIGHBORS_PER_STAR; ++oy) {
      int iy = iy0+oy; if (iy<0||iy>=g.ny) continue;
      for (int ox=-1; ox<=1 && count<LINE_MAX_NEIGHBORS_PER_STAR; ++ox) {
        int ix = ix0+ox; if (ix<0||ix>=g.nx) continue;
        int h = g.head[grid_index(g,ix,iy,iz)];
        while (h != -1 && count<LINE_MAX_NEIGHBORS_PER_STAR) {
          int j = h;
          h = g.next[j];
          if (j <= i) continue;
          if (!d[j].valid) continue;
          if (d[j].camz > zMax) continue;

          float bx=d[j].camx, by=d[j].camy, bz=d[j].camz;
          float dx=bx-ax, dy=by-ay, dz=bz-az;
          float dist2 = dx*dx + dy*dy + dz*dz;
          if (dist2 > thresh2) continue;

          float dist = sqrtf(fmaxf(1e-6f, dist2));
          float dist_factor = 1.0f - fminf(1.0f, dist/thresh);

          float z_avg = 0.5f*(az + bz);
          float z_norm = (z_avg - nearZ) / (zMax - nearZ + 1e-6f);
          z_norm = fminf(1.0f, fmaxf(0.0f, z_norm));
          float z_factor = 1.0f - z_norm;

          float thick_f =
              LINE_THICK_MIN +
              (LINE_THICK_MAX - LINE_THICK_MIN) * (
                LINE_THICK_DIST_WEIGHT*dist_factor +
                LINE_THICK_Z_WEIGHT*z_factor*0.55f
              );

          int thickness = (int)lrintf(fminf(LINE_THICK_MAX, fmaxf(LINE_THICK_MIN, thick_f)));

          float b_avg = 0.5f*(d[i].b + d[j].b);
          float lb = b_avg * LINE_BRIGHT_FACTOR * (0.35f + 0.65f*dist_factor) * (0.60f + 0.40f*z_factor);
          lb = fminf(LINE_BRIGHT_MAX, fmaxf(LINE_BRIGHT_MIN, lb));

          outLines[base + count] = LineSeg{d[i].sx, d[i].sy, d[j].sx, d[j].sy, lb, thickness};
          count++;
        }
      }
    }
  }

  outCounts[i] = count;
}

// Atomic add helper (bounds-check)
__device__ __forceinline__ void add_pixel(float* fb, int x,int y, float v) {
  if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H) {
    atomicAdd(&fb[y*W + x], v);
  }
}

// Draw small disk (for line thickness and star core)
__device__ __forceinline__ void add_disk(float* fb, int cx,int cy, float r, float b) {
  int ir = (int)ceilf(r);
  float rr = r*r;
  for (int y=cy-ir; y<=cy+ir; ++y) {
    float dy = float(y - cy);
    for (int x=cx-ir; x<=cx+ir; ++x) {
      float dx = float(x - cx);
      if (dx*dx + dy*dy <= rr) add_pixel(fb, x,y, b);
    }
  }
}

// Render lines by sampling along segment and splatting a disk
__global__ void render_lines(float* fb,
                            const LineSeg* lines,
                            const int* counts,
                            int nStars) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=nStars) return;

  int count = counts[i];
  int base  = i * LINE_MAX_NEIGHBORS_PER_STAR;

  for (int k=0; k<count; ++k) {
    LineSeg ln = lines[base+k];
    float x1=ln.x1, y1=ln.y1, x2=ln.x2, y2=ln.y2;
    float dx=x2-x1, dy=y2-y1;
    float len = sqrtf(dx*dx + dy*dy);
    if (len < 1e-3f) continue;

    float step = LINE_SAMPLE_STEP_PX;
    int steps = (int)ceilf(len/step);
    float inv = 1.0f / (float)steps;

    // brightness contribution per sample (keep lines subtle)
    float per = ln.b * 0.18f;

    for (int s=0; s<=steps; ++s) {
      float t = s * inv;
      float xf = x1 + dx*t;
      float yf = y1 + dy*t;
      int xi = (int)lrintf(xf);
      int yi = (int)lrintf(yf);
      add_disk(fb, xi, yi, (float)ln.thickness, per);
    }
  }
}

// Render stars: hard core + gaussian glow (analytic)
__global__ void render_stars(float* fb, const DrawStar* d, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  if (!d[i].valid) return;

  float sx = d[i].sx, sy = d[i].sy;
  float b  = d[i].b;
  float r  = d[i].r;

  int cx = (int)lrintf(sx);
  int cy = (int)lrintf(sy);

  // CORE
  float core_r = fminf(CORE_R_MAX, fmaxf(CORE_R_MIN, r * CORE_RADIUS_FACTOR));
  float core_b = b * CORE_FRACTION;

  if (core_r < 1.05f) {
    // bilinear-ish splat to 4 pixels
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - x0;
    float fy = sy - y0;
    float w00=(1-fx)*(1-fy);
    float w10=fx*(1-fy);
    float w01=(1-fx)*fy;
    float w11=fx*fy;
    add_pixel(fb, x0,   y0,   core_b*w00);
    add_pixel(fb, x0+1, y0,   core_b*w10);
    add_pixel(fb, x0,   y0+1, core_b*w01);
    add_pixel(fb, x0+1, y0+1, core_b*w11);
  } else {
    add_disk(fb, cx, cy, core_r, core_b);
  }

  // GLOW
  float glow_r = fminf(GLOW_R_MAX, fmaxf(GLOW_R_MIN, r * GLOW_RADIUS_FACTOR));
  float glow_b = b * GLOW_FRACTION;

  // gaussian sigma
  float sigma = fmaxf(0.35f, glow_r/2.0f);
  float inv2s2 = 1.0f/(2.0f*sigma*sigma);

  int half = (int)ceilf(3.25f * sigma); // cutoff like before
  int x1 = cx - half, x2 = cx + half;
  int y1 = cy - half, y2 = cy + half;

  for (int yy=y1; yy<=y2; ++yy) {
    float dy = float(yy - cy);
    for (int xx=x1; xx<=x2; ++xx) {
      float dx = float(xx - cx);
      float w = expf(-(dx*dx + dy*dy) * inv2s2);
      add_pixel(fb, xx, yy, glow_b * w);
    }
  }
}

// Float framebuffer -> GRAY8
__global__ void fb_to_u8(const float* fb, unsigned char* out, int nPix) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=nPix) return;
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

  std::cout << "CUDA device init...\n";
  ck(cudaSetDevice(0), "cudaSetDevice");

  // Projection constants
  const float fov_rad = FOV_DEG * float(M_PI) / 180.0f;
  const float tan_half = std::tan(fov_rad * 0.5f);
  const float aspect = float(H) / float(W);
  const float f = (W * 0.5f) / tan_half;
  const float cx = W * 0.5f;
  const float cy = H * 0.5f;

  // Allocate device data
  Star* d_stars = nullptr;
  DrawStar* d_draw = nullptr;
  curandState* d_rng = nullptr;
  float* d_fb = nullptr;
  unsigned char* d_u8 = nullptr;

  ck(cudaMalloc(&d_stars, NUM_STARS * sizeof(Star)), "malloc stars");
  ck(cudaMalloc(&d_draw,  NUM_STARS * sizeof(DrawStar)), "malloc draw");
  ck(cudaMalloc(&d_rng,   NUM_STARS * sizeof(curandState)), "malloc rng");
  ck(cudaMalloc(&d_fb,    size_t(W) * size_t(H) * sizeof(float)), "malloc fb");
  ck(cudaMalloc(&d_u8,    size_t(W) * size_t(H) * sizeof(unsigned char)), "malloc u8");

  // 3D grid for neighbor search
  // camera-space bounds for x,y at z=LINE_Z_MAX:
  float maxX = LINE_Z_MAX * tan_half;
  float maxY = maxX * aspect;

  float cell = LINE_THRESH_3D;
  int nx = (int)std::ceil((2*maxX)/cell) + 2;
  int ny = (int)std::ceil((2*maxY)/cell) + 2;
  int nz = (int)std::ceil((LINE_Z_MAX - NEAR_Z)/cell) + 2;

  int cells = nx*ny*nz;
  int* d_head = nullptr;
  int* d_next = nullptr;
  ck(cudaMalloc(&d_head, cells * sizeof(int)), "malloc head");
  ck(cudaMalloc(&d_next, NUM_STARS * sizeof(int)), "malloc next");

  Grid3 g3;
  g3.nx=nx; g3.ny=ny; g3.nz=nz;
  g3.cell=cell;
  g3.minx = -maxX - cell;
  g3.miny = -maxY - cell;
  g3.minz = NEAR_Z - cell;
  g3.head = d_head;
  g3.next = d_next;

  // Lines buffers (fixed slots per star)
  LineSeg* d_lines = nullptr;
  int* d_counts = nullptr;
  ck(cudaMalloc(&d_lines,  NUM_STARS * LINE_MAX_NEIGHBORS_PER_STAR * sizeof(LineSeg)), "malloc lines");
  ck(cudaMalloc(&d_counts, NUM_STARS * sizeof(int)), "malloc counts");

  // Init RNG + stars
  dim3 B(256);
  dim3 Gs((NUM_STARS + B.x - 1)/B.x);
  init_rng<<<Gs,B>>>(d_rng, 123456789ULL, NUM_STARS);
  init_stars<<<Gs,B>>>(d_stars, d_rng, NUM_STARS, NEAR_Z, FAR_Z, tan_half, aspect);
  ck(cudaGetLastError(), "init kernels");
  ck(cudaDeviceSynchronize(), "sync init");

  // Host buffers for preview/encode
  cv::Mat frame_gray(H, W, CV_8UC1);
  std::vector<uint8_t> nv12;

  // Start ffmpeg writer
  const std::string out_path = make_out_path_10h();
  FFmpegPipe writer(out_path, W, H, FPS);
  writer.open();
  std::cout << "Recording (10h): " << out_path << "\n";
  std::cout << "3D grid: " << nx << "x" << ny << "x" << nz
            << " cells=" << cells << " cell=" << cell << "\n";

  if (SHOW_PREVIEW) {
    try {
      cv::namedWindow("starfield", cv::WINDOW_NORMAL);
      cv::resizeWindow("starfield", 1280, 720);
    } catch (...) {
      std::cout << "Preview window failed to create (headless?).\n";
    }
  }

  const int64_t total_frames = int64_t(VIDEO_SECONDS) * int64_t(FPS);
  auto t0 = std::chrono::high_resolution_clock::now();
  float sim_time = 0.0f;

  dim3 Gpix(( (int64_t)W*(int64_t)H + B.x - 1)/B.x);
  dim3 Gcells((cells + B.x - 1)/B.x);

  for (int64_t frame=0; frame<total_frames && !g_stop.load(std::memory_order_relaxed); ++frame) {
    // Update stars on GPU
    update_stars<<<Gs,B>>>(d_stars, d_rng, NUM_STARS, STAR_SPEED, DT, NEAR_Z, FAR_Z, tan_half, aspect);

    // Camera rotation
    Mat3 R = camera_rotation(sim_time);
    Mat3 Rt = mat3_transpose(R);

    // Project
    project_stars<<<Gs,B>>>(d_stars, d_draw, NUM_STARS, Rt, f, cx, cy,
                            NEAR_Z, FAR_Z,
                            BRIGHT_NEAR, BRIGHT_FAR, BRIGHT_MIN, BRIGHT_MAX,
                            RADIUS_SCALE, RADIUS_MIN_F, RADIUS_MAX_F);

    // Build 3D grid and generate lines (camera space)
    grid_reset<<<Gcells,B>>>(d_head, cells);
    grid_build<<<Gs,B>>>(g3, d_draw, NUM_STARS, LINE_Z_MAX);
    gen_lines_3d<<<Gs,B>>>(g3, d_draw, NUM_STARS,
                           d_lines, d_counts,
                           LINE_THRESH_3D, NEAR_Z, LINE_Z_MAX);

    // Clear framebuffer
    clear_fbuffer<<<Gpix,B>>>(d_fb, W*H, 0.0f);

    // Render lines then stars
    render_lines<<<Gs,B>>>(d_fb, d_lines, d_counts, NUM_STARS);
    render_stars<<<Gs,B>>>(d_fb, d_draw, NUM_STARS);

    // Convert to U8
    fb_to_u8<<<Gpix,B>>>(d_fb, d_u8, W*H);

    // Download for preview + encode
    ck(cudaMemcpy(frame_gray.data, d_u8, size_t(W)*size_t(H), cudaMemcpyDeviceToHost), "memcpy u8->host");

    if (SHOW_PREVIEW && (frame % PREVIEW_EVERY_N_FRAMES == 0)) {
      try {
        cv::imshow("starfield", frame_gray);
        int key = cv::waitKey(PREVIEW_WAITKEY_MS);
        if (key == 27) g_stop.store(true, std::memory_order_relaxed);
      } catch (...) {}
    }

    // Encode via ffmpeg pipe
    gray_to_nv12_cpu(frame_gray, nv12);
    writer.write_nv12(nv12.data(), nv12.size());

    sim_time += DT;

    if (frame % 600 == 0 && frame > 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double wall = std::chrono::duration<double>(t1 - t0).count();
      double vid  = double(frame) / double(FPS);
      double speed = (wall > 1e-9) ? (vid / wall) : 0.0;

      double remain_vid = double(total_frames - frame) / double(FPS);
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

  cudaFree(d_counts);
  cudaFree(d_lines);
  cudaFree(d_next);
  cudaFree(d_head);
  cudaFree(d_u8);
  cudaFree(d_fb);
  cudaFree(d_rng);
  cudaFree(d_draw);
  cudaFree(d_stars);

  return 0;
}

