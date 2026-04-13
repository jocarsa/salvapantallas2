// starfield.cpp
//
// Conic-projection starfield, fast offline rendering -> MP4 (10 hours).
// Optimized CPU render (tiled + OpenMP) + GPU encode (NVENC or QSV) via NV12 pipe.
//
// Lines: 3D-space (camera-space) proximity using 3D spatial hash grid.
// Rendering lines: Option B (AA lines in 8-bit overlay -> accumulate into float tile).
//
// Deps (Ubuntu):
//   sudo apt install ffmpeg libopencv-dev
//
// Compile:
//   g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 starfield.cpp \
//     `pkg-config --cflags --libs opencv4` -o starfield_openmp
//
// Run:
//   export OMP_NUM_THREADS=12
//   ./starfield_openmp

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
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

// =========================
// CONFIG
// =========================
static constexpr int   W = 3840;
static constexpr int   H = 2160;
static constexpr int   FPS = 60;
static constexpr float DT = 1.0f / float(FPS);

static constexpr int   NUM_STARS = 28000;

static constexpr float FOV_DEG = 70.0f;
static constexpr float NEAR_Z  = 0.20f;
static constexpr float FAR_Z   = 120.0f;

static constexpr float STAR_SPEED = 22.0f; // units/sec

// Medium star size (visible in 4K, not giant blur blobs)
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
static constexpr int   VIDEO_SECONDS = 10 * 60 * 60; // 36000
static const char*     OUT_DIR = "videos_starfield";

// GPU encoder selection:
static const char* HW_ENCODER = "h264_nvenc";

// NVENC knobs
static const char* NVENC_PRESET = "p1";
static constexpr int NVENC_QP = 18;

// QSV knobs (if using h264_qsv)
static constexpr int QSV_BITRATE_Mbps = 20;

// Gaussian sprite LUT (higher resolution)
static constexpr float SPRITE_STEP   = 0.10f;
static constexpr float SPRITE_CUTOFF = 3.25f;

// Core + Glow model
static constexpr float CORE_FRACTION      = 0.85f;
static constexpr float GLOW_FRACTION      = 0.55f;

static constexpr float CORE_RADIUS_FACTOR = 0.40f;
static constexpr float CORE_R_MIN         = 0.90f;
static constexpr float CORE_R_MAX         = 3.20f;

static constexpr float GLOW_RADIUS_FACTOR = 1.55f;
static constexpr float GLOW_R_MIN         = 2.20f;
static constexpr float GLOW_R_MAX         = 16.0f;

static constexpr float SPRITE_MAX_R  = GLOW_R_MAX;

// Optional global blur: expensive at 4K. Keep 0 for max speed.
static constexpr float GLOBAL_BLUR_SIGMA = 0.0f;

// Tiling (parallel)
static constexpr int TILE_W = 128;
static constexpr int TILE_H = 128;

// =========================
// LINES (3D SPACE)
// =========================
//
// Threshold is in CAMERA-SPACE UNITS (same coordinate system as camx/camy/camz).
// Start around 6..14. Larger => denser.
static constexpr float LINE_THRESH_3D = 4.0f;

// If you don’t want background webs, clamp which depths can contribute lines.
// Keep LINE_Z_MAX <= FAR_Z. Example: 45 makes lines mostly foreground/mid.
static constexpr float LINE_Z_MAX = 60.0f;

// Keep per-star connections capped for stability (speed + density).
static constexpr int   LINE_MAX_NEIGHBORS_PER_STAR = 6;

// Line intensity & thickness control
static constexpr float LINE_BRIGHT_FACTOR = 0.11f;  // relative to star brightness
static constexpr float LINE_BRIGHT_MIN    = 4.0f;
static constexpr float LINE_BRIGHT_MAX    = 65.0f;

static constexpr float LINE_THICK_MIN = 2.0f;
static constexpr float LINE_THICK_MAX = 8.0f;

// thickness increases with:
//  - smaller 3D distance
//  - smaller z (nearer camera)
static constexpr float LINE_THICK_DIST_WEIGHT = 0.85f;
static constexpr float LINE_THICK_Z_WEIGHT    = 1.00f;

// =========================
// LIVE PREVIEW
// =========================
static constexpr bool SHOW_PREVIEW = true;
static constexpr int  PREVIEW_EVERY_N_FRAMES = 2;
static constexpr int  PREVIEW_WAITKEY_MS = 1;

// =========================
// GLOBALS
// =========================
static std::atomic<bool> g_stop{false};

static void handle_sigint(int) {
  g_stop.store(true, std::memory_order_relaxed);
}

static inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

// =========================
// Timestamp / paths
// =========================
static std::string now_stamp() {
  auto t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
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
// FFmpeg pipe writer (NV12 -> GPU encoder)
// =========================
class FFmpegPipe {
public:
  FFmpegPipe(const std::string& out_path, int w, int h, int fps)
    : out_path_(out_path), w_(w), h_(h), fps_(fps) {}

  void open() {
    std::ostringstream cmd;

    if (std::string(HW_ENCODER) == "h264_nvenc") {
      cmd
        << "ffmpeg -y "
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
        << "\"" << out_path_ << "\" "
        << "2>/dev/null";
    } else if (std::string(HW_ENCODER) == "h264_qsv") {
      cmd
        << "ffmpeg -y "
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
        << "\"" << out_path_ << "\" "
        << "2>/dev/null";
    } else {
      throw std::runtime_error("Unknown HW_ENCODER. Use h264_nvenc or h264_qsv.");
    }

    pipe_ = popen(cmd.str().c_str(), "w");
    if (!pipe_) throw std::runtime_error("Failed to open ffmpeg pipe.");
  }

  void write_nv12(const uint8_t* nv12, size_t bytes) {
    if (!pipe_) return;
    std::fwrite(nv12, 1, bytes, pipe_);
  }

  void close() {
    if (!pipe_) return;
    pclose(pipe_);
    pipe_ = nullptr;
  }

  ~FFmpegPipe() { close(); }

private:
  std::string out_path_;
  int w_, h_, fps_;
  FILE* pipe_{nullptr};
};

// =========================
// Rotation matrices
// =========================
static inline cv::Matx33f rot_x(float a) {
  float ca = std::cos(a), sa = std::sin(a);
  return cv::Matx33f(1, 0, 0,
                     0, ca, -sa,
                     0, sa, ca);
}
static inline cv::Matx33f rot_y(float a) {
  float ca = std::cos(a), sa = std::sin(a);
  return cv::Matx33f(ca, 0, sa,
                     0,  1, 0,
                     -sa,0, ca);
}
static inline cv::Matx33f rot_z(float a) {
  float ca = std::cos(a), sa = std::sin(a);
  return cv::Matx33f(ca, -sa, 0,
                     sa,  ca, 0,
                     0,   0,  1);
}
static inline cv::Matx33f camera_rotation(float t_sec) {
  const float amp = (AMP_DEG * float(M_PI) / 180.0f);
  const float yaw   = amp * std::sin(2.0f * float(M_PI) * t_sec / YAW_PERIOD_SEC);
  const float pitch = amp * std::sin(2.0f * float(M_PI) * t_sec / PITCH_PERIOD_SEC);
  const float roll  = amp * std::sin(2.0f * float(M_PI) * t_sec / ROLL_PERIOD_SEC);
  return rot_z(roll) * rot_y(yaw) * rot_x(pitch);
}

// =========================
// RNG / star spawning
// =========================
static std::mt19937 rng_from_rd() {
  std::random_device rd;
  std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
  return std::mt19937(seq);
}

static void spawn_stars(std::vector<cv::Vec3f>& dst, int n,
                        float z_min, float z_max, float fov_rad,
                        std::mt19937& rng)
{
  std::uniform_real_distribution<float> dz(z_min, z_max);
  std::uniform_real_distribution<float> du(-1.0f, 1.0f);

  const float tan_half = std::tan(fov_rad * 0.5f);
  const float aspect = float(H) / float(W);

  dst.resize(n);
  for (int i = 0; i < n; ++i) {
    float z = dz(rng);
    float u = du(rng);
    float v = du(rng);

    float x = u * z * tan_half;
    float y = v * z * tan_half * aspect;

    dst[i] = cv::Vec3f(x, y, z);
  }
}

// =========================
// Gaussian sprite cache
// =========================
struct Sprite {
  cv::Mat1f g; // [0..1]
  int half;
};

static inline int radius_key_int(float r) {
  return int(std::lround(r / SPRITE_STEP));
}
static inline float key_to_radius(int k) {
  return float(k) * SPRITE_STEP;
}

static std::unordered_map<int, Sprite> SPRITES;

static Sprite make_gaussian_sprite(float radius_px) {
  const float sigma = std::max(0.35f, radius_px / 2.0f);
  const int half = int(std::ceil(SPRITE_CUTOFF * sigma));
  const int size = 2 * half + 1;

  cv::Mat1f g(size, size);
  const float inv2s2 = 1.0f / (2.0f * sigma * sigma);

  float maxv = 0.0f;
  for (int y = -half; y <= half; ++y) {
    for (int x = -half; x <= half; ++x) {
      float v = std::exp(-(float(x*x + y*y)) * inv2s2);
      g(y + half, x + half) = v;
      if (v > maxv) maxv = v;
    }
  }
  if (maxv > 0.0f) g *= (1.0f / maxv);

  return Sprite{g, half};
}

static void precompute_sprites() {
  SPRITES.reserve(size_t(radius_key_int(SPRITE_MAX_R) + 16));
  const int k_max = std::max(1, radius_key_int(SPRITE_MAX_R));
  for (int k = 1; k <= k_max; ++k) {
    float r = key_to_radius(k);
    SPRITES.emplace(k, make_gaussian_sprite(r));
  }
}

// =========================
// GRAY -> NV12
// =========================
static inline void gray_to_nv12(const cv::Mat& gray_u8, std::vector<uint8_t>& out_nv12) {
  const int w = gray_u8.cols;
  const int h = gray_u8.rows;

  const size_t y_bytes  = size_t(w) * size_t(h);
  const size_t uv_bytes = size_t(w) * size_t(h / 2);

  out_nv12.resize(y_bytes + uv_bytes);

  std::memcpy(out_nv12.data(), gray_u8.data, y_bytes);
  std::memset(out_nv12.data() + y_bytes, 128, uv_bytes);
}

// =========================
// Render helpers
// =========================
struct StarDraw {
  // screen
  float x, y;
  // brightness
  float b;
  // base radius in px
  float r;
  // camera space (for 3D neighbor checks)
  float camx, camy, camz;
  // glow sprite footprint half-size
  int half;
};

struct LineDraw {
  float x1, y1, x2, y2; // screen coords
  float b;
  int thickness;
  int half;
};

static inline void bilinear_splat_into_roi(float* roi, int roi_w, int roi_h,
                                          int roi_x0, int roi_y0,
                                          float xf, float yf, float brightness)
{
  int x0 = int(std::floor(xf));
  int y0 = int(std::floor(yf));
  float dx = xf - float(x0);
  float dy = yf - float(y0);

  float w00 = (1.0f - dx) * (1.0f - dy);
  float w10 = dx * (1.0f - dy);
  float w01 = (1.0f - dx) * dy;
  float w11 = dx * dy;

  int rx = x0 - roi_x0;
  int ry = y0 - roi_y0;

  if (rx >= 0 && ry >= 0 && rx < roi_w && ry < roi_h)
    roi[ry * roi_w + rx] += brightness * w00;

  if ((rx + 1) >= 0 && ry >= 0 && (rx + 1) < roi_w && ry < roi_h)
    roi[ry * roi_w + (rx + 1)] += brightness * w10;

  if (rx >= 0 && (ry + 1) >= 0 && rx < roi_w && (ry + 1) < roi_h)
    roi[(ry + 1) * roi_w + rx] += brightness * w01;

  if ((rx + 1) >= 0 && (ry + 1) >= 0 && (rx + 1) < roi_w && (ry + 1) < roi_h)
    roi[(ry + 1) * roi_w + (rx + 1)] += brightness * w11;
}

static inline void add_sprite_into_roi(float* roi, int roi_w, int roi_h,
                                       int roi_x0, int roi_y0,
                                       int cx_i, int cy_i,
                                       const Sprite& sp,
                                       float brightness)
{
  const int sw = sp.g.cols;
  const int sh = sp.g.rows;
  const int half = sp.half;

  int x1 = cx_i - half;
  int y1 = cy_i - half;
  int x2 = x1 + sw;
  int y2 = y1 + sh;

  int sx1 = std::max(x1, roi_x0);
  int sy1 = std::max(y1, roi_y0);
  int sx2 = std::min(x2, roi_x0 + roi_w);
  int sy2 = std::min(y2, roi_y0 + roi_h);
  if (sx1 >= sx2 || sy1 >= sy2) return;

  int tx1 = sx1 - x1;
  int ty1 = sy1 - y1;

  int rx1 = sx1 - roi_x0;
  int ry1 = sy1 - roi_y0;

  const int wcopy = sx2 - sx1;
  const int hcopy = sy2 - sy1;

  for (int j = 0; j < hcopy; ++j) {
    const float* sp_row = sp.g.ptr<float>(ty1 + j) + tx1;
    float* out_row = roi + (ry1 + j) * roi_w + rx1;
    for (int i = 0; i < wcopy; ++i) out_row[i] += brightness * sp_row[i];
  }
}

static inline void add_hard_disk_into_roi(float* roi, int roi_w, int roi_h,
                                         int roi_x0, int roi_y0,
                                         int cx_i, int cy_i,
                                         float r, float brightness)
{
  int ir = (int)std::ceil(r);
  int x1 = cx_i - ir, x2 = cx_i + ir;
  int y1 = cy_i - ir, y2 = cy_i + ir;

  int sx1 = std::max(x1, roi_x0);
  int sy1 = std::max(y1, roi_y0);
  int sx2 = std::min(x2 + 1, roi_x0 + roi_w);
  int sy2 = std::min(y2 + 1, roi_y0 + roi_h);
  if (sx1 >= sx2 || sy1 >= sy2) return;

  const float rr = r * r;

  for (int y = sy1; y < sy2; ++y) {
    float* row = roi + (y - roi_y0) * roi_w;
    float dy = float(y - cy_i);
    for (int x = sx1; x < sx2; ++x) {
      float dx = float(x - cx_i);
      if (dx*dx + dy*dy <= rr) row[x - roi_x0] += brightness;
    }
  }
}

// 3D grid for neighbor search
struct Cell3Key {
  int ix, iy, iz;
  bool operator==(const Cell3Key& o) const { return ix == o.ix && iy == o.iy && iz == o.iz; }
};
struct Cell3Hash {
  std::size_t operator()(const Cell3Key& k) const noexcept {
    std::size_t h1 = (std::size_t)(k.ix * 73856093);
    std::size_t h2 = (std::size_t)(k.iy * 19349663);
    std::size_t h3 = (std::size_t)(k.iz * 83492791);
    return h1 ^ h2 ^ h3;
  }
};

static cv::Mat render_frame_tiled(const std::vector<cv::Vec3f>& stars,
                                  float f, float cx, float cy,
                                  const cv::Matx33f& Rt)
{
  cv::Mat1f imgf(H, W, 0.0f);

  const int tiles_x = (W + TILE_W - 1) / TILE_W;
  const int tiles_y = (H + TILE_H - 1) / TILE_H;
  const int tile_count = tiles_x * tiles_y;

  std::vector<std::vector<int>> tile_lists(tile_count);      // stars per tile
  std::vector<std::vector<int>> tile_line_lists(tile_count); // lines per tile

  std::vector<StarDraw> draws;
  draws.reserve(stars.size());

  // Pass 1: project + create draw list
  for (int i = 0; i < (int)stars.size(); ++i) {
    const cv::Vec3f s = stars[size_t(i)];

    const float camx = Rt(0,0)*s[0] + Rt(0,1)*s[1] + Rt(0,2)*s[2];
    const float camy = Rt(1,0)*s[0] + Rt(1,1)*s[1] + Rt(1,2)*s[2];
    const float camz = Rt(2,0)*s[0] + Rt(2,1)*s[1] + Rt(2,2)*s[2];

    if (camz <= NEAR_Z) continue;

    const float invz = f / camz;
    const float x = camx * invz + cx;
    const float y = camy * invz + cy;

    if (x < -128.0f || x > (W + 128.0f) || y < -128.0f || y > (H + 128.0f)) continue;

    float r = RADIUS_SCALE / (camz + 1e-6f);
    r = clampf(r, RADIUS_MIN_F, RADIUS_MAX_F);

    float t = 1.0f - (camz - NEAR_Z) / (FAR_Z - NEAR_Z);
    float b = BRIGHT_NEAR * t + BRIGHT_FAR;
    b = clampf(b, BRIGHT_MIN, BRIGHT_MAX);

    float glow_r = clampf(r * GLOW_RADIUS_FACTOR, GLOW_R_MIN, GLOW_R_MAX);
    int glow_key = std::max(1, radius_key_int(glow_r));
    auto itg = SPRITES.find(glow_key);
    if (itg == SPRITES.end()) continue;

    StarDraw d{};
    d.x = x; d.y = y; d.b = b; d.r = r;
    d.camx = camx; d.camy = camy; d.camz = camz;
    d.half = itg->second.half;

    draws.push_back(d);
  }

  // ---- Lines: 3D neighbor search in camera space ----
  std::vector<LineDraw> lines;
  lines.reserve(draws.size());

  const float thresh = LINE_THRESH_3D;
  const float thresh2 = thresh * thresh;
  const float inv_cell = 1.0f / thresh; // cell size = thresh

  std::unordered_map<Cell3Key, std::vector<int>, Cell3Hash> grid;
  grid.reserve(draws.size() * 2);

  auto cell3_of = [&](float x, float y, float z) -> Cell3Key {
    return Cell3Key{
      (int)std::floor(x * inv_cell),
      (int)std::floor(y * inv_cell),
      (int)std::floor(z * inv_cell)
    };
  };

  for (int i = 0; i < (int)draws.size(); ++i) {
    const StarDraw& a = draws[size_t(i)];
    if (a.camz > LINE_Z_MAX) continue; // optional: ignore far background for lines
    grid[cell3_of(a.camx, a.camy, a.camz)].push_back(i);
  }

  for (int i = 0; i < (int)draws.size(); ++i) {
    const StarDraw& a = draws[size_t(i)];
    if (a.camz > LINE_Z_MAX) continue;

    Cell3Key ck = cell3_of(a.camx, a.camy, a.camz);
    int added = 0;

    for (int oz = -1; oz <= 1 && added < LINE_MAX_NEIGHBORS_PER_STAR; ++oz) {
      for (int oy = -1; oy <= 1 && added < LINE_MAX_NEIGHBORS_PER_STAR; ++oy) {
        for (int ox = -1; ox <= 1 && added < LINE_MAX_NEIGHBORS_PER_STAR; ++ox) {
          Cell3Key nk{ck.ix + ox, ck.iy + oy, ck.iz + oz};
          auto it = grid.find(nk);
          if (it == grid.end()) continue;

          for (int j : it->second) {
            if (j <= i) continue;
            const StarDraw& b = draws[size_t(j)];
            if (b.camz > LINE_Z_MAX) continue;

            float dx = b.camx - a.camx;
            float dy = b.camy - a.camy;
            float dz = b.camz - a.camz;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > thresh2) continue;

            float d = std::sqrt(std::max(1e-6f, d2));
            float dist_factor = 1.0f - clampf(d / thresh, 0.0f, 1.0f); // closer => 1

            float z_avg = 0.5f * (a.camz + b.camz);
            float z_norm = clampf((z_avg - NEAR_Z) / (LINE_Z_MAX - NEAR_Z + 1e-6f), 0.0f, 1.0f);
            float z_factor = 1.0f - z_norm; // near => 1

            float thick_f =
                LINE_THICK_MIN
              + (LINE_THICK_MAX - LINE_THICK_MIN) * (
                    LINE_THICK_DIST_WEIGHT * dist_factor
                  + LINE_THICK_Z_WEIGHT    * z_factor * 0.55f
                );

            int thickness = (int)std::lround(clampf(thick_f, LINE_THICK_MIN, LINE_THICK_MAX));

            float b_avg = 0.5f * (a.b + b.b);
            float lb = b_avg * LINE_BRIGHT_FACTOR * (0.35f + 0.65f * dist_factor) * (0.60f + 0.40f * z_factor);
            lb = clampf(lb, LINE_BRIGHT_MIN, LINE_BRIGHT_MAX);

            LineDraw ln{};
            ln.x1 = a.x; ln.y1 = a.y; ln.x2 = b.x; ln.y2 = b.y;
            ln.b = lb;
            ln.thickness = thickness;
            ln.half = thickness + 3;
            lines.push_back(ln);

            added++;
            if (added >= LINE_MAX_NEIGHBORS_PER_STAR) break;
          }
        }
      }
    }
  }

  // Pass 2: bin stars into tiles (including spill)
  for (int di = 0; di < (int)draws.size(); ++di) {
    const StarDraw& d = draws[size_t(di)];
    int cx_i = int(std::lround(d.x));
    int cy_i = int(std::lround(d.y));

    int x1 = cx_i - d.half;
    int y1 = cy_i - d.half;
    int x2 = cx_i + d.half + 1;
    int y2 = cy_i + d.half + 1;

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(W, x2);
    y2 = std::min(H, y2);
    if (x1 >= x2 || y1 >= y2) continue;

    int tx1 = x1 / TILE_W;
    int ty1 = y1 / TILE_H;
    int tx2 = (x2 - 1) / TILE_W;
    int ty2 = (y2 - 1) / TILE_H;

    for (int ty = ty1; ty <= ty2; ++ty)
      for (int tx = tx1; tx <= tx2; ++tx)
        tile_lists[size_t(ty * tiles_x + tx)].push_back(di);
  }

  // Pass 2b: bin lines into tiles (bbox + thickness)
  for (int li = 0; li < (int)lines.size(); ++li) {
    const LineDraw& ln = lines[size_t(li)];

    int x1 = (int)std::floor(std::min(ln.x1, ln.x2)) - ln.half;
    int y1 = (int)std::floor(std::min(ln.y1, ln.y2)) - ln.half;
    int x2 = (int)std::ceil (std::max(ln.x1, ln.x2)) + ln.half;
    int y2 = (int)std::ceil (std::max(ln.y1, ln.y2)) + ln.half;

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(W - 1, x2);
    y2 = std::min(H - 1, y2);
    if (x1 >= x2 || y1 >= y2) continue;

    int tx1 = x1 / TILE_W;
    int ty1 = y1 / TILE_H;
    int tx2 = x2 / TILE_W;
    int ty2 = y2 / TILE_H;

    for (int ty = ty1; ty <= ty2; ++ty)
      for (int tx = tx1; tx <= tx2; ++tx)
        tile_line_lists[size_t(ty * tiles_x + tx)].push_back(li);
  }

  // Pass 3: render tiles in parallel
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int tile_id = 0; tile_id < tile_count; ++tile_id) {
    const int tx = tile_id % tiles_x;
    const int ty = tile_id / tiles_x;

    const int x0 = tx * TILE_W;
    const int y0 = ty * TILE_H;
    const int tw = std::min(TILE_W, W - x0);
    const int th = std::min(TILE_H, H - y0);

    std::vector<float> tile(size_t(tw * th), 0.0f);
    float* roi = tile.data();

    // Option B: AA lines in 8-bit overlay -> accumulate into float ROI
    cv::Mat1b line8(th, tw, (uint8_t)0);

    const auto& llist = tile_line_lists[size_t(tile_id)];
    for (int li : llist) {
      const LineDraw& ln = lines[size_t(li)];

      cv::Point p1((int)std::lround(ln.x1) - x0, (int)std::lround(ln.y1) - y0);
      cv::Point p2((int)std::lround(ln.x2) - x0, (int)std::lround(ln.y2) - y0);

      int minx = std::min(p1.x, p2.x), maxx = std::max(p1.x, p2.x);
      int miny = std::min(p1.y, p2.y), maxy = std::max(p1.y, p2.y);
      if (maxx < -ln.half || minx > tw - 1 + ln.half || maxy < -ln.half || miny > th - 1 + ln.half) continue;

      cv::line(line8, p1, p2, cv::Scalar(255), ln.thickness, cv::LINE_AA);

      const float scale = ln.b / 255.0f;
      for (int yy = 0; yy < th; ++yy) {
        const uint8_t* m = line8.ptr<uint8_t>(yy);
        float* out = roi + yy * tw;
        for (int xx = 0; xx < tw; ++xx) out[xx] += float(m[xx]) * scale;
      }

      line8.setTo(0);
    }

    // Stars
    const auto& list = tile_lists[size_t(tile_id)];
    for (int di : list) {
      const StarDraw& d = draws[size_t(di)];
      int cx_i = int(std::lround(d.x));
      int cy_i = int(std::lround(d.y));

      float core_r = clampf(d.r * CORE_RADIUS_FACTOR, CORE_R_MIN, CORE_R_MAX);
      float core_b = d.b * CORE_FRACTION;

      if (core_r < 1.05f) bilinear_splat_into_roi(roi, tw, th, x0, y0, d.x, d.y, core_b);
      else add_hard_disk_into_roi(roi, tw, th, x0, y0, cx_i, cy_i, core_r, core_b);

      float glow_r = clampf(d.r * GLOW_RADIUS_FACTOR, GLOW_R_MIN, GLOW_R_MAX);
      int glow_key = std::max(1, radius_key_int(glow_r));
      auto itg = SPRITES.find(glow_key);
      if (itg != SPRITES.end()) add_sprite_into_roi(roi, tw, th, x0, y0, cx_i, cy_i, itg->second, d.b * GLOW_FRACTION);
    }

    // Copy tile to full image
    float* dst = imgf.ptr<float>(y0) + x0;
    for (int yy = 0; yy < th; ++yy) {
      std::memcpy(dst + yy * W, roi + yy * tw, size_t(tw) * sizeof(float));
    }
  }

  if (GLOBAL_BLUR_SIGMA > 1e-6f) {
    cv::GaussianBlur(imgf, imgf, cv::Size(0,0), GLOBAL_BLUR_SIGMA, GLOBAL_BLUR_SIGMA, cv::BORDER_DEFAULT);
  }

  // Float -> U8 gray
  cv::Mat1b img8(H, W);
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < H; ++y) {
    const float* src = imgf.ptr<float>(y);
    uint8_t* dst = img8.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      float v = src[x];
      v = (v < 0.0f) ? 0.0f : (v > 255.0f ? 255.0f : v);
      dst[x] = (uint8_t)(v + 0.5f);
    }
  }

  return img8;
}

// =========================
// MAIN
// =========================
int main() {
  std::signal(SIGINT, handle_sigint);

  std::string mkdir_cmd = std::string("mkdir -p \"") + OUT_DIR + "\"";
  std::system(mkdir_cmd.c_str());

  precompute_sprites();
  std::cout << "Precomputed sprites: " << SPRITES.size()
            << "  (SPRITE_STEP=" << SPRITE_STEP << ")\n";
#ifdef _OPENMP
  std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif
  std::cout << "FFmpeg HW encoder: " << HW_ENCODER << "\n";
  std::cout << "Line 3D thresh: " << LINE_THRESH_3D
            << "  LINE_Z_MAX: " << LINE_Z_MAX
            << "  max neighbors/star: " << LINE_MAX_NEIGHBORS_PER_STAR << "\n";

  const float fov_rad = FOV_DEG * float(M_PI) / 180.0f;
  const float f = (W * 0.5f) / std::tan(fov_rad * 0.5f);
  const float cx = W * 0.5f;
  const float cy = H * 0.5f;

  std::mt19937 rng = rng_from_rd();
  std::vector<cv::Vec3f> stars;
  spawn_stars(stars, NUM_STARS, NEAR_Z, FAR_Z, fov_rad, rng);

  const int64_t total_frames = int64_t(VIDEO_SECONDS) * int64_t(FPS);

  std::vector<uint8_t> nv12;

  const std::string out_path = make_out_path_10h();
  FFmpegPipe writer(out_path, W, H, FPS);
  writer.open();
  std::cout << "Recording (10h): " << out_path << "\n";

  if (SHOW_PREVIEW) {
    try {
      cv::namedWindow("starfield", cv::WINDOW_NORMAL);
      cv::resizeWindow("starfield", 1280, 720);
    } catch (...) {
      std::cout << "Preview window failed to create (headless?).\n";
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  int64_t frames_done = 0;
  float sim_time = 0.0f;

  while (frames_done < total_frames && !g_stop.load(std::memory_order_relaxed)) {
    for (auto& s : stars) s[2] -= STAR_SPEED * DT;

    int dead_count = 0;
    for (const auto& s : stars) if (s[2] <= NEAR_Z) dead_count++;

    if (dead_count > 0) {
      std::vector<cv::Vec3f> newstars;
      spawn_stars(newstars, dead_count, FAR_Z * 0.8f, FAR_Z, fov_rad, rng);

      int k = 0;
      for (auto& s : stars) if (s[2] <= NEAR_Z) s = newstars[size_t(k++)];
    }

    const cv::Matx33f R  = camera_rotation(sim_time);
    const cv::Matx33f Rt = R.t();

    cv::Mat frame_gray = render_frame_tiled(stars, f, cx, cy, Rt);

    if (SHOW_PREVIEW && (frames_done % PREVIEW_EVERY_N_FRAMES == 0)) {
      try {
        cv::imshow("starfield", frame_gray);
        int key = cv::waitKey(PREVIEW_WAITKEY_MS);
        if (key == 27) g_stop.store(true, std::memory_order_relaxed);
      } catch (...) {}
    }

    gray_to_nv12(frame_gray, nv12);
    writer.write_nv12(nv12.data(), nv12.size());

    frames_done++;
    sim_time += DT;

    if (frames_done % 600 == 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double wall = std::chrono::duration<double>(t1 - t0).count();
      double vid  = double(frames_done) / double(FPS);
      double speed = (wall > 1e-9) ? (vid / wall) : 0.0;

      double remain_vid = double(total_frames - frames_done) / double(FPS);
      double remain_wall = (speed > 1e-9) ? (remain_vid / speed) : 0.0;

      std::cout << "Frames: " << frames_done
                << " / " << total_frames
                << "  Video: " << std::fixed << std::setprecision(1) << vid << "s"
                << "  Speed: " << std::setprecision(2) << speed << "x realtime"
                << "  Est. remaining wall: " << std::setprecision(1) << remain_wall << "s\n";
    }
  }

  writer.close();
  std::cout << "Done. Wrote: " << out_path << "\n";
  return 0;
}

