// terrain_cuda.cu
// CUDA terrain renderer -> direct GPU framebuffer -> FFmpeg NVENC MP4
//
// Main design:
// 1) GPU computes terrain / sky / clouds / biome fields
// 2) GPU rasterizes directly into a packed framebuffer
// 3) CPU copies only the final image
// 4) FFmpeg encodes with h264_nvenc
// 5) Every 10 seconds, shows a temporary OpenCV framebuffer preview
//
// Build (Ubuntu):
//   nvcc -O3 -std=c++17 terrain_cuda.cu `pkg-config --cflags --libs opencv4` -o terrain_cuda
//
// Run:
//   ./terrain_cuda
//   ./terrain_cuda 30
//   ./terrain_cuda 30 salida.mp4
//   ./terrain_cuda 30 salida.mp4 0.35 1.0
//   ./terrain_cuda 30 salida.mp4 0.35 1.0 900
//   ./terrain_cuda 30 salida.mp4 0.35 1.0 900 1.0
//   ./terrain_cuda 30 salida.mp4 0.35 1.0 900 1.8
//   ./terrain_cuda 30 salida.mp4 0.35 1.0 900 1.8 0.40
//   ./terrain_cuda 30 salida.mp4 0.35 1.0 900 1.8 0.40 1.35
//
// Args:
//   argv[1] = duration_seconds           (default 3600)
//   argv[2] = output mp4 filename        (default auto-generated)
//   argv[3] = forward_speed              (default 0.35)
//   argv[4] = radius_multiplier          (default 1.0)
//   argv[5] = near_subdivisions          (default 900)
//   argv[6] = shadow_darkness_multiplier (default 1.0)
//   argv[7] = water_reflection_strength  (default 0.40)
//   argv[8] = terrain_height_multiplier  (default 1.0)
//
// This version:
// - Keeps more fractal detail than the original
// - But is much cheaper than the heavy variant
// - Terrain: fine + macro + detail1 + ridged1
// - Clouds: base + detail1 + puffy
// - Water: only two ripple fields
// - Octaves reduced back to 4 for speed
// - Adds a real terrain_height_multiplier via CUDA constant memory
// - Terrain circles are SOLID, with no inner radial gradient

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// =========================
// Configuration
// =========================
static constexpr int width  = 1920;
static constexpr int height = 1080;
static constexpr int output_fps = 60;

// World extents
static constexpr float plane_width = 400.0f;
static constexpr float plane_depth = 220.0f;

// Start sampling just in front of the camera
static constexpr float terrain_sample_near_z = 6.0f;
static constexpr float sky_sample_near_z     = 6.0f;

// Single global density control
static int near_subdivisions = 900;

static constexpr float terrain_visible_overscan = 1.30f;
static constexpr float cloud_visible_overscan   = 1.35f;
static constexpr float sky_visible_overscan     = 1.35f;

struct TerrainModification {
    float value = 0.0f;
    float persistence = 0.99f;
};

// Fog configuration
static constexpr float fog_near = 20.0f;
static constexpr float fog_far  = 420.0f;
static constexpr float fog_density = 1.0f;

// Sky fog
static constexpr float sky_fog_near = 40.0f;
static constexpr float sky_fog_far  = 560.0f;
static constexpr float sky_fog_density = 1.0f;

// Sky world extents
static constexpr float sky_height = 20.0f;
static constexpr float sky_plane_width = 1600.0f;
static constexpr float sky_plane_depth = 900.0f;

// Background
static const cv::Scalar window_bg_color = cv::Scalar(255, 255, 255);

// Camera
static float terrain_height_multiplier = 2.0f;
static cv::Vec3f camera_position = cv::Vec3f(0.0f, 3.0f * terrain_height_multiplier, 3.0f);
static cv::Vec3f camera_rotation = cv::Vec3f(-0.1f, 0.0f, 0.0f);

// Projection / noise
static constexpr float focal_length = 800.0f;

// Balanced fast fractal terrain
static constexpr float noise_scale = 0.15f;
static constexpr float noise_amplitude = 1.35f;
static constexpr int   octaves = 4;
static constexpr float persistence = 0.54f;
static constexpr float lacunarity = 2.10f;

static constexpr float macro_noise_scale = 0.0090f;
static constexpr float macro_noise_amplitude = 3.2f;

// Biome scales
static constexpr float biome_temp_scale      = 0.0024f;
static constexpr float biome_humidity_scale  = 0.0020f;
static constexpr float biome_rugged_scale    = 0.0032f;
static constexpr float biome_region_scale    = 0.0012f;

// Clouds
static constexpr float cloud_height = 6.0f;
static constexpr float cloud_noise_scale = 0.070f;
static constexpr float cloud_noise_amplitude = 1.2f;
static constexpr float cloud_threshold_min = 0.15f;

// Circle radius limits
static constexpr int max_circle_radius = 8;
static constexpr int min_circle_radius = 1;

// Runtime parameters
static float forward_speed = -0.065f;
static float radius_multiplier = 1.0f;
static float shadow_darkness_multiplier = 1.0f;
static float water_reflection_strength = 0.40f;

// Visual tuning
static constexpr float terrain_radius_scale = 0.85f;
static constexpr float cloud_radius_scale   = 1.08f;
static constexpr float sky_radius_scale     = 0.84f;

// Jitter strength in world units
static constexpr float terrain_jitter_x = 0.18f;
static constexpr float terrain_jitter_z = 0.10f;
static constexpr float cloud_jitter_x   = 0.35f;
static constexpr float cloud_jitter_z   = 0.25f;

// Sky jitter is adaptive to local cell size / depth band size
static constexpr float sky_jitter_x_mul = 0.42f;
static constexpr float sky_jitter_z_mul = 0.42f;
static constexpr float sky_row_warp_mul = 0.35f;

// Sun / terrain lighting
static constexpr float sun_azimuth_deg   = -45.0f;
static constexpr float sun_elevation_deg = 16.0f;
static constexpr float ambient_light      = 0.24f;
static constexpr float diffuse_strength   = 1.05f;
static constexpr float backlight_strength = 0.05f;
static constexpr float normal_eps         = 0.75f;

// Cloud shadow projection
static constexpr float cloud_shadow_strength_base  = 0.92f;
static constexpr float cloud_shadow_softness       = 0.07f;
static constexpr float cloud_shadow_min_light_base = 0.08f;

// Water reflection tuning
static constexpr float water_fresnel_bias             = 0.10f;
static constexpr float water_fresnel_power            = 3.8f;
static constexpr float water_specular_strength        = 0.18f;

// Small tighter ripples, but cheap
static constexpr float water_wave_noise_scale         = 0.26f;
static constexpr float water_wave_distort             = 0.10f;

static constexpr float water_max_reflection_dist      = 500.0f;
static constexpr int   water_reflection_steps         = 56;
static constexpr float water_reflection_step_min      = 1.5f;
static constexpr float water_reflection_step_max      = 14.0f;
static constexpr float water_reflection_cloud_gain    = 0.70f;
static constexpr float water_reflection_terrain_gain  = 0.95f;
static constexpr float water_reflection_sky_gain      = 0.85f;

// Preview showing
static constexpr int preview_every_seconds = 2;

// Biome timing
static constexpr int biome_slot_seconds       = 60;
static constexpr int biome_hold_seconds       = 50;
static constexpr int biome_transition_seconds = 10;

// =========================
// Host helpers
// =========================
template<typename T>
static inline T clamp_host(T v, T lo, T hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float lerp_host(float a, float b, float t){
    t = clamp_host(t, 0.0f, 1.0f);
    return a + (b - a) * t;
}

static inline float3 normalize3_host(float3 v){
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 1e-20f) return make_float3(0.f, -1.f, 0.f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

static std::string make_default_output_filename() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif

    std::ostringstream oss;
    oss << "infinite earth terrain simulation "
        << static_cast<long long>(now)
        << " "
        << std::put_time(&tm_buf, "%Y-%m-%d_%H-%M-%S")
        << ".mp4";
    return oss.str();
}

static int compute_inverse_perspective_row_count(float, float, int subdivisions){
    subdivisions = std::max(8, subdivisions);
    return std::max(2, subdivisions);
}

static void show_preview_window(const cv::Mat& frame, long long frame_idx){
    static bool window_created = false;
    static cv::Mat preview;

    if (!window_created) {
        cv::namedWindow("terrain preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("terrain preview", 960, 540);
        window_created = true;
    }

    preview = frame.clone();

    const long long sec = frame_idx / output_fps;

    std::ostringstream oss;
    oss << "preview t=" << sec << "s";

    cv::putText(
        preview,
        oss.str(),
        cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        1.1,
        cv::Scalar(20, 20, 20),
        3,
        cv::LINE_AA
    );

    cv::imshow("terrain preview", preview);
    cv::waitKey(1);
}

// =========================
// Biome style state
// =========================
struct BiomeStyle {
    float global_temperature;
    float global_humidity;
    float global_ruggedness;
    float global_snowline;
    float cloudiness;
};

static BiomeStyle biome_style_desert(){
    return {0.95f, 0.10f, 0.30f, 0.95f, 0.20f};
}

static BiomeStyle biome_style_valley(){
    return {0.55f, 0.85f, 0.20f, 0.75f, 0.75f};
}

static BiomeStyle biome_style_mountain(){
    return {0.35f, 0.35f, 0.95f, 0.45f, 0.45f};
}

static BiomeStyle biome_style_snow(){
    return {0.10f, 0.45f, 0.80f, 0.10f, 0.65f};
}

static BiomeStyle biome_lerp(const BiomeStyle& a, const BiomeStyle& b, float t){
    BiomeStyle r{};
    r.global_temperature = lerp_host(a.global_temperature, b.global_temperature, t);
    r.global_humidity    = lerp_host(a.global_humidity,    b.global_humidity,    t);
    r.global_ruggedness  = lerp_host(a.global_ruggedness,  b.global_ruggedness,  t);
    r.global_snowline    = lerp_host(a.global_snowline,    b.global_snowline,    t);
    r.cloudiness         = lerp_host(a.cloudiness,         b.cloudiness,         t);
    return r;
}

static BiomeStyle random_biome_style(std::mt19937& rng){
    std::uniform_int_distribution<int> dist(0, 3);
    int k = dist(rng);
    switch (k) {
        case 0: return biome_style_desert();
        case 1: return biome_style_valley();
        case 2: return biome_style_mountain();
        default:return biome_style_snow();
    }
}

// =========================
// CUDA utilities
// =========================
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

__device__ __forceinline__ float clampf(float v, float lo, float hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float fade(float t){
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

__device__ __forceinline__ float lerpf(float a, float b, float t){
    return a + t * (b - a);
}

__device__ __forceinline__ float smooth01(float t){
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float3 add3(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float s){
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b){
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__device__ __forceinline__ float3 normalize3(float3 v){
    float len2 = dot3(v, v);
    if (len2 <= 1e-20f) return make_float3(0.f, 1.f, 0.f);
    float inv = rsqrtf(len2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ __forceinline__ float3 reflect3(float3 i, float3 n){
    return sub3(i, mul3(n, 2.0f * dot3(i, n)));
}

__constant__ int d_perm[512];
__constant__ float d_terrain_height_multiplier;

__device__ __forceinline__ float grad2(int hash, float x, float y){
    int h = hash & 3;
    float u = (h < 2) ? x : y;
    float v = (h < 2) ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.f * v : 2.f * v);
}

__device__ float perlin2(float x, float y){
    int xi = ((int)floorf(x)) & 255;
    int yi = ((int)floorf(y)) & 255;
    float xf = x - floorf(x);
    float yf = y - floorf(y);
    float u = fade(xf);
    float v = fade(yf);

    int aa = d_perm[d_perm[xi] + yi];
    int ab = d_perm[d_perm[xi] + yi + 1];
    int ba = d_perm[d_perm[xi + 1] + yi];
    int bb = d_perm[d_perm[xi + 1] + yi + 1];

    float x1 = lerpf(grad2(aa, xf, yf),       grad2(ba, xf - 1.f, yf),       u);
    float x2 = lerpf(grad2(ab, xf, yf - 1.f), grad2(bb, xf - 1.f, yf - 1.f), u);
    return lerpf(x1, x2, v);
}

__device__ float fractal_noise2(float x, float y){
    float total = 0.f;
    float freq  = 1.f;
    float amp   = 1.f;
    float maxa  = 0.f;
    #pragma unroll
    for (int i = 0; i < octaves; i++){
        total += perlin2(x * freq, y * freq) * amp;
        maxa  += amp;
        amp   *= persistence;
        freq  *= lacunarity;
    }
    return total / maxa;
}

// =========================
// Small hash/jitter helpers
// =========================
__device__ __forceinline__ unsigned int hash_u32(unsigned int x){
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float rand01_from_2i(int a, int b, unsigned int seed){
    unsigned int h = hash_u32((unsigned int)a * 73856093u ^ (unsigned int)b * 19349663u ^ seed);
    return (float)(h & 0x00FFFFFFu) / 16777215.0f;
}

__device__ __forceinline__ float rand_signed_from_2i(int a, int b, unsigned int seed){
    return rand01_from_2i(a, b, seed) * 2.0f - 1.0f;
}

// =========================
// GPU fog + color helpers
// =========================
__device__ __forceinline__ void fog_mix(float& b, float& g, float& r, float depth, bool is_sky){
    float fog_factor = 0.f;
    if (!is_sky) {
        if (depth > fog_near) {
            float nd = (depth - fog_near) / (fog_far - fog_near);
            nd = clampf(nd, 0.f, 1.f);
            fog_factor = 1.f - expf(-fog_density * nd * 5.f);
        }
    } else {
        if (depth > sky_fog_near) {
            float nd = (depth - sky_fog_near) / (sky_fog_far - sky_fog_near);
            nd = clampf(nd, 0.f, 1.f);
            fog_factor = 1.f - expf(-sky_fog_density * nd * 5.f);
        }
    }

    b = b * (1.f - fog_factor) + 255.f * fog_factor;
    g = g * (1.f - fog_factor) + 255.f * fog_factor;
    r = r * (1.f - fog_factor) + 255.f * fog_factor;
}

struct DeviceBiomeStyle {
    float global_temperature;
    float global_humidity;
    float global_ruggedness;
    float global_snowline;
    float cloudiness;
};

// =========================
// GPU projection
// =========================
struct Mat3 { float m[9]; };

__device__ __forceinline__ float3 mul(const Mat3& R, float3 v){
    return make_float3(
        R.m[0]*v.x + R.m[1]*v.y + R.m[2]*v.z,
        R.m[3]*v.x + R.m[4]*v.y + R.m[5]*v.z,
        R.m[6]*v.x + R.m[7]*v.y + R.m[8]*v.z
    );
}

__device__ bool project_point_gpu(float3 point3D, int& u, int& v, float& depth_out, float3 camPos, Mat3 R){
    float3 rel = make_float3(point3D.x - camPos.x, point3D.y - camPos.y, point3D.z - camPos.z);
    float3 rot = mul(R, rel);

    float z = rot.z;
    if (z <= 0.01f) return false;

    u = (int)(width  * 0.5f + focal_length * rot.x / z);
    v = (int)(height * 0.5f - focal_length * rot.y / z);

    if ((unsigned)u >= (unsigned)width || (unsigned)v >= (unsigned)height) return false;

    depth_out = z;
    return true;
}

__device__ __forceinline__ int compute_radius_gpu(float rel_z, float radius_mult){
    float z = (rel_z > 1.f) ? rel_z : 1.f;
    float normalized_z = z / plane_depth;
    float perspective_scale = 1.f / normalized_z;
    float t = clampf(perspective_scale, 0.f, 1.f);
    float base = (min_circle_radius + t * (max_circle_radius - min_circle_radius));
    float scaled = base * radius_mult;
    if (scaled < 1.f) scaled = 1.f;
    if (scaled > 64.f) scaled = 64.f;
    return (int)lrintf(scaled);
}

// =========================
// Consistent-density sampling helpers
// =========================
__device__ __forceinline__ float visible_world_width_at_depth(float depth){
    return ((float)width * depth) / focal_length;
}

__device__ __forceinline__ float sample_depth_inverse_perspective(int row, int total_rows, float z_near, float z_far){
    if (total_rows <= 1) return z_near;

    float t = (float)row / (float)(total_rows - 1);
    float inv_near = 1.0f / z_near;
    float inv_far  = 1.0f / z_far;
    float inv_z    = lerpf(inv_near, inv_far, t);
    return 1.0f / inv_z;
}

__device__ __forceinline__ float sample_x_for_column(int col, int total_cols, float world_width){
    if (total_cols <= 1) return 0.0f;
    float t = (float)col / (float)(total_cols - 1);
    return -world_width * 0.5f + t * world_width;
}

__device__ __forceinline__ float row_stagger_offset(int row, int total_cols, float world_width){
    if (total_cols <= 1) return 0.0f;
    float cell = world_width / (float)(total_cols - 1);
    return (row & 1) ? (0.5f * cell) : 0.0f;
}

__device__ __forceinline__ float cell_width_for_cols(int total_cols, float world_width){
    if (total_cols <= 1) return world_width;
    return world_width / (float)(total_cols - 1);
}

__device__ __forceinline__ float depth_band_size_inverse_perspective(int row, int total_rows, float z_near, float z_far){
    if (total_rows <= 1) return z_far - z_near;
    float z0 = sample_depth_inverse_perspective(row, total_rows, z_near, z_far);
    int row1 = min(row + 1, total_rows - 1);
    float z1 = sample_depth_inverse_perspective(row1, total_rows, z_near, z_far);
    float dz = fabsf(z1 - z0);
    return (dz > 1e-6f) ? dz : 1e-6f;
}

// =========================
// Biome helpers
// =========================
__device__ __forceinline__ float local_temperature(float x, float z, const DeviceBiomeStyle& style){
    float t = clampf(
        0.5f + 0.5f * fractal_noise2(x * biome_temp_scale + 130.0f, z * biome_temp_scale + 210.0f),
        0.0f, 1.0f
    );
    return clampf(lerpf(t, style.global_temperature, 0.35f), 0.0f, 1.0f);
}

__device__ __forceinline__ float local_humidity(float x, float z, const DeviceBiomeStyle& style){
    float h = clampf(
        0.5f + 0.5f * fractal_noise2(x * biome_humidity_scale + 700.0f, z * biome_humidity_scale + 1200.0f),
        0.0f, 1.0f
    );
    return clampf(lerpf(h, style.global_humidity, 0.35f), 0.0f, 1.0f);
}

__device__ __forceinline__ float local_ruggedness(float x, float z, const DeviceBiomeStyle& style){
    float r = clampf(
        0.5f + 0.5f * fractal_noise2(x * biome_rugged_scale + 3000.0f, z * biome_rugged_scale + 4000.0f),
        0.0f, 1.0f
    );
    return clampf(lerpf(r, style.global_ruggedness, 0.35f), 0.0f, 1.0f);
}

struct BiomeWeights {
    float desert;
    float valley;
    float mountain;
    float temp;
    float humid;
    float rugged;
};

__device__ __forceinline__ BiomeWeights biome_weights_at(float x, float z, float, const DeviceBiomeStyle& style){
    BiomeWeights bw{};

    float region = 0.5f + 0.5f * fractal_noise2(x * biome_region_scale + 1200.0f, z * biome_region_scale + 3400.0f);

    bw.temp   = local_temperature(x, z, style);
    bw.humid  = local_humidity(x, z, style);
    bw.rugged = local_ruggedness(x, z, style);

    bw.desert   = clampf(1.0f - fabsf(region - 0.15f) / 0.22f, 0.0f, 1.0f);
    bw.valley   = clampf(1.0f - fabsf(region - 0.50f) / 0.20f, 0.0f, 1.0f);
    bw.mountain = clampf(1.0f - fabsf(region - 0.82f) / 0.20f, 0.0f, 1.0f);

    bw.desert   *= bw.temp * (1.0f - bw.humid);
    bw.valley   *= bw.humid * (1.0f - bw.rugged * 0.7f);
    bw.mountain *= bw.rugged;

    float sum = bw.desert + bw.valley + bw.mountain + 1e-6f;
    bw.desert   /= sum;
    bw.valley   /= sum;
    bw.mountain /= sum;

    return bw;
}

// =========================
// Procedural terrain shaping
// =========================
__device__ __forceinline__ float terrain_base_height_biomes(float world_x, float world_z, const DeviceBiomeStyle& style){
    float fine  = fractal_noise2(world_x * noise_scale,       world_z * noise_scale);
    float macro = fractal_noise2(world_x * macro_noise_scale, world_z * macro_noise_scale);

    float detail1 = fractal_noise2(world_x * (noise_scale * 2.0f) + 400.0f,
                                   world_z * (noise_scale * 2.0f) + 900.0f);

    float ridged1 = 1.0f - fabsf(fine);
    ridged1 = ridged1 * ridged1;

    float biomeA = 0.5f + 0.5f * fractal_noise2(
        world_x * 0.0012f + 1200.0f,
        world_z * 0.0012f + 3400.0f
    );

    float desert_region   = clampf(1.0f - fabsf(biomeA - 0.15f) / 0.22f, 0.0f, 1.0f);
    float valley_region   = clampf(1.0f - fabsf(biomeA - 0.50f) / 0.20f, 0.0f, 1.0f);
    float mountain_region = clampf(1.0f - fabsf(biomeA - 0.82f) / 0.20f, 0.0f, 1.0f);

    desert_region   *= lerpf(0.7f, 1.3f, style.global_temperature) * lerpf(1.3f, 0.7f, style.global_humidity);
    valley_region   *= lerpf(0.7f, 1.3f, style.global_humidity);
    mountain_region *= lerpf(0.7f, 1.3f, style.global_ruggedness);

    float sum = desert_region + valley_region + mountain_region + 1e-6f;
    desert_region   /= sum;
    valley_region   /= sum;
    mountain_region /= sum;

    float desert_shape =
        macro   * 1.05f +
        fine    * 0.28f +
        detail1 * 0.12f -
        0.45f;

    float valley_shape =
        macro   * 1.55f +
        fine    * 0.75f +
        detail1 * 0.22f -
        0.18f;

    float mountain_shape =
        macro   * 2.00f +
        ridged1 * 2.00f +
        detail1 * 0.14f +
        0.32f;

    float h =
        desert_shape   * desert_region +
        valley_shape   * valley_region +
        mountain_shape * mountain_region;

    h += (style.global_ruggedness - 0.5f) * 0.62f;
    h -= (style.global_humidity  - 0.5f) * 0.18f;

    return h * d_terrain_height_multiplier;
}

__device__ __forceinline__ float terrain_full_height(
    float world_x,
    float world_z,
    const TerrainModification* mods,
    int row,
    int col,
    int near_subdivs,
    const DeviceBiomeStyle& style)
{
    float h = terrain_base_height_biomes(world_x, world_z, style);
    if (mods && row >= 0 && col >= 0) {
        int midx = row * near_subdivs + col;
        h += mods[midx].value;
    }
    return h;
}

__device__ __forceinline__ float cloud_density_at(float x, float z, float cloud_offset_z, const DeviceBiomeStyle& style){
    float nx = x * cloud_noise_scale + 100.f;
    float nz = (z + cloud_offset_z) * cloud_noise_scale + 200.f;

    float base = fractal_noise2(nx, nz);
    float detail1 = fractal_noise2(nx * 2.0f + 71.0f, nz * 2.0f + 29.0f);

    float puffy = 1.0f - fabsf(base);
    puffy = puffy * puffy;

    float humid = local_humidity(x, z, style);

    float dense =
        base    * 0.78f +
        detail1 * 0.18f +
        puffy   * 0.20f;

    dense = lerpf(dense - 0.10f, dense + 0.24f, 0.35f * humid + 0.65f * style.cloudiness);
    return dense;
}

__device__ __forceinline__ float cloud_shadow_factor(
    float x, float terrain_y, float z,
    float3 sun_dir,
    float cloud_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    const DeviceBiomeStyle& style)
{
    float up_y = -sun_dir.y;
    if (up_y <= 1e-5f) return 1.0f;

    float t = (cloud_height - terrain_y) / up_y;
    if (t <= 0.0f) return 1.0f;

    float xc = x + (-sun_dir.x) * t;
    float zc = z + (-sun_dir.z) * t;

    float cloud = cloud_density_at(xc, zc, cloud_offset_z, style);
    float alpha = clampf((cloud - cloud_threshold) / cloud_shadow_softness, 0.0f, 1.0f);

    float strength  = clampf(cloud_shadow_strength_base * shadow_darkness_mult, 0.0f, 1.20f);
    float min_light = clampf(cloud_shadow_min_light_base / fmaxf(shadow_darkness_mult, 0.05f), 0.02f, 1.0f);

    float shadow = 1.0f - strength * alpha;
    return clampf(shadow, min_light, 1.0f);
}

__device__ __forceinline__ float3 terrain_normal(
    float world_x,
    float world_z,
    const TerrainModification* mods,
    int row,
    int col,
    int near_subdivs,
    float eps,
    const DeviceBiomeStyle& style)
{
    float hL = terrain_full_height(world_x - eps, world_z,       mods, row, col, near_subdivs, style);
    float hR = terrain_full_height(world_x + eps, world_z,       mods, row, col, near_subdivs, style);
    float hD = terrain_full_height(world_x,       world_z - eps, mods, row, col, near_subdivs, style);
    float hU = terrain_full_height(world_x,       world_z + eps, mods, row, col, near_subdivs, style);

    float3 dx = make_float3(2.0f * eps, hR - hL, 0.0f);
    float3 dz = make_float3(0.0f, hU - hD, 2.0f * eps);
    float3 n  = cross3(dz, dx);
    return normalize3(n);
}

__device__ __forceinline__ float snow_amount(float x, float z, float h, float3 normal, const DeviceBiomeStyle& style){
    float t = clampf(
        0.5f + 0.5f * fractal_noise2(x * biome_temp_scale + 130.0f, z * biome_temp_scale + 210.0f),
        0.0f, 1.0f
    );
    float local_temp = lerpf(t, style.global_temperature, 0.40f);
    float coldness = 1.0f - local_temp;

    float snowline = lerpf(4.0f, 1.0f, 1.0f - style.global_snowline);
    float alt = clampf((h - snowline) / 2.5f, 0.0f, 1.0f);
    float flatness = clampf(normal.y, 0.0f, 1.0f);

    float snow = coldness * alt * lerpf(0.45f, 1.0f, flatness);
    return clampf(snow, 0.0f, 1.0f);
}

// =========================
// Biome-aware sky / terrain color
// =========================
__device__ __forceinline__ void sky_color_at_dir(float3 dir, float depth_hint, const DeviceBiomeStyle& style, float& b, float& g, float& r){
    float up = clampf(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);

    float cold_b1 = 255.0f, cold_g1 = 228.0f, cold_r1 = 205.0f;
    float cold_b2 = 255.0f, cold_g2 = 205.0f, cold_r2 = 175.0f;

    float warm_b1 = 255.0f, warm_g1 = 215.0f, warm_r1 = 168.0f;
    float warm_b2 = 255.0f, warm_g2 = 182.0f, warm_r2 = 135.0f;

    float cold_b = lerpf(cold_b2, cold_b1, up);
    float cold_g = lerpf(cold_g2, cold_g1, up);
    float cold_r = lerpf(cold_r2, cold_r1, up);

    float warm_b = lerpf(warm_b2, warm_b1, up);
    float warm_g = lerpf(warm_g2, warm_g1, up);
    float warm_r = lerpf(warm_r2, warm_r1, up);

    b = lerpf(cold_b, warm_b, style.global_temperature);
    g = lerpf(cold_g, warm_g, style.global_temperature);
    r = lerpf(cold_r, warm_r, style.global_temperature);

    fog_mix(b, g, r, depth_hint, true);
}

__device__ __forceinline__ void biome_base_color(
    float x, float z, float h, float3 n, bool is_water,
    const DeviceBiomeStyle& style,
    float& b, float& g, float& r)
{
    if (is_water) {
        float depth_like = clampf((-h) / 2.5f, 0.0f, 1.0f);

        float shallow_b = 210.f, shallow_g = 170.f, shallow_r =  90.f;
        float deep_b    = 140.f, deep_g    =  80.f, deep_r    =  25.f;

        b = lerpf(shallow_b, deep_b, depth_like);
        g = lerpf(shallow_g, deep_g, depth_like);
        r = lerpf(shallow_r, deep_r, depth_like);
        return;
    }

    BiomeWeights bw = biome_weights_at(x, z, h, style);

    float flatness  = clampf(n.y, 0.0f, 1.0f);
    float rockiness = clampf(1.0f - flatness, 0.0f, 1.0f);

    float shore_t = smooth01((h - 0.00f) / 0.35f);
    float rock_t  = smooth01((h - 2.20f) / 1.90f);

    float sand_b = 150.f, sand_g = 205.f, sand_r = 238.f;
    float grass_b =  58.f, grass_g = 138.f, grass_r =  46.f;
    float forest_b =  50.f, forest_g = 110.f, forest_r =  38.f;
    float rock_b = 122.f, rock_g = 122.f, rock_r = 126.f;

    float humid = bw.humid;
    float dry   = 1.0f - humid;

    sand_b = lerpf(sand_b, 135.f, bw.desert * 0.55f);
    sand_g = lerpf(sand_g, 195.f, bw.desert * 0.55f);
    sand_r = lerpf(sand_r, 230.f, bw.desert * 0.55f);

    sand_b = lerpf(sand_b, 120.f, humid * 0.20f);
    sand_g = lerpf(sand_g, 180.f, humid * 0.20f);
    sand_r = lerpf(sand_r, 205.f, humid * 0.20f);

    grass_b = lerpf(grass_b,  48.f, humid * 0.50f);
    grass_g = lerpf(grass_g, 150.f, humid * 0.50f);
    grass_r = lerpf(grass_r,  40.f, humid * 0.50f);

    forest_b = lerpf(forest_b,  40.f, humid * 0.50f);
    forest_g = lerpf(forest_g, 125.f, humid * 0.50f);
    forest_r = lerpf(forest_r,  32.f, humid * 0.50f);

    grass_b = lerpf(grass_b,  70.f, dry * 0.20f);
    grass_g = lerpf(grass_g, 145.f, dry * 0.20f);
    grass_r = lerpf(grass_r,  70.f, dry * 0.20f);

    rock_b = lerpf(rock_b, 135.f, bw.mountain * 0.35f + rockiness * 0.25f);
    rock_g = lerpf(rock_g, 135.f, bw.mountain * 0.35f + rockiness * 0.25f);
    rock_r = lerpf(rock_r, 138.f, bw.mountain * 0.35f + rockiness * 0.25f);

    float low_b = lerpf(sand_b, grass_b, shore_t);
    float low_g = lerpf(sand_g, grass_g, shore_t);
    float low_r = lerpf(sand_r, grass_r, shore_t);

    float forest_t = smooth01((h - 0.70f) / 1.30f);
    low_b = lerpf(low_b, forest_b, forest_t * 0.55f);
    low_g = lerpf(low_g, forest_g, forest_t * 0.55f);
    low_r = lerpf(low_r, forest_r, forest_t * 0.55f);

    b = lerpf(low_b, rock_b, rock_t);
    g = lerpf(low_g, rock_g, rock_t);
    r = lerpf(low_r, rock_r, rock_t);

    float exposed_rock = rockiness * smooth01((h - 1.6f) / 2.0f) * 0.55f;
    b = lerpf(b, rock_b, exposed_rock);
    g = lerpf(g, rock_g, exposed_rock);
    r = lerpf(r, rock_r, exposed_rock);

    float warm_t = style.global_temperature;
    b = lerpf(b, b - 6.f, warm_t * 0.18f);
    g = lerpf(g, g + 4.f, warm_t * 0.10f);
    r = lerpf(r, r + 8.f, warm_t * 0.18f);

    float snow = snow_amount(x, z, h, n, style);

    float high_peak_whiten = smooth01((h - 4.2f) / 1.7f) * (0.25f + 0.45f * bw.mountain);
    snow = clampf(snow + high_peak_whiten * (1.0f - snow), 0.0f, 1.0f);

    b = lerpf(b, 245.f, snow);
    g = lerpf(g, 245.f, snow);
    r = lerpf(r, 245.f, snow);
}

__device__ __forceinline__ void terrain_color_lit(
    float world_x,
    float world_z,
    float terrain_h,
    bool is_water,
    const TerrainModification* mods,
    float3 sun_dir,
    float cloud_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    int near_subdivs,
    const DeviceBiomeStyle& style,
    float& b, float& g, float& r)
{
    float3 n = terrain_normal(world_x, world_z, mods, -1, -1, near_subdivs, normal_eps, style);

    biome_base_color(world_x, world_z, terrain_h, n, is_water, style, b, g, r);

    if (!is_water) {
        float ndotl = clampf(dot3(n, mul3(sun_dir, -1.0f)), 0.0f, 1.0f);
        float slope_backlight = clampf(1.0f - n.y, 0.0f, 1.0f);

        float cloud_shadow = cloud_shadow_factor(
            world_x, terrain_h > 0.0f ? terrain_h : 0.0f, world_z,
            sun_dir, cloud_offset_z, cloud_threshold, shadow_darkness_mult, style
        );

        float lighting =
            ambient_light +
            diffuse_strength * ndotl * cloud_shadow +
            backlight_strength * slope_backlight * 0.35f;

        lighting = clampf(lighting, 0.04f, 1.35f);

        b *= lighting;
        g *= lighting;
        r *= lighting;

        if (cloud_shadow < 0.75f) {
            float shadow_t = clampf((0.75f - cloud_shadow) / 0.75f, 0.0f, 1.0f);
            b *= (1.00f + 0.04f * shadow_t);
            g *= (1.00f - 0.05f * shadow_t);
            r *= (1.00f - 0.10f * shadow_t);
        }
    }
}

// =========================
// Water reflection tracing
// =========================
__device__ __forceinline__ float reflection_step_size(float t){
    float nt = clampf(t / water_max_reflection_dist, 0.0f, 1.0f);
    return lerpf(water_reflection_step_min, water_reflection_step_max, nt);
}

__device__ __forceinline__ void water_reflection_trace(
    float local_x,
    float local_z,
    float world_x,
    float world_z,
    float depth_to_water,
    const TerrainModification* mods,
    int near_subdivs,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float reflection_strength,
    float3 sun_dir,
    float3 camPos,
    const DeviceBiomeStyle& style,
    float& out_b,
    float& out_g,
    float& out_r)
{
    float wave_x = fractal_noise2(
        world_x * water_wave_noise_scale + 401.0f,
        world_z * water_wave_noise_scale + 97.0f
    );

    float wave_z = fractal_noise2(
        world_x * water_wave_noise_scale + 211.0f,
        world_z * water_wave_noise_scale + 501.0f
    );

    float3 water_n = normalize3(make_float3(
        wave_x * water_wave_distort,
        1.0f,
        wave_z * water_wave_distort
    ));

    float3 p = make_float3(local_x, 0.0f, local_z);
    float3 v_to_eye = normalize3(sub3(camPos, p));
    float3 incident = mul3(v_to_eye, -1.0f);
    float3 refl_dir = normalize3(reflect3(incident, water_n));

    if (refl_dir.y < 0.01f) refl_dir.y = 0.01f;
    refl_dir = normalize3(refl_dir);

    float ndotv = clampf(dot3(water_n, v_to_eye), 0.0f, 1.0f);
    float fresnel = water_fresnel_bias + (1.0f - water_fresnel_bias) * powf(1.0f - ndotv, water_fresnel_power);
    fresnel = clampf(fresnel, 0.0f, 1.0f);

    float accum_cloud_alpha = 0.0f;
    float cloud_b = 0.0f, cloud_g = 0.0f, cloud_r = 0.0f;

    bool hit_terrain = false;
    float hit_b = 0.0f, hit_g = 0.0f, hit_r = 0.0f;
    float traced_dist = 0.0f;

    float3 pos = add3(p, mul3(refl_dir, 0.35f));

    #pragma unroll 1
    for (int s = 0; s < water_reflection_steps; ++s) {
        float step_len = reflection_step_size(traced_dist);
        pos = add3(pos, mul3(refl_dir, step_len));
        traced_dist += step_len;

        if (traced_dist > water_max_reflection_dist) break;

        float sample_world_x = pos.x + terrain_offset_x;
        float sample_world_z = pos.z + terrain_offset_z;
        float terrain_h = terrain_full_height(sample_world_x, sample_world_z, mods, -1, -1, near_subdivs, style);

        if (fabsf(pos.y - cloud_height) < 2.5f) {
            float c = cloud_density_at(pos.x, pos.z, cloud_offset_z, style);
            float ca = clampf((c - cloud_threshold) / 0.25f, 0.0f, 1.0f);
            if (ca > 0.001f) {
                float cb = lerpf(215.f, 242.f, ca);
                float cg = lerpf(220.f, 244.f, ca);
                float cr = lerpf(240.f, 255.f, ca);

                float remain = 1.0f - accum_cloud_alpha;
                cloud_b += cb * ca * remain;
                cloud_g += cg * ca * remain;
                cloud_r += cr * ca * remain;
                accum_cloud_alpha += ca * 0.55f * remain;
                accum_cloud_alpha = clampf(accum_cloud_alpha, 0.0f, 1.0f);
            }
        }

        if (terrain_h > 0.0f) {
            float diff = pos.y - terrain_h;
            if (diff <= 0.0f) {
                float3 hit_pos = sub3(pos, mul3(refl_dir, step_len * 0.5f));

                float hit_world_x = hit_pos.x + terrain_offset_x;
                float hit_world_z = hit_pos.z + terrain_offset_z;
                float hit_h = terrain_full_height(hit_world_x, hit_world_z, mods, -1, -1, near_subdivs, style);

                terrain_color_lit(
                    hit_world_x, hit_world_z, hit_h, false,
                    mods, sun_dir, cloud_offset_z, cloud_threshold,
                    shadow_darkness_mult, near_subdivs, style,
                    hit_b, hit_g, hit_r
                );

                fog_mix(hit_b, hit_g, hit_r, depth_to_water + traced_dist, false);
                hit_terrain = true;
                break;
            }
        }
    }

    float sky_b, sky_g, sky_r;
    sky_color_at_dir(refl_dir, depth_to_water + traced_dist, style, sky_b, sky_g, sky_r);

    float rb = sky_b * water_reflection_sky_gain;
    float rg = sky_g * water_reflection_sky_gain;
    float rr = sky_r * water_reflection_sky_gain;

    if (accum_cloud_alpha > 0.001f) {
        rb = lerpf(rb, cloud_b, clampf(accum_cloud_alpha * water_reflection_cloud_gain, 0.0f, 1.0f));
        rg = lerpf(rg, cloud_g, clampf(accum_cloud_alpha * water_reflection_cloud_gain, 0.0f, 1.0f));
        rr = lerpf(rr, cloud_r, clampf(accum_cloud_alpha * water_reflection_cloud_gain, 0.0f, 1.0f));
    }

    if (hit_terrain) {
        rb = lerpf(rb, hit_b, water_reflection_terrain_gain);
        rg = lerpf(rg, hit_g, water_reflection_terrain_gain);
        rr = lerpf(rr, hit_r, water_reflection_terrain_gain);
    }

    float sun_glint = powf(clampf(dot3(v_to_eye, mul3(sun_dir, -1.0f)), 0.0f, 1.0f), 28.0f);
    rb += 255.0f * water_specular_strength * sun_glint;
    rg += 245.0f * water_specular_strength * sun_glint;
    rr += 230.0f * water_specular_strength * sun_glint;

    float final_mix = clampf(reflection_strength * fresnel, 0.0f, 1.0f);

    out_b = clampf(rb * final_mix, 0.0f, 255.0f);
    out_g = clampf(rg * final_mix, 0.0f, 255.0f);
    out_r = clampf(rr * final_mix, 0.0f, 255.0f);
}

// =========================
// Packed framebuffer
// =========================
__device__ __forceinline__ unsigned long long pack_depth_color(float depth, unsigned char b, unsigned char g, unsigned char r){
    float d = clampf(depth, 0.f, 10000.f);
    unsigned int depth_q = (unsigned int)(d * 1024.0f);
    return ((unsigned long long)depth_q << 24) |
           ((unsigned long long)b << 16) |
           ((unsigned long long)g << 8)  |
           ((unsigned long long)r);
}

__device__ __forceinline__ void splat_disk(
    unsigned long long* packed_fb,
    int u, int v, int radius,
    float depth, unsigned char b, unsigned char g, unsigned char r)
{
    unsigned long long packed = pack_depth_color(depth, b, g, r);

    int x0 = max(0, u - radius);
    int x1 = min(width - 1, u + radius);
    int y0 = max(0, v - radius);
    int y1 = min(height - 1, v + radius);

    int rr = radius * radius;

    for (int py = y0; py <= y1; ++py) {
        int dy = py - v;
        int dy2 = dy * dy;
        for (int px = x0; px <= x1; ++px) {
            int dx = px - u;
            if (dx * dx + dy2 > rr) continue;
            atomicMin((unsigned long long*)&packed_fb[py * width + px], packed);
        }
    }
}

// =========================
// GPU kernels
// =========================
__global__ void k_clear_packed_fb(unsigned long long* fb, unsigned long long clear_value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx < n) fb[idx] = clear_value;
}

__global__ void k_unpack_to_bgr(const unsigned long long* packed_fb, unsigned char* out_bgr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    unsigned long long v = packed_fb[idx];

    unsigned char b = (unsigned char)((v >> 16) & 0xFFull);
    unsigned char g = (unsigned char)((v >> 8)  & 0xFFull);
    unsigned char r = (unsigned char)(v & 0xFFull);

    int o = idx * 3;
    out_bgr[o + 0] = b;
    out_bgr[o + 1] = g;
    out_bgr[o + 2] = r;
}

__global__ void k_decay_mods(TerrainModification* mods, int mods_count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mods_count) return;

    float v = mods[idx].value * mods[idx].persistence;
    if (fabsf(v) < 0.01f) v = 0.f;
    mods[idx].value = v;
}

__global__ void k_render_terrain(
    unsigned long long* packed_fb,
    const TerrainModification* mods,
    int terrain_rows_runtime,
    int near_subdivs,
    float terrain_offset_x,
    float terrain_offset_z,
    float radius_mult,
    float cloud_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float reflection_strength,
    float3 sun_dir,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= terrain_rows_runtime || i >= near_subdivs) return;

    float z = sample_depth_inverse_perspective(
        j, terrain_rows_runtime, terrain_sample_near_z, plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(plane_width, visible_w * terrain_visible_overscan);

    float x = sample_x_for_column(i, near_subdivs, row_world_width);
    x += row_stagger_offset(j, near_subdivs, row_world_width);

    x += rand_signed_from_2i(i, j, 0x1234u) * terrain_jitter_x;
    z += rand_signed_from_2i(i, j, 0x5678u) * terrain_jitter_z;

    float world_x = x + terrain_offset_x;
    float world_z = z + terrain_offset_z;

    float theoretical_y = terrain_full_height(world_x, world_z, mods, j, i, near_subdivs, style);

    float actual_y = (theoretical_y > 0.f) ? theoretical_y : 0.f;
    bool is_water = (theoretical_y < 0.f);

    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(x, actual_y, z), u, v, depth, camPos, R)) return;

    float b, g, r;

    if (!is_water) {
        terrain_color_lit(
            world_x, world_z, theoretical_y, false,
            mods, sun_dir, cloud_offset_z, cloud_threshold,
            shadow_darkness_mult, near_subdivs, style,
            b, g, r
        );
    } else {
        biome_base_color(world_x, world_z, theoretical_y, make_float3(0.f, 1.f, 0.f), true, style, b, g, r);

        float cloud_shadow = cloud_shadow_factor(
            world_x, actual_y, world_z, sun_dir, cloud_offset_z, cloud_threshold, shadow_darkness_mult, style
        );

        float water_light = 0.88f * lerpf(0.72f, 1.0f, cloud_shadow);
        b *= water_light;
        g *= water_light;
        r *= water_light;

        float rb, rg, rr;
        water_reflection_trace(
            x, z, world_x, world_z, depth,
            mods, near_subdivs,
            terrain_offset_x, terrain_offset_z,
            cloud_offset_z, cloud_threshold,
            shadow_darkness_mult,
            reflection_strength,
            sun_dir, camPos,
            style,
            rb, rg, rr
        );

        b = clampf(b + rb, 0.0f, 255.0f);
        g = clampf(g + rg, 0.0f, 255.0f);
        r = clampf(r + rr, 0.0f, 255.0f);

        b *= 1.02f;
        g *= 0.98f;
        r *= 0.92f;
    }

    fog_mix(b, g, r, depth, false);

    int rad = compute_radius_gpu(depth, radius_mult * terrain_radius_scale);

    splat_disk(
        packed_fb, u, v, rad, depth,
        (unsigned char)clampf(b, 0.f, 255.f),
        (unsigned char)clampf(g, 0.f, 255.f),
        (unsigned char)clampf(r, 0.f, 255.f)
    );
}

__global__ void k_render_sky(
    unsigned long long* packed_fb,
    int sky_rows_runtime,
    int near_subdivs,
    float radius_mult,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= sky_rows_runtime || i >= near_subdivs) return;

    float z = sample_depth_inverse_perspective(
        j, sky_rows_runtime, sky_sample_near_z, sky_plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(sky_plane_width, visible_w * sky_visible_overscan);

    float cell_w = cell_width_for_cols(near_subdivs, row_world_width);
    float band_d = depth_band_size_inverse_perspective(j, sky_rows_runtime, sky_sample_near_z, sky_plane_depth);

    float x = sample_x_for_column(i, near_subdivs, row_world_width);
    x += row_stagger_offset(j, near_subdivs, row_world_width);

    x += rand_signed_from_2i(i, j, 0x1111u) * cell_w * sky_jitter_x_mul;
    z += rand_signed_from_2i(i, j, 0x2222u) * band_d * sky_jitter_z_mul;
    z += rand_signed_from_2i(j, 17, 0xABCDu) * band_d * sky_row_warp_mul;

    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(x, sky_height, z), u, v, depth, camPos, R)) return;

    float3 dir = normalize3(make_float3(x - camPos.x, sky_height - camPos.y, z - camPos.z));

    float b, g, r;
    sky_color_at_dir(dir, depth, style, b, g, r);

    int rad = compute_radius_gpu(depth, radius_mult * sky_radius_scale);

    splat_disk(
        packed_fb, u, v, rad, depth,
        (unsigned char)clampf(b, 0.f, 255.f),
        (unsigned char)clampf(g, 0.f, 255.f),
        (unsigned char)clampf(r, 0.f, 255.f)
    );
}

__global__ void k_render_clouds(
    unsigned long long* packed_fb,
    int cloud_rows_runtime,
    int near_subdivs,
    float cloud_offset_z,
    float radius_mult,
    float cloud_threshold,
    float3 sun_dir,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cloud_rows_runtime || i >= near_subdivs) return;

    float z = sample_depth_inverse_perspective(
        j, cloud_rows_runtime, terrain_sample_near_z, plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(plane_width, visible_w * cloud_visible_overscan);

    float x = sample_x_for_column(i, near_subdivs, row_world_width);
    x += row_stagger_offset(j, near_subdivs, row_world_width);

    x += rand_signed_from_2i(i, j, 0x3333u) * cloud_jitter_x;
    z += rand_signed_from_2i(i, j, 0x4444u) * cloud_jitter_z;

    float cloud = cloud_density_at(x, z, cloud_offset_z, style);
    if (cloud < cloud_threshold) return;

    float y = cloud_height + (1.f - cloud) * cloud_noise_amplitude;

    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(x, y, z), u, v, depth, camPos, R)) return;

    float cloud_alpha = clampf((cloud - cloud_threshold) / 0.35f, 0.0f, 1.0f);
    float sun_side = clampf(0.6f + 0.4f * (-sun_dir.x * 0.6f + (-sun_dir.y) * 0.4f), 0.0f, 1.0f);

    float warm = style.global_temperature;
    float b = lerpf(220.f, 245.f, cloud_alpha);
    float g = lerpf(224.f, 245.f, cloud_alpha);
    float r = lerpf(238.f, 255.f, cloud_alpha);

    g = lerpf(g, g - 8.0f, 1.0f - warm);
    r = lerpf(r, r + 6.0f, warm);

    b *= (0.96f + 0.08f * sun_side);
    g *= (0.96f + 0.08f * sun_side);
    r *= (0.96f + 0.10f * sun_side);

    fog_mix(b, g, r, depth, false);

    int rad = compute_radius_gpu(depth, radius_mult * cloud_radius_scale);

    splat_disk(
        packed_fb, u, v, rad, depth,
        (unsigned char)clampf(b, 0.f, 255.f),
        (unsigned char)clampf(g, 0.f, 255.f),
        (unsigned char)clampf(r, 0.f, 255.f)
    );
}

// =========================
// Host rotation matrix + conversion
// =========================
static cv::Matx33f get_rotation_matrix_host(const cv::Vec3f& angles) {
    float cx = std::cos(angles[0]); float sx = std::sin(angles[0]);
    float cy = std::cos(angles[1]); float sy = std::sin(angles[1]);
    float cz = std::cos(angles[2]); float sz = std::sin(angles[2]);

    cv::Matx33f Rx(1,0,0,  0,cx,-sx,  0,sx,cx);
    cv::Matx33f Ry(cy,0,sy,  0,1,0,  -sy,0,cy);
    cv::Matx33f Rz(cz,-sz,0,  sz,cz,0,  0,0,1);
    return Rz * Ry * Rx;
}

static Mat3 to_mat3_rowmajor(const cv::Matx33f& R){
    Mat3 m{};
    m.m[0]=R(0,0); m.m[1]=R(0,1); m.m[2]=R(0,2);
    m.m[3]=R(1,0); m.m[4]=R(1,1); m.m[5]=R(1,2);
    m.m[6]=R(2,0); m.m[7]=R(2,1); m.m[8]=R(2,2);
    return m;
}

// =========================
// Permutation init -> GPU constant memory
// =========================
static unsigned int init_permutation_cuda() {
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);

    std::random_device rd;
    unsigned int seed =
        ((unsigned int)rd()) ^
        ((unsigned int)std::time(nullptr)) ^
        ((unsigned int)clock());

    std::mt19937 eng(seed);
    std::shuffle(p.begin(), p.end(), eng);

    int perm512[512];
    for (int i = 0; i < 512; i++) perm512[i] = p[i & 255];

    CUDA_CHECK(cudaMemcpyToSymbol(d_perm, perm512, sizeof(perm512)));
    return seed;
}

// =========================
// FFmpeg / NVENC pipe
// =========================
static std::string shell_escape_single_quotes(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

static FILE* open_ffmpeg_nvenc_pipe(const std::string& output_path) {
    const std::string safe_out = shell_escape_single_quotes(output_path);

    std::ostringstream cmd;
    cmd
        << "ffmpeg -y "
        << "-f rawvideo "
        << "-pix_fmt bgr24 "
        << "-s " << width << "x" << height << " "
        << "-r " << output_fps << " "
        << "-i - "
        << "-an "
        << "-c:v h264_nvenc "
        << "-preset p7 "
        << "-tune hq "
        << "-rc vbr "
        << "-cq 19 "
        << "-b:v 0 "
        << "-pix_fmt yuv420p "
        << "-movflags +faststart "
        << "'" << safe_out << "'";

    std::cout << "FFmpeg command:\n" << cmd.str() << "\n";

    FILE* pipe = popen(cmd.str().c_str(), "w");
    if (!pipe) {
        throw std::runtime_error("Could not open FFmpeg pipe.");
    }
    return pipe;
}

static void write_frame_to_pipe(FILE* pipe, const cv::Mat& frame) {
    if (!pipe) throw std::runtime_error("FFmpeg pipe is null.");
    if (frame.empty()) throw std::runtime_error("Attempted to write empty frame.");
    if (frame.type() != CV_8UC3) throw std::runtime_error("Frame must be CV_8UC3.");
    if (frame.cols != width || frame.rows != height) throw std::runtime_error("Frame size mismatch.");

    const size_t row_bytes = (size_t)width * 3;
    const size_t total_bytes = row_bytes * (size_t)height;

    if (frame.isContinuous()) {
        size_t written = fwrite(frame.data, 1, total_bytes, pipe);
        if (written != total_bytes) throw std::runtime_error("Short write to FFmpeg pipe.");
    } else {
        for (int y = 0; y < frame.rows; ++y) {
            const unsigned char* row = frame.ptr<unsigned char>(y);
            size_t written = fwrite(row, 1, row_bytes, pipe);
            if (written != row_bytes) throw std::runtime_error("Short row write to FFmpeg pipe.");
        }
    }
}

// =========================
// Global buffers
// =========================
static std::vector<TerrainModification> h_mods;
static float g_terrain_offset_x = 0.f;
static float g_terrain_offset_z = 0.f;

// =========================
// Main
// =========================
int main(int argc, char** argv){
    unsigned int noise_seed = init_permutation_cuda();
    std::cout << "Noise seed: " << noise_seed << "\n";

    int duration_seconds = 36000;
    std::string output_file = make_default_output_filename();

    if (argc >= 2) duration_seconds = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) output_file = argv[2];
    if (argc >= 4) forward_speed = std::max(0.0f, static_cast<float>(std::atof(argv[3])));
    if (argc >= 5) radius_multiplier = std::max(0.1f, static_cast<float>(std::atof(argv[4])));
    if (argc >= 6) near_subdivisions = std::max(32, std::atoi(argv[5]));
    if (argc >= 7) shadow_darkness_multiplier = std::max(0.05f, static_cast<float>(std::atof(argv[6])));
    if (argc >= 8) water_reflection_strength = clamp_host(static_cast<float>(std::atof(argv[7])), 0.0f, 1.0f);
    if (argc >= 9) terrain_height_multiplier = std::max(0.0f, static_cast<float>(std::atof(argv[8])));

    CUDA_CHECK(cudaMemcpyToSymbol(d_terrain_height_multiplier, &terrain_height_multiplier, sizeof(float)));

    camera_position = cv::Vec3f(0.0f, 3.0f * terrain_height_multiplier, 5.0f);

    const long long total_frames = (long long)duration_seconds * output_fps;

    const int terrain_rows_runtime = compute_inverse_perspective_row_count(
        terrain_sample_near_z, plane_depth, near_subdivisions
    );
    const int cloud_rows_runtime = terrain_rows_runtime;
    const int sky_rows_runtime = compute_inverse_perspective_row_count(
        sky_sample_near_z, sky_plane_depth, near_subdivisions
    );

    const int terrain_mods_count = terrain_rows_runtime * near_subdivisions;
    h_mods.assign(terrain_mods_count, TerrainModification{0.0f, 0.99f});

    float az = sun_azimuth_deg * 3.14159265358979323846f / 180.0f;
    float el = sun_elevation_deg * 3.14159265358979323846f / 180.0f;

    float3 sun_dir = make_float3(
        cosf(el) * sinf(az),
        -sinf(el),
        cosf(el) * cosf(az)
    );
    sun_dir = normalize3_host(sun_dir);

    std::mt19937 rng(
        ((unsigned int)std::time(nullptr)) ^
        ((unsigned int)clock()) ^
        noise_seed ^
        0xA53219u
    );

    BiomeStyle current_style = random_biome_style(rng);
    BiomeStyle next_style    = random_biome_style(rng);

    std::cout << "Output: " << output_file << "\n";
    std::cout << "Duration: " << duration_seconds << " s\n";
    std::cout << "FPS: " << output_fps << "\n";
    std::cout << "Frames: " << total_frames << "\n";
    std::cout << "Forward speed: " << forward_speed << "\n";
    std::cout << "Radius multiplier: " << radius_multiplier << "\n";
    std::cout << "Near subdivisions: " << near_subdivisions << "\n";
    std::cout << "Shadow darkness multiplier: " << shadow_darkness_multiplier << "\n";
    std::cout << "Water reflection strength: " << water_reflection_strength << "\n";
    std::cout << "Terrain height multiplier: " << terrain_height_multiplier << "\n";
    std::cout << "Terrain rows: " << terrain_rows_runtime << "\n";
    std::cout << "Cloud rows: " << cloud_rows_runtime << "\n";
    std::cout << "Sky rows: " << sky_rows_runtime << "\n";
    std::cout << "Sun dir: (" << sun_dir.x << ", " << sun_dir.y << ", " << sun_dir.z << ")\n";
    std::cout << "Biome slot: " << biome_slot_seconds << " seconds (" << biome_hold_seconds
              << " hold + " << biome_transition_seconds << " transition)\n";
    std::cout << "Preview window interval: " << preview_every_seconds << " seconds\n";

    TerrainModification* d_mods = nullptr;
    unsigned long long* d_packed_fb = nullptr;
    unsigned char* d_bgr = nullptr;

    CUDA_CHECK(cudaMalloc(&d_mods, h_mods.size() * sizeof(TerrainModification)));
    CUDA_CHECK(cudaMemcpy(d_mods, h_mods.data(), h_mods.size() * sizeof(TerrainModification), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_packed_fb, (size_t)width * height * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_bgr, (size_t)width * height * 3));

    cv::Mat frame(height, width, CV_8UC3);

    float cloud_offset_z = 0.f;
    float texture_rotation_angle = 0.f;

    FILE* ffmpeg = nullptr;

    try {
        ffmpeg = open_ffmpeg_nvenc_pipe(output_file);

        dim3 block2d(16, 16);

        dim3 gridTerrain(
            (near_subdivisions + block2d.x - 1) / block2d.x,
            (terrain_rows_runtime + block2d.y - 1) / block2d.y
        );

        dim3 gridClouds(
            (near_subdivisions + block2d.x - 1) / block2d.x,
            (cloud_rows_runtime + block2d.y - 1) / block2d.y
        );

        dim3 gridSky(
            (near_subdivisions + block2d.x - 1) / block2d.x,
            (sky_rows_runtime + block2d.y - 1) / block2d.y
        );

        int pixels = width * height;
        int block1d = 256;
        int gridPixels = (pixels + block1d - 1) / block1d;
        int gridMods = (terrain_mods_count + block1d - 1) / block1d;

        unsigned long long clear_value =
            ((unsigned long long)0xFFFFFFFFu << 24) |
            ((unsigned long long)(unsigned char)window_bg_color[0] << 16) |
            ((unsigned long long)(unsigned char)window_bg_color[1] << 8)  |
            ((unsigned long long)(unsigned char)window_bg_color[2]);

        const long long slot_frames       = (long long)biome_slot_seconds * output_fps;
        const long long hold_frames       = (long long)biome_hold_seconds * output_fps;
        const long long transition_frames = (long long)biome_transition_seconds * output_fps;
        const long long preview_interval_frames = (long long)preview_every_seconds * output_fps;

        for (long long frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
            if (frame_idx > 0 && (frame_idx % slot_frames) == 0) {
                current_style = next_style;
                next_style = random_biome_style(rng);
                std::cout << "\nBiome boundary at t=" << (frame_idx / output_fps) << "s\n";
            }

            long long in_slot = frame_idx % slot_frames;
            float biome_t = 0.0f;
            if (in_slot >= hold_frames) {
                long long trans_pos = in_slot - hold_frames;
                biome_t = (transition_frames > 0)
                    ? clamp_host((float)trans_pos / (float)transition_frames, 0.0f, 1.0f)
                    : 1.0f;
            }

            BiomeStyle blended_style = biome_lerp(current_style, next_style, biome_t);

            DeviceBiomeStyle style{
                blended_style.global_temperature,
                blended_style.global_humidity,
                blended_style.global_ruggedness,
                blended_style.global_snowline,
                blended_style.cloudiness
            };

            g_terrain_offset_z -= forward_speed;
            cloud_offset_z     -= forward_speed * 0.65f;

            k_decay_mods<<<gridMods, block1d>>>(d_mods, terrain_mods_count);
            CUDA_CHECK(cudaGetLastError());

            k_clear_packed_fb<<<gridPixels, block1d>>>(d_packed_fb, clear_value);
            CUDA_CHECK(cudaGetLastError());

            float cos_rot = std::cos(texture_rotation_angle);
            float sin_rot = std::sin(texture_rotation_angle);

            float rotated_x = g_terrain_offset_x * cos_rot - g_terrain_offset_z * sin_rot;
            float rotated_z = g_terrain_offset_x * sin_rot + g_terrain_offset_z * cos_rot;

            cv::Matx33f Rhost = get_rotation_matrix_host(camera_rotation);
            Mat3 R = to_mat3_rowmajor(Rhost);
            float3 camPos = make_float3(camera_position[0], camera_position[1], camera_position[2]);

            float dynamic_cloud_threshold =
                clamp_host(
                    cloud_threshold_min + lerp_host(0.10f, -0.05f, blended_style.cloudiness),
                    0.05f, 0.30f
                );

            k_render_sky<<<gridSky, block2d>>>(
                d_packed_fb,
                sky_rows_runtime,
                near_subdivisions,
                radius_multiplier,
                camPos,
                R,
                style
            );
            CUDA_CHECK(cudaGetLastError());

            k_render_clouds<<<gridClouds, block2d>>>(
                d_packed_fb,
                cloud_rows_runtime,
                near_subdivisions,
                cloud_offset_z,
                radius_multiplier,
                dynamic_cloud_threshold,
                sun_dir,
                camPos,
                R,
                style
            );
            CUDA_CHECK(cudaGetLastError());

            k_render_terrain<<<gridTerrain, block2d>>>(
                d_packed_fb,
                d_mods,
                terrain_rows_runtime,
                near_subdivisions,
                rotated_x,
                rotated_z,
                radius_multiplier,
                cloud_offset_z,
                dynamic_cloud_threshold,
                shadow_darkness_multiplier,
                water_reflection_strength,
                sun_dir,
                camPos,
                R,
                style
            );
            CUDA_CHECK(cudaGetLastError());

            k_unpack_to_bgr<<<gridPixels, block1d>>>(d_packed_fb, d_bgr);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(frame.data, d_bgr, (size_t)width * height * 3, cudaMemcpyDeviceToHost));

            write_frame_to_pipe(ffmpeg, frame);

            if (frame_idx > 0 && (frame_idx % preview_interval_frames) == 0) {
                show_preview_window(frame, frame_idx);
            }

            if ((frame_idx % output_fps) == 0) {
                double pct = 100.0 * (double)frame_idx / (double)total_frames;
                std::cout << "\rFrame " << frame_idx << " / " << total_frames
                          << "  (" << (int)pct << "%)"
                          << " biomeFade=" << std::fixed << std::setprecision(2) << biome_t
                          << std::flush;
            }
        }

        std::cout << "\rFrame " << total_frames << " / " << total_frames << "  (100%)\n";

        int rc = pclose(ffmpeg);
        ffmpeg = nullptr;

        if (rc != 0) {
            std::cerr << "FFmpeg exited with code: " << rc << "\n";
            CUDA_CHECK(cudaFree(d_bgr));
            CUDA_CHECK(cudaFree(d_packed_fb));
            CUDA_CHECK(cudaFree(d_mods));
            cv::destroyAllWindows();
            return 1;
        }

        std::cout << "Video written successfully: " << output_file << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        if (ffmpeg) {
            pclose(ffmpeg);
            ffmpeg = nullptr;
        }
        CUDA_CHECK(cudaFree(d_bgr));
        CUDA_CHECK(cudaFree(d_packed_fb));
        CUDA_CHECK(cudaFree(d_mods));
        cv::destroyAllWindows();
        return 1;
    }

    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_packed_fb));
    CUDA_CHECK(cudaFree(d_mods));
    cv::destroyAllWindows();
    return 0;
}
