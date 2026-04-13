// terrain_raycast_screen_space_volumetric_clouds.cu
// Experimental procedural screen-space raycaster / raymarcher
//
// Improved volumetric-cloud version:
// - Screen-space raycasting
// - One sample every N x N pixels (sample_step parameter)
// - Terrain / water / sky procedural
// - Volumetric integrated clouds with per-pixel jitter
// - Cloud shadows
// - Water reflections
// - Whiter horizon / terrain-sky contact
// - More natural mountains via ridged + warped terrain
// - More terrain detail
// - FIX: terrain geometry no longer morphs during biome transitions
//        (separate stable terrain style from blended visual style)
// - FIX: faster two-tier edge refinement
//        * first pass: 1 sample per block
//        * 4-sample refinement only on likely silhouette / occlusion blocks
// - FEATURE: adjustable deep-water darkening
//        * shallow / near-coast water remains clearer blue
//        * deeper water gets darker according to parameter
// - FIX: reduced ridge crest twinkle / temporal instability
//        * stable block sampling (no frame-varying block jitter)
//        * stable bilinear reconstruction for coherent edge blocks
//        * stronger refinement near horizon silhouettes
// - FIX: reduced far-crest temporal flutter
//        * coarse-vs-full terrain LOD for intersection on far/grazing rays
//        * midpoint-tested raymarch intersection
//        * more conservative silhouette stepping
//        * dynamic normal epsilon for distant terrain
// - FIX: reduced geometry vibrations at ridges / crests
//        * stable trace surface for intersection
//        * crest-clamped far/grazing silhouette tracing
//        * two-stage refinement (stable trace -> full terrain)
//        * calmer fine ridge contribution
//        * camera follow uses coarse terrain mass
//
// Build (Ubuntu):
//   nvcc -O3 -std=c++17 terrain_raycast_screen_space_volumetric_clouds.cu `pkg-config --cflags --libs opencv4` -o terrain_raycast_screen_space_volumetric_clouds
//
// Run examples:
//   ./terrain_raycast_screen_space_volumetric_clouds
//   ./terrain_raycast_screen_space_volumetric_clouds 30
//   ./terrain_raycast_screen_space_volumetric_clouds 30 salida.mp4
//   ./terrain_raycast_screen_space_volumetric_clouds 30 salida.mp4 0.065 2 1.0 0.40 1.0 0.35 1 1 1 1 14.0

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
static constexpr int width  = 960;
static constexpr int height = 540;
static constexpr int output_fps = 60;

// Render sampling
static int sample_step = 2;

// Camera
static float terrain_height_multiplier = 1.0f;
static float camera_y_offset = 0.35f;
static cv::Vec3f camera_position = cv::Vec3f(0.0f, 1.2f, 5.0f);
static cv::Vec3f camera_rotation = cv::Vec3f(-0.10f, 0.0f, 0.0f);

// Dynamic camera height following terrain
static constexpr int   camera_height_sample_count    = 24;
static constexpr float camera_height_lookahead_start = 4.0f;
static constexpr float camera_height_lookahead_step  = 1.25f;
static constexpr float camera_height_clearance_base  = 0.9f;
static constexpr float camera_height_smoothness      = 0.08f;
static constexpr float camera_min_height             = 0.35f;

// Projection
static constexpr float focal_length = 800.0f;

// Terrain preset
static constexpr float noise_scale = 0.22f;
static constexpr int   octaves = 6;
static constexpr float persistence = 0.58f;
static constexpr float lacunarity = 2.15f;
static constexpr float macro_noise_scale = 0.0090f;

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
static constexpr float cloud_wind_speed_factor = 0.05f;

// Volumetric clouds
static constexpr float cloud_layer_base_offset  = -0.45f;
static constexpr float cloud_layer_top_offset   =  1.05f;
static constexpr int   cloud_volume_steps       = 28;
static constexpr float cloud_absorption         = 0.24f;
static constexpr float cloud_density_gain       = 1.55f;
static constexpr float cloud_light_step_dist    = 2.2f;
static constexpr int   cloud_light_steps        = 5;

// Runtime parameters
static float forward_speed = 0.065f;
static float shadow_darkness_multiplier = 1.0f;
static float water_reflection_strength = 0.40f;
static float water_deep_darkness_multiplier = 14.0f;

// Render toggles
static int render_visible_clouds = 1;
static int render_cloud_shadows = 1;
static int render_water_reflections = 1;

// Lighting
static constexpr float sun_azimuth_deg   = -45.0f;
static constexpr float sun_elevation_deg = 16.0f;
static constexpr float ambient_light      = 0.14f;
static constexpr float diffuse_strength   = 1.45f;
static constexpr float backlight_strength = 0.02f;
static constexpr float normal_eps         = 0.18f;
static constexpr float reflection_normal_eps = 0.28f;

// Cloud shadows
static constexpr float cloud_shadow_strength_base  = 0.92f;
static constexpr float cloud_shadow_softness       = 0.07f;
static constexpr float cloud_shadow_min_light_base = 0.08f;

// Water reflection tuning
static constexpr float water_fresnel_bias             = 0.10f;
static constexpr float water_fresnel_power            = 3.8f;
static constexpr float water_specular_strength        = 0.16f;
static constexpr float water_wave_noise_scale         = 0.90f;
static constexpr float water_wave_distort             = 0.08f;
static constexpr float water_max_reflection_dist      = 320.0f;
static constexpr int   water_reflection_steps         = 24;
static constexpr float water_reflection_step_min      = 2.0f;
static constexpr float water_reflection_step_max      = 16.0f;
static constexpr float water_reflection_cloud_gain    = 0.65f;
static constexpr float water_reflection_terrain_gain  = 0.90f;
static constexpr float water_reflection_sky_gain      = 0.85f;

// Sky
static constexpr float sky_fog_near = 20.0f;
static constexpr float sky_fog_far  = 560.0f;
static constexpr float sky_fog_density = 0.2f;

// Terrain fog
static constexpr float fog_near = 22.0f;
static constexpr float fog_far  = 95.0f;
static constexpr float fog_density = 0.22f;

// Preview
static constexpr int preview_every_seconds = 1;

// Raymarching / tracing
static constexpr float ray_t_min = 0.25f;
static constexpr float ray_t_max = 700.0f;
static constexpr int   terrain_primary_steps = 420;
static constexpr int   terrain_refine_steps  = 16;

// Planet curvature
static int   enable_planet_curvature = 1;
static constexpr float planet_radius = 7000.0f;

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
    oss << "screen space terrain raycaster volumetric clouds "
        << static_cast<long long>(now)
        << " "
        << std::put_time(&tm_buf, "%Y-%m-%d_%H-%M-%S")
        << ".mp4";
    return oss.str();
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
    oss << "preview t=" << sec << "s | step=" << sample_step;

    cv::putText(
        preview,
        oss.str(),
        cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
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

static BiomeStyle biome_style_desert(){   return {0.95f, 0.10f, 0.30f, 0.95f, 0.20f}; }
static BiomeStyle biome_style_valley(){   return {0.55f, 0.85f, 0.20f, 0.75f, 0.75f}; }
static BiomeStyle biome_style_mountain(){ return {0.35f, 0.35f, 0.95f, 0.45f, 0.45f}; }
static BiomeStyle biome_style_snow(){     return {0.10f, 0.45f, 0.80f, 0.10f, 0.65f}; }

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
    switch (dist(rng)) {
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

__device__ __forceinline__ int clampi(int v, int lo, int hi){
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

__host__ __device__ __forceinline__ unsigned int hash_u32(unsigned int x){
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

__constant__ int d_perm[512];
__constant__ float d_terrain_height_multiplier;

// =========================
// Projection basis
// =========================
struct Mat3 { float m[9]; };

__device__ __forceinline__ float3 mul(const Mat3& R, float3 v){
    return make_float3(
        R.m[0]*v.x + R.m[1]*v.y + R.m[2]*v.z,
        R.m[3]*v.x + R.m[4]*v.y + R.m[5]*v.z,
        R.m[6]*v.x + R.m[7]*v.y + R.m[8]*v.z
    );
}

// =========================
// Noise
// =========================
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

__device__ float fractal_noise2_7(float x, float y){
    float total = 0.f;
    float freq  = 1.f;
    float amp   = 1.f;
    float maxa  = 0.f;
    #pragma unroll
    for (int i = 0; i < 7; i++){
        total += perlin2(x * freq, y * freq) * amp;
        maxa  += amp;
        amp   *= 0.56f;
        freq  *= 2.10f;
    }
    return total / maxa;
}

__device__ float ridged_noise2(float x, float y, int octv, float gain, float lac){
    float sum = 0.0f;
    float freq = 1.0f;
    float amp  = 0.5f;
    float prev = 1.0f;

    #pragma unroll 1
    for (int i = 0; i < octv; ++i) {
        float n = perlin2(x * freq, y * freq);
        n = 1.0f - fabsf(n);
        n *= n;
        sum += n * amp * prev;
        prev = clampf(n * gain, 0.0f, 1.0f);
        freq *= lac;
        amp  *= 0.55f;
    }
    return sum;
}

__device__ float warped_fbm(float x, float y){
    float wx = fractal_noise2(x * 0.55f + 31.7f, y * 0.55f + 91.3f);
    float wy = fractal_noise2(x * 0.55f + 173.1f, y * 0.55f + 11.9f);
    return fractal_noise2_7(x + wx * 1.35f, y + wy * 1.35f);
}

// =========================
// Curvature helpers
// =========================
__device__ __forceinline__ float curvature_drop_local(float local_x, float local_z, int enabled){
    if (!enabled) return 0.0f;
    return (local_x * local_x + local_z * local_z) / (2.0f * planet_radius);
}

__device__ __forceinline__ float curved_surface_y(float base_y, float local_x, float local_z, int enabled){
    return base_y - curvature_drop_local(local_x, local_z, enabled);
}

__device__ __forceinline__ float3 curvature_normal_local(float local_x, float local_z, int enabled){
    if (!enabled) return make_float3(0.0f, 1.0f, 0.0f);
    return normalize3(make_float3(local_x / planet_radius, 1.0f, local_z / planet_radius));
}

// =========================
// Fog
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
// Terrain / clouds
// =========================
__device__ __forceinline__ float terrain_base_height_biomes_coarse(
    float world_x,
    float world_z,
    const DeviceBiomeStyle& terrain_style)
{
    float warp_x = fractal_noise2(world_x * 0.014f + 120.0f, world_z * 0.014f + 310.0f);
    float warp_z = fractal_noise2(world_x * 0.014f + 870.0f, world_z * 0.014f + 140.0f);

    float wx = world_x + warp_x * 6.0f;
    float wz = world_z + warp_z * 6.0f;

    float continental = warped_fbm(wx * 0.0060f, wz * 0.0060f);
    float broad       = warped_fbm(wx * 0.0180f, wz * 0.0180f);
    float rolling     = warped_fbm(wx * 0.0550f, wz * 0.0550f);

    float base_land =
        continental * 1.45f +
        broad       * 0.65f +
        rolling     * 0.18f -
        0.28f;

    float biomeA = 0.5f + 0.5f * fractal_noise2(
        world_x * 0.0012f + 1200.0f,
        world_z * 0.0012f + 3400.0f
    );

    float desert_region   = clampf(1.0f - fabsf(biomeA - 0.15f) / 0.22f, 0.0f, 1.0f);
    float valley_region   = clampf(1.0f - fabsf(biomeA - 0.50f) / 0.20f, 0.0f, 1.0f);
    float mountain_region = clampf(1.0f - fabsf(biomeA - 0.82f) / 0.20f, 0.0f, 1.0f);

    float region_sum = desert_region + valley_region + mountain_region + 1e-6f;
    desert_region   /= region_sum;
    valley_region   /= region_sum;
    mountain_region /= region_sum;

    float inlandness = smooth01((base_land + 0.05f) / 1.60f);
    inlandness = inlandness * inlandness;

    float altitude_drive = smooth01((base_land - 0.15f) / 2.40f);

    float relief_drive = clampf(
        0.55f * inlandness +
        0.45f * altitude_drive,
        0.0f, 1.0f
    );

    float coarse_ridge =
        ridged_noise2(wx * 0.16f + 800.0f,  wz * 0.16f + 1200.0f, 5, 2.0f, 2.05f) * 1.00f +
        ridged_noise2(wx * 0.30f + 2100.0f, wz * 0.30f + 3300.0f, 4, 2.0f, 2.10f) * 0.55f;

    float coarse_relief = coarse_ridge - fabsf(rolling) * 0.16f;

    float relief_strength =
        lerpf(0.18f, 0.55f, valley_region) +
        lerpf(0.10f, 1.25f, mountain_region) * relief_drive +
        lerpf(0.05f, 0.18f, desert_region);

    float h = base_land + coarse_relief * relief_strength;

    float coast_flatten = 1.0f - smooth01((h + 0.08f) / 0.90f);
    coast_flatten = clampf(coast_flatten, 0.0f, 1.0f);

    h = lerpf(h, h * 0.72f + 0.03f, coast_flatten * 0.45f);

    float valley_soften =
        valley_region *
        (1.0f - mountain_region * 0.65f) *
        smooth01((h - 0.00f) / 1.60f) *
        (1.0f - smooth01((h - 1.8f) / 2.4f));

    h -= 0.10f * valley_soften;
    h = lerpf(h, base_land * 0.82f + h * 0.18f, coast_flatten * 0.35f);

    float beach_mask = 1.0f - smooth01((h + 0.04f) / 0.55f);
    h += 0.045f * beach_mask * (1.0f - beach_mask);

    h += (terrain_style.global_ruggedness - 0.5f) * 0.06f;
    h -= (terrain_style.global_humidity  - 0.5f) * 0.02f;

    return h * d_terrain_height_multiplier;
}

__device__ __forceinline__ float terrain_base_height_biomes(float world_x, float world_z, const DeviceBiomeStyle& terrain_style){
    float warp_x = fractal_noise2(world_x * 0.014f + 120.0f, world_z * 0.014f + 310.0f);
    float warp_z = fractal_noise2(world_x * 0.014f + 870.0f, world_z * 0.014f + 140.0f);

    float wx = world_x + warp_x * 6.0f;
    float wz = world_z + warp_z * 6.0f;

    float continental = warped_fbm(wx * 0.0060f, wz * 0.0060f);
    float broad       = warped_fbm(wx * 0.0180f, wz * 0.0180f);
    float rolling     = warped_fbm(wx * 0.0550f, wz * 0.0550f);

    float base_land =
        continental * 1.45f +
        broad       * 0.65f +
        rolling     * 0.18f -
        0.28f;

    float biomeA = 0.5f + 0.5f * fractal_noise2(
        world_x * 0.0012f + 1200.0f,
        world_z * 0.0012f + 3400.0f
    );

    float desert_region   = clampf(1.0f - fabsf(biomeA - 0.15f) / 0.22f, 0.0f, 1.0f);
    float valley_region   = clampf(1.0f - fabsf(biomeA - 0.50f) / 0.20f, 0.0f, 1.0f);
    float mountain_region = clampf(1.0f - fabsf(biomeA - 0.82f) / 0.20f, 0.0f, 1.0f);

    float region_sum = desert_region + valley_region + mountain_region + 1e-6f;
    desert_region   /= region_sum;
    valley_region   /= region_sum;
    mountain_region /= region_sum;

    float inlandness = smooth01((base_land + 0.05f) / 1.60f);
    inlandness = inlandness * inlandness;

    float altitude_drive = smooth01((base_land - 0.15f) / 2.40f);

    float relief_drive = clampf(
        0.55f * inlandness +
        0.45f * altitude_drive,
        0.0f, 1.0f
    );

    float coarse_ridge =
        ridged_noise2(wx * 0.16f + 800.0f,  wz * 0.16f + 1200.0f, 5, 2.0f, 2.05f) * 1.00f +
        ridged_noise2(wx * 0.30f + 2100.0f, wz * 0.30f + 3300.0f, 4, 2.0f, 2.10f) * 0.55f;

    float coarse_relief = coarse_ridge - fabsf(rolling) * 0.16f;

    float relief_strength =
        lerpf(0.18f, 0.55f, valley_region) +
        lerpf(0.10f, 1.25f, mountain_region) * relief_drive +
        lerpf(0.05f, 0.18f, desert_region);

    float h = base_land + coarse_relief * relief_strength;

    // Calmer fine detail to reduce subpixel far-crest chatter.
    float fine_fbm =
        fractal_noise2_7(wx * 0.34f + 4100.0f, wz * 0.34f + 5100.0f) * 0.030f;

    float fine_ridge =
        ridged_noise2(wx * 0.40f + 7100.0f, wz * 0.40f + 8100.0f, 3, 1.85f, 2.00f) * 0.018f;

    float fine_detail_mask =
        mountain_region *
        smooth01((h - 0.45f) / 2.20f) *
        smooth01((base_land - 0.05f) / 1.40f);

    float fine_detail = fine_fbm + fine_ridge;

    // Reduce thin positive skyline spikes more than negative relief.
    float positive_crest_soften =
        mountain_region *
        smooth01((h - 0.55f) / 2.10f);

    fine_detail = lerpf(fine_detail, fminf(fine_detail, 0.0f), positive_crest_soften * 0.45f);

    h += fine_detail * fine_detail_mask;

    float coast_flatten = 1.0f - smooth01((h + 0.08f) / 0.90f);
    coast_flatten = clampf(coast_flatten, 0.0f, 1.0f);

    h = lerpf(h, h * 0.72f + 0.03f, coast_flatten * 0.45f);

    float valley_soften =
        valley_region *
        (1.0f - mountain_region * 0.65f) *
        smooth01((h - 0.00f) / 1.60f) *
        (1.0f - smooth01((h - 1.8f) / 2.4f));

    h -= 0.10f * valley_soften;
    h = lerpf(h, base_land * 0.82f + h * 0.18f, coast_flatten * 0.35f);

    float beach_mask = 1.0f - smooth01((h + 0.04f) / 0.55f);
    h += 0.045f * beach_mask * (1.0f - beach_mask);

    h += (terrain_style.global_ruggedness - 0.5f) * 0.06f;
    h -= (terrain_style.global_humidity  - 0.5f) * 0.02f;

    return h * d_terrain_height_multiplier;
}

__device__ __forceinline__ float terrain_height_lod(
    float world_x,
    float world_z,
    float t,
    float ray_dir_y_abs,
    const DeviceBiomeStyle& terrain_style)
{
    float h_coarse = terrain_base_height_biomes_coarse(world_x, world_z, terrain_style);
    float h_full   = terrain_base_height_biomes(world_x, world_z, terrain_style);

    float dist_fade = smooth01((t - 35.0f) / 120.0f);
    float grazing   = 1.0f - clampf(ray_dir_y_abs / 0.22f, 0.0f, 1.0f);
    grazing = smooth01(grazing);

    float lod_t = clampf(dist_fade * grazing, 0.0f, 1.0f);

    return lerpf(h_full, h_coarse, lod_t);
}

// Stable tracing surface used only for intersections to calm ridge/crest flicker.
__device__ __forceinline__ float terrain_height_trace_stable(
    float world_x,
    float world_z,
    float t,
    float ray_dir_y_abs,
    const DeviceBiomeStyle& terrain_style)
{
    float h_coarse = terrain_base_height_biomes_coarse(world_x, world_z, terrain_style);
    float h_full   = terrain_base_height_biomes(world_x, world_z, terrain_style);

    float dist_fade = smooth01((t - 28.0f) / 110.0f);
    float grazing   = 1.0f - clampf(ray_dir_y_abs / 0.28f, 0.0f, 1.0f);
    grazing = smooth01(grazing);

    float far_grazing = clampf(dist_fade * grazing, 0.0f, 1.0f);

    // Prevent thin positive skyline spikes on far/grazing rays.
    float crest_clamped = fminf(h_full, h_coarse + 0.025f);

    return lerpf(h_full, crest_clamped, far_grazing);
}

__device__ __forceinline__ float cloud_density_at(float world_x, float world_z, float cloud_wind_offset_z, const DeviceBiomeStyle& visual_style){
    float nx = world_x * cloud_noise_scale + 100.f;
    float nz = (world_z + cloud_wind_offset_z) * cloud_noise_scale + 200.f;

    float base    = fractal_noise2(nx, nz);
    float detail1 = fractal_noise2(nx * 1.8f + 71.0f, nz * 1.8f + 29.0f);
    float detail2 = fractal_noise2(nx * 3.2f + 211.0f, nz * 3.2f + 503.0f);

    float macro = fractal_noise2(
        world_x * 0.010f + 900.0f,
        (world_z + cloud_wind_offset_z * 0.35f) * 0.010f + 1300.0f
    );
    macro = 0.5f + 0.5f * macro;

    float puffy = 1.0f - fabsf(base);
    puffy = puffy * puffy;

    float humid = local_humidity(world_x, world_z, visual_style);

    float dense =
        base    * 0.52f +
        detail1 * 0.20f +
        detail2 * 0.08f +
        puffy   * 0.35f;

    dense = lerpf(dense - 0.18f, dense + 0.18f, 0.35f * humid + 0.65f * visual_style.cloudiness);

    float coverage = smooth01((macro - 0.42f) / 0.30f);
    dense *= coverage;

    return dense;
}

__device__ __forceinline__ float cloud_vertical_profile(float y_local, float x_local, float z_local, int curvature_enabled){
    float base_y = curved_surface_y(cloud_height + cloud_layer_base_offset, x_local, z_local, curvature_enabled);
    float top_y  = curved_surface_y(cloud_height + cloud_layer_top_offset,  x_local, z_local, curvature_enabled);

    float thick = fmaxf(top_y - base_y, 1e-4f);
    float t = (y_local - base_y) / thick;
    if (t <= 0.0f || t >= 1.0f) return 0.0f;

    float center = 1.0f - fabsf(t * 2.0f - 1.0f);
    center = smooth01(center);
    center = center * center;
    return center;
}

__device__ __forceinline__ float cloud_density_volume(
    float x_local,
    float y_local,
    float z_local,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    const DeviceBiomeStyle& visual_style,
    int curvature_enabled)
{
    float prof = cloud_vertical_profile(y_local, x_local, z_local, curvature_enabled);
    if (prof <= 0.0f) return 0.0f;

    float world_x = x_local + terrain_offset_x;
    float world_z = z_local + terrain_offset_z;

    float base = cloud_density_at(world_x, world_z, cloud_wind_offset_z, visual_style);

    float d = (base - cloud_threshold) / fmaxf(0.20f, 1e-5f);
    d = clampf(d, 0.0f, 1.0f);
    d = d * d;
    d *= prof;
    d *= cloud_density_gain;

    return d;
}

// =========================
// Sky / terrain colors
// =========================
__device__ __forceinline__ void sky_color_at_dir(float3 dir, float depth_hint, const DeviceBiomeStyle& visual_style, float& b, float& g, float& r){
    float up = clampf(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);

    float cold_b1 = 185.0f, cold_g1 =  80.0f, cold_r1 =  18.0f;
    float cold_b2 = 240.0f, cold_g2 = 170.0f, cold_r2 = 105.0f;

    float warm_b1 = 170.0f, warm_g1 =  90.0f, warm_r1 =  28.0f;
    float warm_b2 = 232.0f, warm_g2 = 175.0f, warm_r2 = 120.0f;

    float cold_b = lerpf(cold_b2, cold_b1, up);
    float cold_g = lerpf(cold_g2, cold_g1, up);
    float cold_r = lerpf(cold_r2, cold_r1, up);

    float warm_b = lerpf(warm_b2, warm_b1, up);
    float warm_g = lerpf(warm_g2, warm_g1, up);
    float warm_r = lerpf(warm_r2, warm_r1, up);

    b = lerpf(cold_b, warm_b, visual_style.global_temperature);
    g = lerpf(cold_g, warm_g, visual_style.global_temperature);
    r = lerpf(cold_r, warm_r, visual_style.global_temperature);

    fog_mix(b, g, r, depth_hint, true);

    float horizon_band = 1.0f - clampf(fabsf(dir.y) / 0.16f, 0.0f, 1.0f);
    horizon_band = smooth01(horizon_band);
    horizon_band = powf(horizon_band, 1.35f);

    float horizon_strength = lerpf(
        0.78f,
        0.92f,
        visual_style.global_temperature * 0.45f + visual_style.global_humidity * 0.55f
    );

    b = lerpf(b, 255.0f, horizon_band * horizon_strength);
    g = lerpf(g, 255.0f, horizon_band * horizon_strength);
    r = lerpf(r, 255.0f, horizon_band * horizon_strength);
}

__device__ __forceinline__ float snow_amount(float x, float z, float h, float3 normal, const DeviceBiomeStyle& visual_style){
    float t = clampf(
        0.5f + 0.5f * fractal_noise2(x * biome_temp_scale + 130.0f, z * biome_temp_scale + 210.0f),
        0.0f, 1.0f
    );
    float local_temp = lerpf(t, visual_style.global_temperature, 0.40f);
    float coldness = 1.0f - local_temp;

    float snowline = lerpf(4.0f, 1.0f, 1.0f - visual_style.global_snowline);
    float alt = clampf((h - snowline) / 2.5f, 0.0f, 1.0f);
    float flatness = clampf(normal.y, 0.0f, 1.0f);

    float snow = coldness * alt * lerpf(0.45f, 1.0f, flatness);
    return clampf(snow, 0.0f, 1.0f);
}

__device__ __forceinline__ void biome_base_color(
    float x, float z, float h, float3 n, bool is_water,
    const DeviceBiomeStyle& visual_style,
    float water_dark_mult,
    float& b, float& g, float& r)
{
    if (is_water) {
        float depth_like = clampf((-h) / 2.5f, 0.0f, 1.0f);

        float shallow_b = 220.f, shallow_g = 180.f, shallow_r =  96.f;
        float deep_b    =  78.f, deep_g    =  34.f, deep_r    =   8.f;

        float deep_mask = smooth01((depth_like - 0.08f) / 0.42f);
        float deep_scale = 1.0f / fmaxf(water_dark_mult, 0.001f);

        deep_b *= deep_scale;
        deep_g *= deep_scale;
        deep_r *= deep_scale;

        deep_b = clampf(deep_b, 18.0f, 255.0f);
        deep_g = clampf(deep_g, 10.0f, 255.0f);
        deep_r = clampf(deep_r,  4.0f, 255.0f);

        b = lerpf(shallow_b, deep_b, deep_mask);
        g = lerpf(shallow_g, deep_g, deep_mask);
        r = lerpf(shallow_r, deep_r, deep_mask);
        return;
    }

    BiomeWeights bw = biome_weights_at(x, z, h, visual_style);

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

    float warm_t = visual_style.global_temperature;
    b = lerpf(b, b - 6.f, warm_t * 0.18f);
    g = lerpf(g, g + 4.f, warm_t * 0.10f);
    r = lerpf(r, r + 8.f, warm_t * 0.18f);

    float rock_detail = fractal_noise2(x * 0.95f + 8000.0f, z * 0.95f + 9100.0f);
    rock_detail = 0.5f + 0.5f * rock_detail;

    float rocky_mask = rockiness * smooth01((h - 1.8f) / 2.2f);
    float rock_mod = lerpf(0.92f, 1.08f, rock_detail);

    b *= lerpf(1.0f, rock_mod * 0.98f, rocky_mask * 0.35f);
    g *= lerpf(1.0f, rock_mod * 1.00f, rocky_mask * 0.35f);
    r *= lerpf(1.0f, rock_mod * 1.03f, rocky_mask * 0.35f);

    float snow = snow_amount(x, z, h, n, visual_style);
    float high_peak_whiten = smooth01((h - 4.2f) / 1.7f) * (0.25f + 0.45f * bw.mountain);
    snow = clampf(snow + high_peak_whiten * (1.0f - snow), 0.0f, 1.0f);

    b = lerpf(b, 245.f, snow);
    g = lerpf(g, 245.f, snow);
    r = lerpf(r, 245.f, snow);
}

// =========================
// Normals / lighting
// =========================
__device__ __forceinline__ float3 terrain_normal_fast_curved(
    float world_x,
    float world_z,
    float local_x,
    float local_z,
    const DeviceBiomeStyle& terrain_style,
    int curvature_enabled,
    float eps)
{
    float hL = curved_surface_y(terrain_base_height_biomes(world_x - eps, world_z,       terrain_style), local_x - eps, local_z, curvature_enabled);
    float hR = curved_surface_y(terrain_base_height_biomes(world_x + eps, world_z,       terrain_style), local_x + eps, local_z, curvature_enabled);
    float hD = curved_surface_y(terrain_base_height_biomes(world_x,       world_z - eps, terrain_style), local_x, local_z - eps, curvature_enabled);
    float hU = curved_surface_y(terrain_base_height_biomes(world_x,       world_z + eps, terrain_style), local_x, local_z + eps, curvature_enabled);

    float3 dx = make_float3(2.0f * eps, hR - hL, 0.0f);
    float3 dz = make_float3(0.0f, hU - hD, 2.0f * eps);
    float3 n  = cross3(dz, dx);
    return normalize3(n);
}

__device__ __forceinline__ float cloud_shadow_factor(
    float world_x, float terrain_y, float world_z,
    float3 sun_dir,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    const DeviceBiomeStyle& visual_style)
{
    float up_y = -sun_dir.y;
    if (up_y <= 1e-5f) return 1.0f;

    float t = (cloud_height - terrain_y) / up_y;
    if (t <= 0.0f) return 1.0f;

    float xc_world = world_x + (-sun_dir.x) * t;
    float zc_world = world_z + (-sun_dir.z) * t;

    float cloud = cloud_density_at(xc_world, zc_world, cloud_wind_offset_z, visual_style);
    float alpha = clampf((cloud - cloud_threshold) / cloud_shadow_softness, 0.0f, 1.0f);

    float strength  = clampf(cloud_shadow_strength_base * shadow_darkness_mult, 0.0f, 1.20f);
    float min_light = clampf(cloud_shadow_min_light_base / fmaxf(shadow_darkness_mult, 0.05f), 0.02f, 1.0f);

    float shadow = 1.0f - strength * alpha;
    return clampf(shadow, min_light, 1.0f);
}

__device__ __forceinline__ void terrain_color_lit_from_normal(
    float world_x,
    float world_z,
    float terrain_h,
    bool is_water,
    float3 n,
    float3 sun_dir,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    const DeviceBiomeStyle& visual_style,
    float water_dark_mult,
    int enable_cloud_shadows_flag,
    float& b, float& g, float& r)
{
    biome_base_color(world_x, world_z, terrain_h, n, is_water, visual_style, water_dark_mult, b, g, r);

    if (!is_water) {
        float ndotl = clampf(dot3(n, mul3(sun_dir, -1.0f)), 0.0f, 1.0f);
        float ndotl_shaped = powf(ndotl, 1.15f);
        float slope_backlight = clampf(1.0f - n.y, 0.0f, 1.0f);

        float cloud_shadow = 1.0f;
        if (enable_cloud_shadows_flag) {
            cloud_shadow = cloud_shadow_factor(
                world_x, terrain_h > 0.0f ? terrain_h : 0.0f, world_z,
                sun_dir, cloud_wind_offset_z, cloud_threshold, shadow_darkness_mult, visual_style
            );
        }

        float lighting =
            ambient_light +
            diffuse_strength * ndotl_shaped * cloud_shadow +
            backlight_strength * slope_backlight * 0.20f;

        lighting = clampf(lighting, 0.02f, 1.55f);

        b *= lighting;
        g *= lighting;
        r *= lighting;

        if (enable_cloud_shadows_flag && cloud_shadow < 0.75f) {
            float shadow_t = clampf((0.75f - cloud_shadow) / 0.75f, 0.0f, 1.0f);
            b *= (1.00f + 0.04f * shadow_t);
            g *= (1.00f - 0.05f * shadow_t);
            r *= (1.00f - 0.10f * shadow_t);
        }
    }
}

// =========================
// Ray helpers
// =========================
__device__ __forceinline__ float ray_step_primary(float t){
    float nt = clampf(t / ray_t_max, 0.0f, 1.0f);
    float shaped = nt * nt;
    return lerpf(0.55f, 4.20f, shaped);
}

__device__ __forceinline__ float reflection_step_size(float t){
    float nt = clampf(t / water_max_reflection_dist, 0.0f, 1.0f);
    return lerpf(water_reflection_step_min, water_reflection_step_max, nt);
}

struct HitInfo {
    int   hit_type;   // 0=sky, 1=terrain, 2=water
    float t;
    float3 pos_local;
    float world_x;
    float world_z;
    float terrain_h;
};

struct CloudComposite {
    float transmittance;
    float accum_b;
    float accum_g;
    float accum_r;
    float first_t;
    float alpha;
};

struct ShadedSample {
    float b, g, r;
    float depth;
    int hit_type;
    int is_candidate_edge;
};

// =========================
// Water reflection
// =========================
__device__ __forceinline__ void water_reflection_trace(
    float3 p_local,
    float depth_to_water,
    float3 water_n,
    float3 sun_dir,
    float3 camPos,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float reflection_strength,
    const DeviceBiomeStyle& terrain_style,
    const DeviceBiomeStyle& visual_style,
    float water_dark_mult,
    int curvature_enabled,
    int visible_clouds_enabled,
    float& out_b,
    float& out_g,
    float& out_r)
{
    (void)water_dark_mult;

    float3 v_to_eye = normalize3(sub3(camPos, p_local));
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

    float3 pos = add3(p_local, mul3(refl_dir, 0.40f));

    #pragma unroll 1
    for (int s = 0; s < water_reflection_steps; ++s) {
        float step_len = reflection_step_size(traced_dist);
        pos = add3(pos, mul3(refl_dir, step_len));
        traced_dist += step_len;
        if (traced_dist > water_max_reflection_dist) break;

        float sample_world_x = pos.x + terrain_offset_x;
        float sample_world_z = pos.z + terrain_offset_z;

        float terrain_h = terrain_base_height_biomes(sample_world_x, sample_world_z, terrain_style);
        float terrain_y_curved = curved_surface_y(terrain_h > 0.0f ? terrain_h : 0.0f, pos.x, pos.z, curvature_enabled);

        if (visible_clouds_enabled) {
            float cd = cloud_density_volume(
                pos.x, pos.y, pos.z,
                terrain_offset_x, terrain_offset_z,
                cloud_wind_offset_z,
                cloud_threshold,
                visual_style,
                curvature_enabled
            );
            float ca = clampf(cd * 0.30f, 0.0f, 1.0f);
            if (ca > 0.001f) {
                float cb = lerpf(210.f, 238.f, ca);
                float cg = lerpf(214.f, 240.f, ca);
                float cr = lerpf(218.f, 242.f, ca);

                float remain = 1.0f - accum_cloud_alpha;
                cloud_b += cb * ca * remain;
                cloud_g += cg * ca * remain;
                cloud_r += cr * ca * remain;
                accum_cloud_alpha += ca * 0.50f * remain;
                accum_cloud_alpha = clampf(accum_cloud_alpha, 0.0f, 1.0f);
            }
        }

        if (terrain_h > 0.0f && pos.y <= terrain_y_curved) {
            float3 n = terrain_normal_fast_curved(sample_world_x, sample_world_z, pos.x, pos.z, terrain_style, curvature_enabled, reflection_normal_eps);

            terrain_color_lit_from_normal(
                sample_world_x, sample_world_z, terrain_h, false,
                n, sun_dir, cloud_wind_offset_z, cloud_threshold,
                shadow_darkness_mult, visual_style, water_dark_mult,
                0,
                hit_b, hit_g, hit_r
            );
            fog_mix(hit_b, hit_g, hit_r, depth_to_water + traced_dist, false);
            hit_terrain = true;
            break;
        }
    }

    float sky_b, sky_g, sky_r;
    sky_color_at_dir(refl_dir, depth_to_water + traced_dist, visual_style, sky_b, sky_g, sky_r);

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
// Intersections
// =========================
__device__ __forceinline__ bool intersect_terrain_or_water(
    float3 ray_origin,
    float3 ray_dir,
    float terrain_offset_x,
    float terrain_offset_z,
    const DeviceBiomeStyle& terrain_style,
    int curvature_enabled,
    HitInfo& hit)
{
    float ray_dir_y_abs = fabsf(ray_dir.y);

    float t_prev = ray_t_min;
    float3 p_prev = add3(ray_origin, mul3(ray_dir, t_prev));

    float world_x_prev = p_prev.x + terrain_offset_x;
    float world_z_prev = p_prev.z + terrain_offset_z;
    float terrain_h_prev = terrain_height_trace_stable(world_x_prev, world_z_prev, t_prev, ray_dir_y_abs, terrain_style);
    float surface_base_prev = terrain_h_prev > 0.0f ? terrain_h_prev : 0.0f;
    float surface_y_prev = curved_surface_y(surface_base_prev, p_prev.x, p_prev.z, curvature_enabled);
    float signed_prev = p_prev.y - surface_y_prev;

    float t = t_prev;

    #pragma unroll 1
    for (int i = 0; i < terrain_primary_steps; ++i) {
        float base_step = ray_step_primary(t);

        float grazing = 1.0f - clampf(ray_dir_y_abs / 0.22f, 0.0f, 1.0f);
        grazing = smooth01(grazing);

        float distance_factor = smooth01((t - 35.0f) / 160.0f);
        float refine_factor = grazing * distance_factor;

        float step_len = lerpf(base_step, base_step * 0.20f, refine_factor);

        float max_horizon_step = lerpf(4.0f, 0.75f, grazing);
        step_len = fminf(step_len, max_horizon_step);
        step_len = fmaxf(step_len, 0.12f);

        float t_now = t + step_len;
        if (t_now > ray_t_max) break;

        float3 p_now = add3(ray_origin, mul3(ray_dir, t_now));
        float world_x_now = p_now.x + terrain_offset_x;
        float world_z_now = p_now.z + terrain_offset_z;
        float terrain_h_now = terrain_height_trace_stable(world_x_now, world_z_now, t_now, ray_dir_y_abs, terrain_style);
        float surface_base_now = terrain_h_now > 0.0f ? terrain_h_now : 0.0f;
        float surface_y_now = curved_surface_y(surface_base_now, p_now.x, p_now.z, curvature_enabled);
        float signed_now = p_now.y - surface_y_now;

        float t_mid = 0.5f * (t_prev + t_now);
        float3 p_mid = add3(ray_origin, mul3(ray_dir, t_mid));
        float world_x_mid = p_mid.x + terrain_offset_x;
        float world_z_mid = p_mid.z + terrain_offset_z;
        float terrain_h_mid = terrain_height_trace_stable(world_x_mid, world_z_mid, t_mid, ray_dir_y_abs, terrain_style);
        float surface_base_mid = terrain_h_mid > 0.0f ? terrain_h_mid : 0.0f;
        float surface_y_mid = curved_surface_y(surface_base_mid, p_mid.x, p_mid.z, curvature_enabled);
        float signed_mid = p_mid.y - surface_y_mid;

        bool crossed = false;
        float a = t_prev;
        float b = t_now;

        if (signed_prev > 0.0f && signed_mid <= 0.0f) {
            crossed = true;
            a = t_prev;
            b = t_mid;
        }
        else if (signed_mid > 0.0f && signed_now <= 0.0f) {
            crossed = true;
            a = t_mid;
            b = t_now;
        }
        else if (signed_prev > 0.0f && signed_now <= 0.0f) {
            crossed = true;
            a = t_prev;
            b = t_now;
        }

        if (crossed) {
            // Stage 1: stable trace-surface refinement for temporal stability.
            #pragma unroll
            for (int k = 0; k < terrain_refine_steps / 2; ++k) {
                float m = 0.5f * (a + b);
                float3 pm = add3(ray_origin, mul3(ray_dir, m));

                float wx = pm.x + terrain_offset_x;
                float wz = pm.z + terrain_offset_z;
                float th = terrain_height_trace_stable(wx, wz, m, ray_dir_y_abs, terrain_style);
                float sb = th > 0.0f ? th : 0.0f;
                float sy = curved_surface_y(sb, pm.x, pm.z, curvature_enabled);
                float sgn = pm.y - sy;

                if (sgn > 0.0f) a = m;
                else b = m;
            }

            // Stage 2: final refinement against full terrain for local accuracy.
            #pragma unroll
            for (int k = 0; k < terrain_refine_steps / 2; ++k) {
                float m = 0.5f * (a + b);
                float3 pm = add3(ray_origin, mul3(ray_dir, m));

                float wx = pm.x + terrain_offset_x;
                float wz = pm.z + terrain_offset_z;
                float th = terrain_base_height_biomes(wx, wz, terrain_style);
                float sb = th > 0.0f ? th : 0.0f;
                float sy = curved_surface_y(sb, pm.x, pm.z, curvature_enabled);
                float sgn = pm.y - sy;

                if (sgn > 0.0f) a = m;
                else b = m;
            }

            float t_hit = 0.5f * (a + b);
            float3 p_hit = add3(ray_origin, mul3(ray_dir, t_hit));
            float wx = p_hit.x + terrain_offset_x;
            float wz = p_hit.z + terrain_offset_z;

            float th_full = terrain_base_height_biomes(wx, wz, terrain_style);

            hit.hit_type = (th_full > 0.0f) ? 1 : 2;
            hit.t = t_hit;
            hit.pos_local = p_hit;
            hit.world_x = wx;
            hit.world_z = wz;
            hit.terrain_h = th_full;
            return true;
        }

        t_prev = t_now;
        t = t_now;
        signed_prev = signed_now;
    }

    return false;
}

// =========================
// Volumetric clouds
// =========================
__device__ __forceinline__ float cloud_light_visibility(
    float3 sample_pos,
    float3 sun_dir,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    const DeviceBiomeStyle& visual_style,
    int curvature_enabled)
{
    float trans = 1.0f;
    float3 light_dir = mul3(sun_dir, -1.0f);

    #pragma unroll
    for (int i = 0; i < cloud_light_steps; ++i) {
        float t = (float)(i + 1) * cloud_light_step_dist;
        float3 p = add3(sample_pos, mul3(light_dir, t));

        float d = cloud_density_volume(
            p.x, p.y, p.z,
            terrain_offset_x, terrain_offset_z,
            cloud_wind_offset_z,
            cloud_threshold,
            visual_style,
            curvature_enabled
        );

        float a = 1.0f - expf(-d * cloud_absorption * 0.9f);
        trans *= (1.0f - clampf(a, 0.0f, 0.95f));
        if (trans < 0.05f) break;
    }

    return clampf(trans, 0.0f, 1.0f);
}

__device__ __forceinline__ CloudComposite integrate_volumetric_clouds(
    float3 ray_origin,
    float3 ray_dir,
    float max_t,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float3 sun_dir,
    const DeviceBiomeStyle& visual_style,
    int curvature_enabled,
    int sample_x,
    int sample_y)
{
    CloudComposite cc{};
    cc.transmittance = 1.0f;
    cc.accum_b = 0.0f;
    cc.accum_g = 0.0f;
    cc.accum_r = 0.0f;
    cc.first_t = max_t;
    cc.alpha = 0.0f;

    if (max_t <= ray_t_min) return cc;

    float t0 = fmaxf(ray_t_min, 18.0f);
    float t1 = fminf(max_t, 260.0f);
    if (t1 <= t0) return cc;

    float dt = (t1 - t0) / (float)cloud_volume_steps;
    if (dt <= 1e-5f) return cc;

    float jitter = rand01_from_2i(sample_x, sample_y, 0xC1A0u);
    bool found_any = false;

    #pragma unroll 1
    for (int i = 0; i < cloud_volume_steps; ++i) {
        float t = t0 + ((float)i + jitter) * dt;
        float3 p = add3(ray_origin, mul3(ray_dir, t));

        float d = cloud_density_volume(
            p.x, p.y, p.z,
            terrain_offset_x, terrain_offset_z,
            cloud_wind_offset_z,
            cloud_threshold,
            visual_style,
            curvature_enabled
        );

        if (d <= 1e-5f) continue;
        if (!found_any) {
            cc.first_t = t;
            found_any = true;
        }

        float light_vis = cloud_light_visibility(
            p, sun_dir,
            terrain_offset_x, terrain_offset_z,
            cloud_wind_offset_z,
            cloud_threshold,
            visual_style,
            curvature_enabled
        );

        float up_tint = clampf(ray_dir.y * 0.5f + 0.5f, 0.0f, 1.0f);

        float cb = lerpf(210.0f, 238.0f, light_vis);
        float cg = lerpf(214.0f, 240.0f, light_vis);
        float cr = lerpf(218.0f, 242.0f, light_vis);

        cb = lerpf(cb, cb - 6.0f, up_tint * 0.15f);
        cg = lerpf(cg, cg - 4.0f, up_tint * 0.12f);
        cr = lerpf(cr, cr + 2.0f, visual_style.global_temperature * 0.10f);

        float sample_alpha = 1.0f - expf(-d * cloud_absorption * dt);
        sample_alpha = clampf(sample_alpha, 0.0f, 0.98f);

        float contrib = cc.transmittance * sample_alpha;

        cc.accum_b += cb * contrib;
        cc.accum_g += cg * contrib;
        cc.accum_r += cr * contrib;

        cc.transmittance *= (1.0f - sample_alpha);
        if (cc.transmittance < 0.01f) {
            cc.transmittance = 0.01f;
            break;
        }
    }

    cc.alpha = 1.0f - cc.transmittance;
    return cc;
}

// =========================
// Screen ray generation
// =========================
__device__ __forceinline__ float3 screen_ray_dir(
    float sample_x,
    float sample_y,
    Mat3 R)
{
    float px = (sample_x + 0.5f) - (float)width * 0.5f;
    float py = (float)height * 0.5f - (sample_y + 0.5f);

    float3 cam_dir = normalize3(make_float3(px / focal_length, py / focal_length, 1.0f));
    return normalize3(mul(R, cam_dir));
}

// =========================
// Block paint
// =========================
__device__ __forceinline__ void paint_block_bgr(
    unsigned char* out_bgr,
    int bx, int by,
    int step,
    unsigned char b,
    unsigned char g,
    unsigned char r)
{
    int x0 = bx * step;
    int y0 = by * step;
    int x1 = min(width,  x0 + step);
    int y1 = min(height, y0 + step);

    for (int y = y0; y < y1; ++y) {
        int row = y * width * 3;
        for (int x = x0; x < x1; ++x) {
            int o = row + x * 3;
            out_bgr[o + 0] = b;
            out_bgr[o + 1] = g;
            out_bgr[o + 2] = r;
        }
    }
}

// =========================
// Shading
// =========================
__device__ __forceinline__ ShadedSample shade_sample(
    float sample_x_f,
    float sample_y_f,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float water_reflect_strength,
    float water_dark_mult,
    float3 sun_dir,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle terrain_style,
    DeviceBiomeStyle visual_style,
    int visible_clouds_enabled,
    int cloud_shadows_enabled,
    int water_reflections_enabled,
    int curvature_enabled)
{
    ShadedSample s{};
    s.b = s.g = s.r = 0.0f;
    s.depth = ray_t_max;
    s.hit_type = 0;
    s.is_candidate_edge = 0;

    int sample_x_i = clampi((int)floorf(sample_x_f), 0, width  - 1);
    int sample_y_i = clampi((int)floorf(sample_y_f), 0, height - 1);

    float3 ray_origin = camPos;
    float3 ray_dir = screen_ray_dir(sample_x_f, sample_y_f, R);

    float surf_b = 0.0f, surf_g = 0.0f, surf_r = 0.0f;

    HitInfo surf_hit{};
    bool has_surface = intersect_terrain_or_water(
        ray_origin, ray_dir,
        terrain_offset_x, terrain_offset_z,
        terrain_style, curvature_enabled,
        surf_hit
    );

    if (has_surface) {
        s.depth = surf_hit.t;
        s.hit_type = surf_hit.hit_type;

        bool is_water = (surf_hit.hit_type == 2);
        float3 p = surf_hit.pos_local;

        if (!is_water) {
            float dyn_normal_eps = lerpf(
                normal_eps,
                normal_eps * 3.5f,
                smooth01((surf_hit.t - 35.0f) / 140.0f)
            );

            float3 n = terrain_normal_fast_curved(
                surf_hit.world_x, surf_hit.world_z,
                p.x, p.z,
                terrain_style, curvature_enabled,
                dyn_normal_eps
            );

            terrain_color_lit_from_normal(
                surf_hit.world_x, surf_hit.world_z,
                surf_hit.terrain_h, false,
                n, sun_dir,
                cloud_wind_offset_z, cloud_threshold,
                shadow_darkness_mult, visual_style,
                water_dark_mult,
                cloud_shadows_enabled,
                surf_b, surf_g, surf_r
            );

            fog_mix(surf_b, surf_g, surf_r, surf_hit.t, false);

            float horizon_contact = 1.0f - clampf(fabsf(ray_dir.y) / 0.12f, 0.0f, 1.0f);
            horizon_contact = smooth01(horizon_contact);
            horizon_contact *= smooth01((surf_hit.t - 18.0f) / 55.0f);

            surf_b = lerpf(surf_b, 255.0f, horizon_contact * 0.58f);
            surf_g = lerpf(surf_g, 255.0f, horizon_contact * 0.58f);
            surf_r = lerpf(surf_r, 255.0f, horizon_contact * 0.58f);
        } else {
            float wave_x = fractal_noise2(
                surf_hit.world_x * water_wave_noise_scale + 401.0f,
                surf_hit.world_z * water_wave_noise_scale + 97.0f
            );

            float wave_z = fractal_noise2(
                surf_hit.world_x * water_wave_noise_scale + 211.0f,
                surf_hit.world_z * water_wave_noise_scale + 501.0f
            );

            float3 water_n = normalize3(add3(
                curvature_normal_local(p.x, p.z, curvature_enabled),
                make_float3(
                    wave_x * water_wave_distort,
                    0.0f,
                    wave_z * water_wave_distort
                )
            ));

            float seabed_depth = clampf(-surf_hit.terrain_h, 0.0f, 12.0f);
            float deep_t = smooth01((seabed_depth - 0.25f) / 5.5f);

            biome_base_color(
                surf_hit.world_x, surf_hit.world_z, surf_hit.terrain_h,
                water_n, true, visual_style, water_dark_mult,
                surf_b, surf_g, surf_r
            );

            float cloud_shadow = 1.0f;
            if (cloud_shadows_enabled) {
                cloud_shadow = cloud_shadow_factor(
                    surf_hit.world_x, 0.0f, surf_hit.world_z,
                    sun_dir, cloud_wind_offset_z, cloud_threshold,
                    shadow_darkness_mult, visual_style
                );
            }

            float coastal_light = 0.96f;
            float deep_light    = 0.42f;
            float water_light = lerpf(coastal_light, deep_light, deep_t);
            water_light *= lerpf(0.72f, 1.0f, cloud_shadow);

            surf_b *= water_light;
            surf_g *= water_light;
            surf_r *= water_light;

            if (water_reflections_enabled) {
                float rb, rg, rr;

                float depth_reflection_strength =
                    water_reflect_strength * lerpf(1.00f, 0.28f, deep_t);

                water_reflection_trace(
                    p,
                    surf_hit.t,
                    water_n,
                    sun_dir, camPos,
                    terrain_offset_x, terrain_offset_z,
                    cloud_wind_offset_z,
                    cloud_threshold,
                    shadow_darkness_mult,
                    depth_reflection_strength,
                    terrain_style,
                    visual_style,
                    water_dark_mult,
                    curvature_enabled,
                    visible_clouds_enabled,
                    rb, rg, rr
                );

                surf_b = clampf(surf_b + rb, 0.0f, 255.0f);
                surf_g = clampf(surf_g + rg, 0.0f, 255.0f);
                surf_r = clampf(surf_r + rr, 0.0f, 255.0f);
            }

            float absorb = clampf(deep_t * clampf(water_dark_mult, 0.0f, 50.0f) * 0.085f, 0.0f, 0.92f);

            float target_b = 38.0f;
            float target_g = 16.0f;
            float target_r =  6.0f;

            surf_b = lerpf(surf_b, target_b, absorb);
            surf_g = lerpf(surf_g, target_g, absorb);
            surf_r = lerpf(surf_r, target_r, absorb);

            surf_b *= lerpf(1.00f, 1.08f, deep_t);
            surf_g *= lerpf(1.00f, 0.92f, deep_t);
            surf_r *= lerpf(1.00f, 0.78f, deep_t);

            fog_mix(surf_b, surf_g, surf_r, surf_hit.t, false);
        }
    } else {
        float sky_depth_hint = lerpf(35.0f, 220.0f, clampf(1.0f - fmaxf(ray_dir.y, 0.0f), 0.0f, 1.0f));
        sky_color_at_dir(ray_dir, sky_depth_hint, visual_style, surf_b, surf_g, surf_r);
        s.depth = ray_t_max;
        s.hit_type = 0;
    }

    s.b = surf_b;
    s.g = surf_g;
    s.r = surf_r;

    if (visible_clouds_enabled) {
        float max_cloud_t = has_surface ? surf_hit.t : ray_t_max;

        CloudComposite cc = integrate_volumetric_clouds(
            ray_origin,
            ray_dir,
            max_cloud_t,
            terrain_offset_x,
            terrain_offset_z,
            cloud_wind_offset_z,
            cloud_threshold,
            sun_dir,
            visual_style,
            curvature_enabled,
            sample_x_i,
            sample_y_i
        );

        if (cc.alpha > 0.02f) {
            s.b = cc.accum_b + cc.transmittance * s.b;
            s.g = cc.accum_g + cc.transmittance * s.g;
            s.r = cc.accum_r + cc.transmittance * s.r;
        }
    }

    if (has_surface) {
        float horizon_factor = 1.0f - clampf(fabsf(ray_dir.y) / 0.22f, 0.0f, 1.0f);
        horizon_factor = smooth01(horizon_factor);

        float distance_factor = smooth01((s.depth - 18.0f) / 90.0f);
        float crest_bias = smooth01((s.depth - 28.0f) / 110.0f);

        s.is_candidate_edge =
            (horizon_factor * distance_factor > 0.06f || horizon_factor * crest_bias > 0.05f) ? 1 : 0;
    } else {
        float horizon_factor = 1.0f - clampf(fabsf(ray_dir.y) / 0.18f, 0.0f, 1.0f);
        horizon_factor = smooth01(horizon_factor);
        s.is_candidate_edge = (horizon_factor > 0.30f) ? 1 : 0;
    }

    return s;
}

// =========================
// Main render kernel
// =========================
__global__ void k_render_screen_space(
    unsigned char* out_bgr,
    int step,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float water_reflect_strength,
    float water_dark_mult,
    float3 sun_dir,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle terrain_style,
    DeviceBiomeStyle visual_style,
    int visible_clouds_enabled,
    int cloud_shadows_enabled,
    int water_reflections_enabled,
    int curvature_enabled,
    unsigned int frame_seed)
{
    (void)frame_seed;

    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;

    int blocks_x = (width  + step - 1) / step;
    int blocks_y = (height + step - 1) / step;

    if (bx >= blocks_x || by >= blocks_y) return;

    int x0 = bx * step;
    int y0 = by * step;
    int x1 = min(width,  x0 + step);
    int y1 = min(height, y0 + step);

    if (step <= 1) {
        ShadedSample s = shade_sample(
            (float)x0, (float)y0,
            terrain_offset_x, terrain_offset_z,
            cloud_wind_offset_z, cloud_threshold,
            shadow_darkness_mult, water_reflect_strength, water_dark_mult,
            sun_dir, camPos, R,
            terrain_style, visual_style,
            visible_clouds_enabled, cloud_shadows_enabled,
            water_reflections_enabled, curvature_enabled
        );

        int o = (y0 * width + x0) * 3;
        out_bgr[o + 0] = (unsigned char)clampf(s.b, 0.0f, 255.0f);
        out_bgr[o + 1] = (unsigned char)clampf(s.g, 0.0f, 255.0f);
        out_bgr[o + 2] = (unsigned char)clampf(s.r, 0.0f, 255.0f);
        return;
    }

    float jx = 0.0f;
    float jy = 0.0f;

    float center_x = clampf((float)x0 + 0.50f * step + jx * 0.18f * step, 0.0f, (float)(width  - 1));
    float center_y = clampf((float)y0 + 0.50f * step + jy * 0.18f * step, 0.0f, (float)(height - 1));

    ShadedSample sc = shade_sample(
        center_x, center_y,
        terrain_offset_x, terrain_offset_z,
        cloud_wind_offset_z, cloud_threshold,
        shadow_darkness_mult, water_reflect_strength, water_dark_mult,
        sun_dir, camPos, R,
        terrain_style, visual_style,
        visible_clouds_enabled, cloud_shadows_enabled,
        water_reflections_enabled, curvature_enabled
    );

    if (!sc.is_candidate_edge) {
        paint_block_bgr(
            out_bgr,
            bx, by, step,
            (unsigned char)clampf(sc.b, 0.0f, 255.0f),
            (unsigned char)clampf(sc.g, 0.0f, 255.0f),
            (unsigned char)clampf(sc.r, 0.0f, 255.0f)
        );
        return;
    }

    float x_left   = clampf((float)x0 + 0.25f * step + jx * 0.20f * step, 0.0f, (float)(width  - 1));
    float x_right  = clampf((float)x0 + 0.75f * step + jx * 0.20f * step, 0.0f, (float)(width  - 1));
    float y_top    = clampf((float)y0 + 0.25f * step + jy * 0.20f * step, 0.0f, (float)(height - 1));
    float y_bottom = clampf((float)y0 + 0.75f * step + jy * 0.20f * step, 0.0f, (float)(height - 1));

    ShadedSample s00 = shade_sample(
        x_left, y_top,
        terrain_offset_x, terrain_offset_z,
        cloud_wind_offset_z, cloud_threshold,
        shadow_darkness_mult, water_reflect_strength, water_dark_mult,
        sun_dir, camPos, R,
        terrain_style, visual_style,
        visible_clouds_enabled, cloud_shadows_enabled,
        water_reflections_enabled, curvature_enabled
    );

    ShadedSample s10 = shade_sample(
        x_right, y_top,
        terrain_offset_x, terrain_offset_z,
        cloud_wind_offset_z, cloud_threshold,
        shadow_darkness_mult, water_reflect_strength, water_dark_mult,
        sun_dir, camPos, R,
        terrain_style, visual_style,
        visible_clouds_enabled, cloud_shadows_enabled,
        water_reflections_enabled, curvature_enabled
    );

    ShadedSample s01 = shade_sample(
        x_left, y_bottom,
        terrain_offset_x, terrain_offset_z,
        cloud_wind_offset_z, cloud_threshold,
        shadow_darkness_mult, water_reflect_strength, water_dark_mult,
        sun_dir, camPos, R,
        terrain_style, visual_style,
        visible_clouds_enabled, cloud_shadows_enabled,
        water_reflections_enabled, curvature_enabled
    );

    ShadedSample s11 = shade_sample(
        x_right, y_bottom,
        terrain_offset_x, terrain_offset_z,
        cloud_wind_offset_z, cloud_threshold,
        shadow_darkness_mult, water_reflect_strength, water_dark_mult,
        sun_dir, camPos, R,
        terrain_style, visual_style,
        visible_clouds_enabled, cloud_shadows_enabled,
        water_reflections_enabled, curvature_enabled
    );

    float dmin = fminf(fminf(s00.depth, s10.depth), fminf(s01.depth, s11.depth));
    float dmax = fmaxf(fmaxf(s00.depth, s10.depth), fmaxf(s01.depth, s11.depth));

    int mixed_hit_type =
        (s00.hit_type != s10.hit_type) ||
        (s00.hit_type != s01.hit_type) ||
        (s00.hit_type != s11.hit_type);

    float depth_span = dmax - dmin;

    float horizon_boost = 0.0f;
    {
        float cx = (float)x0 + 0.5f * step;
        float cy = (float)y0 + 0.5f * step;
        float3 block_ray = screen_ray_dir(cx, cy, R);
        float h = 1.0f - clampf(fabsf(block_ray.y) / 0.18f, 0.0f, 1.0f);
        horizon_boost = smooth01(h);
    }

    float edge_threshold = lerpf(1.5f, 14.0f, clampf(dmin / 220.0f, 0.0f, 1.0f));
    edge_threshold = lerpf(edge_threshold, edge_threshold * 0.35f, horizon_boost);

    bool edge_block =
        mixed_hit_type ||
        (depth_span > edge_threshold) ||
        s00.is_candidate_edge ||
        s10.is_candidate_edge ||
        s01.is_candidate_edge ||
        s11.is_candidate_edge;

    if (!edge_block) {
        float final_b = 0.25f * (s00.b + s10.b + s01.b + s11.b);
        float final_g = 0.25f * (s00.g + s10.g + s01.g + s11.g);
        float final_r = 0.25f * (s00.r + s10.r + s01.r + s11.r);

        paint_block_bgr(
            out_bgr,
            bx, by, step,
            (unsigned char)clampf(final_b, 0.0f, 255.0f),
            (unsigned char)clampf(final_g, 0.0f, 255.0f),
            (unsigned char)clampf(final_r, 0.0f, 255.0f)
        );
        return;
    }

    bool coherent_block = (!mixed_hit_type && depth_span <= edge_threshold * 1.75f);

    if (coherent_block) {
        for (int y = y0; y < y1; ++y) {
            float fy = (y1 - y0 > 1) ? ((float)(y - y0) / (float)(y1 - y0 - 1)) : 0.5f;
            int row = y * width * 3;

            for (int x = x0; x < x1; ++x) {
                float fx = (x1 - x0 > 1) ? ((float)(x - x0) / (float)(x1 - x0 - 1)) : 0.5f;

                float w00 = (1.0f - fx) * (1.0f - fy);
                float w10 = fx * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                float pb = s00.b * w00 + s10.b * w10 + s01.b * w01 + s11.b * w11;
                float pg = s00.g * w00 + s10.g * w10 + s01.g * w01 + s11.g * w11;
                float pr = s00.r * w00 + s10.r * w10 + s01.r * w01 + s11.r * w11;

                int o = row + x * 3;
                out_bgr[o + 0] = (unsigned char)clampf(pb, 0.0f, 255.0f);
                out_bgr[o + 1] = (unsigned char)clampf(pg, 0.0f, 255.0f);
                out_bgr[o + 2] = (unsigned char)clampf(pr, 0.0f, 255.0f);
            }
        }
        return;
    }

    for (int y = y0; y < y1; ++y) {
        int row = y * width * 3;

        for (int x = x0; x < x1; ++x) {
            ShadedSample sp = shade_sample(
                (float)x, (float)y,
                terrain_offset_x, terrain_offset_z,
                cloud_wind_offset_z, cloud_threshold,
                shadow_darkness_mult, water_reflect_strength, water_dark_mult,
                sun_dir, camPos, R,
                terrain_style, visual_style,
                visible_clouds_enabled, cloud_shadows_enabled,
                water_reflections_enabled, curvature_enabled
            );

            int o = row + x * 3;
            out_bgr[o + 0] = (unsigned char)clampf(sp.b, 0.0f, 255.0f);
            out_bgr[o + 1] = (unsigned char)clampf(sp.g, 0.0f, 255.0f);
            out_bgr[o + 2] = (unsigned char)clampf(sp.r, 0.0f, 255.0f);
        }
    }
}

// =========================
// Camera target height kernel
// =========================
__global__ void k_sample_camera_target_height(
    float* out_avg_height,
    float terrain_offset_x,
    float terrain_offset_z,
    DeviceBiomeStyle terrain_style)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float camera_local_x = 0.0f;
    float camera_local_z = 5.0f;

    float world_x0 = camera_local_x + terrain_offset_x;
    float world_z0 = camera_local_z + terrain_offset_z;

    // Use coarse terrain mass for camera follow stability.
    float h0 = terrain_base_height_biomes_coarse(world_x0, world_z0, terrain_style);
    if (h0 < 0.0f) h0 = 0.0f;

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;

    for (int i = 0; i < camera_height_sample_count; ++i) {
        float sample_local_z = camera_local_z
            + camera_height_lookahead_start
            + (float)i * camera_height_lookahead_step;

        float world_x = camera_local_x + terrain_offset_x;
        float world_z = sample_local_z + terrain_offset_z;

        float h = terrain_base_height_biomes_coarse(world_x, world_z, terrain_style);
        if (h < 0.0f) h = 0.0f;

        float t = (camera_height_sample_count > 1)
            ? (float)i / (float)(camera_height_sample_count - 1)
            : 0.0f;

        float w = lerpf(2.25f, 0.65f, t);

        weighted_sum += h * w;
        weight_total += w;
    }

    float h_ahead = (weight_total > 0.0f) ? (weighted_sum / weight_total) : h0;
    *out_avg_height = h0 * 0.60f + h_ahead * 0.40f;
}

// =========================
// Host rotation matrix
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
// Permutation init
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
// FFmpeg pipe
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
    if (!pipe) throw std::runtime_error("Could not open FFmpeg pipe.");
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
// Main
// =========================
int main(int argc, char** argv){
    unsigned int noise_seed = init_permutation_cuda();
    std::cout << "Noise seed: " << noise_seed << "\n";

    int duration_seconds = 60;
    std::string output_file = make_default_output_filename();

    if (argc >= 2)  duration_seconds = std::max(1, std::atoi(argv[1]));
    if (argc >= 3)  output_file = argv[2];
    if (argc >= 4)  forward_speed = std::max(0.0f, static_cast<float>(std::atof(argv[3])));
    if (argc >= 5)  sample_step = std::max(1, std::atoi(argv[4]));
    if (argc >= 6)  shadow_darkness_multiplier = std::max(0.05f, static_cast<float>(std::atof(argv[5])));
    if (argc >= 7)  water_reflection_strength = clamp_host(static_cast<float>(std::atof(argv[6])), 0.0f, 1.0f);
    if (argc >= 8)  terrain_height_multiplier = std::max(0.0f, static_cast<float>(std::atof(argv[7])));
    if (argc >= 9)  camera_y_offset = static_cast<float>(std::atof(argv[8]));
    if (argc >= 10) render_visible_clouds = std::atoi(argv[9]) ? 1 : 0;
    if (argc >= 11) render_cloud_shadows = std::atoi(argv[10]) ? 1 : 0;
    if (argc >= 12) render_water_reflections = std::atoi(argv[11]) ? 1 : 0;
    if (argc >= 13) enable_planet_curvature = std::atoi(argv[12]) ? 1 : 0;
    if (argc >= 14) water_deep_darkness_multiplier = std::max(0.0f, static_cast<float>(std::atof(argv[13])));

    CUDA_CHECK(cudaMemcpyToSymbol(d_terrain_height_multiplier, &terrain_height_multiplier, sizeof(float)));

    camera_position = cv::Vec3f(0.0f, 1.2f, 5.0f);
    const long long total_frames = (long long)duration_seconds * output_fps;

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
    std::cout << "Sample step: " << sample_step << " (" << sample_step << "x" << sample_step << ")\n";
    std::cout << "Shadow darkness multiplier: " << shadow_darkness_multiplier << "\n";
    std::cout << "Water reflection strength: " << water_reflection_strength << "\n";
    std::cout << "Water deep darkness multiplier: " << water_deep_darkness_multiplier << "\n";
    std::cout << "Terrain height multiplier: " << terrain_height_multiplier << "\n";
    std::cout << "Camera Y offset: " << camera_y_offset << "\n";
    std::cout << "Visible clouds: " << (render_visible_clouds ? "on" : "off") << "\n";
    std::cout << "Cloud shadows: " << (render_cloud_shadows ? "on" : "off") << "\n";
    std::cout << "Water reflections: " << (render_water_reflections ? "on" : "off") << "\n";
    std::cout << "Planet curvature: " << (enable_planet_curvature ? "on" : "off")
              << " (radius=" << planet_radius << ")\n";

    unsigned char* d_bgr = nullptr;
    float* d_camera_avg_height = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bgr, (size_t)width * height * 3));
    CUDA_CHECK(cudaMalloc(&d_camera_avg_height, sizeof(float)));

    cv::Mat frame(height, width, CV_8UC3);

    float g_terrain_offset_x = 0.f;
    float g_terrain_offset_z = 0.f;
    float cloud_wind_offset_z = 0.f;
    float texture_rotation_angle = 0.f;

    FILE* ffmpeg = nullptr;

    try {
        ffmpeg = open_ffmpeg_nvenc_pipe(output_file);

        int blocks_x = (width  + sample_step - 1) / sample_step;
        int blocks_y = (height + sample_step - 1) / sample_step;

        dim3 block2d(16, 16);
        dim3 grid2d(
            (blocks_x + block2d.x - 1) / block2d.x,
            (blocks_y + block2d.y - 1) / block2d.y
        );

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

            BiomeStyle blended_visual_style = biome_lerp(current_style, next_style, biome_t);
            BiomeStyle frozen_terrain_style = current_style;

            DeviceBiomeStyle terrain_style{
                frozen_terrain_style.global_temperature,
                frozen_terrain_style.global_humidity,
                frozen_terrain_style.global_ruggedness,
                frozen_terrain_style.global_snowline,
                frozen_terrain_style.cloudiness
            };

            DeviceBiomeStyle visual_style{
                blended_visual_style.global_temperature,
                blended_visual_style.global_humidity,
                blended_visual_style.global_ruggedness,
                blended_visual_style.global_snowline,
                blended_visual_style.cloudiness
            };

            g_terrain_offset_z += forward_speed;
            cloud_wind_offset_z += forward_speed * cloud_wind_speed_factor;

            float cos_rot = std::cos(texture_rotation_angle);
            float sin_rot = std::sin(texture_rotation_angle);

            float rotated_x = g_terrain_offset_x * cos_rot - g_terrain_offset_z * sin_rot;
            float rotated_z = g_terrain_offset_x * sin_rot + g_terrain_offset_z * cos_rot;

            k_sample_camera_target_height<<<1, 1>>>(
                d_camera_avg_height,
                rotated_x,
                rotated_z,
                terrain_style
            );
            CUDA_CHECK(cudaGetLastError());

            float h_camera_avg = 0.0f;
            CUDA_CHECK(cudaMemcpy(&h_camera_avg, d_camera_avg_height, sizeof(float), cudaMemcpyDeviceToHost));

            float desired_camera_y = h_camera_avg + camera_height_clearance_base + camera_y_offset;
            desired_camera_y = std::max(desired_camera_y, camera_min_height);

            camera_position[1] = lerp_host(
                camera_position[1],
                desired_camera_y,
                camera_height_smoothness
            );

            cv::Matx33f Rhost = get_rotation_matrix_host(camera_rotation);
            Mat3 R = to_mat3_rowmajor(Rhost);
            float3 camPos = make_float3(camera_position[0], camera_position[1], camera_position[2]);

            float dynamic_cloud_threshold =
                clamp_host(
                    cloud_threshold_min + lerp_host(0.10f, -0.05f, blended_visual_style.cloudiness),
                    0.05f, 0.30f
                );

            unsigned int frame_seed = hash_u32((unsigned int)frame_idx ^ noise_seed ^ 0x9E3779B9u);

            k_render_screen_space<<<grid2d, block2d>>>(
                d_bgr,
                sample_step,
                rotated_x,
                rotated_z,
                cloud_wind_offset_z,
                dynamic_cloud_threshold,
                shadow_darkness_multiplier,
                water_reflection_strength,
                water_deep_darkness_multiplier,
                sun_dir,
                camPos,
                R,
                terrain_style,
                visual_style,
                render_visible_clouds,
                render_cloud_shadows,
                render_water_reflections,
                enable_planet_curvature,
                frame_seed
            );
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(frame.data, d_bgr, (size_t)width * height * 3, cudaMemcpyDeviceToHost));

            write_frame_to_pipe(ffmpeg, frame);

            if (preview_interval_frames > 0 && frame_idx > 0 && (frame_idx % preview_interval_frames) == 0) {
                show_preview_window(frame, frame_idx);
            }

            if ((frame_idx % output_fps) == 0) {
                double pct = 100.0 * (double)frame_idx / (double)total_frames;
                std::cout << "\rFrame " << frame_idx << " / " << total_frames
                          << "  (" << (int)pct << "%)"
                          << " biomeFade=" << std::fixed << std::setprecision(2) << biome_t
                          << " camY=" << std::fixed << std::setprecision(2) << camera_position[1]
                          << std::flush;
            }
        }

        std::cout << "\rFrame " << total_frames << " / " << total_frames << "  (100%)\n";

        int rc = pclose(ffmpeg);
        ffmpeg = nullptr;

        if (rc != 0) {
            std::cerr << "FFmpeg exited with code: " << rc << "\n";
            CUDA_CHECK(cudaFree(d_camera_avg_height));
            CUDA_CHECK(cudaFree(d_bgr));
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
        CUDA_CHECK(cudaFree(d_camera_avg_height));
        CUDA_CHECK(cudaFree(d_bgr));
        cv::destroyAllWindows();
        return 1;
    }

    CUDA_CHECK(cudaFree(d_camera_avg_height));
    CUDA_CHECK(cudaFree(d_bgr));
    cv::destroyAllWindows();
    return 0;
}
