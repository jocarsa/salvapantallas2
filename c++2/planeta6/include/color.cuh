#pragma once
#include "config.hpp"
#include "types.hpp"
#include "cuda_common.cuh"
#include "noise.cuh"
#include "curvature.cuh"

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

__device__ __forceinline__ float rand01_from_3i(int a, int b, int c, unsigned int seed){
    unsigned int h =
        (unsigned int)a * 73856093u ^
        (unsigned int)b * 19349663u ^
        (unsigned int)c * 83492791u ^
        seed;
    h = hash_u32(h);
    return (float)(h & 0x00FFFFFFu) / 16777215.0f;
}

__device__ __forceinline__ float rand_signed_from_3i(int a, int b, int c, unsigned int seed){
    return rand01_from_3i(a, b, c, seed) * 2.0f - 1.0f;
}

__device__ __forceinline__ void fog_mix(float& b, float& g, float& r, float depth, bool is_sky){
    float fog_factor = 0.f;
    if (!is_sky) {
        if (depth > cfg::fog_near) {
            float nd = (depth - cfg::fog_near) / (cfg::fog_far - cfg::fog_near);
            nd = clampf(nd, 0.f, 1.f);
            fog_factor = 1.f - expf(-cfg::fog_density * nd * 5.f);
        }
    } else {
        if (depth > cfg::sky_fog_near) {
            float nd = (depth - cfg::sky_fog_near) / (cfg::sky_fog_far - cfg::sky_fog_near);
            nd = clampf(nd, 0.f, 1.f);
            fog_factor = 1.f - expf(-cfg::sky_fog_density * nd * 5.f);
        }
    }

    b = b * (1.f - fog_factor) + 255.f * fog_factor;
    g = g * (1.f - fog_factor) + 255.f * fog_factor;
    r = r * (1.f - fog_factor) + 255.f * fog_factor;
}

__device__ __forceinline__ float local_temperature(float x, float z, const DeviceBiomeStyle& style){
    float t = clampf(
        0.5f + 0.5f * fractal_noise2(x * cfg::biome_temp_scale + 130.0f, z * cfg::biome_temp_scale + 210.0f),
        0.0f, 1.0f
    );
    return clampf(lerpf(t, style.global_temperature, 0.35f), 0.0f, 1.0f);
}

__device__ __forceinline__ float local_humidity(float x, float z, const DeviceBiomeStyle& style){
    float h = clampf(
        0.5f + 0.5f * fractal_noise2(x * cfg::biome_humidity_scale + 700.0f, z * cfg::biome_humidity_scale + 1200.0f),
        0.0f, 1.0f
    );
    return clampf(lerpf(h, style.global_humidity, 0.35f), 0.0f, 1.0f);
}

__device__ __forceinline__ float local_ruggedness(float x, float z, const DeviceBiomeStyle& style){
    float r = clampf(
        0.5f + 0.5f * fractal_noise2(x * cfg::biome_rugged_scale + 3000.0f, z * cfg::biome_rugged_scale + 4000.0f),
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

    float region = 0.5f + 0.5f * fractal_noise2(x * cfg::biome_region_scale + 1200.0f, z * cfg::biome_region_scale + 3400.0f);

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

__device__ __forceinline__ float terrain_base_height_biomes(float world_x, float world_z, const DeviceBiomeStyle& style){
    float fine  = fractal_noise2(world_x * cfg::noise_scale,       world_z * cfg::noise_scale);
    float macro = fractal_noise2(world_x * cfg::macro_noise_scale, world_z * cfg::macro_noise_scale);

    float detail1 = fractal_noise2(world_x * (cfg::noise_scale * 2.0f) + 400.0f,
                                   world_z * (cfg::noise_scale * 2.0f) + 900.0f);

    float detail2 = fractal_noise2(world_x * (cfg::noise_scale * 3.2f) + 1700.0f,
                                   world_z * (cfg::noise_scale * 3.2f) + 2300.0f);

    float ridged_src = fractal_noise2(world_x * (cfg::noise_scale * 1.5f) + 800.0f,
                                      world_z * (cfg::noise_scale * 1.5f) + 1200.0f);

    float ridged1 = 1.0f - fabsf(ridged_src);
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
        fine    * 0.30f +
        detail1 * 0.16f +
        detail2 * 0.05f -
        0.45f;

    float valley_shape =
        macro   * 1.55f +
        fine    * 0.78f +
        detail1 * 0.28f +
        detail2 * 0.08f -
        0.18f;

    float mountain_shape =
        macro   * 2.00f +
        ridged1 * 2.10f +
        detail1 * 0.24f +
        detail2 * 0.10f +
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

__device__ __forceinline__ float cloud_density_at(float world_x, float world_z, float cloud_wind_offset_z, const DeviceBiomeStyle& style){
    float nx = world_x * cfg::cloud_noise_scale + 100.f;
    float nz = (world_z + cloud_wind_offset_z) * cfg::cloud_noise_scale + 200.f;

    float base = fractal_noise2(nx, nz);
    float detail1 = fractal_noise2(nx * 1.8f + 71.0f, nz * 1.8f + 29.0f);

    float puffy = 1.0f - fabsf(base);
    puffy = puffy * puffy;

    float humid = local_humidity(world_x, world_z, style);

    float dense =
        base    * 0.78f +
        detail1 * 0.16f +
        puffy   * 0.18f;

    dense = lerpf(dense - 0.10f, dense + 0.24f, 0.35f * humid + 0.65f * style.cloudiness);
    return dense;
}

__device__ __forceinline__ float cloud_shadow_factor(
    float world_x, float terrain_y, float world_z,
    float3 sun_dir,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    const DeviceBiomeStyle& style)
{
    float up_y = -sun_dir.y;
    if (up_y <= 1e-5f) return 1.0f;

    float t = (cfg::cloud_height - terrain_y) / up_y;
    if (t <= 0.0f) return 1.0f;

    float xc_world = world_x + (-sun_dir.x) * t;
    float zc_world = world_z + (-sun_dir.z) * t;

    float cloud = cloud_density_at(xc_world, zc_world, cloud_wind_offset_z, style);
    float alpha = clampf((cloud - cloud_threshold) / cfg::cloud_shadow_softness, 0.0f, 1.0f);

    float strength  = clampf(cfg::cloud_shadow_strength_base * shadow_darkness_mult, 0.0f, 1.20f);
    float min_light = clampf(cfg::cloud_shadow_min_light_base / fmaxf(shadow_darkness_mult, 0.05f), 0.02f, 1.0f);

    float shadow = 1.0f - strength * alpha;
    return clampf(shadow, min_light, 1.0f);
}

__device__ __forceinline__ float3 terrain_normal_fast_curved(
    float world_x,
    float world_z,
    float local_x,
    float local_z,
    const DeviceBiomeStyle& style,
    float eps)
{
    float hL = curved_surface_y(terrain_base_height_biomes(world_x - eps, world_z,       style), local_x - eps, local_z);
    float hR = curved_surface_y(terrain_base_height_biomes(world_x + eps, world_z,       style), local_x + eps, local_z);
    float hD = curved_surface_y(terrain_base_height_biomes(world_x,       world_z - eps, style), local_x, local_z - eps);
    float hU = curved_surface_y(terrain_base_height_biomes(world_x,       world_z + eps, style), local_x, local_z + eps);

    float3 dx = make_float3(2.0f * eps, hR - hL, 0.0f);
    float3 dz = make_float3(0.0f, hU - hD, 2.0f * eps);
    float3 n  = cross3(dz, dx);
    return normalize3(n);
}

__device__ __forceinline__ float terrain_flatness_at(
    float world_x,
    float world_z,
    const DeviceBiomeStyle& style)
{
    float eps = cfg::normal_eps;

    float hL = terrain_base_height_biomes(world_x - eps, world_z, style);
    float hR = terrain_base_height_biomes(world_x + eps, world_z, style);
    float hD = terrain_base_height_biomes(world_x, world_z - eps, style);
    float hU = terrain_base_height_biomes(world_x, world_z + eps, style);

    float dx = (hR - hL) / (2.0f * eps);
    float dz = (hU - hD) / (2.0f * eps);

    float3 n = normalize3(make_float3(-dx, 1.0f, -dz));
    return clampf(n.y, 0.0f, 1.0f);
}

__device__ __forceinline__ float tree_rand01_from_cell(int cx, int cz, unsigned int seed){
    return rand01_from_2i(cx, cz, seed);
}

__device__ __forceinline__ float tree_rand01_from_cell_clump(int cx, int cz, int clump_idx, unsigned int seed){
    return rand01_from_3i(cx, cz, clump_idx, seed);
}

__device__ __forceinline__ float tree_rand_signed_from_cell_clump(int cx, int cz, int clump_idx, unsigned int seed){
    return rand_signed_from_3i(cx, cz, clump_idx, seed);
}

__device__ __forceinline__ bool tree_candidate_exists(
    int cx,
    int cz,
    const DeviceBiomeStyle& style)
{
    float base_density = clampf(cfg::tree_density, 0.0f, 1.0f);

    float px = ((float)cx + 0.5f) * cfg::tree_cell_size;
    float pz = ((float)cz + 0.5f) * cfg::tree_cell_size;

    float humid = local_humidity(px, pz, style);
    float rugged = local_ruggedness(px, pz, style);

    float cluster_noise =
        0.5f + 0.5f * fractal_noise2(
            px * cfg::tree_cluster_scale + 3100.0f,
            pz * cfg::tree_cluster_scale + 4700.0f
        );

    float cluster_boost = powf(cluster_noise, 1.6f) * (cfg::tree_cluster_strength - 1.0f);
    float density = base_density * (1.0f + cluster_boost);

    density *= lerpf(0.72f, 1.20f, humid);
    density *= lerpf(1.08f, 0.78f, rugged);
    density = clampf(density, 0.0f, 1.0f);

    return tree_rand01_from_cell(cx, cz, 0xA531u) < density;
}

__device__ __forceinline__ void tree_cell_center(int cx, int cz, float& x, float& z){
    x = ((float)cx + 0.5f) * cfg::tree_cell_size;
    z = ((float)cz + 0.5f) * cfg::tree_cell_size;
}

__device__ __forceinline__ TreeInstance make_tree_instance(
    int cx,
    int cz,
    const DeviceBiomeStyle& style)
{
    TreeInstance t{};

    if (!cfg::enable_trees) return t;
    if (!tree_candidate_exists(cx, cz, style)) return t;

    float base_x, base_z;
    tree_cell_center(cx, cz, base_x, base_z);

    float jitter_x = rand_signed_from_2i(cx, cz, 0xB111u) * cfg::tree_cell_size * 0.32f;
    float jitter_z = rand_signed_from_2i(cx, cz, 0xB222u) * cfg::tree_cell_size * 0.32f;

    t.world_x = base_x + jitter_x;
    t.world_z = base_z + jitter_z;

    float h = terrain_base_height_biomes(t.world_x, t.world_z, style);
    if (h < cfg::tree_min_terrain_height || h > cfg::tree_max_terrain_height) return t;

    float flatness = terrain_flatness_at(t.world_x, t.world_z, style);
    if (flatness < cfg::tree_min_flatness) return t;

    BiomeWeights bw = biome_weights_at(t.world_x, t.world_z, h, style);
    if (bw.humid < cfg::tree_min_humidity) return t;
    if (bw.valley < cfg::tree_min_valley_weight) return t;

    t.exists = true;
    t.ground_y = h;
    t.kind = TREE_BROADLEAF;

    float size_u = tree_rand01_from_cell(cx, cz, 0xB333u);

    float trunk_h = lerpf(cfg::tree_trunk_height_min, cfg::tree_trunk_height_max, size_u);
    float canopy_r = lerpf(
        cfg::tree_canopy_radius_min,
        cfg::tree_canopy_radius_max,
        tree_rand01_from_cell(cx, cz, 0xB444u)
    );

    trunk_h *= cfg::tree_size_multiplier;
    canopy_r *= cfg::tree_size_multiplier;

    t.trunk_h = fminf(trunk_h, cfg::tree_max_world_height);
    t.canopy_r = fminf(canopy_r, cfg::tree_max_world_canopy_radius);

    float green_jitter = tree_rand01_from_cell(cx, cz, 0xB555u);
    t.canopy_b     = lerpf(32.0f,  58.0f, green_jitter);
    t.canopy_g     = lerpf(95.0f, 150.0f, green_jitter);
    t.canopy_r_col = lerpf(26.0f,  52.0f, green_jitter);

    float trunk_jitter = tree_rand01_from_cell(cx, cz, 0xB666u);
    t.trunk_b     = lerpf(28.0f, 42.0f, trunk_jitter);
    t.trunk_g     = lerpf(44.0f, 62.0f, trunk_jitter);
    t.trunk_r_col = lerpf(58.0f, 86.0f, trunk_jitter);

    return t;
}

__device__ __forceinline__ float tree_shadow_factor_at(
    float world_x,
    float world_y,
    float world_z,
    float3 sun_dir,
    const DeviceBiomeStyle& style)
{
    if (!cfg::enable_trees) return 1.0f;

    float up_y = -sun_dir.y;
    if (up_y <= 1e-5f) return 1.0f;

    int base_cx = (int)floorf(world_x / cfg::tree_cell_size);
    int base_cz = (int)floorf(world_z / cfg::tree_cell_size);

    float best = 1.0f;

    for (int dz = -cfg::tree_shadow_search_radius; dz <= cfg::tree_shadow_search_radius; ++dz) {
        for (int dx = -cfg::tree_shadow_search_radius; dx <= cfg::tree_shadow_search_radius; ++dx) {
            int cx = base_cx + dx;
            int cz = base_cz + dz;

            TreeInstance t = make_tree_instance(cx, cz, style);
            if (!t.exists) continue;

            float canopy_y = t.ground_y + t.trunk_h + t.canopy_r * 0.65f;

            float ray_t = (canopy_y - world_y) / up_y;
            if (ray_t <= 0.0f) continue;

            float sx = world_x + (-sun_dir.x) * ray_t;
            float sz = world_z + (-sun_dir.z) * ray_t;

            float ddx = sx - t.world_x;
            float ddz = sz - t.world_z;
            float dist2 = ddx * ddx + ddz * ddz;

            float shadow_r = t.canopy_r + cfg::tree_shadow_softness;
            if (dist2 < shadow_r * shadow_r) {
                float d = sqrtf(dist2);
                float k = 1.0f - clampf(d / fmaxf(shadow_r, 1e-5f), 0.0f, 1.0f);
                float candidate = 1.0f - cfg::tree_shadow_strength * k;
                if (candidate < best) best = candidate;
            }
        }
    }

    return clampf(best, 0.0f, 1.0f);
}

__device__ __forceinline__ float snow_amount(float x, float z, float h, float3 normal, const DeviceBiomeStyle& style){
    float t = clampf(
        0.5f + 0.5f * fractal_noise2(x * cfg::biome_temp_scale + 130.0f, z * cfg::biome_temp_scale + 210.0f),
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

__device__ __forceinline__ void sky_color_at_dir(float3 dir, float depth_hint, const DeviceBiomeStyle& style, float& b, float& g, float& r){
    float up = clampf(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);

    float cold_b1 = 210.0f, cold_g1 =  95.0f, cold_r1 =  25.0f;
    float cold_b2 = 245.0f, cold_g2 = 165.0f, cold_r2 =  95.0f;

    float warm_b1 = 195.0f, warm_g1 = 105.0f, warm_r1 =  45.0f;
    float warm_b2 = 235.0f, warm_g2 = 170.0f, warm_r2 = 110.0f;

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

        float shallow_b = 150.f, shallow_g = 110.f, shallow_r =  45.f;
        float deep_b    =  70.f, deep_g    =  35.f, deep_r    =  10.f;

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
    const DeviceBiomeStyle& style,
    bool enable_cloud_shadows,
    float& b, float& g, float& r)
{
    biome_base_color(world_x, world_z, terrain_h, n, is_water, style, b, g, r);

    if (!is_water) {
        float ndotl = clampf(dot3(n, mul3(sun_dir, -1.0f)), 0.0f, 1.0f);
        float ndotl_shaped = powf(ndotl, 1.15f);
        float slope_backlight = clampf(1.0f - n.y, 0.0f, 1.0f);

        float cloud_shadow = 1.0f;
        if (enable_cloud_shadows) {
            cloud_shadow = cloud_shadow_factor(
                world_x, terrain_h > 0.0f ? terrain_h : 0.0f, world_z,
                sun_dir, cloud_wind_offset_z, cloud_threshold, shadow_darkness_mult, style
            );
        }

        float tree_shadow = tree_shadow_factor_at(
            world_x,
            terrain_h > 0.0f ? terrain_h : 0.0f,
            world_z,
            sun_dir,
            style
        );

        float combined_shadow = cloud_shadow * tree_shadow;

        float lighting =
            cfg::ambient_light +
            cfg::diffuse_strength * ndotl_shaped * combined_shadow +
            cfg::backlight_strength * slope_backlight * 0.20f;

        lighting = clampf(lighting, 0.02f, 1.55f);

        b *= lighting;
        g *= lighting;
        r *= lighting;

        if (enable_cloud_shadows && cloud_shadow < 0.75f) {
            float shadow_t = clampf((0.75f - cloud_shadow) / 0.75f, 0.0f, 1.0f);
            b *= (1.00f + 0.04f * shadow_t);
            g *= (1.00f - 0.05f * shadow_t);
            r *= (1.00f - 0.10f * shadow_t);
        }

        if (tree_shadow < 0.98f) {
            float shadow_t = clampf((1.0f - tree_shadow) / fmaxf(cfg::tree_shadow_strength, 1e-5f), 0.0f, 1.0f);
            b *= (1.0f + cfg::tree_shadow_cool_b * shadow_t);
            g *= (1.0f + cfg::tree_shadow_cool_g * shadow_t);
            r *= (1.0f + cfg::tree_shadow_cool_r * shadow_t);
        }
    }
}
