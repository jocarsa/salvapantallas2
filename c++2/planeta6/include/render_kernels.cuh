#pragma once
#include "config.hpp"
#include "types.hpp"
#include "cuda_common.cuh"
#include "sampling.cuh"
#include "projection.cuh"
#include "color.cuh"
#include "water.cuh"

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

    if (radius <= 1) {
        atomicMin((unsigned long long*)&packed_fb[v * cfg::width + u], packed);
        return;
    }

    int x0 = max(0, u - radius);
    int x1 = min(cfg::width - 1, u + radius);
    int y0 = max(0, v - radius);
    int y1 = min(cfg::height - 1, v + radius);

    int rr = radius * radius;

    for (int py = y0; py <= y1; ++py) {
        int dy = py - v;
        int dy2 = dy * dy;
        for (int px = x0; px <= x1; ++px) {
            int dx = px - u;
            if (dx * dx + dy2 > rr) continue;
            atomicMin((unsigned long long*)&packed_fb[py * cfg::width + px], packed);
        }
    }
}

__device__ __forceinline__ void splat_rect(
    unsigned long long* packed_fb,
    int x0, int y0, int x1, int y1,
    float depth, unsigned char b, unsigned char g, unsigned char r)
{
    if (x0 > x1 || y0 > y1) return;

    x0 = max(0, x0);
    y0 = max(0, y0);
    x1 = min(cfg::width  - 1, x1);
    y1 = min(cfg::height - 1, y1);

    unsigned long long packed = pack_depth_color(depth, b, g, r);

    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            atomicMin((unsigned long long*)&packed_fb[py * cfg::width + px], packed);
        }
    }
}

__global__ void k_clear_packed_fb(unsigned long long* fb, unsigned long long clear_value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = cfg::width * cfg::height;
    if (idx < n) fb[idx] = clear_value;
}

__global__ void k_unpack_to_bgr(const unsigned long long* packed_fb, unsigned char* out_bgr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = cfg::width * cfg::height;
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

__global__ void k_unpack_to_depth_f32(const unsigned long long* packed_fb, float* out_depth){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = cfg::width * cfg::height;
    if (idx >= n) return;

    unsigned long long v = packed_fb[idx];
    unsigned int depth_q = (unsigned int)(v >> 24);
    out_depth[idx] = (float)depth_q / 1024.0f;
}

__global__ void k_fill_sky_background(
    unsigned long long* packed_fb,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= cfg::width || v >= cfg::height) return;

    const float cx = 0.5f * (float)cfg::width;
    const float cy = 0.5f * (float)cfg::height;

    float x = ((float)u - cx) / cfg::focal_length;
    float y = (cy - (float)v) / cfg::focal_length;

    float3 dir_cam = normalize3(make_float3(x, y, 1.0f));

    float3 dir_world = make_float3(
        R.m[0] * dir_cam.x + R.m[3] * dir_cam.y + R.m[6] * dir_cam.z,
        R.m[1] * dir_cam.x + R.m[4] * dir_cam.y + R.m[7] * dir_cam.z,
        R.m[2] * dir_cam.x + R.m[5] * dir_cam.y + R.m[8] * dir_cam.z
    );
    dir_world = normalize3(dir_world);

    float b, g, r;
    sky_color_at_dir(dir_world, cfg::sky_plane_depth, style, b, g, r);

    packed_fb[v * cfg::width + u] = pack_depth_color(
        9999.0f,
        (unsigned char)clampf(b, 0.f, 255.f),
        (unsigned char)clampf(g, 0.f, 255.f),
        (unsigned char)clampf(r, 0.f, 255.f)
    );
}

__global__ void k_decay_mods(TerrainModification* mods, int mods_count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mods_count) return;

    float v = mods[idx].value * mods[idx].persistence;
    if (fabsf(v) < 0.01f) v = 0.f;
    mods[idx].value = v;
}

__global__ void k_sample_camera_target_height(
    float* out_avg_height,
    const TerrainModification* mods,
    int near_subdivs,
    float terrain_offset_x,
    float terrain_offset_z,
    DeviceBiomeStyle style,
    float camera_local_x,
    float camera_local_z)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float world_x0 = camera_local_x + terrain_offset_x;
    float world_z0 = camera_local_z + terrain_offset_z;

    float h0 = terrain_full_height(world_x0, world_z0, mods, -1, -1, near_subdivs, style);
    if (h0 < 0.0f) h0 = 0.0f;

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;

    for (int i = 0; i < cfg::camera_height_sample_count; ++i) {
        float sample_local_z = camera_local_z
            + cfg::camera_height_lookahead_start
            + (float)i * cfg::camera_height_lookahead_step;

        float sample_local_x = camera_local_x;

        float world_x = sample_local_x + terrain_offset_x;
        float world_z = sample_local_z + terrain_offset_z;

        float h = terrain_full_height(world_x, world_z, mods, -1, -1, near_subdivs, style);
        if (h < 0.0f) h = 0.0f;

        float t = (cfg::camera_height_sample_count > 1)
            ? (float)i / (float)(cfg::camera_height_sample_count - 1)
            : 0.0f;

        float w = lerpf(2.25f, 0.65f, t);

        weighted_sum += h * w;
        weight_total += w;
    }

    float h_ahead = (weight_total > 0.0f) ? (weighted_sum / weight_total) : h0;

    *out_avg_height = h0 * 0.60f + h_ahead * 0.40f;
}

__global__ void k_render_terrain(
    unsigned long long* packed_fb,
    const TerrainModification* mods,
    int terrain_rows_runtime,
    int near_subdivs,
    float terrain_offset_x,
    float terrain_offset_z,
    float radius_mult,
    float cloud_wind_offset_z,
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
        j, terrain_rows_runtime, cfg::terrain_sample_near_z, cfg::plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(cfg::plane_width, visible_w * cfg::terrain_visible_overscan);
    int cols_for_row = compute_cols_for_row(row_world_width, cfg::plane_width, near_subdivs);
    if (i >= cols_for_row) return;

    float x = sample_x_for_column(i, cols_for_row, row_world_width);
    x += row_stagger_offset(j, cols_for_row, row_world_width);

    x += rand_signed_from_2i(i, j, 0x1234u) * cfg::terrain_jitter_x;
    z += rand_signed_from_2i(i, j, 0x5678u) * cfg::terrain_jitter_z;

    float world_x = x + terrain_offset_x;
    float world_z = z + terrain_offset_z;

    float theoretical_y = terrain_full_height(world_x, world_z, mods, j, min(i, near_subdivs - 1), near_subdivs, style);

    float actual_y = (theoretical_y > 0.f) ? theoretical_y : 0.f;
    bool is_water = (theoretical_y < 0.f);

    // FIX:
    // Curvature must be applied in camera-local coordinates, not in world coordinates.
    float bent_y = curved_surface_y(actual_y, x, z);

    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(world_x, bent_y, world_z), u, v, depth, camPos, R)) return;

    float b, g, r;

    if (!is_water) {
        float3 n = terrain_normal_fast_curved(world_x, world_z, x, z, style, cfg::normal_eps);

        terrain_color_lit_from_normal(
            world_x, world_z, theoretical_y, false,
            n, sun_dir, cloud_wind_offset_z, cloud_threshold,
            shadow_darkness_mult, style,
            true,
            b, g, r
        );
    } else {
        float3 water_n = normalize3(add3(
            curvature_normal_local(x, z),
            make_float3(0.0f, 0.0f, 0.0f)
        ));

        biome_base_color(world_x, world_z, theoretical_y, water_n, true, style, b, g, r);

        float cloud_shadow = cloud_shadow_factor(
            world_x, actual_y, world_z, sun_dir, cloud_wind_offset_z, cloud_threshold, shadow_darkness_mult, style
        );

        float water_light = 0.62f * lerpf(0.65f, 0.90f, cloud_shadow);
        b *= water_light;
        g *= water_light;
        r *= water_light;

        float rb, rg, rr;
        water_reflection_trace(
            x, z, world_x, world_z, depth,
            mods, near_subdivs,
            terrain_offset_x, terrain_offset_z,
            cloud_wind_offset_z, cloud_threshold,
            shadow_darkness_mult,
            reflection_strength,
            sun_dir, camPos,
            style,
            rb, rg, rr
        );

        b = clampf(b + rb * 1.30f, 0.0f, 255.0f);
        g = clampf(g + rg * 1.30f, 0.0f, 255.0f);
        r = clampf(r + rr * 1.30f, 0.0f, 255.0f);

        b *= 0.92f;
        g *= 0.89f;
        r *= 0.86f;
    }

    fog_mix(b, g, r, depth, false);

    int rad = compute_radius_gpu(depth, radius_mult * cfg::terrain_radius_scale);

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
        j, sky_rows_runtime, cfg::sky_sample_near_z, cfg::sky_plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(cfg::sky_plane_width, visible_w * cfg::sky_visible_overscan);
    int cols_for_row = compute_cols_for_row(row_world_width, cfg::sky_plane_width, near_subdivs);
    if (i >= cols_for_row) return;

    float x = sample_x_for_column(i, cols_for_row, row_world_width);
    x += row_stagger_offset(j, cols_for_row, row_world_width);

    x += rand_signed_from_2i(i, j, 0x9A31u) * (cfg::sky_jitter_x_mul * cell_width_for_cols(cols_for_row, row_world_width));
    z += rand_signed_from_2i(i, j, 0x9A32u) * (cfg::sky_jitter_z_mul * depth_band_size_inverse_perspective(j, sky_rows_runtime, cfg::sky_sample_near_z, cfg::sky_plane_depth));
    x += fractal_noise2(x * 0.003f + 800.0f, z * 0.003f + 200.0f) * cfg::sky_row_warp_mul;

    float y = cfg::sky_height;
    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(x, y, z), u, v, depth, camPos, R)) return;

    float b, g, r;
    sky_color_at_dir(normalize3(make_float3(x, y, z)), depth, style, b, g, r);

    int rad = compute_radius_gpu(depth, radius_mult * cfg::sky_radius_scale);

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
    float terrain_offset_x,
    float terrain_offset_z,
    float radius_mult,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cloud_rows_runtime || i >= near_subdivs) return;

    float z = sample_depth_inverse_perspective(
        j, cloud_rows_runtime, cfg::terrain_sample_near_z, cfg::plane_depth
    );

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(cfg::plane_width, visible_w * cfg::cloud_visible_overscan);
    int cols_for_row = compute_cols_for_row(row_world_width, cfg::plane_width, near_subdivs);
    if (i >= cols_for_row) return;

    float x = sample_x_for_column(i, cols_for_row, row_world_width);
    x += row_stagger_offset(j, cols_for_row, row_world_width);

    x += rand_signed_from_2i(i, j, 0x1112u) * cfg::cloud_jitter_x;
    z += rand_signed_from_2i(i, j, 0x2223u) * cfg::cloud_jitter_z;

    // Clouds must live in the same world space as terrain.
    float world_x = x + terrain_offset_x;
    float world_z = z + terrain_offset_z;

    // IMPORTANT: sample the cloud field in WORLD coordinates, not local coordinates.
    float cloud = cloud_density_at(world_x, world_z, cloud_wind_offset_z, style);
    if (cloud < cloud_threshold) return;

    float density = clampf(
        (cloud - cloud_threshold) / fmaxf(1.0f - cloud_threshold, 1e-5f),
        0.0f,
        1.0f
    );

    float y = cfg::cloud_height;
    int u, v;
    float depth;
    if (!project_point_gpu(make_float3(world_x, y, world_z), u, v, depth, camPos, R)) return;

    float thinness = 1.0f - density;

    float brightness = lerpf(
        cfg::cloud_brightness_dense,
        cfg::cloud_brightness_thin,
        powf(thinness, cfg::cloud_brightness_curve)
    );

    float grey_bias = lerpf(
        cfg::cloud_grey_dense,
        cfg::cloud_grey_thin,
        powf(thinness, cfg::cloud_grey_curve)
    );

    float cool_bias = lerpf(
        cfg::cloud_cool_dense,
        cfg::cloud_cool_thin,
        thinness
    );

    float b = brightness * cool_bias;
    float g = brightness * grey_bias;
    float r = brightness * grey_bias;

    fog_mix(b, g, r, depth, false);

    int rad = compute_radius_gpu(
        depth,
        radius_mult * cfg::cloud_radius_scale *
        lerpf(cfg::cloud_radius_thin_mul, cfg::cloud_radius_dense_mul, density)
    );

    splat_disk(
        packed_fb, u, v, rad, depth,
        (unsigned char)clampf(b, 0.f, 255.f),
        (unsigned char)clampf(g, 0.f, 255.f),
        (unsigned char)clampf(r, 0.f, 255.f)
    );
}

__global__ void k_render_trees(
    unsigned long long* packed_fb,
    int terrain_rows_runtime,
    int near_subdivs,
    float terrain_offset_x,
    float terrain_offset_z,
    float cloud_wind_offset_z,
    float cloud_threshold,
    float shadow_darkness_mult,
    float3 sun_dir,
    float3 camPos,
    Mat3 R,
    DeviceBiomeStyle style)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= terrain_rows_runtime || i >= near_subdivs) return;
    if (!cfg::enable_trees) return;

    float z = sample_depth_inverse_perspective(
        j, terrain_rows_runtime, cfg::terrain_sample_near_z, cfg::plane_depth
    );

    if (z > cfg::tree_line_max_distance) return;

    float visible_w = visible_world_width_at_depth(z);
    float row_world_width = fminf(cfg::plane_width, visible_w * cfg::terrain_visible_overscan);
    int cols_for_row = compute_cols_for_row(row_world_width, cfg::plane_width, near_subdivs);
    if (i >= cols_for_row) return;

    float cell_w = cell_width_for_cols(cols_for_row, row_world_width);
    float band_d = depth_band_size_inverse_perspective(
        j, terrain_rows_runtime, cfg::terrain_sample_near_z, cfg::plane_depth
    );

    float x_center = sample_x_for_column(i, cols_for_row, row_world_width);
    x_center += row_stagger_offset(j, cols_for_row, row_world_width);

    float world_x_center = x_center + terrain_offset_x;
    float world_z_center = z + terrain_offset_z;

    int base_cx = (int)floorf(world_x_center / cfg::tree_cell_size);
    int base_cz = (int)floorf(world_z_center / cfg::tree_cell_size);

    for (int dz = -2; dz <= 2; ++dz) {
        for (int dx = -2; dx <= 2; ++dx) {
            int cx = base_cx + dx;
            int cz = base_cz + dz;

            TreeInstance t = make_tree_instance(cx, cz, style);
            if (!t.exists) continue;

            float local_x = t.world_x - terrain_offset_x;
            float local_z = t.world_z - terrain_offset_z;

            if (local_z <= cfg::terrain_sample_near_z || local_z >= cfg::plane_depth) continue;

            float dx_local = fabsf(local_x - x_center);
            float dz_local = fabsf(local_z - z);

            if (dx_local > cell_w * 1.65f) continue;
            if (dz_local > band_d * 1.65f) continue;

            // FIX:
            // Tree curvature must also use local coordinates.
            float bent_ground = curved_surface_y(t.ground_y, local_x, local_z);
            float bent_top    = curved_surface_y(t.ground_y + t.trunk_h, local_x, local_z);
            float bent_canopy = curved_surface_y(t.ground_y + t.trunk_h + t.canopy_r * 0.55f, local_x, local_z);

            int ub, vb, ut, vt, uc, vc;
            float db, dt, dc;

            if (!project_point_gpu(make_float3(t.world_x, bent_ground, t.world_z), ub, vb, db, camPos, R)) continue;
            if (!project_point_gpu(make_float3(t.world_x, bent_top,    t.world_z), ut, vt, dt, camPos, R)) continue;
            if (!project_point_gpu(make_float3(t.world_x, bent_canopy, t.world_z), uc, vc, dc, camPos, R)) continue;

            int trunk_h_px = max(2, abs(vb - vt));
            int trunk_w_px = max(1, trunk_h_px / 5);

            float dist_scale = clampf(1.0f - dc / cfg::tree_line_max_distance, 0.12f, 1.0f);
            int canopy_rad_px = max(
                2,
                (int)lrintf((float)trunk_h_px * lerpf(0.34f, 0.52f, dist_scale))
            );

            float tree_shadow = tree_shadow_factor_at(
                t.world_x,
                t.ground_y + t.trunk_h * 0.35f,
                t.world_z,
                sun_dir,
                style
            );

            float trunk_cloud_shadow = cloud_shadow_factor(
                t.world_x,
                t.ground_y + t.trunk_h * 0.35f,
                t.world_z,
                sun_dir,
                cloud_wind_offset_z,
                cloud_threshold,
                shadow_darkness_mult,
                style
            );

            float canopy_cloud_shadow = cloud_shadow_factor(
                t.world_x,
                t.ground_y + t.trunk_h + t.canopy_r * 0.55f,
                t.world_z,
                sun_dir,
                cloud_wind_offset_z,
                cloud_threshold,
                shadow_darkness_mult,
                style
            );

            float trunk_combined_shadow  = tree_shadow * trunk_cloud_shadow;
            float canopy_combined_shadow = tree_shadow * canopy_cloud_shadow;

            float trunk_n_dot = clampf(
                dot3(normalize3(make_float3(-sun_dir.x * 0.35f, 1.0f, -sun_dir.z * 0.35f)), mul3(sun_dir, -1.0f)),
                0.0f, 1.0f
            );

            float canopy_n_dot = clampf(
                dot3(normalize3(make_float3(-sun_dir.x * 0.20f, 1.0f, -sun_dir.z * 0.20f)), mul3(sun_dir, -1.0f)),
                0.0f, 1.0f
            );

            float trunk_light =
                cfg::ambient_light +
                cfg::diffuse_strength * powf(trunk_n_dot, 1.10f) * trunk_combined_shadow;

            float canopy_light =
                cfg::ambient_light +
                cfg::diffuse_strength * powf(canopy_n_dot, 1.05f) * canopy_combined_shadow +
                0.08f;

            trunk_light = clampf(trunk_light, 0.06f, 1.35f);
            canopy_light = clampf(canopy_light, 0.08f, 1.45f);

            float trunk_b = t.trunk_b * trunk_light;
            float trunk_g = t.trunk_g * trunk_light;
            float trunk_r = t.trunk_r_col * trunk_light;

            if (trunk_cloud_shadow < 0.98f) {
                float shadow_t = clampf((1.0f - trunk_cloud_shadow) / 0.92f, 0.0f, 1.0f);
                trunk_b *= (1.00f + 0.04f * shadow_t);
                trunk_g *= (1.00f - 0.05f * shadow_t);
                trunk_r *= (1.00f - 0.10f * shadow_t);
            }

            if (tree_shadow < 0.98f) {
                float shadow_t = 1.0f - tree_shadow;
                trunk_b *= (1.0f + cfg::tree_shadow_cool_b * shadow_t);
                trunk_g *= (1.0f + cfg::tree_shadow_cool_g * shadow_t);
                trunk_r *= (1.0f + cfg::tree_shadow_cool_r * shadow_t);
            }

            fog_mix(trunk_b, trunk_g, trunk_r, db, false);

            float trunk_draw_depth = dt + 0.02f;

            splat_rect(
                packed_fb,
                ut - trunk_w_px / 2, vt,
                ut + trunk_w_px / 2, vb,
                trunk_draw_depth,
                (unsigned char)clampf(trunk_b, 0.f, 255.f),
                (unsigned char)clampf(trunk_g, 0.f, 255.f),
                (unsigned char)clampf(trunk_r, 0.f, 255.f)
            );

            int clump_count = (int)lrintf(lerpf(
                (float)cfg::tree_canopy_clump_count_min,
                (float)cfg::tree_canopy_clump_count_max,
                tree_rand01_from_cell(cx, cz, 0xC101u)
            ));
            clump_count = max(cfg::tree_canopy_clump_count_min, min(clump_count, cfg::tree_canopy_clump_count_max));

            int detail_count = 10;

            for (int clump = 0; clump < clump_count; ++clump) {
                float angle = tree_rand01_from_cell_clump(cx, cz, clump, 0xC201u) * 6.28318530718f;
                float radial_u = tree_rand01_from_cell_clump(cx, cz, clump, 0xC202u);
                float radial = sqrtf(radial_u);

                float spread = t.canopy_r * cfg::tree_canopy_clump_spread;
                float inner  = t.canopy_r * cfg::tree_canopy_clump_inner_fill;

                float off_x_world =
                    cosf(angle) * lerpf(inner * 0.25f, spread, radial) +
                    tree_rand_signed_from_cell_clump(cx, cz, clump, 0xC203u) * t.canopy_r * 0.08f;

                float off_y_world =
                    tree_rand_signed_from_cell_clump(cx, cz, clump, 0xC204u) * t.canopy_r * 0.18f +
                    t.canopy_r * cfg::tree_canopy_clump_upward_bias;

                float clump_world_y =
                    t.ground_y + t.trunk_h + off_y_world;

                float clump_world_x = t.world_x + off_x_world;

                // FIX:
                float clump_local_x = clump_world_x - terrain_offset_x;
                float clump_local_z = t.world_z - terrain_offset_z;
                float bent_clump_y = curved_surface_y(clump_world_y, clump_local_x, clump_local_z);

                int clu, clv;
                float cld;
                if (!project_point_gpu(
                    make_float3(clump_world_x, bent_clump_y, t.world_z),
                    clu, clv, cld, camPos, R
                )) {
                    continue;
                }

                float clump_radius_world = lerpf(
                    cfg::tree_canopy_clump_radius_min,
                    cfg::tree_canopy_clump_radius_max,
                    tree_rand01_from_cell_clump(cx, cz, clump, 0xC205u)
                ) * fmaxf(t.canopy_r / fmaxf(cfg::tree_canopy_radius_min * cfg::tree_size_multiplier, 1e-5f), 0.6f);

                clump_radius_world = fminf(clump_radius_world, t.canopy_r * 0.95f);

                float radius_variation = lerpf(0.55f, 1.05f, tree_rand01_from_cell_clump(cx, cz, clump, 0xC206u));
                int clump_rad_px = max(
                    1,
                    (int)lrintf((float)canopy_rad_px * radius_variation * (clump_radius_world / fmaxf(t.canopy_r, 1e-5f)))
                );

                float clump_b = t.canopy_b;
                float clump_g = t.canopy_g;
                float clump_r = t.canopy_r_col;

                float tint = tree_rand_signed_from_cell_clump(cx, cz, clump, 0xC207u);
                clump_b += tint * 6.0f;
                clump_g += tint * 14.0f;
                clump_r += tint * 5.0f;

                float top_bias = clampf((off_y_world / fmaxf(t.canopy_r, 1e-5f)) * 0.5f + 0.5f, 0.0f, 1.0f);
                clump_b *= lerpf(0.96f, 1.05f, top_bias);
                clump_g *= lerpf(0.96f, 1.10f, top_bias);
                clump_r *= lerpf(0.96f, 1.04f, top_bias);

                float clump_cloud_shadow = cloud_shadow_factor(
                    t.world_x + off_x_world,
                    clump_world_y,
                    t.world_z,
                    sun_dir,
                    cloud_wind_offset_z,
                    cloud_threshold,
                    shadow_darkness_mult,
                    style
                );

                float clump_combined_shadow = tree_shadow * clump_cloud_shadow;

                float clump_light =
                    cfg::ambient_light +
                    cfg::diffuse_strength * powf(canopy_n_dot, 1.05f) * clump_combined_shadow +
                    0.08f;

                clump_light = clampf(clump_light, 0.08f, 1.45f);

                clump_b *= clump_light;
                clump_g *= clump_light;
                clump_r *= clump_light;

                if (clump_cloud_shadow < 0.98f) {
                    float shadow_t = clampf((1.0f - clump_cloud_shadow) / 0.92f, 0.0f, 1.0f);
                    clump_b *= (1.00f + 0.04f * shadow_t);
                    clump_g *= (1.00f - 0.05f * shadow_t);
                    clump_r *= (1.00f - 0.10f * shadow_t);
                }

                if (tree_shadow < 0.98f) {
                    float shadow_t = 1.0f - tree_shadow;
                    clump_b *= (1.0f + cfg::tree_shadow_cool_b * shadow_t);
                    clump_g *= (1.0f + cfg::tree_shadow_cool_g * shadow_t);
                    clump_r *= (1.0f + cfg::tree_shadow_cool_r * shadow_t);
                }

                fog_mix(clump_b, clump_g, clump_r, cld, false);

                splat_disk(
                    packed_fb,
                    clu, clv, clump_rad_px, cld,
                    (unsigned char)clampf(clump_b, 0.f, 255.f),
                    (unsigned char)clampf(clump_g, 0.f, 255.f),
                    (unsigned char)clampf(clump_r, 0.f, 255.f)
                );
            }

            for (int detail = 0; detail < detail_count; ++detail) {
                float angle = tree_rand01_from_cell_clump(cx, cz, detail, 0xD101u) * 6.28318530718f;
                float radial_u = tree_rand01_from_cell_clump(cx, cz, detail, 0xD102u);
                float radial = sqrtf(radial_u);

                float detail_spread = t.canopy_r * 1.82f;
                float detail_inner  = t.canopy_r * 0.50f;

                float off_x_world =
                    cosf(angle) * lerpf(detail_inner, detail_spread, radial) +
                    tree_rand_signed_from_cell_clump(cx, cz, detail, 0xD103u) * t.canopy_r * 0.06f;

                float off_y_world =
                    tree_rand_signed_from_cell_clump(cx, cz, detail, 0xD104u) * t.canopy_r * 0.26f +
                    t.canopy_r * 0.10f;

                float detail_world_y = t.ground_y + t.trunk_h + off_y_world;
                float detail_world_x = t.world_x + off_x_world;

                // FIX:
                float detail_local_x = detail_world_x - terrain_offset_x;
                float detail_local_z = t.world_z - terrain_offset_z;
                float bent_detail_y = curved_surface_y(detail_world_y, detail_local_x, detail_local_z);

                int du, dv;
                float dd;
                if (!project_point_gpu(
                    make_float3(detail_world_x, bent_detail_y, t.world_z),
                    du, dv, dd, camPos, R
                )) {
                    continue;
                }

                int detail_rad_px = max(
                    1,
                    (int)lrintf((float)canopy_rad_px * lerpf(0.22f, 0.42f, tree_rand01_from_cell_clump(cx, cz, detail, 0xD105u)))
                );

                float detail_b = t.canopy_b;
                float detail_g = t.canopy_g;
                float detail_r = t.canopy_r_col;

                float tint = tree_rand_signed_from_cell_clump(cx, cz, detail, 0xD106u);
                detail_b += tint * 8.0f;
                detail_g += tint * 18.0f;
                detail_r += tint * 7.0f;

                float brightness = lerpf(0.88f, 1.14f, tree_rand01_from_cell_clump(cx, cz, detail, 0xD107u));
                detail_b *= brightness;
                detail_g *= brightness;
                detail_r *= brightness;

                float detail_cloud_shadow = cloud_shadow_factor(
                    t.world_x + off_x_world,
                    detail_world_y,
                    t.world_z,
                    sun_dir,
                    cloud_wind_offset_z,
                    cloud_threshold,
                    shadow_darkness_mult,
                    style
                );

                float detail_combined_shadow = tree_shadow * detail_cloud_shadow;

                float detail_light =
                    cfg::ambient_light +
                    cfg::diffuse_strength * powf(canopy_n_dot, 1.05f) * detail_combined_shadow +
                    0.10f;

                detail_light = clampf(detail_light, 0.08f, 1.50f);

                detail_b *= detail_light;
                detail_g *= detail_light;
                detail_r *= detail_light;

                if (detail_cloud_shadow < 0.98f) {
                    float shadow_t = clampf((1.0f - detail_cloud_shadow) / 0.92f, 0.0f, 1.0f);
                    detail_b *= (1.00f + 0.04f * shadow_t);
                    detail_g *= (1.00f - 0.05f * shadow_t);
                    detail_r *= (1.00f - 0.10f * shadow_t);
                }

                if (tree_shadow < 0.98f) {
                    float shadow_t = 1.0f - tree_shadow;
                    detail_b *= (1.0f + cfg::tree_shadow_cool_b * shadow_t);
                    detail_g *= (1.0f + cfg::tree_shadow_cool_g * shadow_t);
                    detail_r *= (1.0f + cfg::tree_shadow_cool_r * shadow_t);
                }

                fog_mix(detail_b, detail_g, detail_r, dd, false);

                splat_disk(
                    packed_fb,
                    du, dv, detail_rad_px, dd,
                    (unsigned char)clampf(detail_b, 0.f, 255.f),
                    (unsigned char)clampf(detail_g, 0.f, 255.f),
                    (unsigned char)clampf(detail_r, 0.f, 255.f)
                );
            }
        }
    }
}
