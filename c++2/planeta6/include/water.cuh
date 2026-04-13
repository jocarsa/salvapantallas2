#pragma once
#include "config.hpp"
#include "types.hpp"
#include "cuda_common.cuh"
#include "color.cuh"

__device__ __forceinline__ float reflection_step_size(float t){
    float nt = clampf(t / cfg::water_max_reflection_dist, 0.0f, 1.0f);
    return lerpf(cfg::water_reflection_step_min, cfg::water_reflection_step_max, nt);
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
    float cloud_wind_offset_z,
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
    (void)mods;
    (void)near_subdivs;

    float wave_x = fractal_noise2(
        world_x * cfg::water_wave_noise_scale + 401.0f,
        world_z * cfg::water_wave_noise_scale + 97.0f
    );

    float wave_z = fractal_noise2(
        world_x * cfg::water_wave_noise_scale + 211.0f,
        world_z * cfg::water_wave_noise_scale + 501.0f
    );

    float water_surface_y = curved_surface_y(0.0f, local_x, local_z);
    float3 base_n = curvature_normal_local(local_x, local_z);

    float3 water_n = normalize3(add3(
        mul3(base_n, 1.0f),
        make_float3(
            wave_x * cfg::water_wave_distort,
            0.0f,
            wave_z * cfg::water_wave_distort
        )
    ));

    float3 p = make_float3(local_x, water_surface_y, local_z);
    float3 v_to_eye = normalize3(sub3(camPos, p));
    float3 incident = mul3(v_to_eye, -1.0f);
    float3 refl_dir = normalize3(reflect3(incident, water_n));

    if (refl_dir.y < 0.01f) refl_dir.y = 0.01f;
    refl_dir = normalize3(refl_dir);

    float ndotv = clampf(dot3(water_n, v_to_eye), 0.0f, 1.0f);
    float fresnel = cfg::water_fresnel_bias + (1.0f - cfg::water_fresnel_bias) * powf(1.0f - ndotv, cfg::water_fresnel_power);
    fresnel = clampf(fresnel, 0.0f, 1.0f);

    float accum_cloud_alpha = 0.0f;
    float cloud_b = 0.0f, cloud_g = 0.0f, cloud_r = 0.0f;

    bool hit_terrain = false;
    float hit_b = 0.0f, hit_g = 0.0f, hit_r = 0.0f;
    float traced_dist = 0.0f;

    float3 pos = add3(p, mul3(refl_dir, 0.40f));

    #pragma unroll 1
    for (int s = 0; s < cfg::water_reflection_steps; ++s) {
        float step_len = reflection_step_size(traced_dist);
        pos = add3(pos, mul3(refl_dir, step_len));
        traced_dist += step_len;

        if (traced_dist > cfg::water_max_reflection_dist) break;

        float sample_world_x = pos.x + terrain_offset_x;
        float sample_world_z = pos.z + terrain_offset_z;

        float terrain_h = terrain_base_height_biomes(sample_world_x, sample_world_z, style);
        float terrain_y_curved = curved_surface_y(terrain_h > 0.0f ? terrain_h : 0.0f, pos.x, pos.z);

        if (fabsf(pos.y - curved_surface_y(cfg::cloud_height, pos.x, pos.z)) < 2.2f) {
            float c = cloud_density_at(sample_world_x, sample_world_z, cloud_wind_offset_z, style);
            float ca = clampf((c - cloud_threshold) / 0.25f, 0.0f, 1.0f);
            if (ca > 0.001f) {
                float cb = lerpf(215.f, 242.f, ca);
                float cg = lerpf(220.f, 244.f, ca);
                float cr = lerpf(240.f, 255.f, ca);

                float remain = 1.0f - accum_cloud_alpha;
                cloud_b += cb * ca * remain;
                cloud_g += cg * ca * remain;
                cloud_r += cr * ca * remain;
                accum_cloud_alpha += ca * 0.50f * remain;
                accum_cloud_alpha = clampf(accum_cloud_alpha, 0.0f, 1.0f);
            }
        }

        if (terrain_h > 0.0f && pos.y <= terrain_y_curved) {
            float hit_world_x = sample_world_x;
            float hit_world_z = sample_world_z;
            float hit_h = terrain_h;
            float3 n = terrain_normal_fast_curved(hit_world_x, hit_world_z, pos.x, pos.z, style, cfg::normal_eps_reflection);

            terrain_color_lit_from_normal(
                hit_world_x, hit_world_z, hit_h, false,
                n, sun_dir, cloud_wind_offset_z, cloud_threshold,
                shadow_darkness_mult, style,
                false,
                hit_b, hit_g, hit_r
            );

            fog_mix(hit_b, hit_g, hit_r, depth_to_water + traced_dist, false);
            hit_terrain = true;
            break;
        }
    }

    float sky_b, sky_g, sky_r;
    sky_color_at_dir(refl_dir, depth_to_water + traced_dist, style, sky_b, sky_g, sky_r);

    float rb = sky_b * cfg::water_reflection_sky_gain;
    float rg = sky_g * cfg::water_reflection_sky_gain;
    float rr = sky_r * cfg::water_reflection_sky_gain;

    if (accum_cloud_alpha > 0.001f) {
        rb = lerpf(rb, cloud_b, clampf(accum_cloud_alpha * cfg::water_reflection_cloud_gain, 0.0f, 1.0f));
        rg = lerpf(rg, cloud_g, clampf(accum_cloud_alpha * cfg::water_reflection_cloud_gain, 0.0f, 1.0f));
        rr = lerpf(rr, cloud_r, clampf(accum_cloud_alpha * cfg::water_reflection_cloud_gain, 0.0f, 1.0f));
    }

    if (hit_terrain) {
        rb = lerpf(rb, hit_b, cfg::water_reflection_terrain_gain);
        rg = lerpf(rg, hit_g, cfg::water_reflection_terrain_gain);
        rr = lerpf(rr, hit_r, cfg::water_reflection_terrain_gain);
    }

    float sun_glint = powf(clampf(dot3(v_to_eye, mul3(sun_dir, -1.0f)), 0.0f, 1.0f), 28.0f);
    rb += 255.0f * cfg::water_specular_strength * sun_glint;
    rg += 245.0f * cfg::water_specular_strength * sun_glint;
    rr += 230.0f * cfg::water_specular_strength * sun_glint;

    float final_mix = clampf(reflection_strength * fresnel, 0.0f, 1.0f);

    out_b = clampf(rb * final_mix, 0.0f, 255.0f);
    out_g = clampf(rg * final_mix, 0.0f, 255.0f);
    out_r = clampf(rr * final_mix, 0.0f, 255.0f);
}
