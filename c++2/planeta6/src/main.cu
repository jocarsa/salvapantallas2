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

#include "config.hpp"
#include "types.hpp"
#include "biome.hpp"
#include "host_utils.hpp"
#include "preview.hpp"
#include "ffmpeg_writer.hpp"

#include "cuda_common.cuh"
#include "noise.cuh"
#include "curvature.cuh"
#include "sampling.cuh"
#include "projection.cuh"
#include "color.cuh"
#include "water.cuh"
#include "render_kernels.cuh"

// Must match declaration in include/host_utils.hpp
unsigned int init_permutation_cuda() {
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

static float deg2rad_host(float d){
    return d * 3.14159265358979323846f / 180.0f;
}

int main(int argc, char** argv) {
    using namespace cfg;

    unsigned int noise_seed = init_permutation_cuda();
    std::cout << "Noise seed: " << noise_seed << "\n";

    int duration_seconds = 60;
    std::string output_file = make_default_output_filename();

    if (argc >= 2)  duration_seconds = std::max(1, std::atoi(argv[1]));
    if (argc >= 3)  output_file = argv[2];
    if (argc >= 4)  forward_speed = std::max(0.0f, static_cast<float>(std::atof(argv[3])));
    if (argc >= 5)  radius_multiplier = std::max(0.1f, static_cast<float>(std::atof(argv[4])));
    if (argc >= 6)  near_subdivisions = std::max(32, std::atoi(argv[5]));
    if (argc >= 7)  shadow_darkness_multiplier = std::max(0.05f, static_cast<float>(std::atof(argv[6])));
    if (argc >= 8)  water_reflection_strength = clamp_host(static_cast<float>(std::atof(argv[7])), 0.0f, 1.0f);
    if (argc >= 9)  terrain_height_multiplier = std::max(0.0f, static_cast<float>(std::atof(argv[8])));
    if (argc >= 10) camera_y_offset = static_cast<float>(std::atof(argv[9]));

    CUDA_CHECK(cudaMemcpyToSymbol(
        d_terrain_height_multiplier,
        &terrain_height_multiplier,
        sizeof(float)
    ));

    camera_position = cv::Vec3f(0.0f, 1.2f, 1.0f);
    camera_rotation = cv::Vec3f(0.0f, 0.0f, 0.0f);

    const long long total_frames = (long long)duration_seconds * output_fps;
    const int pixel_count = width * height;

    const int terrain_rows_runtime = compute_inverse_perspective_row_count(terrain_sample_near_z, plane_depth, near_subdivisions);
    const int sky_rows_runtime     = compute_inverse_perspective_row_count(sky_sample_near_z, sky_plane_depth, near_subdivisions);
    const int cloud_rows_runtime   = terrain_rows_runtime;

    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Frames: " << total_frames << "\n";
    std::cout << "Terrain rows: " << terrain_rows_runtime << "\n";
    std::cout << "Sky rows: " << sky_rows_runtime << "\n";
    std::cout << "Near subdivisions: " << near_subdivisions << "\n";

    unsigned long long* d_packed_fb = nullptr;
    unsigned char* d_bgr = nullptr;
    float* d_depth = nullptr;
    float* d_camera_target_height = nullptr;

    CUDA_CHECK(cudaMalloc(&d_packed_fb, (size_t)pixel_count * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_bgr, (size_t)pixel_count * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_depth, (size_t)pixel_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_camera_target_height, sizeof(float)));

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat depth_map(height, width, CV_32FC1);
    cv::Mat prev_frame;

    FILE* ffmpeg = open_ffmpeg_nvenc_pipe(output_file);
    if (!ffmpeg) {
        CUDA_CHECK(cudaFree(d_camera_target_height));
        CUDA_CHECK(cudaFree(d_depth));
        CUDA_CHECK(cudaFree(d_bgr));
        CUDA_CHECK(cudaFree(d_packed_fb));
        throw std::runtime_error("Could not open FFmpeg pipe");
    }

    const unsigned long long clear_value =
        (((unsigned long long)((unsigned int)(9999.0f * 1024.0f))) << 24);

    TerrainModification* d_mods = nullptr;
    const int mods_count = terrain_rows_runtime * near_subdivisions;
    CUDA_CHECK(cudaMalloc(&d_mods, (size_t)mods_count * sizeof(TerrainModification)));

    std::vector<TerrainModification> init_mods((size_t)mods_count);
    for (int i = 0; i < mods_count; ++i) {
        init_mods[(size_t)i].value = 0.0f;
        init_mods[(size_t)i].persistence = 0.985f;
    }
    CUDA_CHECK(cudaMemcpy(
        d_mods,
        init_mods.data(),
        (size_t)mods_count * sizeof(TerrainModification),
        cudaMemcpyHostToDevice
    ));

    const float sun_az = deg2rad_host(sun_azimuth_deg);
    const float sun_el = deg2rad_host(sun_elevation_deg);

    float3 sun_dir = normalize3_host(make_float3(
        std::cos(sun_el) * std::sin(sun_az),
        -std::sin(sun_el),
        std::cos(sun_el) * std::cos(sun_az)
    ));

    cv::Vec3f prev_camera_position = camera_position;
    float prev_rotated_x = 0.0f;
    float prev_rotated_z = 0.0f;

    std::mt19937 biome_rng(noise_seed ^ 0x9E3779B9u);
    BiomeStyle biome_current = random_biome_style(biome_rng);
    BiomeStyle biome_next    = random_biome_style(biome_rng);

    const dim3 block2d(16, 16);
    const dim3 gridSky(
        (unsigned int)((width + block2d.x - 1) / block2d.x),
        (unsigned int)((height + block2d.y - 1) / block2d.y)
    );

    const dim3 blockRows(16, 16);
    const dim3 gridTerrain(
        (unsigned int)((near_subdivisions + blockRows.x - 1) / blockRows.x),
        (unsigned int)((terrain_rows_runtime + blockRows.y - 1) / blockRows.y)
    );
    const dim3 gridSkyRows(
        (unsigned int)((near_subdivisions + blockRows.x - 1) / blockRows.x),
        (unsigned int)((sky_rows_runtime + blockRows.y - 1) / blockRows.y)
    );
    const dim3 gridCloudRows(
        (unsigned int)((near_subdivisions + blockRows.x - 1) / blockRows.x),
        (unsigned int)((cloud_rows_runtime + blockRows.y - 1) / blockRows.y)
    );

    const int threads1d = 256;
    const int blocksPixels = (pixel_count + threads1d - 1) / threads1d;
    const int blocksMods = (mods_count + threads1d - 1) / threads1d;

    for (long long frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
        const float time_sec = (float)frame_idx / (float)output_fps;

        // biome timeline
        {
            const int slot_total = biome_slot_seconds;
            const int hold = biome_hold_seconds;
            const int transition = std::max(1, biome_transition_seconds);

            int slot_idx = (int)(time_sec / (float)slot_total);
            float slot_time = std::fmod(time_sec, (float)slot_total);

            if (slot_idx > 0 && slot_time < (1.0f / (float)output_fps)) {
                biome_current = biome_next;
                biome_next = random_biome_style(biome_rng);
            }

            float t = 0.0f;
            if (slot_time > (float)hold) {
                t = clamp_host((slot_time - (float)hold) / (float)transition, 0.0f, 1.0f);
            }

            biome_current = biome_lerp(biome_current, biome_next, t);
        }

        DeviceBiomeStyle device_biome{};
        device_biome.global_temperature = biome_current.global_temperature;
        device_biome.global_humidity    = biome_current.global_humidity;
        device_biome.global_ruggedness  = biome_current.global_ruggedness;
        device_biome.global_snowline    = biome_current.global_snowline;
        device_biome.cloudiness         = biome_current.cloudiness;

        const float rotated_x = 0.0f;
        const float rotated_z = time_sec * forward_speed;

        // CAMBIO: las nubes usan el mismo desplazamiento longitudinal
        // que el terreno y la cámara, evitando que aparenten ir en
        // dirección opuesta o a velocidad distinta.
        const float cloud_wind_offset_z = rotated_z;
        const float cloud_threshold = cloud_threshold_min;

        const float terrain_offset_x = rotated_x;
        const float terrain_offset_z = rotated_z;

        // camera target height
        k_sample_camera_target_height<<<1, 1>>>(
            d_camera_target_height,
            d_mods,
            near_subdivisions,
            terrain_offset_x,
            terrain_offset_z,
            device_biome,
            0.0f,
            0.0f
        );
        CUDA_CHECK(cudaGetLastError());

        float target_ground_y = 0.0f;
        CUDA_CHECK(cudaMemcpy(
            &target_ground_y,
            d_camera_target_height,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        const float target_camera_y = std::max(
            camera_min_height,
            target_ground_y + camera_height_clearance_base + camera_y_offset
        );

        camera_position[0] = rotated_x;
        camera_position[1] = lerp_host(camera_position[1], target_camera_y, camera_height_smoothness);
        camera_position[2] = rotated_z;

        cv::Matx33f Rcv = get_rotation_matrix_host(camera_rotation);
        Mat3 R = to_mat3_rowmajor(Rcv);

        // decay mods
        k_decay_mods<<<blocksMods, threads1d>>>(d_mods, mods_count);
        CUDA_CHECK(cudaGetLastError());

        // clear + background
        k_clear_packed_fb<<<blocksPixels, threads1d>>>(d_packed_fb, clear_value);
        CUDA_CHECK(cudaGetLastError());

        k_fill_sky_background<<<gridSky, block2d>>>(
            d_packed_fb,
            make_float3(camera_position[0], camera_position[1], camera_position[2]),
            R,
            device_biome
        );
        CUDA_CHECK(cudaGetLastError());

        // render sky / clouds / terrain / trees
        k_render_sky<<<gridSkyRows, blockRows>>>(
            d_packed_fb,
            sky_rows_runtime,
            near_subdivisions,
            radius_multiplier,
            make_float3(camera_position[0], camera_position[1], camera_position[2]),
            R,
            device_biome
        );
        CUDA_CHECK(cudaGetLastError());

        k_render_clouds<<<gridCloudRows, blockRows>>>(
            d_packed_fb,
            cloud_rows_runtime,
            near_subdivisions,
            terrain_offset_x,
            terrain_offset_z,
            radius_multiplier,
            cloud_wind_offset_z,
            cloud_threshold,
            make_float3(camera_position[0], camera_position[1], camera_position[2]),
            R,
            device_biome
        );
        CUDA_CHECK(cudaGetLastError());

        k_render_terrain<<<gridTerrain, blockRows>>>(
            d_packed_fb,
            d_mods,
            terrain_rows_runtime,
            near_subdivisions,
            terrain_offset_x,
            terrain_offset_z,
            radius_multiplier,
            cloud_wind_offset_z,
            cloud_threshold,
            shadow_darkness_multiplier,
            water_reflection_strength,
            sun_dir,
            make_float3(camera_position[0], camera_position[1], camera_position[2]),
            R,
            device_biome
        );
        CUDA_CHECK(cudaGetLastError());

        k_render_trees<<<gridTerrain, blockRows>>>(
            d_packed_fb,
            terrain_rows_runtime,
            near_subdivisions,
            terrain_offset_x,
            terrain_offset_z,
            cloud_wind_offset_z,
            cloud_threshold,
            shadow_darkness_multiplier,
            sun_dir,
            make_float3(camera_position[0], camera_position[1], camera_position[2]),
            R,
            device_biome
        );
        CUDA_CHECK(cudaGetLastError());

        // unpack color + depth
        k_unpack_to_bgr<<<blocksPixels, threads1d>>>(d_packed_fb, d_bgr);
        CUDA_CHECK(cudaGetLastError());

        k_unpack_to_depth_f32<<<blocksPixels, threads1d>>>(d_packed_fb, d_depth);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            frame.data,
            d_bgr,
            (size_t)pixel_count * 3 * sizeof(unsigned char),
            cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            depth_map.data,
            d_depth,
            (size_t)pixel_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // postprocess
        {
            const float motion_amount = compute_motion_amount(
                prev_rotated_x,
                prev_rotated_z,
                prev_camera_position,
                rotated_x,
                rotated_z,
                camera_position
            );

            apply_motion_blur(frame, prev_frame, motion_amount);
            apply_depth_blur(frame, depth_map);
            apply_glow(frame);
        }

        // preview
        if (preview_every_seconds > 0) {
            long long every_n = (long long)preview_every_seconds * (long long)output_fps;
            if (every_n > 0 && (frame_idx % every_n) == 0) {
                show_preview_window(frame, frame_idx);
            }
        }

        // output
        write_frame_to_pipe(ffmpeg, frame);

        prev_frame = frame.clone();
        prev_camera_position = camera_position;
        prev_rotated_x = rotated_x;
        prev_rotated_z = rotated_z;

        if ((frame_idx % output_fps) == 0) {
            double pct = 100.0 * (double)frame_idx / std::max(1.0, (double)total_frames);
            std::cout
                << "\rFrame " << frame_idx << "/" << total_frames
                << "  (" << std::fixed << std::setprecision(1) << pct << "%)"
                << std::flush;
        }
    }

    std::cout << "\n";

    if (ffmpeg) {
        fflush(ffmpeg);
        pclose(ffmpeg);
        ffmpeg = nullptr;
    }

    CUDA_CHECK(cudaFree(d_mods));
    CUDA_CHECK(cudaFree(d_camera_target_height));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_packed_fb));

    return 0;
}
