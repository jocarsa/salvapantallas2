#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace cfg {

// output
inline constexpr int width = 1080;
inline constexpr int height = 1080;
inline constexpr int output_fps = 60;

// world
inline constexpr float plane_width = 400.0f;
inline constexpr float plane_depth = 220.0f;
inline constexpr float terrain_sample_near_z = 1.0f;
inline constexpr float sky_sample_near_z = 6.0f;

// runtime-tunable globals
extern int near_subdivisions;
extern float terrain_height_multiplier;
extern float camera_y_offset;
extern cv::Vec3f camera_position;
extern cv::Vec3f camera_rotation;
extern float forward_speed;
extern float radius_multiplier;
extern float shadow_darkness_multiplier;
extern float water_reflection_strength;

// camera follow
inline constexpr int   camera_height_sample_count    = 48;
inline constexpr float camera_height_lookahead_start = 4.0f;
inline constexpr float camera_height_lookahead_step  = 1.75f;
inline constexpr float camera_height_clearance_base  = 0.9f;
inline constexpr float camera_height_smoothness      = 0.04f;
inline constexpr float camera_min_height             = 0.35f;

// projection / world appearance
inline constexpr float focal_length = 800.0f;
inline constexpr float terrain_visible_overscan = 1.25f;
inline constexpr float cloud_visible_overscan   = 1.28f;
inline constexpr float sky_visible_overscan     = 1.28f;

// fog
inline constexpr float fog_near = 0.0f;
inline constexpr float fog_far  = 65.0f;
inline constexpr float fog_density = 0.1f;

inline constexpr float sky_fog_near = 24.0f;
inline constexpr float sky_fog_far  = 420.0f;
inline constexpr float sky_fog_density = 0.1f;

// sky
inline constexpr float sky_height = 20.0f;
inline constexpr float sky_plane_width = 1600.0f;
inline constexpr float sky_plane_depth = 900.0f;

// terrain noise
inline constexpr float noise_scale = 0.22f;
inline constexpr float noise_amplitude = 1.35f;
inline constexpr int   octaves = 5;
inline constexpr float persistence = 0.58f;
inline constexpr float lacunarity = 2.15f;
inline constexpr float macro_noise_scale = 0.0090f;
inline constexpr float macro_noise_amplitude = 3.2f;

// biome
inline constexpr float biome_temp_scale      = 0.0024f;
inline constexpr float biome_humidity_scale  = 0.0020f;
inline constexpr float biome_rugged_scale    = 0.0032f;
inline constexpr float biome_region_scale    = 0.0012f;

// clouds
inline constexpr float cloud_height = 6.0f;
inline constexpr float cloud_noise_scale = 0.070f;
inline constexpr float cloud_noise_amplitude = 1.2f;
inline constexpr float cloud_threshold_min = 0.15f;
inline constexpr float cloud_wind_speed_factor = 0.05f;

// stormy cloud appearance controls
// Dense clouds should be darker and greyer.
// Thin clouds can still catch a little more light and a subtle cool tint.
inline constexpr float cloud_brightness_dense = 118.0f;
inline constexpr float cloud_brightness_thin  = 178.0f;

inline constexpr float cloud_grey_dense = 0.68f;
inline constexpr float cloud_grey_thin  = 0.88f;

// Slight blue lift for thinner areas; dense areas stay more neutral/dark.
inline constexpr float cloud_cool_dense = 1.00f;
inline constexpr float cloud_cool_thin  = 1.04f;

// Shape controls for tonal response.
inline constexpr float cloud_brightness_curve = 0.62f;
inline constexpr float cloud_grey_curve       = 0.82f;

// Radius response: dense clouds read as heavier/broader masses.
inline constexpr float cloud_radius_dense_mul = 1.42f;
inline constexpr float cloud_radius_thin_mul  = 0.82f;

// splats
inline constexpr int max_circle_radius = 6;
inline constexpr int min_circle_radius = 1;
inline constexpr float terrain_radius_scale = 0.55f;
inline constexpr float cloud_radius_scale   = 0.55f;
inline constexpr float sky_radius_scale     = 0.56f;

// jitter
inline constexpr float terrain_jitter_x = 0.18f;
inline constexpr float terrain_jitter_z = 0.10f;
inline constexpr float cloud_jitter_x   = 0.30f;
inline constexpr float cloud_jitter_z   = 0.22f;
inline constexpr float sky_jitter_x_mul = 0.38f;
inline constexpr float sky_jitter_z_mul = 0.38f;
inline constexpr float sky_row_warp_mul = 0.30f;

// sun
inline constexpr float sun_azimuth_deg   = -45.0f;
inline constexpr float sun_elevation_deg = 16.0f;
inline constexpr float ambient_light      = 0.14f;
inline constexpr float diffuse_strength   = 1.45f;
inline constexpr float backlight_strength = 0.02f;
inline constexpr float normal_eps         = 0.30f;
inline constexpr float normal_eps_reflection = 0.45f;

// cloud shadows
inline constexpr float cloud_shadow_strength_base  = 0.62f;
inline constexpr float cloud_shadow_softness       = 0.07f;
inline constexpr float cloud_shadow_min_light_base = 0.08f;

// water
inline constexpr float water_fresnel_bias             = 0.10f;
inline constexpr float water_fresnel_power            = 3.8f;
inline constexpr float water_specular_strength        = 0.16f;
inline constexpr float water_wave_noise_scale         = 0.90f;
inline constexpr float water_wave_distort             = 0.08f;
inline constexpr float water_max_reflection_dist      = 320.0f;
inline constexpr int   water_reflection_steps         = 24;
inline constexpr float water_reflection_step_min      = 2.0f;
inline constexpr float water_reflection_step_max      = 16.0f;
inline constexpr float water_reflection_cloud_gain    = 0.65f;
inline constexpr float water_reflection_terrain_gain  = 0.90f;
inline constexpr float water_reflection_sky_gain      = 0.85f;

// trees
inline constexpr bool  enable_trees = true;

// Base spacing between tree-grid cells.
// Lower = denser possible placement.
inline constexpr float tree_cell_size = 0.12f;

// Render trees farther away.
inline constexpr float tree_line_max_distance = 210.0f;

// Looser conditions = more coverage.
inline constexpr float tree_min_terrain_height = 0.05f;
inline constexpr float tree_max_terrain_height = 5.40f;
inline constexpr float tree_min_flatness = 0.42f;
inline constexpr float tree_min_humidity = 0.16f;
inline constexpr float tree_min_valley_weight = 0.04f;

// Probability-like occupancy in [0,1].
inline constexpr float tree_density = 0.92f;

// Forest clustering.
inline constexpr float tree_cluster_strength = 1.65f;
inline constexpr float tree_cluster_scale = 0.055f;

// Base size range.
inline constexpr float tree_trunk_height_min = 0.65f;
inline constexpr float tree_trunk_height_max = 1.55f;
inline constexpr float tree_canopy_radius_min = 0.42f;
inline constexpr float tree_canopy_radius_max = 1.05f;

// Global size control.
inline constexpr float tree_size_multiplier = 0.08f;

// Clamp.
inline constexpr float tree_max_world_height = 4.50f;
inline constexpr float tree_max_world_canopy_radius = 2.20f;

// -----------------------------------------------------------------------------
// Fractal tree controls
// -----------------------------------------------------------------------------
inline constexpr bool  tree_use_pythagoras_fractal = true;

// Detail based on projected trunk size in pixels.
inline constexpr int   tree_fractal_depth_px_1 = 6;
inline constexpr int   tree_fractal_depth_px_2 = 12;
inline constexpr int   tree_fractal_depth_px_3 = 24;
inline constexpr int   tree_fractal_depth_px_4 = 40;
inline constexpr int   tree_fractal_depth_px_5 = 70;

inline constexpr int   tree_fractal_depth_min = 1;
inline constexpr int   tree_fractal_depth_max = 6;

// Branch geometry.
inline constexpr float tree_fractal_branch_angle_min_deg = 18.0f;
inline constexpr float tree_fractal_branch_angle_max_deg = 34.0f;
inline constexpr float tree_fractal_branch_ratio_min = 0.62f;
inline constexpr float tree_fractal_branch_ratio_max = 0.78f;
inline constexpr float tree_fractal_trunk_width_ratio = 0.22f;
inline constexpr float tree_fractal_branch_width_decay = 0.72f;
inline constexpr float tree_fractal_vertical_bias = 0.92f;
inline constexpr float tree_fractal_lean_max = 0.18f;

// Endpoint leaves for near trees only.
inline constexpr bool  tree_fractal_add_leaf_disks = true;
inline constexpr int   tree_fractal_leaf_min_depth = 4;
inline constexpr float tree_fractal_leaf_radius_min_mul = 0.12f;
inline constexpr float tree_fractal_leaf_radius_max_mul = 0.24f;

// Synthetic canopy footprint retained for lighting/shadow envelope.
inline constexpr float tree_fractal_shadow_radius_mul = 1.10f;

// tree shadows
inline constexpr float tree_shadow_strength = 0.26f;
inline constexpr float tree_shadow_softness = 0.85f;
inline constexpr int   tree_shadow_search_radius = 2;
inline constexpr float tree_shadow_cool_b = 0.03f;
inline constexpr float tree_shadow_cool_g = -0.03f;
inline constexpr float tree_shadow_cool_r = -0.08f;

// preview and post
inline constexpr int preview_every_seconds = 1;

// motion blur
inline constexpr bool  enable_motion_blur = true;
inline constexpr float motion_blur_strength = 3.60f;
inline constexpr float motion_blur_max_alpha = 0.40f;
inline constexpr float motion_blur_movement_scale = 2.0f;

// depth blur / DOF
inline constexpr bool  enable_depth_blur = false;
inline constexpr float depth_blur_focus_depth = 28.0f;
inline constexpr float depth_blur_focus_range = 9.0f;
inline constexpr float depth_blur_near_strength = 0.18f;
inline constexpr float depth_blur_far_strength  = 0.90f;
inline constexpr float depth_blur_mid_sigma  = 3.0f;
inline constexpr float depth_blur_strong_sigma = 8.0f;

// glow
inline constexpr bool  enable_glow = true;
inline constexpr float glow_threshold = 120.0f;
inline constexpr int   glow_blur_kernel = 0;
inline constexpr float glow_blur_sigma = 54.0f;
inline constexpr float glow_intensity = 0.5f;

// biome timing
inline constexpr int biome_slot_seconds = 60;
inline constexpr int biome_hold_seconds = 50;
inline constexpr int biome_transition_seconds = 10;

// curvature
inline constexpr bool  enable_planet_curvature = true;
inline constexpr float planet_radius = 2000.0f;

// misc
inline const cv::Scalar window_bg_color = cv::Scalar(0,0,0);

std::string make_default_output_filename();

} // namespace cfg
