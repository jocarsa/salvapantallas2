#pragma once
#include <opencv2/opencv.hpp>

struct TerrainModification {
    float value = 0.0f;
    float persistence = 0.99f;
};

struct BiomeStyle {
    float global_temperature;
    float global_humidity;
    float global_ruggedness;
    float global_snowline;
    float cloudiness;
};

struct DeviceBiomeStyle {
    float global_temperature;
    float global_humidity;
    float global_ruggedness;
    float global_snowline;
    float cloudiness;
};

struct Mat3 {
    float m[9];
};

struct RenderState {
    float terrain_offset_x = 0.0f;
    float terrain_offset_z = 0.0f;
    float cloud_wind_offset_z = 0.0f;
    float texture_rotation_angle = 0.0f;
};

enum TreeKind : int {
    TREE_NONE = 0,
    TREE_BROADLEAF = 1,
    TREE_PINE = 2,
    TREE_SHRUB = 3
};

struct TreeInstance {
    bool exists = false;
    int kind = TREE_NONE;

    float world_x = 0.0f;
    float world_z = 0.0f;
    float ground_y = 0.0f;

    float trunk_h = 0.0f;
    float canopy_r = 0.0f;

    float canopy_b = 0.0f;
    float canopy_g = 0.0f;
    float canopy_r_col = 0.0f;

    float trunk_b = 0.0f;
    float trunk_g = 0.0f;
    float trunk_r_col = 0.0f;
};
