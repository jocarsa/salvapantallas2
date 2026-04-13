#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "types.hpp"

template<typename T>
inline T clamp_host(T v, T lo, T hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

float lerp_host(float a, float b, float t);
float3 normalize3_host(float3 v);

cv::Matx33f get_rotation_matrix_host(const cv::Vec3f& angles);
Mat3 to_mat3_rowmajor(const cv::Matx33f& R);

float compute_motion_amount(
    float prev_rotated_x,
    float prev_rotated_z,
    const cv::Vec3f& prev_camera_position,
    float rotated_x,
    float rotated_z,
    const cv::Vec3f& current_camera_position
);

void apply_motion_blur(cv::Mat& frame, const cv::Mat& prev_frame, float amount);
void apply_depth_blur(cv::Mat& frame, const cv::Mat& depth_map);
void apply_glow(cv::Mat& frame);

// Implemented in src/main.cu
unsigned int init_permutation_cuda();
