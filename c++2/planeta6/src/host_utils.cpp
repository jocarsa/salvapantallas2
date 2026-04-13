#include "host_utils.hpp"
#include "config.hpp"

#include <cmath>
#include <stdexcept>

float lerp_host(float a, float b, float t){
    return a + (b - a) * t;
}

float3 normalize3_host(float3 v){
    float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 <= 1e-20f) return make_float3(0.0f, 1.0f, 0.0f);
    float inv = 1.0f / std::sqrt(len2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

cv::Matx33f get_rotation_matrix_host(const cv::Vec3f& angles){
    const float pitch = angles[0];
    const float yaw   = angles[1];
    const float roll  = angles[2];

    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float cr = std::cos(roll);
    const float sr = std::sin(roll);

    const cv::Matx33f Rx(
        1,  0,   0,
        0, cp, -sp,
        0, sp,  cp
    );

    const cv::Matx33f Ry(
         cy, 0, sy,
          0, 1,  0,
        -sy, 0, cy
    );

    const cv::Matx33f Rz(
        cr, -sr, 0,
        sr,  cr, 0,
         0,   0, 1
    );

    return Rz * Ry * Rx;
}

Mat3 to_mat3_rowmajor(const cv::Matx33f& R){
    Mat3 out{};
    out.m[0] = R(0,0); out.m[1] = R(0,1); out.m[2] = R(0,2);
    out.m[3] = R(1,0); out.m[4] = R(1,1); out.m[5] = R(1,2);
    out.m[6] = R(2,0); out.m[7] = R(2,1); out.m[8] = R(2,2);
    return out;
}

float compute_motion_amount(
    float prev_rotated_x,
    float prev_rotated_z,
    const cv::Vec3f& prev_camera_position,
    float rotated_x,
    float rotated_z,
    const cv::Vec3f& current_camera_position)
{
    float dx_world = rotated_x - prev_rotated_x;
    float dz_world = rotated_z - prev_rotated_z;
    float dy_cam   = current_camera_position[1] - prev_camera_position[1];

    float movement = std::sqrt(dx_world * dx_world + dz_world * dz_world + dy_cam * dy_cam);
    float norm = movement * cfg::motion_blur_movement_scale;
    return clamp_host(norm * cfg::motion_blur_strength, 0.0f, cfg::motion_blur_max_alpha);
}

void apply_motion_blur(cv::Mat& frame, const cv::Mat& prev_frame, float amount){
    if (!cfg::enable_motion_blur) return;
    if (prev_frame.empty()) return;
    if (amount <= 1e-6f) return;
    if (frame.empty()) return;
    if (frame.size() != prev_frame.size() || frame.type() != prev_frame.type()) return;

    cv::Mat blended;
    cv::addWeighted(frame, 1.0 - amount, prev_frame, amount, 0.0, blended);
    blended.copyTo(frame);
}

static inline float blur_amount_from_depth(float depth){
    const float focus_depth = cfg::depth_blur_focus_depth;
    const float focus_range = std::max(1e-5f, cfg::depth_blur_focus_range);

    if (depth <= 0.0f) return 0.0f;

    const float delta = depth - focus_depth;

    if (delta >= 0.0f) {
        const float far_t = clamp_host(delta / focus_range, 0.0f, 1.0f);
        return far_t * cfg::depth_blur_far_strength;
    } else {
        const float near_t = clamp_host((-delta) / focus_range, 0.0f, 1.0f);
        return near_t * cfg::depth_blur_near_strength;
    }
}

void apply_depth_blur(cv::Mat& frame, const cv::Mat& depth_map){
    if (!cfg::enable_depth_blur) return;
    if (frame.empty() || depth_map.empty()) return;
    if (frame.size() != depth_map.size()) return;
    if (frame.type() != CV_8UC3) return;
    if (depth_map.type() != CV_32FC1) return;

    cv::Mat blur_mid;
    cv::Mat blur_strong;

    cv::GaussianBlur(
        frame,
        blur_mid,
        cv::Size(0, 0),
        cfg::depth_blur_mid_sigma,
        cfg::depth_blur_mid_sigma,
        cv::BORDER_REPLICATE
    );

    cv::GaussianBlur(
        frame,
        blur_strong,
        cv::Size(0, 0),
        cfg::depth_blur_strong_sigma,
        cfg::depth_blur_strong_sigma,
        cv::BORDER_REPLICATE
    );

    cv::Mat out(frame.size(), frame.type());

    for (int y = 0; y < frame.rows; ++y) {
        const float* drow = depth_map.ptr<float>(y);
        const cv::Vec3b* src = frame.ptr<cv::Vec3b>(y);
        const cv::Vec3b* mid = blur_mid.ptr<cv::Vec3b>(y);
        const cv::Vec3b* strong = blur_strong.ptr<cv::Vec3b>(y);
        cv::Vec3b* dst = out.ptr<cv::Vec3b>(y);

        for (int x = 0; x < frame.cols; ++x) {
            const float k = clamp_host(blur_amount_from_depth(drow[x]), 0.0f, 1.0f);

            const float k_mid = clamp_host(k * 1.35f, 0.0f, 1.0f);
            const float k_strong = clamp_host((k - 0.35f) / 0.65f, 0.0f, 1.0f);

            cv::Vec3f a(
                (float)src[x][0],
                (float)src[x][1],
                (float)src[x][2]
            );

            cv::Vec3f b(
                lerp_host(a[0], (float)mid[x][0], k_mid),
                lerp_host(a[1], (float)mid[x][1], k_mid),
                lerp_host(a[2], (float)mid[x][2], k_mid)
            );

            cv::Vec3f c(
                lerp_host(b[0], (float)strong[x][0], k_strong),
                lerp_host(b[1], (float)strong[x][1], k_strong),
                lerp_host(b[2], (float)strong[x][2], k_strong)
            );

            dst[x][0] = (unsigned char)clamp_host((int)std::lround(c[0]), 0, 255);
            dst[x][1] = (unsigned char)clamp_host((int)std::lround(c[1]), 0, 255);
            dst[x][2] = (unsigned char)clamp_host((int)std::lround(c[2]), 0, 255);
        }
    }

    out.copyTo(frame);
}

void apply_glow(cv::Mat& frame){
    if (!cfg::enable_glow) return;
    if (frame.empty()) return;

    cv::Mat gray, mask, bright, glow;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, cfg::glow_threshold, 255, cv::THRESH_BINARY);

    bright = cv::Mat::zeros(frame.size(), frame.type());
    frame.copyTo(bright, mask);

    cv::GaussianBlur(
        bright,
        glow,
        cv::Size(cfg::glow_blur_kernel, cfg::glow_blur_kernel),
        cfg::glow_blur_sigma,
        cfg::glow_blur_sigma,
        cv::BORDER_REPLICATE
    );

    cv::addWeighted(frame, 1.0, glow, cfg::glow_intensity, 0.0, frame);
}
