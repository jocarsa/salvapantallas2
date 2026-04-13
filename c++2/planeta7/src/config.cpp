#include "config.hpp"
#include <ctime>
#include <iomanip>
#include <sstream>

namespace cfg {

int near_subdivisions = 1900;
float terrain_height_multiplier = 1.0f;
float camera_y_offset = 0.35f;
cv::Vec3f camera_position = cv::Vec3f(0.0f, 1.2f, 5.0f);
cv::Vec3f camera_rotation = cv::Vec3f(-0.1f, 0.0f, 0.0f);
float forward_speed = 1.165f;
float radius_multiplier = 4.0f;
float shadow_darkness_multiplier = 1.0f;
float water_reflection_strength = 0.40f;

std::string make_default_output_filename() {
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

} // namespace cfg
