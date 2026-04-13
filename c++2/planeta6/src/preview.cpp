#include "preview.hpp"
#include "config.hpp"
#include <sstream>

void show_preview_window(const cv::Mat& frame, long long frame_idx){
    static bool window_created = false;
    static cv::Mat preview;

    if (!window_created) {
        cv::namedWindow("terrain preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("terrain preview", 960, 540);
        window_created = true;
    }

    preview = frame.clone();

    const long long sec = frame_idx / cfg::output_fps;

    std::ostringstream oss;
    oss << "preview t=" << sec << "s";

    cv::putText(
        preview,
        oss.str(),
        cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        1.1,
        cv::Scalar(20, 20, 20),
        3,
        cv::LINE_AA
    );

    cv::imshow("terrain preview", preview);
    cv::waitKey(1);
}
