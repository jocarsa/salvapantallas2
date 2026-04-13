#include "ffmpeg_writer.hpp"
#include "config.hpp"
#include <sstream>
#include <stdexcept>
#include <iostream>

std::string shell_escape_single_quotes(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

FILE* open_ffmpeg_nvenc_pipe(const std::string& output_path) {
    const std::string safe_out = shell_escape_single_quotes(output_path);

    std::ostringstream cmd;
    cmd
        << "ffmpeg -y "
        << "-f rawvideo "
        << "-pix_fmt bgr24 "
        << "-s " << cfg::width << "x" << cfg::height << " "
        << "-r " << cfg::output_fps << " "
        << "-i - "
        << "-an "
        << "-c:v h264_nvenc "
        << "-preset p7 "
        << "-tune hq "
        << "-rc vbr "
        << "-cq 19 "
        << "-b:v 0 "
        << "-pix_fmt yuv420p "
        << "-movflags +faststart "
        << "'" << safe_out << "'";

    std::cout << "FFmpeg command:\n" << cmd.str() << "\n";

    FILE* pipe = popen(cmd.str().c_str(), "w");
    if (!pipe) throw std::runtime_error("Could not open FFmpeg pipe.");
    return pipe;
}

void write_frame_to_pipe(FILE* pipe, const cv::Mat& frame) {
    if (!pipe) throw std::runtime_error("FFmpeg pipe is null.");
    if (frame.empty()) throw std::runtime_error("Attempted to write empty frame.");
    if (frame.type() != CV_8UC3) throw std::runtime_error("Frame must be CV_8UC3.");
    if (frame.cols != cfg::width || frame.rows != cfg::height) throw std::runtime_error("Frame size mismatch.");

    const size_t row_bytes = (size_t)cfg::width * 3;
    const size_t total_bytes = row_bytes * (size_t)cfg::height;

    if (frame.isContinuous()) {
        size_t written = fwrite(frame.data, 1, total_bytes, pipe);
        if (written != total_bytes) throw std::runtime_error("Short write to FFmpeg pipe.");
    } else {
        for (int y = 0; y < frame.rows; ++y) {
            const unsigned char* row = frame.ptr<unsigned char>(y);
            size_t written = fwrite(row, 1, row_bytes, pipe);
            if (written != row_bytes) throw std::runtime_error("Short row write to FFmpeg pipe.");
        }
    }
}
