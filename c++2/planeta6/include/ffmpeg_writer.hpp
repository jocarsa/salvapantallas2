#pragma once
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>

std::string shell_escape_single_quotes(const std::string& s);
FILE* open_ffmpeg_nvenc_pipe(const std::string& output_path);
void write_frame_to_pipe(FILE* pipe, const cv::Mat& frame);
