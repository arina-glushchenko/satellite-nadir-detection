#ifndef DRAW_UTILS_STB_H
#define DRAW_UTILS_STB_H

#include <opencv2/opencv.hpp>
#include "postprocess.h"  // Added for detect_result_group_t

void draw_boxes_with_stb(const cv::Mat& input_image, detect_result_group_t* detect_result_group, const char* output_path);

#endif