#include "preprocess.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <sys/time.h>
#include <libgen.h>
#include "postprocess.h"
#include "draw_utils_stb.h"

// Include STB headers without defining implementation
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i)
    {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
}

static void *load_file(const char *file_path, size_t *file_size)
{
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL)
    {
        printf("failed to open file: %s\n", file_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void *file_data = malloc(size);
    if (file_data == NULL)
    {
        fclose(fp);
        printf("failed allocate file size: %zu\n", size);
        return NULL;
    }

    if (fread(file_data, 1, size, fp) != size)
    {
        fclose(fp);
        free(file_data);
        printf("failed to read file data!\n");
        return NULL;
    }

    fclose(fp);

    *file_size = size;

    return file_data;
}

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr, int *img_height, int *img_width)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[2];
        req_channel = input_attr->dims[3];
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[3];
        req_channel = input_attr->dims[1];
        break;
    default:
        printf("meet unsupported layout: %d\n", input_attr->fmt);
        return NULL;
    }

    int channel = 0;
    unsigned char *image_data = stbi_load(image_path, img_width, img_height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed: %s, reason: %s\n", image_path, stbi_failure_reason());
        FILE* fp = fopen(image_path, "rb");
        if (!fp) {
            printf("Cannot open file %s for reading\n", image_path);
        } else {
            printf("File %s exists but failed to load as image\n", image_path);
            fclose(fp);
        }
        return NULL;
    }

    if (*img_width != req_width || *img_height != req_height)
    {
        unsigned char *image_resized = (unsigned char *)malloc(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            free(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, *img_width, *img_height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            free(image_data);
            free(image_resized);
            return NULL;
        }
        free(image_data);
        image_data = image_resized;
    }

    return image_data;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s model_path input_path.c [loop_count]\n", argv[0]);
        return -1;
    }

    char *model_path = argv[1];
    char *input_path = argv[2];
    int loop_count = (argc > 3) ? atoi(argv[3]) : 1;

    // Timing variables
    int64_t preprocess_start_us, preprocess_elapse_us;
    int64_t inference_start_us, inference_elapse_us;
    int64_t postprocess_start_us, postprocess_elapse_us;

    // Preprocessing
    preprocess_start_us = getCurrentTimeUs();

    cv::Mat preprocessed_image;
    std::pair<int, int> center_top_left;
    std::vector<RelDetection> detections_relative;
    std::pair<int, int> center_dims;
    std::tuple<float, float, float> vector;
    char *preprocessed_image_path = NULL; // To store the new input path
    try {
        auto photos_result = read_photos_from_c_file(input_path);
        auto photos = photos_result.first;
        auto face_numbers = photos_result.second;

        printf("[%s] Read %zu faces: ", input_path, photos.size());
        for (int num : face_numbers) printf("%d ", num);
        printf("\n");

        int center_face = auto_select_center_face_index_by_hot_pixel_count(photos) + 1;
        printf("Auto selected central face: %d\n", center_face);

        if (center_face < 1 || center_face > 6) {
            throw std::invalid_argument("Central face must be between 1 and 6");
        }

        auto result = create_combined_canvas(photos, face_numbers, center_face, "inferno");
        preprocessed_image = std::get<0>(result);
        center_top_left = std::get<1>(result);
        detections_relative = std::get<2>(result);
        center_dims = std::get<3>(result);
        vector = std::get<4>(result);

        // Validate preprocessed_image
        if (preprocessed_image.empty()) {
            printf("Error: preprocessed_image is empty\n");
            return -1;
        }
        if (preprocessed_image.type() != CV_8UC3) {
            cv::Mat temp;
            preprocessed_image.convertTo(temp, CV_8UC3);
            preprocessed_image = temp;
            if (preprocessed_image.empty()) {
                printf("Error: Failed to convert preprocessed_image to CV_8UC3\n");
                return -1;
            }
            printf("Converted preprocessed_image to CV_8UC3\n");
        }

        // Save preprocessed image for RKNN input
        std::string output_path = "./preprocessed_image.png";
        try {
            std::remove(output_path.c_str()); // Remove existing file to avoid conflicts
            if (!cv::imwrite(output_path, preprocessed_image)) {
                printf("Error: Failed to write preprocessed_image.png to %s\n", output_path.c_str());
                return -1;
            }
            // Verify the file exists and is readable
            FILE* fp = fopen(output_path.c_str(), "rb");
            if (!fp) {
                printf("Error: Cannot open preprocessed_image.png for reading after writing\n");
                return -1;
            }
            fclose(fp);
            printf("Successfully saved preprocessed_image.png\n");
            // Create a persistent copy of the output path
            preprocessed_image_path = strdup(output_path.c_str());
            if (!preprocessed_image_path) {
                printf("Error: Failed to allocate memory for preprocessed_image_path\n");
                return -1;
            }
            input_path = preprocessed_image_path;
        } catch (const cv::Exception& e) {
            printf("Error saving preprocessed_image.png: %s\n", e.what());
            return -1;
        }
    } catch (const std::exception& e) {
        printf("Preprocessing failed: %s\n", e.what());
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    preprocess_elapse_us = getCurrentTimeUs() - preprocess_start_us;
    printf("Preprocessing Time = %.2fms\n", preprocess_elapse_us / 1000.f);

    // Initialize RKNN
    rknn_context ctx = 0;
    size_t model_size = 0;
    void *model_data = load_file(model_path, &model_size);
    if (!model_data) {
        printf("Failed to load model file: %s\n", model_path);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    // Check if model has expected number of outputs
    if (io_num.n_output != 1 && io_num.n_output != 3) {
        printf("Error: Model output count must be 1 or 3, but got %d\n", io_num.n_output);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query error! ret=%d\n", ret);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    // Load preprocessed image
    int img_width = 0, img_height = 0;
    unsigned char *input_data = load_image(input_path, &input_attrs[0], &img_height, &img_width);
    if (!input_data) {
        printf("Failed to load image: %s\n", input_path);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    rknn_tensor_mem *input_mems[1];
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;
    input_attrs[0].type = input_type;
    input_attrs[0].fmt = input_layout;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!input_mems[0]) {
        printf("Failed to create input memory\n");
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    int width = input_attrs[0].dims[2];
    int stride = input_attrs[0].w_stride;

    if (width == stride) {
        memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
    } else {
        int height = input_attrs[0].dims[1];
        int channel = input_attrs[0].dims[3];
        uint8_t *src_ptr = input_data;
        uint8_t *dst_ptr = (uint8_t *)input_mems[0]->virt_addr;
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }

    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            printf("Failed to create output memory %d\n", i);
            for (uint32_t j = 0; j < i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
    }

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            for (uint32_t j = 0; j <= i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
    }

    // Run inference
    inference_start_us = getCurrentTimeUs();
    for (int i = 0; i < loop_count; ++i) {
        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            printf("rknn run error %d\n", ret);
            for (uint32_t j = 0; j < io_num.n_output; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
    }
    inference_elapse_us = getCurrentTimeUs() - inference_start_us;
    printf("Inference Time = %.2fms, FPS = %.2f\n", inference_elapse_us / 1000.f, 1000.f * 1000.f / inference_elapse_us);

    // Post-process
    postprocess_start_us = getCurrentTimeUs();

    int model_width = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[2] : input_attrs[0].dims[1];
    int model_height = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[3] : input_attrs[0].dims[2];
    float scale_w = (float)model_width / img_width;
    float scale_h = (float)model_height / img_height;

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    detect_result_group_t detect_result_group;
    memset(&detect_result_group, 0, sizeof(detect_result_group_t));
    
    int dim[5 * 3] = {0};
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        dim[5 * i + 0] = (int)output_attrs[i].dims[0];
        dim[5 * i + 1] = (int)output_attrs[i].dims[1];
        dim[5 * i + 2] = (int)output_attrs[i].dims[2];
        dim[5 * i + 3] = (int)output_attrs[i].dims[3];
        dim[5 * i + 4] = (int)output_attrs[i].dims[4];
    }

    // Validate output memory
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        if (!output_mems[i]->virt_addr) {
            printf("Error: output_mems[%d]->virt_addr is null\n", i);
            for (uint32_t j = 0; j < io_num.n_output; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
    }

    // Call post_process
    if (io_num.n_output == 1) {
        ret = post_process((int8_t *)output_mems[0]->virt_addr, NULL, NULL,
                          model_height, model_width, BOX_THRESH, NMS_THRESH, 
                          scale_w, scale_h, out_zps, out_scales,
                          &detect_result_group, dim);
    } else if (io_num.n_output == 3) {
        ret = post_process((int8_t *)output_mems[0]->virt_addr, 
                          (int8_t *)output_mems[1]->virt_addr, 
                          (int8_t *)output_mems[2]->virt_addr,
                          model_height, model_width, BOX_THRESH, NMS_THRESH, 
                          scale_w, scale_h, out_zps, out_scales,
                          &detect_result_group, dim);
    } else {
        printf("Unsupported output number: %d\n", io_num.n_output);
        ret = -1;
    }
    
    if (ret < 0) {
        printf("post_process failed! ret=%d\n", ret);
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    // Draw detections
    if (preprocessed_image.empty()) {
        printf("Error: preprocessed_image is empty\n");
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }
    cv::Mat to_save = preprocessed_image.clone();
    if (to_save.empty()) {
        printf("Error: to_save image is empty after cloning\n");
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    // Ensure to_save is CV_8UC3
    if (to_save.type() != CV_8UC3) {
        cv::Mat temp;
        to_save.convertTo(temp, CV_8UC3);
        to_save = temp;
        printf("Converted to_save to CV_8UC3\n");
    }

    bool plotted_any = false;
    float x_det, y_det;
    if (!detect_result_group.count) {
        float cx_rel = center_dims.first / 2.0f;
        float cy_rel = center_dims.second / 2.0f;
        x_det = center_top_left.first + cx_rel;
        y_det = center_top_left.second + cy_rel;
        try {
            if (x_det < 0 || x_det >= to_save.cols || y_det < 0 || y_det >= to_save.rows) {
                printf("Warning: Invalid no-detection coordinates (x_det=%.2f, y_det=%.2f)\n", x_det, y_det);
            } else {
                cv::circle(to_save, cv::Point(static_cast<int>(x_det), static_cast<int>(y_det)), 2, cv::Scalar(0, 0, 255), -1);
                printf("No detections. center_rel=(%.2f, %.2f)\n", cx_rel, cy_rel);
            }
        } catch (const cv::Exception& e) {
            printf("Error drawing circle: %s\n", e.what());
            for (uint32_t i = 0; i < io_num.n_output; ++i) {
                rknn_destroy_mem(ctx, output_mems[i]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
    } else {
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            if (!det_result || !det_result->name) {
                printf("Error: det_result or det_result->name is null for detection %d\n", i);
                continue;
            }
            
            // Draw bounding box
            cv::Rect bbox(
                det_result->box.left,
                det_result->box.top,
                det_result->box.right - det_result->box.left,
                det_result->box.bottom - det_result->box.top
            );
            
            // Check if bounding box is within image bounds
            if (bbox.x >= 0 && bbox.y >= 0 && 
                bbox.x + bbox.width <= to_save.cols && 
                bbox.y + bbox.height <= to_save.rows) {
                
                // Draw rectangle
                cv::rectangle(to_save, bbox, cv::Scalar(0, 255, 0), 2);
                
                // Add label with class and confidence
                std::string label = std::string(det_result->name) + " " + 
                                   std::to_string(det_result->prop).substr(0, 4);
            } else {
                printf("Warning: Bounding box out of image bounds for detection %d\n", i);
            }
            
            float cx = (det_result->box.left + det_result->box.right) / 2.0f;
            float cy = (det_result->box.top + det_result->box.bottom) / 2.0f;
            float rel_cx = cx - center_top_left.first;
            float rel_cy = cy - center_top_left.second;
            try {
                if (cx < 0 || cx >= to_save.cols || cy < 0 || cy >= to_save.rows) {
                    printf("Warning: Invalid detection coordinates (cx=%.2f, cy=%.2f) for detection %d\n", cx, cy, i);
                    continue;
                }
                cv::circle(to_save, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), 2, cv::Scalar(0, 0, 255), -1);
                printf("Detection class %s conf %.2f center_rel=(%.2f, %.2f)\n",
                       det_result->name, det_result->prop, rel_cx, rel_cy);
                if (i == 0) {
                    x_det = cx;
                    y_det = cy;
                }
                plotted_any = true;
            } catch (const cv::Exception& e) {
                printf("Error drawing circle for detection %d: %s\n", i, e.what());
                continue;
            }
        }
    }

    float vx = std::get<0>(vector);
    float vy = std::get<1>(vector);
    float vz = std::get<2>(vector);
    printf("Vector: (%.2f, %.2f, %.2f)\n", vx, vy, vz);

    // Save the output image
    char output_image_path[256];
    char jpg_path[256];
    
    char *input_path_copy = strdup(input_path);
    if (input_path_copy) {
        char *base_name = basename(input_path_copy);
        snprintf(output_image_path, sizeof(output_image_path), "output_%s", base_name);
        snprintf(jpg_path, sizeof(jpg_path), "output_%s.jpg", base_name);
        free(input_path_copy);
    } else {
        snprintf(output_image_path, sizeof(output_image_path), "output_image.png");
        snprintf(jpg_path, sizeof(jpg_path), "output_image.jpg");
    }

    try {
        if (to_save.empty()) {
            printf("Error: to_save is empty before saving\n");
            for (uint32_t i = 0; i < io_num.n_output; ++i) {
                rknn_destroy_mem(ctx, output_mems[i]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            rknn_destroy(ctx);
            if (preprocessed_image_path) free(preprocessed_image_path);
            return -1;
        }
        printf("Saving image: %s, rows=%d, cols=%d, type=%d\n", 
               output_image_path, to_save.rows, to_save.cols, to_save.type());

        if (!cv::imwrite(output_image_path, to_save)) {
            printf("Error: Failed to save image as PNG: %s\n", output_image_path);
            if (!cv::imwrite(jpg_path, to_save)) {
                printf("Error: Failed to save image as JPEG: %s\n", jpg_path);
                for (uint32_t i = 0; i < io_num.n_output; ++i) {
                    rknn_destroy_mem(ctx, output_mems[i]);
                }
                rknn_destroy_mem(ctx, input_mems[0]);
                free(input_data);
                rknn_destroy(ctx);
                if (preprocessed_image_path) free(preprocessed_image_path);
                return -1;
            }
            printf("Saved image as JPEG: %s\n", jpg_path);
        } else {
            printf("Saved image as PNG: %s\n", output_image_path);
            cv::Mat verify_output = cv::imread(output_image_path, cv::IMREAD_COLOR);
            if (verify_output.empty()) {
                printf("Warning: Failed to read back output image: %s\n", output_image_path);
            }
        }
    } catch (const cv::Exception& e) {
        printf("Error saving image: %s\n", e.what());
        if (!cv::imwrite(jpg_path, to_save)) {
            printf("Error: Failed to save image as JPEG: %s\n", jpg_path);
        } else {
            printf("Saved image as JPEG: %s\n", jpg_path);
        }
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        free(input_data);
        rknn_destroy(ctx);
        if (preprocessed_image_path) free(preprocessed_image_path);
        return -1;
    }

    postprocess_elapse_us = getCurrentTimeUs() - postprocess_start_us;
    printf("Postprocessing Time = %.2fms, FPS = %.2f\n", postprocess_elapse_us / 1000.f, 1000.f * 1000.f / postprocess_elapse_us);

    // Cleanup
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        if (output_mems[i]) {
            rknn_destroy_mem(ctx, output_mems[i]);
            output_mems[i] = nullptr;
        }
    }
    if (input_mems[0]) {
        rknn_destroy_mem(ctx, input_mems[0]);
        input_mems[0] = nullptr;
    }
    if (input_data) {
        free(input_data);
        input_data = nullptr;
    }
    rknn_destroy(ctx);
    if (preprocessed_image_path) {
        free(preprocessed_image_path);
        preprocessed_image_path = nullptr;
    }

    return 0;
}