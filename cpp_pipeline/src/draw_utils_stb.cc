// #include "draw_utils_stb.h"
// #include <libgen.h>
// #include <string.h>
// #include "stb/stb_image.h"
// #include "stb/stb_image_write.h"
// #include "stb_truetype.h"


// // Функция для рисования линии
// static void draw_line(unsigned char* image, int width, int height, int channels,
//                      int x1, int y1, int x2, int y2,
//                      unsigned char r, unsigned char g, unsigned char b) {
//     int dx = abs(x2 - x1);
//     int dy = abs(y2 - y1);
//     int sx = x1 < x2 ? 1 : -1;
//     int sy = y1 < y2 ? 1 : -1;
//     int err = (dx > dy ? dx : -dy) / 2;
    
//     while (1) {
//         if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
//             int idx = (y1 * width + x1) * channels;
//             image[idx] = r;
//             if (channels > 1) image[idx+1] = g;
//             if (channels > 2) image[idx+2] = b;
//         }
        
//         if (x1 == x2 && y1 == y2) break;
//         int e2 = err;
//         if (e2 > -dx) { err -= dy; x1 += sx; }
//         if (e2 < dy) { err += dx; y1 += sy; }
//     }
// }

// void draw_boxes_with_stb(const char* image_path, 
//                         detect_result_group_t* detect_result_group, 
//                         const char* output_path) {
//     int width, height, channels;
//     unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 0);
//     if (!image_data) {
//         printf("Failed to load image for drawing boxes!\n");
//         return;
//     }

//     // Цвета для bounding box'ов (R, G, B)
//     unsigned char colors[][3] = {
//         {255, 0, 0},   // Красный
//         {0, 255, 0},   // Зеленый
//         {0, 0, 255},   // Синий
//         {255, 255, 0}, // Желтый
//         {255, 0, 255}  // Пурпурный
//     };
//     int num_colors = sizeof(colors) / sizeof(colors[0]);

//     // Толщина линий
//     int thickness = 2;

//     // Рисуем каждый bounding box
//     for (int i = 0; i < detect_result_group->count; i++) {
//         detect_result_t* det_result = &(detect_result_group->results[i]);
        
//         int x1 = det_result->box.left;
//         int y1 = det_result->box.top;
//         int x2 = det_result->box.right;
//         int y2 = det_result->box.bottom;
        
//         // Выбираем цвет
//         unsigned char r = colors[i % num_colors][0];
//         unsigned char g = colors[i % num_colors][1];
//         unsigned char b = colors[i % num_colors][2];
        
//         // Рисуем 4 линии для bounding box'а
//         for (int t = 0; t < thickness; t++) {
//             // Верхняя линия
//             draw_line(image_data, width, height, channels, 
//                      x1+t, y1+t, x2-t, y1+t, r, g, b);
//             // Правая линия
//             draw_line(image_data, width, height, channels, 
//                      x2-t, y1+t, x2-t, y2-t, r, g, b);
//             // Нижняя линия
//             draw_line(image_data, width, height, channels, 
//                      x1+t, y2-t, x2-t, y2-t, r, g, b);
//             // Левая линия
//             draw_line(image_data, width, height, channels, 
//                      x1+t, y1+t, x1+t, y2-t, r, g, b);
//         }
//     }

//     // Сохраняем изображение (используем тот же формат, что и входной файл)
//     const char* ext = strrchr(output_path, '.');
//     int success = 0;
    
//     if (ext != NULL) {
//         if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0) {
//             success = stbi_write_jpg(output_path, width, height, channels, image_data, 90);
//         } 
//         else if (strcasecmp(ext, ".png") == 0) {
//             success = stbi_write_png(output_path, width, height, channels, image_data, width * channels);
//         }
//     }
    
//     // Если не распознали формат или сохранение не удалось, пробуем PNG
//     if (!success) {
//         char new_path[256];
//         snprintf(new_path, sizeof(new_path), "%s.png", output_path);
//         success = stbi_write_png(new_path, width, height, channels, image_data, width * channels);
//     }

//     if (!success) {
//         printf("Failed to save image with bounding boxes!\n");
//     } else {
//         printf("Image with bounding boxes saved successfully.\n");
//     }

//     stbi_image_free(image_data);
// }

#include "draw_utils_stb.h"
#include <opencv2/opencv.hpp>
#include "stb_image_write.h"
#include "postprocess.h"

void draw_boxes_with_stb(const cv::Mat& input_image, detect_result_group_t* detect_result_group, const char* output_path) {
    cv::Mat image = input_image.clone();
    for (int i = 0; i < detect_result_group->count; i++) {
        detect_result_t* det_result = &(detect_result_group->results[i]);
        cv::rectangle(image, 
                     cv::Point(det_result->box.left, det_result->box.top),
                     cv::Point(det_result->box.right, det_result->box.bottom),
                     cv::Scalar(0, 255, 0), 2);
        char text[256];
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        cv::putText(image, text, 
                   cv::Point(det_result->box.left, det_result->box.top - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    std::vector<unsigned char> buffer;
    cv::imencode(".png", image, buffer);
    stbi_write_png(output_path, image.cols, image.rows, 3, buffer.data(), image.cols * 3);
}