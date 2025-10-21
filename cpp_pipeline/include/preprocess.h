#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>
#include <cstring>
#include <chrono> 

#define M_PI 3.14159265358979323846

/**
 * @brief Ограничивает значение заданными границами
 * @tparam T Тип значения
 * @param value Исходное значение
 * @param min_val Нижняя граница
 * @param max_val Верхняя граница
 * @return Значение в диапазоне [min_val, max_val]
 */
template<typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(max_val, value));
}

/**
 * @brief Структура для хранения позиций граней куба
 */
struct Pos {
    int center;  ///< Центральная грань
    int up;      ///< Верхняя грань
    int down;    ///< Нижняя грань
    int left;    ///< Левая грань
    int right;   ///< Правая грань
    int back;    ///< Задняя грань
};

/**
 * @brief Получает расположение граней относительно центральной
 * @param center_face Номер центральной грани (1-6)
 * @return Структура Pos с расположением граней
 * @throw std::invalid_argument Если номер грани не от 1 до 6
 */
Pos get_pos_by_center(int center_face) {
    std::map<int, std::vector<int>> face_adjacency = {
        {1, {1, 4, 2, 6, 5, 3}},
        {2, {2, 5, 6, 1, 3, 4}},
        {3, {3, 4, 2, 5, 6, 1}},
        {4, {4, 6, 5, 1, 3, 2}},
        {5, {5, 4, 2, 1, 3, 6}},
        {6, {6, 4, 2, 3, 1, 5}}
    };
    
    if (face_adjacency.find(center_face) == face_adjacency.end()) {
        throw std::invalid_argument("Central face must be between 1 and 6");
    }
    
    auto layout = face_adjacency[center_face];
    Pos pos;
    pos.center = layout[0];
    pos.up = layout[1];
    pos.down = layout[2];
    pos.left = layout[3];
    pos.right = layout[4];
    pos.back = layout[5];
    return pos;
}

// Функции поворота граней
cv::Mat rotate_face_1(const cv::Mat& photo, int face_number) {
    if (face_number == 3) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    return photo.clone();
}

cv::Mat rotate_face_2(const cv::Mat& photo, int face_number) {
    if (face_number >= 1 && face_number <= 4) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rot;
    }
    return photo.clone();
}

cv::Mat rotate_face_3(const cv::Mat& photo, int face_number) {
    if (face_number == 2 || face_number == 3) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    if (face_number == 4) {
        cv::Mat flipped;
        cv::flip(photo, flipped, 1);
        return flipped;
    }
    return photo.clone();
}

cv::Mat rotate_face_4(const cv::Mat& photo, int face_number) {
    if (face_number >= 1 && face_number <= 4) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_CLOCKWISE);
        return rot;
    }
    if (face_number == 6) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    return photo.clone();
}

cv::Mat rotate_face_5(const cv::Mat& photo, int face_number) {
    if (face_number == 2) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rot;
    }
    if (face_number == 3) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    if (face_number == 4) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_CLOCKWISE);
        return rot;
    }
    if (face_number == 6) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    return photo.clone();
}

cv::Mat rotate_face_6(const cv::Mat& photo, int face_number) {
    if (face_number == 2) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_CLOCKWISE);
        return rot;
    }
    if (face_number == 3) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    if (face_number == 4) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rot;
    }
    if (face_number == 6) {
        cv::Mat rot;
        cv::rotate(photo, rot, cv::ROTATE_180);
        return rot;
    }
    return photo.clone();
}

/// Тип для функций поворота граней
typedef cv::Mat (*RotationStrategy)(const cv::Mat&, int);

/**
 * @brief Получает стратегию поворота для заданной центральной грани
 * @param center_face Номер центральной грани
 * @return Указатель на функцию поворота
 * @throw std::invalid_argument Если номер грани невалидный
 */
RotationStrategy get_rotation_strategy(int center_face) {
    std::map<int, RotationStrategy> strategies = {
        {1, rotate_face_1},
        {2, rotate_face_2},
        {3, rotate_face_3},
        {4, rotate_face_4},
        {5, rotate_face_5},
        {6, rotate_face_6}
    };
    
    if (strategies.find(center_face) == strategies.end()) {
        throw std::invalid_argument("Invalid center face");
    }
    return strategies[center_face];
}

/**
 * @brief Читает все строки из файла
 * @param file_path Путь к файлу
 * @return Вектор строк
 * @throw std::runtime_error Если файл не может быть открыт
 */
std::vector<std::string> read_lines_from_file(const std::string& file_path) {
    std::ifstream f(file_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        lines.push_back(line);
    }

    return lines;
}

/**
 * @brief Парсит фотографии и номера граней из строк
 * @param lines Вектор строк для парсинга
 * @param file_path Путь к файлу (для сообщений об ошибках)
 * @return Пара: вектор изображений и вектор номеров граней
 * @throw std::runtime_error Если не найдено валидных массивов
 */
std::pair<std::vector<cv::Mat>, std::vector<int>> parse_photos_from_lines(
    const std::vector<std::string>& lines,
    const std::string& file_path
) {
    std::vector<cv::Mat> photos;
    photos.reserve(10);
    std::vector<int> face_numbers;
    face_numbers.reserve(10);
    std::vector<float> current_array;
    current_array.reserve(768);

    bool in_photo_block = false;
    int current_face_number = 1;

    const size_t target_size = 768;
    const char* delimiters = " \t\n\r,{}fF";
    
    for (const auto& line : lines) {
        if (line.empty()) continue;
        
        // Пропуск начальных пробелов
        size_t start = 0;
        while (start < line.length() && (line[start] == ' ' || line[start] == '\t')) {
            ++start;
        }
        if (start >= line.length()) continue;

        // Обработка комментариев с номером грани
        if (line[start] == '/' && line[start + 1] == '/') {
            size_t pos = start + 2;
            while (pos < line.length() - 1) {
                if (line[pos] == 'h' && line[pos + 1] == 's') {
                    pos += 2;
                    while (pos < line.length() && !std::isdigit(static_cast<unsigned char>(line[pos]))) {
                        ++pos;
                    }
                    int number = 0;
                    while (pos < line.length() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
                        number = number * 10 + (line[pos] - '0');
                        ++pos;
                    }
                    if (number > 0) {
                        current_face_number = number;
                        face_numbers.push_back(number);
                        in_photo_block = true;
                        current_array.clear();
                    }
                    break;
                }
                ++pos;
            }
            continue;
        }

        // Обработка данных массива
        if (in_photo_block) {
            bool has_data = false;
            for (size_t i = start; i < line.length(); ++i) {
                char c = line[i];
                if (c == '{' || c == '.' || (c >= '0' && c <= '9')) {
                    has_data = true;
                    break;
                }
            }
            
            if (has_data) {
                const char* ptr = line.c_str() + start;
                const char* end = line.c_str() + line.length();
                
                while (ptr < end && current_array.size() < target_size) {
                    // Пропуск разделителей
                    while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == ',' || 
                           *ptr == '{' || *ptr == '}' || *ptr == 'f' || *ptr == 'F')) {
                        ++ptr;
                    }
                    
                    if (ptr >= end) break;
                    
                    // Парсинг числа
                    char* next_ptr;
                    float value = std::strtof(ptr, &next_ptr);
                    
                    if (next_ptr == ptr) break;
                    
                    current_array.push_back(value);
                    ptr = next_ptr;
                    
                    // Создание матрицы при достижении целевого размера
                    if (current_array.size() == target_size) {
                        cv::Mat mat(24, 32, CV_32F);
                        std::memcpy(mat.data, current_array.data(), target_size * sizeof(float));
                        photos.push_back(std::move(mat));
                        current_array.clear();
                        in_photo_block = false;
                        break;
                    }
                }
            }
        }

        // Завершение блока при обнаружении закрывающей скобки
        if (in_photo_block) {
            for (size_t i = start; i < line.length(); ++i) {
                if (line[i] == '}') {
                    if (current_array.size() != target_size) {
                        std::cerr << "Warning: Array for face " << current_face_number << " in file " << file_path
                                  << " has " << current_array.size() << " elements, expected 768, skipping." << std::endl;
                    }
                    current_array.clear();
                    in_photo_block = false;
                    break;
                }
            }
        }
    }

    // Синхронизация размеров массивов
    if (photos.size() != face_numbers.size()) {
        if (face_numbers.size() > photos.size()) {
            face_numbers.resize(photos.size());
        } else {
            for (size_t i = face_numbers.size(); i < photos.size(); ++i) {
                face_numbers.push_back(static_cast<int>(i + 1));
            }
        }
    }

    if (photos.empty()) {
        throw std::runtime_error("No valid arrays found in file: " + file_path);
    }

    return {std::move(photos), std::move(face_numbers)};
}

/**
 * @brief Добавляет изображение на canvas с учетом перекрытий
 * @param sum_canvas Canvas для суммирования значений
 * @param count_canvas Canvas для подсчета перекрытий
 * @param img Изображение для добавления
 * @param y0 Начальная Y-координата
 * @param x0 Начальная X-координата
 */
void add_to_canvas(cv::Mat& sum_canvas, cv::Mat& count_canvas, const cv::Mat& img, int y0, int x0) {
    if (img.empty()) return;
    
    int H = sum_canvas.rows, W = sum_canvas.cols;
    int h = img.rows, w = img.cols;
    
    // Вычисление области перекрытия
    int y_overlap_start = std::max(y0, 0);
    int y_overlap_end = std::min(y0 + h, H);
    int x_overlap_start = std::max(x0, 0);
    int x_overlap_end = std::min(x0 + w, W);
    
    if (y_overlap_start < y_overlap_end && x_overlap_start < x_overlap_end) {
        int img_y_start = y_overlap_start - y0;
        int img_y_end = img_y_start + (y_overlap_end - y_overlap_start);
        int img_x_start = x_overlap_start - x0;
        int img_x_end = img_x_start + (x_overlap_end - x_overlap_start);
        
        cv::Mat sum_roi = sum_canvas(cv::Rect(x_overlap_start, y_overlap_start, 
                                            x_overlap_end - x_overlap_start, 
                                            y_overlap_end - y_overlap_start));
        cv::Mat count_roi = count_canvas(cv::Rect(x_overlap_start, y_overlap_start, 
                                                x_overlap_end - x_overlap_start, 
                                                y_overlap_end - y_overlap_start));
        cv::Mat img_roi = img(cv::Rect(img_x_start, img_y_start, 
                                     img_x_end - img_x_start, 
                                     img_y_end - img_y_start));
        
        sum_roi += img_roi;
        count_roi += cv::Scalar(1.0f);
    }
}

/**
 * @brief Автоматически выбирает центральную грань по количеству "горячих" пикселей
 * @param photos Вектор изображений граней
 * @param percentile Процентиль для определения порога (по умолчанию 80)
 * @return Индекс центральной грани в векторе
 */
int auto_select_center_face_index_by_hot_pixel_count(const std::vector<cv::Mat>& photos, int percentile = 80) {
    // Сбор всех пикселей
    std::vector<float> all_pixels;
    for (const auto& p : photos) {
        const float* data = p.ptr<float>(0);
        for (size_t j = 0; j < p.total(); ++j) {
            all_pixels.push_back(data[j]);
        }
    }
    
    // Вычисление порога по процентилю
    std::sort(all_pixels.begin(), all_pixels.end());
    size_t index = static_cast<size_t>(all_pixels.size() * percentile / 100.0);
    float threshold = all_pixels[index];
    
    // Подсчет пикселей выше порога для каждого изображения
    std::vector<int> counts(photos.size(), 0);
    for (size_t i = 0; i < photos.size(); ++i) {
        const float* data = photos[i].ptr<float>(0);
        for (size_t j = 0; j < photos[i].total(); ++j) {
            if (data[j] > threshold) ++counts[i];
        }
    }
    
    // Обработка случая, когда все счетчики нулевые
    bool all_zero = true;
    for (int c : counts) {
        if (c != 0) all_zero = false;
    }
    
    if (all_zero) {
        std::vector<float> maxs;
        for (const auto& p : photos) {
            double max_val;
            cv::minMaxLoc(p, nullptr, &max_val);
            maxs.push_back(static_cast<float>(max_val));
        }
        auto it = std::max_element(maxs.begin(), maxs.end());
        return static_cast<int>(std::distance(maxs.begin(), it));
    }
    
    // Выбор кандидатов с максимальным счетчиком
    int max_count = *std::max_element(counts.begin(), counts.end());
    std::vector<int> candidates;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] == max_count) candidates.push_back(static_cast<int>(i));
    }
    
    if (candidates.size() == 1) return candidates[0];
    
    // Разрешение неоднозначности по среднему значению
    std::vector<float> means;
    for (int idx : candidates) {
        cv::Scalar mean = cv::mean(photos[idx]);
        means.push_back(static_cast<float>(mean[0]));
    }
    auto it = std::max_element(means.begin(), means.end());
    return candidates[std::distance(means.begin(), it)];
}

/**
 * @brief Структура для относительного обнаружения
 */
struct RelDetection {
    int cls;                    ///< Класс обнаружения
    float conf;                 ///< Уверенность
    std::vector<float> center_rel;  ///< Относительные координаты центра
    std::vector<float> xyxy;    ///< Координаты bounding box
};

/**
 * @brief Создает объединенное изображение кубической проекции
 * @param photos Вектор изображений граней
 * @param face_numbers Вектор номеров граней
 * @param center_face Номер центральной грани
 * @param cmap_name Имя цветовой карты
 * @return Кортеж с результатами:
 *         - Объединенное RGB изображение
 *         - Координаты центральной грани
 *         - Вектор обнаружений
 *         - Размеры центральной грани
 *         - Вектор направления
 */
std::tuple<cv::Mat, std::pair<int, int>, std::vector<RelDetection>, std::pair<int, int>, 
           std::tuple<float, float, float>> create_combined_canvas(
    const std::vector<cv::Mat>& photos,
    const std::vector<int>& face_numbers,
    int center_face,
    const std::string& cmap_name
) {
    // Создание словаря грань->изображение
    std::map<int, cv::Mat> face_dict;
    for (size_t i = 0; i < face_numbers.size(); ++i) {
        face_dict[face_numbers[i]] = photos[i];
    }
    
    if (face_dict.find(center_face) == face_dict.end()) {
        throw std::invalid_argument("Central face not found");
    }
    
    Pos pos = get_pos_by_center(center_face);

    // Определение сдвигов для разных центральных граней
    int shift_x = 0, shift_y = 0;
    if (face_numbers.size() >= 6) {
        if (center_face == 1 || center_face == 3) {
            shift_x = 3;
            shift_y = -2;
        } else if (center_face == 2 || center_face == 4) {
            shift_x = -2;
            shift_y = 0;
        } else if (center_face == 5 || center_face == 6) {
            shift_x = 3;
            shift_y = 0;
        }
    }

    RotationStrategy rotation_func = get_rotation_strategy(center_face);

    // Применение поворотов ко всем граням
    std::map<int, cv::Mat> rotated_dict;
    for (const auto& kv : face_dict) {
        rotated_dict[kv.first] = rotation_func(kv.second, kv.first);
    }

    // Создание изображений по умолчанию для отсутствующих граней
    cv::Mat default_img(24, 32, CV_32F, cv::Scalar(0.0f));

    cv::Mat center_img = (rotated_dict.find(center_face) != rotated_dict.end()) ? 
                         rotated_dict[center_face] : default_img.clone();
    cv::Mat left_img = (rotated_dict.find(pos.left) != rotated_dict.end()) ? 
                       rotated_dict[pos.left] : default_img.clone();
    cv::Mat right_img = (rotated_dict.find(pos.right) != rotated_dict.end()) ? 
                        rotated_dict[pos.right] : default_img.clone();
    cv::Mat up_img = (rotated_dict.find(pos.up) != rotated_dict.end()) ? 
                     rotated_dict[pos.up] : default_img.clone();
    cv::Mat down_img = (rotated_dict.find(pos.down) != rotated_dict.end()) ? 
                       rotated_dict[pos.down] : default_img.clone();
    cv::Mat back_img = (rotated_dict.find(pos.back) != rotated_dict.end()) ? 
                       rotated_dict[pos.back] : default_img.clone();

    // Подготовка задней грани (разделение на части)
    int h = back_img.rows, w = back_img.cols;
    int half_h = h / 2, half_w = w / 2;
    cv::Mat left_half_back = back_img.colRange(0, half_w);
    cv::Mat right_half_back = back_img.colRange(half_w, w);
    cv::Mat back_photo_2;
    cv::flip(back_img, back_photo_2, 1);
    cv::Mat top_half_back;
    cv::flip(back_photo_2.rowRange(0, half_h), top_half_back, 0);
    cv::Mat bottom_half_back;
    cv::flip(back_photo_2.rowRange(half_h, h), bottom_half_back, 0);

    // Создание canvas для объединения
    int H = 128, W = 128;
    cv::Mat sum_canvas(H, W, CV_32F, cv::Scalar(0.0f));
    cv::Mat count_canvas(H, W, CV_32F, cv::Scalar(0.0f));

    // Вычисление координат для различных частей
    int center_h = center_img.rows, center_w = center_img.cols;
    int y_base = (H - center_h) / 2;
    int x_center_start = (W - center_w) / 2;

    int left_h = left_img.rows, left_w = left_img.cols;
    int y_left_start = y_base + (center_h - left_h) / 2;
    int x_left_start = (shift_x < 0) ? x_center_start - left_w - std::abs(shift_x) : 
                                       x_center_start - left_w + shift_x;

    int lhb_h = left_half_back.rows, lhb_w = left_half_back.cols;
    int y_lhb_start = y_base + (center_h - lhb_h) / 2;
    int x_lhb_start = (shift_x < 0) ? x_left_start - lhb_w - std::abs(shift_x) : 
                                      x_left_start - lhb_w + shift_x;

    int right_h = right_img.rows, right_w = right_img.cols;
    int y_right_start = y_base + (center_h - right_h) / 2;
    int x_right_start = (shift_x < 0) ? x_center_start + center_w + std::abs(shift_x) : 
                                        x_center_start + center_w - shift_x;

    int rhb_h = right_half_back.rows, rhb_w = right_half_back.cols;
    int y_rhb_start = y_base + (center_h - rhb_h) / 2;
    int x_rhb_start = (shift_x < 0) ? x_right_start + right_w + std::abs(shift_x) : 
                                      x_right_start + right_w - shift_x;

    int up_h = up_img.rows, up_w = up_img.cols;
    int x_up_start = x_center_start + (center_w - up_w) / 2;
    int y_up_start = (shift_y < 0) ? y_base - up_h - std::abs(shift_y) : 
                                     y_base - up_h + shift_y;

    int thb_h = top_half_back.rows, thb_w = top_half_back.cols;
    int x_thb_start = x_center_start + (center_w - thb_w) / 2;
    int y_thb_start = (shift_y < 0) ? y_up_start - thb_h - std::abs(shift_y) : 
                                      y_up_start - thb_h + shift_y;

    int down_h = down_img.rows, down_w = down_img.cols;
    int x_down_start = x_center_start + (center_w - down_w) / 2;
    int y_down_start = (shift_y < 0) ? y_base + center_h + std::abs(shift_y) : 
                                       y_base + center_h - shift_y;

    int bhb_h = bottom_half_back.rows, bhb_w = bottom_half_back.cols;
    int x_bhb_start = x_center_start + (center_w - bhb_w) / 2;
    int y_bhb_start = (shift_y < 0) ? y_down_start + down_h + std::abs(shift_y) : 
                                      y_down_start + down_h - shift_y;

    // Обработка перекрытий для 6 граней
    if (face_numbers.size() == 6) {
        if (shift_x > 0) {
            // Обработка горизонтальных перекрытий
            cv::Mat l_edge = left_half_back.colRange(lhb_w - shift_x, lhb_w);
            cv::Mat li_edge = left_img.colRange(0, shift_x);
            (l_edge + li_edge) / 2.0f;
            l_edge.copyTo(left_half_back.colRange(lhb_w - shift_x, lhb_w));
            li_edge.copyTo(left_img.colRange(0, shift_x));

            cv::Mat li_edge2 = left_img.colRange(left_w - shift_x, left_w);
            cv::Mat ci_edge = center_img.colRange(0, shift_x);
            (li_edge2 + ci_edge) / 2.0f;
            li_edge2.copyTo(left_img.colRange(left_w - shift_x, left_w));
            ci_edge.copyTo(center_img.colRange(0, shift_x));

            cv::Mat ci_edge_right = center_img.colRange(center_w - shift_x, center_w);
            cv::Mat ri_edge = right_img.colRange(0, shift_x);
            (ci_edge_right + ri_edge) / 2.0f;
            ci_edge_right.copyTo(center_img.colRange(center_w - shift_x, center_w));
            ri_edge.copyTo(right_img.colRange(0, shift_x));

            cv::Mat ri_edge2 = right_img.colRange(right_w - shift_x, right_w);
            cv::Mat rb_edge = right_half_back.colRange(0, shift_x);
            (ri_edge2 + rb_edge) / 2.0f;
            ri_edge2.copyTo(right_img.colRange(right_w - shift_x, right_w));
            rb_edge.copyTo(right_half_back.colRange(0, shift_x));
        }

        if (shift_y > 0) {
            // Обработка вертикальных перекрытий
            cv::Mat tb_edge = top_half_back.rowRange(thb_h - shift_y, thb_h);
            cv::Mat ui_edge = up_img.rowRange(0, shift_y);
            (tb_edge + ui_edge) / 2.0f;
            tb_edge.copyTo(top_half_back.rowRange(thb_h - shift_y, thb_h));
            ui_edge.copyTo(up_img.rowRange(0, shift_y));

            cv::Mat ui_edge2 = up_img.rowRange(up_h - shift_y, up_h);
            cv::Mat ci_edge_top = center_img.rowRange(0, shift_y);
            (ui_edge2 + ci_edge_top) / 2.0f;
            ui_edge2.copyTo(up_img.rowRange(up_h - shift_y, up_h));
            ci_edge_top.copyTo(center_img.rowRange(0, shift_y));

            cv::Mat ci_edge_bot = center_img.rowRange(center_h - shift_y, center_h);
            cv::Mat di_edge = down_img.rowRange(0, shift_y);
            (ci_edge_bot + di_edge) / 2.0f;
            ci_edge_bot.copyTo(center_img.rowRange(center_h - shift_y, center_h));
            di_edge.copyTo(down_img.rowRange(0, shift_y));

            cv::Mat di_edge2 = down_img.rowRange(down_h - shift_y, down_h);
            cv::Mat bb_edge = bottom_half_back.rowRange(0, shift_y);
            (di_edge2 + bb_edge) / 2.0f;
            di_edge2.copyTo(down_img.rowRange(down_h - shift_y, down_h));
            bb_edge.copyTo(bottom_half_back.rowRange(0, shift_y));
        }
    }

    // Добавление интерполяционных блоков для отрицательных сдвигов
    if (face_numbers.size() == 6) {
        if (shift_x < 0) {
            int gap = std::abs(shift_x);
            
            cv::Mat interp_block1(lhb_h, gap, CV_32F);
            for (int col = 0; col < gap; ++col) {
                float alpha = static_cast<float>(col + 1) / (gap + 1);
                interp_block1.col(col) = (1.0f - alpha) * left_half_back.col(lhb_w - 1) + 
                                         alpha * left_img.col(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block1, y_lhb_start, x_lhb_start + lhb_w);

            cv::Mat interp_block2(left_h, gap, CV_32F);
            for (int col = 0; col < gap; ++col) {
                float alpha = static_cast<float>(col + 1) / (gap + 1);
                interp_block2.col(col) = (1.0f - alpha) * left_img.col(left_w - 1) + 
                                         alpha * center_img.col(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block2, y_left_start, x_left_start + left_w);

            cv::Mat interp_block_right1(center_h, gap, CV_32F);
            for (int col = 0; col < gap; ++col) {
                float alpha = static_cast<float>(col + 1) / (gap + 1);
                interp_block_right1.col(col) = (1.0f - alpha) * center_img.col(center_w - 1) + 
                                               alpha * right_img.col(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block_right1, y_base, x_center_start + center_w);

            cv::Mat interp_block_right2(right_h, gap, CV_32F);
            for (int col = 0; col < gap; ++col) {
                float alpha = static_cast<float>(col + 1) / (gap + 1);
                interp_block_right2.col(col) = (1.0f - alpha) * right_img.col(right_w - 1) + 
                                               alpha * right_half_back.col(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block_right2, y_right_start, x_right_start + right_w);
        }

        if (shift_y < 0) {
            int gap = std::abs(shift_y);
            
            cv::Mat interp_block1(gap, thb_w, CV_32F);
            for (int row = 0; row < gap; ++row) {
                float alpha = static_cast<float>(row + 1) / (gap + 1);
                interp_block1.row(row) = (1.0f - alpha) * top_half_back.row(thb_h - 1) + 
                                         alpha * up_img.row(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block1, y_thb_start + thb_h, x_thb_start);

            cv::Mat interp_block2(gap, up_w, CV_32F);
            for (int row = 0; row < gap; ++row) {
                float alpha = static_cast<float>(row + 1) / (gap + 1);
                interp_block2.row(row) = (1.0f - alpha) * up_img.row(up_h - 1) + 
                                         alpha * center_img.row(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block2, y_up_start + up_h, x_up_start);

            cv::Mat interp_block_bottom1(gap, center_w, CV_32F);
            for (int row = 0; row < gap; ++row) {
                float alpha = static_cast<float>(row + 1) / (gap + 1);
                interp_block_bottom1.row(row) = (1.0f - alpha) * center_img.row(center_h - 1) + 
                                                alpha * down_img.row(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block_bottom1, y_base + center_h, x_center_start);

            cv::Mat interp_block_bottom2(gap, down_w, CV_32F);
            for (int row = 0; row < gap; ++row) {
                float alpha = static_cast<float>(row + 1) / (gap + 1);
                interp_block_bottom2.row(row) = (1.0f - alpha) * down_img.row(down_h - 1) + 
                                                alpha * bottom_half_back.row(0);
            }
            add_to_canvas(sum_canvas, count_canvas, interp_block_bottom2, y_down_start + down_h, x_bhb_start);
        }
    }

    // Добавление всех частей на canvas
    add_to_canvas(sum_canvas, count_canvas, center_img, y_base, x_center_start);
    add_to_canvas(sum_canvas, count_canvas, left_half_back, y_lhb_start, x_lhb_start);
    add_to_canvas(sum_canvas, count_canvas, left_img, y_left_start, x_left_start);
    add_to_canvas(sum_canvas, count_canvas, right_img, y_right_start, x_right_start);
    add_to_canvas(sum_canvas, count_canvas, right_half_back, y_rhb_start, x_rhb_start);
    add_to_canvas(sum_canvas, count_canvas, top_half_back, y_thb_start, x_thb_start);
    add_to_canvas(sum_canvas, count_canvas, up_img, y_up_start, x_up_start);
    add_to_canvas(sum_canvas, count_canvas, down_img, y_down_start, x_down_start);
    add_to_canvas(sum_canvas, count_canvas, bottom_half_back, y_bhb_start, x_bhb_start);

    // Вычисление средних значений
    cv::Mat value_canvas = cv::Mat::zeros(H, W, CV_32F);
    cv::divide(sum_canvas, count_canvas, value_canvas, 1.0, -1);

    // Нормализация значений
    std::vector<float> valid_values;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            if (count_canvas.at<float>(row, col) > 0) {
                valid_values.push_back(value_canvas.at<float>(row, col));
            }
        }
    }
    
    float vmin = 0.0f, vmax = 1.0f;
    if (!valid_values.empty()) {
        vmin = *std::min_element(valid_values.begin(), valid_values.end());
        vmax = *std::max_element(valid_values.begin(), valid_values.end());
    }

    // Преобразование в grayscale и применение цветовой карты
    cv::Mat gray(H, W, CV_8U, cv::Scalar(0));
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            if (count_canvas.at<float>(row, col) > 0) {
                float val = (value_canvas.at<float>(row, col) - vmin) / (vmax - vmin + 1e-6) * 255.0f;
                gray.at<uint8_t>(row, col) = static_cast<uint8_t>(clamp(val, 0.0f, 255.0f));
            }
        }
    }

    cv::equalizeHist(gray, gray);
    cv::Mat rgb_canvas;
    cv::applyColorMap(gray, rgb_canvas, cv::COLORMAP_INFERNO);

    // Обрезка пустых областей
    cv::Mat row_sum;
    cv::reduce(count_canvas, row_sum, 1, cv::REDUCE_SUM);
    cv::Mat col_sum;
    cv::reduce(count_canvas, col_sum, 0, cv::REDUCE_SUM);

    std::vector<int> y_used;
    for (int i = 0; i < row_sum.rows; ++i) {
        if (row_sum.at<float>(i, 0) > 0) y_used.push_back(i);
    }
    std::vector<int> x_used;
    for (int i = 0; i < col_sum.cols; ++i) {
        if (col_sum.at<float>(0, i) > 0) x_used.push_back(i);
    }

    int y0_crop = 0, x0_crop = 0;
    if (!y_used.empty() && !x_used.empty()) {
        y0_crop = y_used[0];
        x0_crop = x_used[0];
        rgb_canvas = rgb_canvas(cv::Range(y_used[0], y_used.back() + 1), 
                               cv::Range(x_used[0], x_used.back() + 1));
    }

    // Вычисление координат центральной грани
    int center_top_left_x = x_center_start - x0_crop;
    int center_top_left_y = y_base - y0_crop;
    std::pair<int, int> center_top_left = {center_top_left_x, center_top_left_y};

    std::vector<RelDetection> detections_relative;
    std::pair<int, int> center_dims = {center_w, center_h};

    // Вычисление вектора направления
    float x_det = center_top_left.first + center_w / 2.0f;
    float y_det = center_top_left.second + center_h / 2.0f;

    float x_cor = 16.0f - x_det;
    float y_cor = 12.0f - y_det;
    float r = std::sqrt(x_cor * x_cor + y_cor * y_cor);
    float zenith = r / 16.0f * 55.0f * static_cast<float>(M_PI) / 180.0f;
    float azimuth = std::atan2(y_cor, x_cor);
    float vx = std::sin(zenith) * std::cos(azimuth);
    float vy = std::sin(zenith) * std::sin(azimuth);
    float vz = std::cos(zenith);

    std::tuple<float, float, float> vector = {vx, vy, vz};

    return std::make_tuple(rgb_canvas, center_top_left, detections_relative, center_dims, vector);
}

#endif // PREPROCESS_H