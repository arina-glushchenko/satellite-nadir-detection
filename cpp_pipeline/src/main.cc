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

#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"

/**
 * @brief Получает текущее время в микросекундах
 * @return Текущее время в микросекундах
 */
static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief Выводит информацию о тензоре (атрибуты)
 * @param attr Указатель на структуру атрибутов тензора
 */
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i)
    {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    // Для отладки можно раскомментировать:
    // printf("Tensor shape: [%s]\n", dims);
}

/**
 * @brief Загружает файл в память
 * @param file_path Путь к файлу
 * @param file_size Указатель для сохранения размера файла
 * @return Указатель на данные файла или NULL при ошибке
 */
static void *load_file(const char *file_path, size_t *file_size)
{
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL)
    {
        printf("Не удалось открыть файл: %s\n", file_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void *file_data = malloc(size);
    if (file_data == NULL)
    {
        fclose(fp);
        printf("Не удалось выделить память для файла размером: %zu\n", size);
        return NULL;
    }

    if (fread(file_data, 1, size, fp) != size)
    {
        fclose(fp);
        free(file_data);
        printf("Ошибка чтения данных файла!\n");
        return NULL;
    }

    fclose(fp);
    *file_size = size;

    return file_data;
}

/**
 * @brief Загружает и обрабатывает изображение
 * @param image_path Путь к изображению
 * @param input_attr Атрибуты входного тензора модели
 * @param img_height Указатель для сохранения высоты изображения
 * @param img_width Указатель для сохранения ширины изображения
 * @return Указатель на данные изображения или NULL при ошибке
 */
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
        printf("Неподдерживаемый формат тензора: %d\n", input_attr->fmt);
        return NULL;
    }

    int channel = 0;
    unsigned char *image_data = stbi_load(image_path, img_width, img_height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("Ошибка загрузки изображения: %s, причина: %s\n", image_path, stbi_failure_reason());
        FILE* fp = fopen(image_path, "rb");
        if (!fp) {
            printf("Файл %s не существует\n", image_path);
        } else {
            printf("Файл %s существует, но не может быть загружен как изображение\n", image_path);
            fclose(fp);
        }
        return NULL;
    }

    if (*img_width != req_width || *img_height != req_height)
    {
        unsigned char *image_resized = (unsigned char *)malloc(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("Ошибка выделения памяти для resize изображения!\n");
            free(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, *img_width, *img_height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("Ошибка resize изображения!\n");
            free(image_data);
            free(image_resized);
            return NULL;
        }
        free(image_data);
        image_data = image_resized;
    }

    return image_data;
}

/**
 * @brief Выводит информацию об использовании оперативной памяти
 */
static void print_memory_usage() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long total = 0, available = 0;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %ld kB", &total);
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %ld kB", &available);
        }
    }
    if (total > 0) {
        double used = total - available;
        double percent = (used / total) * 100.0;
        printf("Использование RAM: %.2f%% (использовано: %.1f MB / всего: %.1f MB)\n", 
               percent, used / 1024.0, total / 1024.0);
    } else {
        printf("Не удалось прочитать /proc/meminfo\n");
    }
}

/**
 * @brief Основная функция программы
 * @param argc Количество аргументов командной строки
 * @param argv Аргументы командной строки:
 *             [0] - имя программы
 *             [1] - путь к модели RKNN
 *             [2] - путь к входному файлу
 *             [3] - (опционально) "debug" для сохранения отладочных изображений
 *             [4] - (опционально) количество циклов выполнения
 * @return Код возврата: 0 при успехе, -1 при ошибке
 */
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Использование: %s model_path input_path [debug] [loop_count]\n", argv[0]);
        printf("  model_path - путь к файлу модели RKNN\n");
        printf("  input_path - путь к входному файлу с данными\n");
        printf("  debug - (опционально) если указан, сохраняются отладочные изображения\n");
        printf("  loop_count - (опционально) количество циклов выполнения\n");
        return -1;
    }

    // Парсинг аргументов командной строки
    char *model_path = argv[1];
    char *original_input_path = argv[2];
    bool debug_mode = false;
    int loop_count = -1;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "debug") {
            debug_mode = true;
            printf("Режим отладки включен - отладочные изображения будут сохраняться\n");
        } else {
            loop_count = atoi(argv[i]);
            if (loop_count > 0) {
                printf("Количество циклов выполнения: %d\n", loop_count);
            }
        }
    }

    // Инициализация контекста RKNN
    rknn_context ctx = 0;
    size_t model_size = 0;
    void *model_data = load_file(model_path, &model_size);
    if (!model_data) {
        printf("Не удалось загрузить файл модели: %s\n", model_path);
        return -1;
    }

    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("Ошибка rknn_init! ret=%d\n", ret);
        return -1;
    }

    // Получение информации о количестве входов/выходов модели
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("Ошибка rknn_query! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    if (io_num.n_output != 1 && io_num.n_output != 3) {
        printf("Ошибка: количество выходов модели должно быть 1 или 3, получено %d\n", io_num.n_output);
        rknn_destroy(ctx);
        return -1;
    }

    // Получение атрибутов входных тензоров
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("Ошибка rknn_query! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    // Получение атрибутов выходных тензоров
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("Ошибка rknn_query! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    // Получение пользовательской строки (если есть)
    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC) {
        printf("Ошибка rknn_query! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    // Выделение памяти для входных данных
    rknn_tensor_mem *input_mems[1];
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;
    input_attrs[0].type = input_type;
    input_attrs[0].fmt = input_layout;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!input_mems[0]) {
        printf("Не удалось создать память для входа\n");
        rknn_destroy(ctx);
        return -1;
    }

    // Выделение памяти для выходных данных
    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            printf("Не удалось создать память для выхода %d\n", i);
            for (uint32_t j = 0; j < i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            rknn_destroy(ctx);
            return -1;
        }
    }

    // Привязка памяти к входам/выходам модели
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("Ошибка rknn_set_io_mem! ret=%d\n", ret);
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        rknn_destroy_mem(ctx, input_mems[0]);
        rknn_destroy(ctx);
        return -1;
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("Ошибка rknn_set_io_mem! ret=%d\n", ret);
            for (uint32_t j = 0; j <= i; ++j) {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            rknn_destroy(ctx);
            return -1;
        }
    }

    // Переменные для обработки данных
    std::vector<cv::Mat> photos;
    std::vector<int> face_numbers;
    cv::Mat preprocessed_image;
    cv::Mat rgb_image;
    cv::Mat resized_image;
    std::vector<RelDetection> detections_relative;
    std::pair<int, int> center_top_left;
    std::pair<int, int> center_dims;
    std::tuple<float, float, float> vector;

    // Переменные для измерения времени
    int iteration = 0;
    int64_t preprocess_start_us, preprocess_elapse_us;
    int64_t inference_start_us, inference_elapse_us;
    int64_t postprocess_start_us, postprocess_elapse_us;
    int64_t cycle_start_us, cycle_elapse_us;
    unsigned char *input_data = NULL;
    cv::Mat input_mat;
    int64_t read_lines_start_us, read_lines_elapse_us;
    int64_t parse_lines_start_us, parse_lines_elapse_us;
    int64_t select_start_us, select_elapse_us;
    int64_t canvas_start_us, canvas_elapse_us;
    int64_t validate_start_us, validate_elapse_us;
    int64_t prepare_start_us, prepare_elapse_us;
    int64_t memcpy_start_us, memcpy_elapse_us;

    cv::setUseOptimized(true);

    // Основной цикл обработки
    while (true) {
        ++iteration;

        // Проверка количества циклов (если указано)
        if (loop_count > 0 && iteration > loop_count) {
            printf("Достигнуто заданное количество циклов (%d). Завершение.\n", loop_count);
            break;
        }

        cycle_start_us = getCurrentTimeUs();

        // Этап предобработки
        preprocess_start_us = getCurrentTimeUs();

        try {
            // Чтение и парсинг входных данных
            read_lines_start_us = getCurrentTimeUs();
            photos.clear();
            face_numbers.clear();
            std::vector<std::string> lines = read_lines_from_file(original_input_path);
            read_lines_elapse_us = getCurrentTimeUs() - read_lines_start_us;
            
            parse_lines_start_us = getCurrentTimeUs();
            auto photos_result = parse_photos_from_lines(lines, original_input_path);
            photos = std::move(photos_result.first);
            face_numbers = std::move(photos_result.second);
            parse_lines_elapse_us = getCurrentTimeUs() - parse_lines_start_us;

            // Выбор центрального лица
            select_start_us = getCurrentTimeUs();
            int center_face = auto_select_center_face_index_by_hot_pixel_count(photos) + 1;
            select_elapse_us = getCurrentTimeUs() - select_start_us;

            if (center_face < 1 || center_face > 6) {
                throw std::invalid_argument("Центральное лицо должно быть в диапазоне от 1 до 6");
            }

            // Создание комбинированного изображения
            canvas_start_us = getCurrentTimeUs();
            detections_relative.clear();
            preprocessed_image.release(); 
            auto result = create_combined_canvas(photos, face_numbers, center_face, "inferno");
            preprocessed_image = std::move(std::get<0>(result));
            center_top_left = std::move(std::get<1>(result));
            detections_relative = std::move(std::get<2>(result));
            center_dims = std::move(std::get<3>(result));
            vector = std::move(std::get<4>(result));
            canvas_elapse_us = getCurrentTimeUs() - canvas_start_us;

            // Сохранение отладочного изображения если включен режим отладки
            if (debug_mode) {
                std::string debug_path = "debug_preprocessed_" + std::to_string(iteration) + ".jpg";
                if (cv::imwrite(debug_path, preprocessed_image)) {
                    printf("Сохранено отладочное изображение: %s\n", debug_path.c_str());
                } else {
                    printf("Не удалось сохранить отладочное изображение: %s\n", debug_path.c_str());
                }
            }

            // Валидация предобработанного изображения
            validate_start_us = getCurrentTimeUs();
            if (preprocessed_image.empty()) {
                printf("Ошибка: preprocessed_image пустое\n");
                break;
            }
            if (preprocessed_image.type() != CV_8UC3) {
                cv::Mat temp;
                preprocessed_image.convertTo(temp, CV_8UC3);
                preprocessed_image = std::move(temp);
                if (preprocessed_image.empty()) {
                    printf("Ошибка: Не удалось конвертировать preprocessed_image в CV_8UC3\n");
                    break;
                }
                printf("Конвертировано preprocessed_image в CV_8UC3\n");
            }
            validate_elapse_us = getCurrentTimeUs() - validate_start_us;

            // Подготовка данных для модели
            prepare_start_us = getCurrentTimeUs();
            cv::cvtColor(preprocessed_image, rgb_image, cv::COLOR_BGR2RGB);
            input_mat = rgb_image.clone();
            prepare_elapse_us = getCurrentTimeUs() - prepare_start_us;

        } catch (const std::exception& e) {
            printf("Ошибка предобработки: %s\n", e.what());
            break;
        }

        preprocess_elapse_us = getCurrentTimeUs() - preprocess_start_us;

        // Копирование данных в память модели
        memcpy_start_us = getCurrentTimeUs();
        input_data = input_mat.data;

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
        memcpy_elapse_us = getCurrentTimeUs() - memcpy_start_us;

        // Очистка промежуточных данных
        preprocessed_image.release();
        rgb_image.release();
        resized_image.release();
        photos.clear();
        detections_relative.clear();

        // Вывод модели (инференс)
        inference_start_us = getCurrentTimeUs();
        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            printf("Ошибка rknn_run %d\n", ret);
            break;
        }
        inference_elapse_us = getCurrentTimeUs() - inference_start_us;

        // Постобработка результатов
        postprocess_start_us = getCurrentTimeUs();
        float vx = std::get<0>(vector);
        float vy = std::get<1>(vector);
        float vz = std::get<2>(vector);
        printf("Вектор: (%.2f, %.2f, %.2f)\n", vx, vy, vz);

        postprocess_elapse_us = getCurrentTimeUs() - postprocess_start_us;

        // Расчет общего времени цикла
        cycle_elapse_us = getCurrentTimeUs() - cycle_start_us;

        // Очистка входных данных
        input_mat.release();
        
        // Вывод статистики
        print_memory_usage();
        printf("Итерация %d: Время предобработки = %.2fms, Время инференса = %.2fms, Время постобработки = %.2fms, Общее время цикла = %.2fms\n", 
                iteration, preprocess_elapse_us / 1000.f, inference_elapse_us / 1000.f, postprocess_elapse_us / 1000.f, cycle_elapse_us / 1000.f);
        printf("  - read_lines_from_file: %.2fms\n", read_lines_elapse_us / 1000.f);
        printf("  - parse_photos_from_lines: %.2fms\n", parse_lines_elapse_us / 1000.f);
        printf("  - auto_select_center_face_index_by_hot_pixel_count: %.2fms\n", select_elapse_us / 1000.f);
        printf("  - create_combined_canvas: %.2fms\n", canvas_elapse_us / 1000.f);
        printf("  - rknn_run (инференс): %.2fms\n", inference_elapse_us / 1000.f);
        printf("  - постобработка: %.2fms\n", postprocess_elapse_us / 1000.f);
    }

    // Освобождение ресурсов
    if (input_data) {
        input_data = NULL;
    }
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        if (output_mems[i]) {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
    }
    if (input_mems[0]) {
        rknn_destroy_mem(ctx, input_mems[0]);
    }
    rknn_destroy(ctx);

    return 0;
}