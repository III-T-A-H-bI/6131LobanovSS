#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Структуры для работы с BMP (8-bit Grayscale) ---
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t type;          // Тип файла (должен быть 0x4D42 для BMP)
    uint32_t size;          // Размер файла в байтах
    uint16_t reserved1;     // Зарезервировано
    uint16_t reserved2;     // Зарезервировано
    uint32_t offset;        // Смещение до пиксельных данных
};

struct BMPInfoHeader {
    uint32_t size;          // Размер заголовка
    int32_t width;          // Ширина изображения
    int32_t height;         // Высота изображения (положительная - снизу вверх)
    uint16_t planes;        // Количество плоскостей (всегда 1)
    uint16_t bitCount;      // Бит на пиксель (8 для grayscale)
    uint32_t compression;   // Тип сжатия (0 - без сжатия)
    uint32_t sizeImage;     // Размер изображения в байтах
    int32_t xPelsPerMeter;  // Горизонтальное разрешение
    int32_t yPelsPerMeter;  // Вертикальное разрешение
    uint32_t clrUsed;       // Количество используемых цветов
    uint32_t clrImportant;  // Количество важных цветов
};
#pragma pack(pop)

// Класс для работы с 8-битными grayscale BMP изображениями
class SimpleBMP {
public:
    int width, height;
    std::vector<unsigned char> pixels;

    // Загрузка BMP файла
    bool load(const std::string& filename) {
        FILE* f = fopen(filename.c_str(), "rb");
        if (!f) return false;

        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;
        fread(&fileHeader, sizeof(BMPFileHeader), 1, f);
        fread(&infoHeader, sizeof(BMPInfoHeader), 1, f);

        // Проверка, что файл является 8-битным grayscale BMP
        if (fileHeader.type != 0x4D42 || infoHeader.bitCount != 8) {
            fclose(f);
            std::cerr << "Ошибка: Поддерживаются только 8-битные grayscale BMP!" << std::endl;
            return false;
        }

        width = infoHeader.width;
        height = abs(infoHeader.height);
        
        // Переход к пиксельным данным (пропуск палитры)
        fseek(f, fileHeader.offset, SEEK_SET);
        
        // Вычисление размера строки с учетом выравнивания до 4 байт
        int rowSize = (width + 3) & ~3;
        pixels.resize(width * height);

        std::vector<unsigned char> rowBuffer(rowSize);
        // Чтение строк снизу вверх (особенность BMP формата)
        for (int y = 0; y < height; ++y) {
            fread(rowBuffer.data(), 1, rowSize, f);
            for (int x = 0; x < width; ++x) {
                // Переворачиваем изображение (BMP хранит снизу вверх)
                pixels[(height - 1 - y) * width + x] = rowBuffer[x];
            }
        }
        fclose(f);
        return true;
    }

    // Сохранение BMP файла
    bool save(const std::string& filename) const {
        FILE* f = fopen(filename.c_str(), "wb");
        if (!f) return false;

        BMPFileHeader fileHeader = {0};
        BMPInfoHeader infoHeader = {0};

        int rowSize = (width + 3) & ~3;  // Выравнивание строки до 4 байт
        uint32_t paletteSize = 256 * 4;   // 256 цветов по 4 байта (BGRA)
        uint32_t imageSize = rowSize * height;

        // Заполнение заголовка файла
        fileHeader.type = 0x4D42;  // "BM" в little-endian
        fileHeader.size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize + imageSize;
        fileHeader.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize;

        // Заполнение информационного заголовка
        infoHeader.size = sizeof(BMPInfoHeader);
        infoHeader.width = width;
        infoHeader.height = height;  // Положительное значение - изображение снизу вверх
        infoHeader.planes = 1;
        infoHeader.bitCount = 8;
        infoHeader.compression = 0;
        infoHeader.sizeImage = imageSize;

        // Запись заголовков
        fwrite(&fileHeader, sizeof(BMPFileHeader), 1, f);
        fwrite(&infoHeader, sizeof(BMPInfoHeader), 1, f);

        // Запись градационной палитры (оттенки серого)
        unsigned char palette[1024] = {0};
        for (int i = 0; i < 256; ++i) {
            palette[i * 4 + 0] = i; // B (синий)
            palette[i * 4 + 1] = i; // G (зеленый)
            palette[i * 4 + 2] = i; // R (красный)
            palette[i * 4 + 3] = 0; // Зарезервировано
        }
        fwrite(palette, 1, 1024, f);

        // Запись пиксельных данных (снизу вверх, как требует BMP формат)
        std::vector<unsigned char> rowBuffer(rowSize, 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                rowBuffer[x] = pixels[(height - 1 - y) * width + x];
            }
            fwrite(rowBuffer.data(), 1, rowSize, f);
        }

        fclose(f);
        return true;
    }
};

// --- CUDA Kernel для билатеральной фильтрации ---
// Входные данные в текстуре нормализованы к диапазону [0.0, 1.0]
// sigma_r_normalized должна быть в том же диапазоне (исходная sigma_r / 255.0)
__global__ void bilateralFilterKernel(unsigned char* d_out, int width, int height, 
                                      float sigma_d, float sigma_r_normalized, 
                                      cudaTextureObject_t texObj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Текстура возвращает значения в диапазоне 0.0-1.0 (благодаря cudaReadModeNormalizedFloat)
    float center_val = tex2D<float>(texObj, x + 0.5f, y + 0.5f);
    
    float sum_weighted = 0.0f;
    float sum_weights = 0.0f;
    int valid_pixels = 0;

    // Предвычисление констант для ускорения (2 * sigma^2)
    float sigma_d_sq_2 = 2.0f * sigma_d * sigma_d;
    float sigma_r_sq_2 = 2.0f * sigma_r_normalized * sigma_r_normalized;

    // Используем маску 3x3 для соответствия CPU версии
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Явная проверка границ для точного соответствия CPU
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float val = tex2D<float>(texObj, nx + 0.5f, ny + 0.5f);
                
                // Вычисление пространственного расстояния (в пикселях)
                float dist_sq = dx * dx + dy * dy;
                // Вычисление разницы яркостей (в нормализованном диапазоне)
                float val_diff_sq = (val - center_val) * (val - center_val);

                // Гауссовы веса
                float w_spatial = expf(-dist_sq / sigma_d_sq_2);
                float w_range = expf(-val_diff_sq / sigma_r_sq_2);
                float weight = w_spatial * w_range;

                sum_weighted += val * weight;
                sum_weights += weight;
                valid_pixels++;
            }
        }
    }
    
    // Защита от деления на ноль (на случай, если все пиксели вышли за границы)
    float result;
    if (valid_pixels > 0 && sum_weights > 0) {
        result = sum_weighted / sum_weights;
    } else {
        result = center_val;
    }
    
    // Конвертация обратно в диапазон [0, 255] для сохранения в BMP
    d_out[y * width + x] = (unsigned char)(result * 255.0f);
}

// --- CPU Реализация билатеральной фильтрации для сравнения ---
// Работает с данными в диапазоне [0, 255]
void bilateralFilterCPU(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, 
                        int width, int height, float sigma_d, float sigma_r) {
    // Копируем входные данные в выходной массив
    out = in;
    
    // Предвычисление констант для ускорения (2 * sigma^2)
    float sigma_d_sq_2 = 2.0f * sigma_d * sigma_d;
    float sigma_r_sq_2 = 2.0f * sigma_r * sigma_r;

    // Обход всех пикселей изображения
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float center_val = in[y * width + x];
            float sum_weighted = 0.0f;
            float sum_weights = 0.0f;

            // Ядро 3x3
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    // Обработка краев: clamp к ближайшему существующему пикселю
                    int nx = std::max(0, std::min(width - 1, x + dx));
                    int ny = std::max(0, std::min(height - 1, y + dy));
                    
                    float val = in[ny * width + nx];
                    float dist_sq = dx * dx + dy * dy;
                    float val_diff_sq = (val - center_val) * (val - center_val);

                    // Гауссовы веса
                    float w_spatial = std::exp(-dist_sq / sigma_d_sq_2);
                    float w_range = std::exp(-val_diff_sq / sigma_r_sq_2);
                    float weight = w_spatial * w_range;

                    sum_weighted += val * weight;
                    sum_weights += weight;
                }
            }
            // Результат сразу в диапазоне 0-255
            out[y * width + x] = static_cast<unsigned char>(sum_weighted / sum_weights);
        }
    }
}

int main(int argc, char** argv) {
    // Проверка аргументов командной строки
    if (argc < 4) {
        std::cerr << "Использование: " << argv[0] << " <input.bmp> <output_gpu.bmp> <output_cpu.bmp> [sigma_d] [sigma_r]" << std::endl;
        std::cerr << "Значения по умолчанию: sigma_d = 1.5, sigma_r = 25.0" << std::endl;
        return 1;
    }

    // Парсинг аргументов
    std::string inputFile = argv[1];
    std::string gpuOutputFile = argv[2];
    std::string cpuOutputFile = argv[3];
    
    float sigma_d = (argc >= 5) ? std::stof(argv[4]) : 1.5f;
    float sigma_r = (argc >= 6) ? std::stof(argv[5]) : 25.0f;

    // Загрузка изображения
    SimpleBMP img;
    if (!img.load(inputFile)) {
        std::cerr << "Не удалось загрузить изображение: " << inputFile << std::endl;
        return 1;
    }

    std::cout << "Изображение загружено: " << img.width << "x" << img.height << std::endl;
    std::cout << "Параметры: sigma_d = " << sigma_d << ", sigma_r = " << sigma_r << std::endl;
    std::cout << "Размер изображения: " << (img.width * img.height * sizeof(unsigned char)) / 1024 << " KB" << std::endl;

    // ========== CPU Вычисления ==========
    std::cout << "\n--- Выполнение CPU фильтрации ---" << std::endl;
    
    // Сохраняем оригинальные пиксели для GPU (важно!)
    std::vector<unsigned char> originalPixels = img.pixels;
    
    std::vector<unsigned char> cpuResult;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    bilateralFilterCPU(originalPixels, cpuResult, img.width, img.height, sigma_d, sigma_r);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;
    std::cout << "Время выполнения на CPU: " << cpuTime.count() << " мс" << std::endl;
    
    // Сохранение CPU результата
    SimpleBMP cpuImg = img;
    cpuImg.pixels = cpuResult;
    cpuImg.save(cpuOutputFile);
    std::cout << "Результат CPU сохранен в: " << cpuOutputFile << std::endl;

    // ========== GPU Вычисления ==========
    std::cout << "\n--- Выполнение GPU фильтрации ---" << std::endl;
    
    // Масштабирование sigma_r для нормализованного диапазона [0.0-1.0]
    // CPU работает с диапазоном яркостей [0-255], GPU с [0.0-1.0]
    // Поэтому sigma_r на GPU должна быть в 255 раз меньше
    float sigma_r_normalized = sigma_r / 255.0f;
    std::cout << "Нормализованная sigma_r для GPU: " << sigma_r_normalized << std::endl;
    
    // Выделение памяти на GPU для выходного изображения
    unsigned char* d_out = nullptr;
    size_t imageSize = img.width * img.height * sizeof(unsigned char);
    cudaMalloc(&d_out, imageSize);

    // Создание массива для текстуры
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, img.width, img.height);

    // Копирование оригинальных данных (не обработанных CPU) в текстуру
    cudaMemcpyToArray(cuArray, 0, 0, originalPixels.data(), imageSize, cudaMemcpyHostToDevice);

    // Описание ресурса для текстуры
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Описание параметров текстуры
    struct cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Clamp по краям (повтор последнего пикселя)
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;       // Точечная фильтрация (без интерполяции)
    texDesc.readMode = cudaReadModeNormalizedFloat; // ВАЖНО: конвертация 0-255 в 0.0-1.0
    texDesc.normalizedCoords = 0;                   // Координаты в пикселях, а не в normalized

    // Создание текстурного объекта
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Настройка сетки блоков и потоков
    dim3 blockSize(16, 16);  // 16x16 = 256 потоков на блок
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, 
                  (img.height + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;

    // Синхронизация перед замером времени
    cudaDeviceSynchronize();
    auto gpuStart = std::chrono::high_resolution_clock::now();
    
    // Запуск CUDA ядра (используем нормализованную sigma_r)
    bilateralFilterKernel<<<gridSize, blockSize>>>(d_out, img.width, img.height, 
                                                    sigma_d, sigma_r_normalized, texObj);
    
    // Ожидание завершения ядра
    cudaDeviceSynchronize();
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpuTime = gpuEnd - gpuStart;
    std::cout << "Время выполнения на GPU: " << gpuTime.count() << " мс" << std::endl;

    // Проверка ошибок CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA ошибка при запуске ядра: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Копирование результата обратно на хост
    std::vector<unsigned char> gpuResult(img.width * img.height);
    cudaMemcpy(gpuResult.data(), d_out, imageSize, cudaMemcpyDeviceToHost);
    
    // Проверка ошибок при копировании
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA ошибка при копировании результата: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Сохранение GPU результата
    SimpleBMP gpuImg = img;
    gpuImg.pixels = gpuResult;
    gpuImg.save(gpuOutputFile);
    std::cout << "Результат GPU сохранен в: " << gpuOutputFile << std::endl;

    // ========== Сравнение результатов ==========
    std::cout << "\n--- Сравнение результатов ---" << std::endl;
    
    // Вывод первых 20 пикселей для визуальной проверки
    std::cout << "Первые 20 пикселей CPU: ";
    for(int i = 0; i < 20 && i < cpuResult.size(); i++) {
        std::cout << (int)cpuResult[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Первые 20 пикселей GPU: ";
    for(int i = 0; i < 20 && i < gpuResult.size(); i++) {
        std::cout << (int)gpuResult[i] << " ";
    }
    std::cout << std::endl;

    // Вычисление статистики различий между CPU и GPU результатами
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;
    int pixelsWithDiff = 0;
    
    for (size_t i = 0; i < cpuResult.size(); ++i) {
        float diff = std::abs((int)cpuResult[i] - (int)gpuResult[i]);
        sumDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 1.0f) pixelsWithDiff++;
    }
    float avgDiff = sumDiff / cpuResult.size();
    float percentDiff = (pixelsWithDiff * 100.0f) / cpuResult.size();
    
    std::cout << "\nСтатистика различий:" << std::endl;
    std::cout << "Максимальная разница: " << maxDiff << std::endl;
    std::cout << "Средняя разница: " << avgDiff << std::endl;
    std::cout << "Пикселей с разницей > 1: " << percentDiff << "% (" << pixelsWithDiff << " из " << cpuResult.size() << ")" << std::endl;
    
    if (maxDiff > 5.0f) {
        std::cout << "ВНИМАНИЕ: Результаты CPU и GPU значительно различаются!" << std::endl;
    } else if (maxDiff > 1.0f) {
        std::cout << "Результаты CPU и GPU имеют небольшие различия (возможно из-за округления)." << std::endl;
    } else {
        std::cout << "Результаты CPU и GPU совпадают (в пределах погрешности округления)." << std::endl;
    }

    // Вычисление ускорения
    std::cout << "\n--- Производительность ---" << std::endl;
    std::cout << "CPU время: " << cpuTime.count() << " мс" << std::endl;
    std::cout << "GPU время: " << gpuTime.count() << " мс" << std::endl;
    std::cout << "Ускорение (Speedup): " << (cpuTime.count() / gpuTime.count()) << "x" << std::endl;

    // ========== Очистка ресурсов ==========
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);

    std::cout << "\nГотово!" << std::endl;
    return 0;
}