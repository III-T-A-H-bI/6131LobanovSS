#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Оптимизированные ограничения для constant memory (64KB)
#define MAX_LOOKUP_ENTRIES 8000  // 8000 * 8 байт = 64KB
#define MAX_N 2000
#define MAX_H 200000

// Структура для хранения ссылки на символ в подстроке (упакованная в 4 байта)
struct LookupEntryPacked {
    int packed; // Старшие 16 бит: n_idx, младшие 16 бит: k
};

// Constant memory для быстрого доступа к таблице переходов
__constant__ int d_offsets[257];
__constant__ LookupEntryPacked d_lookup_data[MAX_LOOKUP_ENTRIES];

// Вспомогательные функции для упаковки/распаковки
__device__ inline void unpackEntry(LookupEntryPacked entry, int& n_idx, int& k) {
    n_idx = entry.packed >> 16;
    k = entry.packed & 0xFFFF;
}

// --- CUDA Kernel ---
__global__ void massSearchKernel(const unsigned char* d_H, int h_len, int n_count, 
                                 const int* d_lengths, int* d_R) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= h_len) return;

    unsigned char c = d_H[j];
    int start = d_offsets[c];
    int end = d_offsets[c + 1];

    // Для каждого вхождения символа c в наборе подстрок
    for (int idx = start; idx < end; ++idx) {
        int n_idx, k;
        unpackEntry(d_lookup_data[idx], n_idx, k);
        int pos = j - k;

        // Если начальная позиция корректна, инкрементируем счетчик совпадений
        if (pos >= 0) {
            atomicAdd(&d_R[n_idx * h_len + pos], 1);
        }
    }
}

// --- CPU Реализация (для верификации) ---
void massSearchCPU(const std::vector<unsigned char>& H, 
                   const std::vector<std::vector<unsigned char>>& N,
                   const std::vector<int>& lengths,
                   std::vector<std::vector<int>>& results) {
    int h_len = H.size();
    int n_count = N.size();
    
    // Инициализация матрицы R нулями
    std::vector<std::vector<int>> R(n_count, std::vector<int>(h_len, 0));

    // Основная итерация
    for (int j = 0; j < h_len; ++j) {
        unsigned char c = H[j];
        for (int i = 0; i < n_count; ++i) {
            int len = lengths[i];
            for (int k = 0; k < len; ++k) {
                if (N[i][k] == c) {
                    int pos = j - k;
                    if (pos >= 0) {
                        R[i][pos]++;
                    }
                }
            }
        }
    }

    // Интерпретация результатов
    results.resize(n_count);
    for (int i = 0; i < n_count; ++i) {
        int len = lengths[i];
        for (int pos = 0; pos <= h_len - len; ++pos) {
            if (R[i][pos] == len) {
                results[i].push_back(pos);
            }
        }
    }
}

// --- Вспомогательные функции ---
void saveDataToFile(const std::string& filename, const std::vector<unsigned char>& data) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (f) {
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }
}

void savePatternsToFile(const std::string& filename, const std::vector<std::vector<unsigned char>>& N) {
    FILE* f = fopen(filename.c_str(), "w");
    if (f) {
        for (size_t i = 0; i < N.size(); ++i) {
            fprintf(f, "%zu:", i);
            for (unsigned char c : N[i]) {
                fprintf(f, "%02X", c);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
}

int main(int argc, char** argv) {
    // Параметры по умолчанию (уменьшены для constant memory)
    int h_len = 50000;        // Длина буфера H
    int n_count = 100;        // Количество подстрок N
    int min_len = 5;          // Минимальная длина подстроки
    int max_len = 15;         // Максимальная длина подстроки
    int search_mode = 1;      // 0 - факт присутствия, 1 - все позиции

    // Парсинг аргументов
    if (argc >= 6) {
        h_len = std::stoi(argv[1]);
        n_count = std::stoi(argv[2]);
        min_len = std::stoi(argv[3]);
        max_len = std::stoi(argv[4]);
        search_mode = std::stoi(argv[5]);
    }

    std::cout << "Параметры: H=" << h_len << ", N=" << n_count 
              << ", len=[" << min_len << "," << max_len << "], mode=" << search_mode << std::endl;

    if (h_len > MAX_H || n_count > MAX_N) {
        std::cerr << "Ошибка: Превышены максимальные лимиты (H=" << MAX_H << ", N=" << MAX_N << ")" << std::endl;
        return 1;
    }

    // 1. Генерация данных
    std::mt19937 gen(42);
    std::uniform_int_distribution<> byte_dist(0, 255);
    std::uniform_int_distribution<> len_dist(min_len, max_len);

    std::vector<unsigned char> H(h_len);
    for (int i = 0; i < h_len; ++i) H[i] = byte_dist(gen);

    std::vector<std::vector<unsigned char>> N(n_count);
    std::vector<int> lengths(n_count);
    int total_entries_check = 0;
    for (int i = 0; i < n_count; ++i) {
        int len = len_dist(gen);
        lengths[i] = len;
        N[i].resize(len);
        for (int j = 0; j < len; ++j) {
            N[i][j] = byte_dist(gen);
            total_entries_check++;
        }
    }

    std::cout << "Всего записей в lookup table: " << total_entries_check << std::endl;
    
    if (total_entries_check > MAX_LOOKUP_ENTRIES) {
        std::cerr << "Ошибка: Слишком много записей для constant memory! (" << total_entries_check 
                  << " > " << MAX_LOOKUP_ENTRIES << ")" << std::endl;
        std::cerr << "Уменьшите количество или длину подстрок." << std::endl;
        return 1;
    }

    // Сохранение для верификации
    saveDataToFile("buffer_H.bin", H);
    savePatternsToFile("patterns_N.txt", N);
    std::cout << "Данные сохранены в buffer_H.bin и patterns_N.txt" << std::endl;

    // 2. Подготовка Lookup Table для GPU (упакованная версия)
    std::vector<LookupEntryPacked> h_lookup[256];
    for (int i = 0; i < n_count; ++i) {
        for (int k = 0; k < lengths[i]; ++k) {
            unsigned char c = N[i][k];
            LookupEntryPacked packed;
            packed.packed = (i << 16) | (k & 0xFFFF);
            h_lookup[c].push_back(packed);
        }
    }

    std::vector<int> h_offsets(257, 0);
    std::vector<LookupEntryPacked> h_data(total_entries_check);
    int current_offset = 0;
    for (int c = 0; c < 256; ++c) {
        h_offsets[c] = current_offset;
        for (const auto& entry : h_lookup[c]) {
            h_data[current_offset++] = entry;
        }
    }
    h_offsets[256] = total_entries_check;

    // 3. Выделение памяти на GPU
    unsigned char* d_H;
    int* d_lengths;
    int* d_R;
    
    cudaMalloc(&d_H, h_len * sizeof(unsigned char));
    cudaMalloc(&d_lengths, n_count * sizeof(int));
    cudaMalloc(&d_R, n_count * h_len * sizeof(int));

    cudaMemcpy(d_H, H.data(), h_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), n_count * sizeof(int), cudaMemcpyHostToDevice);
    
    // Инициализация матрицы R нулями
    cudaMemset(d_R, 0, n_count * h_len * sizeof(int));

    // Копирование lookup table в constant memory
    cudaMemcpyToSymbol(d_offsets, h_offsets.data(), 257 * sizeof(int));
    cudaMemcpyToSymbol(d_lookup_data, h_data.data(), total_entries_check * sizeof(LookupEntryPacked));

    // 4. Запуск CPU версии и замер времени
    std::vector<std::vector<int>> cpu_results;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    massSearchCPU(H, N, lengths, cpu_results);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "Время выполнения на CPU: " << cpu_time.count() << " мс" << std::endl;

    // 5. Запуск GPU версии и замер времени
    int threadsPerBlock = 256;
    int blocksPerGrid = (h_len + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    massSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_H, h_len, n_count, d_lengths, d_R);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    std::cout << "Время выполнения на GPU: " << gpu_time.count() << " мс" << std::endl;

    // 6. Чтение результатов с GPU и интерпретация
    std::vector<int> d_R_host(n_count * h_len);
    cudaMemcpy(d_R_host.data(), d_R, n_count * h_len * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<std::vector<int>> gpu_results(n_count);
    for (int i = 0; i < n_count; ++i) {
        int len = lengths[i];
        for (int pos = 0; pos <= h_len - len; ++pos) {
            if (d_R_host[i * h_len + pos] == len) {
                gpu_results[i].push_back(pos);
            }
        }
    }

    // 7. Верификация результатов
    bool match = true;
    int total_matches = 0;
    for (int i = 0; i < n_count; ++i) {
        if (cpu_results[i].size() != gpu_results[i].size()) {
            match = false;
            std::cout << "Несоответствие для подстроки " << i << ": CPU=" << cpu_results[i].size() 
                      << ", GPU=" << gpu_results[i].size() << std::endl;
            break;
        }
        for (size_t j = 0; j < cpu_results[i].size(); ++j) {
            if (cpu_results[i][j] != gpu_results[i][j]) {
                match = false;
                std::cout << "Несоответствие позиции для подстроки " << i << std::endl;
                break;
            }
        }
        total_matches += cpu_results[i].size();
    }

    std::cout << "--------------------------------------------------" << std::endl;
    if (match) {
        std::cout << "ОТЧЕТ: Результаты CPU и GPU ПОЛНОСТЬЮ СОВПАДАЮТ." << std::endl;
    } else {
        std::cout << "ОТЧЕТ: ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ между CPU и GPU!" << std::endl;
    }
    
    if (search_mode == 0) {
        int found_count = 0;
        for (int i = 0; i < n_count; ++i) {
            if (!gpu_results[i].empty()) found_count++;
        }
        std::cout << "Режим 'Факт присутствия': Найдено " << found_count << " уникальных подстрок." << std::endl;
    } else {
        std::cout << "Режим 'Все позиции': Всего найдено " << total_matches << " вхождений." << std::endl;
    }
    
    std::cout << "Ускорение (Speedup): " << cpu_time.count() / gpu_time.count() << "x" << std::endl;

    // 8. Сохранение результатов в файл для визуализации
    FILE* res_file = fopen("search_results.txt", "w");
    if (res_file) {
        fprintf(res_file, "=== Results of substring search ===\n");
        fprintf(res_file, "H_len=%d N_count=%d\n\n", h_len, n_count);
        
        // Сохраняем буфер H в читаемом виде (первые 500 байт)
        fprintf(res_file, "Buffer H (first 500 bytes):\n");
        for (int i = 0; i < std::min(500, h_len); ++i) {
            fprintf(res_file, "%02X ", H[i]);
            if ((i + 1) % 32 == 0) fprintf(res_file, "\n");
        }
        fprintf(res_file, "\n\n");
        
        // Сохраняем паттерны и результаты
        for (int i = 0; i < n_count; ++i) {
            fprintf(res_file, "Pattern %d (len=%d): ", i, lengths[i]);
            for (unsigned char c : N[i]) {
                fprintf(res_file, "%02X", c);
            }
            fprintf(res_file, "\nPositions: ");
            if (gpu_results[i].empty()) {
                fprintf(res_file, "NOT FOUND");
            } else {
                for (int pos : gpu_results[i]) {
                    fprintf(res_file, "%d ", pos);
                }
            }
            fprintf(res_file, "\n\n");
        }
        fclose(res_file);
        std::cout << "Результаты сохранены в search_results.txt" << std::endl;
    }

    // Очистка памяти
    cudaFree(d_H);
    cudaFree(d_lengths);
    cudaFree(d_R);

    return 0;
}