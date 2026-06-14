#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_LOOKUP_ENTRIES 8000
#define MAX_N 2000
#define MAX_H 200000

struct LookupEntryPacked {
    int packed;
};

__constant__ int d_offsets[257];
__constant__ LookupEntryPacked d_lookup_data[MAX_LOOKUP_ENTRIES];

__device__ inline void unpackEntry(LookupEntryPacked entry, int& n_idx, int& k) {
    n_idx = entry.packed >> 16;
    k = entry.packed & 0xFFFF;
}

__global__ void massSearchKernel(const unsigned char* d_H, int h_len, int n_count, 
                                 const int* d_lengths, int* d_R) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= h_len) return;

    unsigned char c = d_H[j];
    int start = d_offsets[c];
    int end = d_offsets[c + 1];

    for (int idx = start; idx < end; ++idx) {
        int n_idx, k;
        unpackEntry(d_lookup_data[idx], n_idx, k);
        int pos = j - k;
        if (pos >= 0) {
            atomicAdd(&d_R[n_idx * h_len + pos], 1);
        }
    }
}

void massSearchCPU(const std::vector<unsigned char>& H, 
                   const std::vector<std::vector<unsigned char>>& N,
                   const std::vector<int>& lengths,
                   std::vector<std::vector<int>>& results) {
    int h_len = H.size();
    int n_count = N.size();
    std::vector<std::vector<int>> R(n_count, std::vector<int>(h_len, 0));

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

int main(int argc, char** argv) {
    int h_len = 50000;
    int n_count = 50;
    int min_len = 8;
    int max_len = 12;
    int search_mode = 1;

    if (argc >= 6) {
        h_len = std::stoi(argv[1]);
        n_count = std::stoi(argv[2]);
        min_len = std::stoi(argv[3]);
        max_len = std::stoi(argv[4]);
        search_mode = std::stoi(argv[5]);
    }

    std::cout << "=== ТЕСТОВЫЙ РЕЖИМ С ГАРАНТИРОВАННЫМИ СОВПАДЕНИЯМИ ===" << std::endl;
    std::cout << "Параметры: H=" << h_len << ", N=" << n_count 
              << ", len=[" << min_len << "," << max_len << "]" << std::endl;

    std::mt19937 gen(42);
    std::uniform_int_distribution<> byte_dist(0, 255);
    std::uniform_int_distribution<> len_dist(min_len, max_len);
    std::uniform_int_distribution<> pos_dist(0, h_len - max_len);

    // Генерируем буфер со случайными данными
    std::vector<unsigned char> H(h_len);
    for (int i = 0; i < h_len; ++i) H[i] = byte_dist(gen);

    // Генерируем паттерны и вставляем их в буфер
    std::vector<std::vector<unsigned char>> N(n_count);
    std::vector<int> lengths(n_count);
    std::vector<std::vector<int>> expected_positions(n_count);
    
    int inserted_count = 0;
    
    for (int i = 0; i < n_count; ++i) {
        int len = len_dist(gen);
        lengths[i] = len;
        N[i].resize(len);
        
        // Генерируем случайный паттерн
        for (int j = 0; j < len; ++j) {
            N[i][j] = byte_dist(gen);
        }
        
        // Вставляем паттерн в буфер (для 40% паттернов)
        if (i < n_count * 0.4) {
            int num_insertions = 1 + (i % 3); // 1-3 вставки на паттерн
            for (int ins = 0; ins < num_insertions; ++ins) {
                int pos = pos_dist(gen);
                for (int j = 0; j < len; ++j) {
                    if (pos + j < h_len) {
                        H[pos + j] = N[i][j];
                    }
                }
                expected_positions[i].push_back(pos);
                inserted_count++;
            }
            std::cout << "  Паттерн " << i << " (len=" << len << ") вставлен в позиции: ";
            for (int pos : expected_positions[i]) std::cout << pos << " ";
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n✓ Вставлено " << inserted_count << " копий паттернов" << std::endl;

    // Сохраняем данные
    FILE* f = fopen("buffer_H.bin", "wb");
    if (f) { fwrite(H.data(), 1, H.size(), f); fclose(f); }
    
    f = fopen("patterns_N.txt", "w");
    if (f) {
        for (size_t i = 0; i < N.size(); ++i) {
            fprintf(f, "%zu:", i);
            for (unsigned char c : N[i]) fprintf(f, "%02X", c);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    // Подготовка Lookup Table
    std::vector<LookupEntryPacked> h_lookup[256];
    int total_entries = 0;
    for (int i = 0; i < n_count; ++i) {
        for (int k = 0; k < lengths[i]; ++k) {
            unsigned char c = N[i][k];
            LookupEntryPacked packed;
            packed.packed = (i << 16) | (k & 0xFFFF);
            h_lookup[c].push_back(packed);
            total_entries++;
        }
    }

    std::vector<int> h_offsets(257, 0);
    std::vector<LookupEntryPacked> h_data(total_entries);
    int current_offset = 0;
    for (int c = 0; c < 256; ++c) {
        h_offsets[c] = current_offset;
        for (const auto& entry : h_lookup[c]) {
            h_data[current_offset++] = entry;
        }
    }
    h_offsets[256] = total_entries;

    // CUDA память
    unsigned char* d_H;
    int* d_lengths;
    int* d_R;
    
    cudaMalloc(&d_H, h_len * sizeof(unsigned char));
    cudaMalloc(&d_lengths, n_count * sizeof(int));
    cudaMalloc(&d_R, n_count * h_len * sizeof(int));

    cudaMemcpy(d_H, H.data(), h_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), n_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_R, 0, n_count * h_len * sizeof(int));

    cudaMemcpyToSymbol(d_offsets, h_offsets.data(), 257 * sizeof(int));
    cudaMemcpyToSymbol(d_lookup_data, h_data.data(), total_entries * sizeof(LookupEntryPacked));

    // CPU
    std::vector<std::vector<int>> cpu_results;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    massSearchCPU(H, N, lengths, cpu_results);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (h_len + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    auto gpu_start = std::chrono::high_resolution_clock::now();
    massSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_H, h_len, n_count, d_lengths, d_R);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    // Получение результатов
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

    // Верификация
    bool match = true;
    int total_matches = 0;
    for (int i = 0; i < n_count; ++i) {
        if (cpu_results[i].size() != gpu_results[i].size()) {
            match = false;
            break;
        }
        total_matches += cpu_results[i].size();
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Время CPU: " << cpu_time.count() << " мс" << std::endl;
    std::cout << "Время GPU: " << gpu_time.count() << " мс" << std::endl;
    std::cout << "Ускорение: " << (cpu_time.count() / gpu_time.count()) << "x" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    if (match) {
        std::cout << "✓ Результаты CPU и GPU СОВПАДАЮТ" << std::endl;
    } else {
        std::cout << "✗ ОШИБКА: результаты не совпадают!" << std::endl;
    }
    
    std::cout << "Найдено вхождений: " << total_matches << std::endl;
    std::cout << "Ожидалось вхождений: " << inserted_count << std::endl;

    // Сохранение результатов
    FILE* res_file = fopen("search_results.txt", "w");
    if (res_file) {
        fprintf(res_file, "=== Results of substring search ===\n");
        fprintf(res_file, "H_len=%d N_count=%d\n\n", h_len, n_count);
        
        for (int i = 0; i < n_count; ++i) {
            fprintf(res_file, "Pattern %d (len=%d): ", i, lengths[i]);
            for (unsigned char c : N[i]) fprintf(res_file, "%02X", c);
            fprintf(res_file, "\nPositions: ");
            if (gpu_results[i].empty()) {
                fprintf(res_file, "NOT FOUND");
            } else {
                for (int pos : gpu_results[i]) fprintf(res_file, "%d ", pos);
            }
            fprintf(res_file, "\n\n");
        }
        fclose(res_file);
    }

    cudaFree(d_H);
    cudaFree(d_lengths);
    cudaFree(d_R);

    return 0;
}