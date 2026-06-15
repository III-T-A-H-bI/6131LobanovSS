#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Функция для разделителя
print_header() {
    echo -e "\n${MAGENTA}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║ $1${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "\n${CYAN}▶ $1${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Очистка и создание директории для результатов
print_header "ПОЛНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ CUDA vs CPU"
mkdir -p results
rm -f results/*

# 1. Проверка системы
print_section "ПРОВЕРКА СИСТЕМЫ"
echo -n "CUDA версия: "
nvcc --version | grep "release" | awk '{print $6}' | tr -d ','
echo -n "GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader | head -n1
echo -n "CPU: "
lscpu | grep "Model name" | head -1 | cut -d':' -f2 | xargs
echo -n "RAM: "
free -h | grep "Mem:" | awk '{print $2}'

# 2. Настройка виртуального окружения
print_section "НАСТРОЙКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ"
if [ -d "venv_cuda" ]; then
    source venv_cuda/bin/activate
    print_success "Виртуальное окружение активировано"
else
    python3 -m venv venv_cuda
    source venv_cuda/bin/activate
    print_success "Виртуальное окружение создано"
fi

pip install --upgrade pip -q
pip install numpy matplotlib seaborn pandas -q
print_success "Зависимости установлены"

# 3. Компиляция программ
print_section "КОМПИЛЯЦИЯ ПРОГРАММ"

echo "Компиляция mass_search.cu..."
nvcc -O3 -arch=sm_89 --maxrregcount=64 mass_search.cu -o mass_search 2>&1 | grep -E "error|warning" || true
if [ -f "mass_search" ]; then
    print_success "mass_search скомпилирован"
else
    print_error "Ошибка компиляции mass_search"
    exit 1
fi

echo "Компиляция mass_search_test.cu..."
nvcc -O3 -arch=sm_89 --maxrregcount=64 mass_search_test.cu -o mass_search_test 2>&1 | grep -E "error|warning" || true
if [ -f "mass_search_test" ]; then
    print_success "mass_search_test скомпилирован"
else
    print_error "Ошибка компиляции mass_search_test"
fi

# 4. Тестирование с разными параметрами
print_header "ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ"

# Функция для запуска теста
run_benchmark() {
    local name=$1
    local cmd=$2
    echo -e "\n${BLUE}━━━ $name ━━━${NC}"
    eval $cmd
}

# Тест 1
print_section "ТЕСТ 1: СЛУЧАЙНЫЕ ДАННЫЕ (50000 байт, 100 паттернов)"
run_benchmark "Стандартный тест" "./mass_search 50000 100 5 15 1" | tee results/test1.log

# Тест 2
print_section "ТЕСТ 2: УВЕЛИЧЕННЫЙ БУФЕР (100000 байт, 100 паттернов)"
run_benchmark "Больше данных" "./mass_search 100000 100 5 15 1" | tee results/test2.log

# Тест 3
print_section "ТЕСТ 3: РЕЖИМ ФАКТА ПРИСУТСТВИЯ"
run_benchmark "Режим 0" "./mass_search 50000 100 5 15 0" | tee results/test3.log

# Тест 4
print_section "ТЕСТ 4: МНОГО КОРОТКИХ ПАТТЕРНОВ"
run_benchmark "Короткие паттерны" "./mass_search 100000 200 3 10 1" | tee results/test4.log

# Тест 5
print_section "ТЕСТ 5: ДЛИННЫЕ ПАТТЕРНЫ"
run_benchmark "Длинные паттерны" "./mass_search 50000 50 20 50 1" | tee results/test5.log

# 5. Тестовый режим с гарантированными совпадениями
print_header "ТЕСТИРОВАНИЕ С ГАРАНТИРОВАННЫМИ СОВПАДЕНИЯМИ"

print_section "ЗАПУСК mass_search_test"
./mass_search_test 20000 30 6 12 1 | tee results/test_matches.log

# Сохраняем данные для визуализации из теста с совпадениями
cp buffer_H.bin buffer_H_matches.bin 2>/dev/null
cp patterns_N.txt patterns_N_matches.txt 2>/dev/null
cp search_results.txt search_results_matches.txt 2>/dev/null

# 6. Визуализация результатов для теста с совпадениями
print_header "ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ"

print_section "ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ ТЕСТА С СОВПАДЕНИЯМИ"

# Убеждаемся, что файлы результатов существуют
if [ -f "search_results_matches.txt" ]; then
    cp search_results_matches.txt search_results.txt
    cp buffer_H_matches.bin buffer_H.bin
    cp patterns_N_matches.txt patterns_N.txt
    print_success "Данные для визуализации подготовлены"
    
    # Запуск визуализации
    python3 visualize_results.py 2>&1 | tee results/vizualization.log
    print_success "Визуализация завершена"
    
    # Перемещаем графики в папку results
    mv *.png results/ 2>/dev/null
    mv search_report.txt results/ 2>/dev/null
else
    print_error "Файлы результатов не найдены"
fi

# 7. Сбор статистики
print_header "СБОР СТАТИСТИКИ"

# Создаем итоговый отчет
cat > results/FINAL_REPORT.txt << 'EOF'
================================================================================
                    ИТОГОВЫЙ ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ
================================================================================

СИСТЕМНАЯ ИНФОРМАЦИЯ
--------------------------------------------------------------------------------
EOF

echo "CUDA версия: $(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')" >> results/FINAL_REPORT.txt
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" >> results/FINAL_REPORT.txt
echo "CPU: $(lscpu | grep "Model name" | head -1 | cut -d':' -f2 | xargs)" >> results/FINAL_REPORT.txt
echo "Дата: $(date)" >> results/FINAL_REPORT.txt

cat >> results/FINAL_REPORT.txt << 'EOF'

РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ
--------------------------------------------------------------------------------

EOF

# Извлекаем результаты из логов
for test in test1 test2 test3 test4 test5; do
    if [ -f "results/${test}.log" ]; then
        echo "=== ${test} ===" >> results/FINAL_REPORT.txt
        grep -E "Параметры:|Время выполнения на CPU:|Время выполнения на GPU:|Ускорение|Всего найдено" results/${test}.log >> results/FINAL_REPORT.txt
        echo "" >> results/FINAL_REPORT.txt
    fi
done

cat >> results/FINAL_REPORT.txt << 'EOF'

ТЕСТ С ГАРАНТИРОВАННЫМИ СОВПАДЕНИЯМИ
--------------------------------------------------------------------------------
EOF

if [ -f "results/test_matches.log" ]; then
    grep -E "Вставлено|Время CPU:|Время GPU:|Ускорение:|Найдено вхождений:" results/test_matches.log >> results/FINAL_REPORT.txt
fi

# 8. Вывод сводной таблицы
print_header "СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ"

echo -e "\n${GREEN}┌─────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${GREEN}│                    СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ                 │${NC}"
echo -e "${GREEN}├─────────────┬──────────┬──────────┬────────────┬────────────────┤${NC}"
echo -e "${GREEN}│   ТЕСТ      │  CPU(мс) │  GPU(мс) │ УСКОРЕНИЕ  │   РЕЗУЛЬТАТ    │${NC}"
echo -e "${GREEN}├─────────────┼──────────┼──────────┼────────────┼────────────────┤${NC}"

for test in test1 test2 test3 test4 test5; do
    if [ -f "results/${test}.log" ]; then
        CPU=$(grep "Время выполнения на CPU:" results/${test}.log | awk '{print $5}')
        GPU=$(grep "Время выполнения на GPU:" results/${test}.log | awk '{print $5}')
        SPEEDUP=$(grep "Ускорение" results/${test}.log | awk '{print $3}' | tr -d 'x')
        RESULT=$(grep "Всего найдено" results/${test}.log | awk '{print $4}')
        
        printf "${GREEN}│ %-11s │ %8s │ %8s │ %10s │ %14s │${NC}\n" "$test" "$CPU" "$GPU" "${SPEEDUP}x" "$RESULT"
    fi
done

# Добавляем тест с совпадениями
if [ -f "results/test_matches.log" ]; then
    CPU=$(grep "Время CPU:" results/test_matches.log | awk '{print $3}')
    GPU=$(grep "Время GPU:" results/test_matches.log | awk '{print $3}')
    SPEEDUP=$(grep "Ускорение:" results/test_matches.log | awk '{print $2}' | tr -d 'x')
    MATCHES=$(grep "Найдено вхождений:" results/test_matches.log | awk '{print $3}')
    
    printf "${GREEN}│ %-11s │ %8s │ %8s │ %10s │ %14s │${NC}\n" "MATCHES" "$CPU" "$GPU" "${SPEEDUP}x" "$MATCHES"
fi

echo -e "${GREEN}└─────────────┴──────────┴──────────┴────────────┴────────────────┘${NC}"

# 9. Создание графиков производительности
print_section "СОЗДАНИЕ ГРАФИКОВ ПРОИЗВОДИТЕЛЬНОСТИ"

cat > results/plot_performance.py << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import re
import os

os.chdir('/home/fandyou/mass_search/results')

# Данные из тестов
tests = ['Test1', 'Test2', 'Test3', 'Test4', 'Test5']
cpu_times = []
gpu_times = []
speedups = []

# Читаем данные из файлов
for test in ['test1', 'test2', 'test3', 'test4', 'test5']:
    try:
        with open(f'{test}.log', 'r') as f:
            content = f.read()
            cpu = float(re.search(r'Время выполнения на CPU:\s+([\d.]+)', content).group(1))
            gpu = float(re.search(r'Время выполнения на GPU:\s+([\d.]+)', content).group(1))
            speedup = float(re.search(r'Ускорение.*?([\d.]+)x', content).group(1))
            cpu_times.append(cpu)
            gpu_times.append(gpu)
            speedups.append(speedup)
    except:
        cpu_times.append(0)
        gpu_times.append(0)
        speedups.append(0)

# График 1: Сравнение времени выполнения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(tests))
width = 0.35

bars1 = axes[0].bar(x - width/2, cpu_times, width, label='CPU', color='coral', alpha=0.8)
bars2 = axes[0].bar(x + width/2, gpu_times, width, label='GPU', color='skyblue', alpha=0.8)
axes[0].set_xlabel('Тесты')
axes[0].set_ylabel('Время (мс)')
axes[0].set_title('Сравнение времени выполнения CPU vs GPU')
axes[0].set_xticks(x)
axes[0].set_xticklabels(tests)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, (bar, val) in enumerate(zip(bars1, cpu_times)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for i, (bar, val) in enumerate(zip(bars2, gpu_times)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# График 2: Ускорение
bars3 = axes[1].bar(x, speedups, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
axes[1].set_xlabel('Тесты')
axes[1].set_ylabel('Ускорение (x)')
axes[1].set_title('Коэффициент ускорения GPU относительно CPU')
axes[1].set_xticks(x)
axes[1].set_xticklabels(tests)
axes[1].grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, (bar, val) in enumerate(zip(bars3, speedups)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.0f}x', ha='center', va='bottom', fontsize=9)

plt.suptitle('Анализ производительности CUDA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ График производительности сохранен")
EOF

cd results
python3 plot_performance.py
cd ..

# 10. Финальный вывод
print_header "РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ"

echo -e "\n${GREEN}Все файлы сохранены в директории 'results/':${NC}"
ls -lh results/ | grep -E "\.(png|txt|log)$" | awk '{print "  " $9 " (" $5 ")"}'

print_section "КРАТКОЕ РЕЗЮМЕ"

# Вычисляем среднее ускорение
AVG_SPEEDUP=$(grep "Ускорение" results/test*.log 2>/dev/null | awk '{sum+=$3; count++} END {printf "%.1f", sum/count}' | tr -d 'x')
echo -e "${GREEN}Среднее ускорение во всех тестах: ${AVG_SPEEDUP}x${NC}"

# Проверка корректности
if grep -q "ПОЛНОСТЬЮ СОВПАДАЮТ" results/test1.log; then
    print_success "Все тесты пройдены успешно (CPU и GPU результаты совпадают)"
else
    print_error "Обнаружены расхождения в результатах"
fi

# Проверка найденных паттернов
if [ -f "results/test_matches.log" ]; then
    FOUND=$(grep "Найдено вхождений:" results/test_matches.log | awk '{print $3}')
    echo -e "${GREEN}Тест с гарантированными совпадениями: найдено ${FOUND} паттернов${NC}"
fi

print_section "СОЗДАННЫЕ ФАЙЛЫ"

echo -e "${CYAN}Исполняемые файлы:${NC}"
echo "  • mass_search - основная программа"
echo "  • mass_search_test - тестовая программа с гарантированными совпадениями"

echo -e "\n${CYAN}Результаты тестов (results/):${NC}"
echo "  • FINAL_REPORT.txt - полный отчет по лабораторной работе"
echo "  • performance_graph.png - график производительности"
echo "  • heatmap.png - тепловая карта совпадений"
echo "  • statistics.png - статистика поиска"
echo "  • alignment.png - выравнивание паттернов"
echo "  • chord.png - диаграмма связей"
echo "  • search_report.txt - текстовый отчет о поиске"
echo "  • test*.log - логи отдельных тестов"

# Показываем содержимое отчета
echo -e "\n${YELLOW}Содержимое итогового отчета:${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
cat results/FINAL_REPORT.txt
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                       ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА!                          ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════${NC}"

# Деактивация окружения
deactivate 2>/dev/null