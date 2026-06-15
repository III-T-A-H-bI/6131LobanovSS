import subprocess
import sys
import os

def ensure_packages():
    """Принудительная установка пакетов в текущее окружение"""
    packages = ['numpy', 'matplotlib', 'seaborn', 'pandas']
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Установка {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            print(f"{package} установлен")

# Устанавливаем пакеты перед импортом
ensure_packages()

# Теперь импортируем
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Tuple
import pandas as pd

def read_buffer_h(filename="buffer_H.bin"):
    """Чтение буфера H из бинарного файла"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None
    with open(filename, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)

def read_patterns(filename="patterns_N.txt"):
    """Чтение паттернов из текстового файла"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None
    
    patterns = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                idx, hex_str = line.strip().split(':')
                # Конвертируем HEX строку в байты
                try:
                    pattern = bytes.fromhex(hex_str)
                    patterns.append(list(pattern))
                except:
                    patterns.append([])
    return patterns

def read_results(filename="search_results.txt"):
    """Чтение результатов поиска из текстового файла"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None
    
    results = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Разбиваем на секции паттернов
    import re
    pattern_sections = re.findall(r'Pattern (\d+) \(len=(\d+)\): ([0-9A-F]+)\nPositions: (.*?)\n\n', content, re.DOTALL)
    
    for match in pattern_sections:
        idx, length, hex_pattern, positions_str = match
        positions_str = positions_str.strip()
        if positions_str == "NOT FOUND":
            results.append([])
        else:
            positions = [int(x) for x in positions_str.split() if x.isdigit()]
            results.append(positions)
    
    return results

def visualize_heatmap(H, patterns, results, output_file="heatmap.png"):
    """Визуализация в виде тепловой карты совпадений"""
    if H is None or patterns is None or results is None:
        print("Нет данных для визуализации")
        return
    
    h_len = len(H)
    n_count = len(patterns)
    
    if n_count == 0:
        print("Нет паттернов для отображения")
        return
    
    # Ограничиваем размер для лучшей визуализации
    max_patterns = min(n_count, 100)
    max_positions = min(h_len, 1000)
    
    # Создаем матрицу совпадений
    match_matrix = np.zeros((max_patterns, max_positions), dtype=int)
    
    for i in range(max_patterns):
        if i < len(results) and results[i]:
            pattern_len = len(patterns[i])
            for pos in results[i]:
                if pos < max_positions:
                    for offset in range(min(pattern_len, max_positions - pos)):
                        match_matrix[i, pos + offset] += 1
    
    # Создаем тепловую карту
    plt.figure(figsize=(20, 12))
    
    # Используем нормализацию для лучшего отображения
    if np.max(match_matrix) > 0:
        im = plt.imshow(match_matrix, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest', norm='log')
    else:
        im = plt.imshow(match_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    cbar = plt.colorbar(im, label='Количество совпадений', shrink=0.8)
    plt.xlabel('Позиция в буфере H', fontsize=12)
    plt.ylabel('Индекс паттерна', fontsize=12)
    plt.title(f'Тепловая карта совпадений паттернов\n(Показаны первые {max_patterns} паттернов и {max_positions} позиций)', 
              fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Тепловая карта сохранена в {output_file}")

def visualize_alignment(H, patterns, results, output_file="alignment.png"):
    """Визуализация выравнивания паттернов"""
    if H is None or patterns is None or results is None:
        return
    
    # Показываем только паттерны, которые были найдены
    found_patterns = [(i, p, r) for i, (p, r) in enumerate(zip(patterns, results)) if r]
    
    if not found_patterns:
        print("Нет найденных паттернов для визуализации выравнивания")
        return
    
    max_display = min(30, len(found_patterns))
    found_patterns = found_patterns[:max_display]
    
    fig, axes = plt.subplots(max_display, 1, figsize=(20, max_display * 0.8))
    if max_display == 1:
        axes = [axes]
    
    # Показываем первые 200 символов буфера
    display_len = min(200, len(H))
    buffer_str = ' '.join(f'{b:02X}' for b in H[:display_len])
    
    for idx, (i, pattern, positions) in enumerate(found_patterns):
        ax = axes[idx]
        
        # Создаем строку для визуализации
        text_line = [' '] * display_len
        
        # Отмечаем совпадения
        for pos in positions:
            if pos < display_len:
                for offset in range(len(pattern)):
                    if pos + offset < display_len:
                        text_line[pos + offset] = '█'
        
        # Отмечаем начала паттернов
        for pos in positions:
            if pos < display_len:
                text_line[pos] = '▼'
        
        display_str = ''.join(text_line)
        
        # Формируем информацию о паттерне
        pattern_hex = ''.join(f'{b:02X}' for b in pattern[:20])
        if len(pattern) > 20:
            pattern_hex += "..."
        
        ax.text(0.02, 0.7, f"Pattern {i} (len={len(pattern)}): {pattern_hex}", 
                transform=ax.transAxes, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        ax.text(0.02, 0.3, display_str, transform=ax.transAxes, 
                fontsize=7, fontfamily='monospace', verticalalignment='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Добавляем счетчик вхождений
        if positions:
            ax.text(0.98, 0.85, f"Found: {len(positions)} times", 
                   transform=ax.transAxes, fontsize=8, ha='right',
                   bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle(f'Выравнивание найденных паттернов в буфере H\n(первые {display_len} позиций)', 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Выравнивание сохранено в {output_file}")

def visualize_statistics(patterns, results, output_file="statistics.png"):
    """Визуализация статистики"""
    if patterns is None or results is None:
        print("Нет данных для статистики")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение длин паттернов
    lengths = [len(p) for p in patterns if p]
    if lengths:
        axes[0, 0].hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Длина паттерна')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].set_title('Распределение длин паттернов')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Среднее: {np.mean(lengths):.1f}')
        axes[0, 0].legend()
    
    # 2. Количество вхождений на паттерн
    occurrences = [len(r) for r in results]
    if occurrences:
        axes[0, 1].hist(occurrences, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('Количество вхождений')
        axes[0, 1].set_ylabel('Количество паттернов')
        axes[0, 1].set_title('Распределение вхождений паттернов')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Статистика
        found_count = sum(1 for o in occurrences if o > 0)
        stats_text = f'Найдено: {found_count}/{len(occurrences)}\n'
        stats_text += f'Макс: {max(occurrences)}\n'
        stats_text += f'Среднее: {np.mean(occurrences):.2f}\n'
        stats_text += f'Медиана: {np.median(occurrences):.0f}'
        axes[0, 1].text(0.7, 0.9, stats_text, transform=axes[0, 1].transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8), fontsize=10)
    
    # 3. Топ паттернов по вхождениям
    top_n = min(20, len(occurrences))
    if top_n > 0 and max(occurrences) > 0:
        top_indices = np.argsort(occurrences)[-top_n:][::-1]
        top_occurrences = [occurrences[i] for i in top_indices]
        top_lengths = [len(patterns[i]) for i in top_indices]
        
        x = range(len(top_occurrences))
        bars = axes[1, 0].bar(x, top_occurrences, color='coral', alpha=0.8)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'P{i}\n(len={top_lengths[j]})' for j, i in enumerate(top_indices)], 
                                  rotation=45, ha='right', fontsize=8)
        axes[1, 0].set_ylabel('Количество вхождений')
        axes[1, 0].set_title(f'Топ-{top_n} паттернов по вхождениям')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, (bar, val) in enumerate(zip(bars, top_occurrences)):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(val), ha='center', va='bottom', fontsize=8)
    
    # 4. Зависимость вхождений от длины паттерна
    if lengths and occurrences:
        scatter = axes[1, 1].scatter(lengths, occurrences, alpha=0.6, c=occurrences, 
                                    cmap='viridis', s=50)
        axes[1, 1].set_xlabel('Длина паттерна')
        axes[1, 1].set_ylabel('Количество вхождений')
        axes[1, 1].set_title('Вхождения vs Длина паттерна')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        if len(lengths) > 1:
            z = np.polyfit(lengths, occurrences, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(lengths), max(lengths), 100)
            axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Тренд (slope={z[0]:.2f})')
            axes[1, 1].legend()
        
        plt.colorbar(scatter, ax=axes[1, 1], label='Вхождения')
    
    plt.suptitle('Статистика поиска паттернов', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Статистика сохранена в {output_file}")

def visualize_chord_diagram(H, patterns, results, output_file="chord.png"):
    """Диаграмма связей между паттернами и позициями"""
    if not results or not patterns:
        return
    
    # Собираем данные для круговой диаграммы
    pattern_counts = [len(r) for r in results]
    found_patterns = [i for i, count in enumerate(pattern_counts) if count > 0]
    
    if not found_patterns:
        print("Нет найденных паттернов для диаграммы")
        return
    
    fig = plt.figure(figsize=(14, 7))
    
    # Левая часть: круговая диаграмма вхождений
    ax1 = plt.subplot(121)
    counts = [pattern_counts[i] for i in found_patterns[:20]]  # Топ-20 для читаемости
    labels = [f'P{i}' for i in found_patterns[:20]]
    
    if counts and sum(counts) > 0:
        wedges, texts, autotexts = ax1.pie(counts, labels=labels, autopct='%1.1f%%',
                                            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(counts))))
        ax1.set_title('Распределение вхождений по паттернам', fontsize=12)
    
    # Правая часть: гистограмма позиций
    ax2 = plt.subplot(122)
    all_positions = []
    for positions in results:
        all_positions.extend(positions)
    
    if all_positions:
        ax2.hist(all_positions, bins=50, edgecolor='black', alpha=0.7, color='teal')
        ax2.set_xlabel('Позиция в буфере')
        ax2.set_ylabel('Количество вхождений')
        ax2.set_title('Распределение вхождений по позициям')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем статистику
        ax2.text(0.7, 0.95, f'Всего вхождений: {len(all_positions)}\n'
                           f'Мин позиция: {min(all_positions)}\n'
                           f'Макс позиция: {max(all_positions)}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Анализ связей паттернов с позициями', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Диаграмма связей сохранена в {output_file}")

def create_report(results, patterns, cpu_time=None, gpu_time=None, speedup=None, output_file="search_report.txt"):
    """Создание текстового отчета"""
    if results is None:
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ О ПОИСКЕ ПОДСТРОК\n")
        f.write("=" * 60 + "\n\n")
        
        total_patterns = len(results)
        found_patterns = sum(1 for r in results if r)
        total_matches = sum(len(r) for r in results)
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Всего паттернов: {total_patterns}\n")
        f.write(f"Найдено паттернов: {found_patterns}\n")
        f.write(f"Не найдено паттернов: {total_patterns - found_patterns}\n")
        f.write(f"Всего вхождений: {total_matches}\n")
        
        if found_patterns > 0:
            f.write(f"Процент найденных: {found_patterns/total_patterns*100:.1f}%\n")
            f.write(f"Среднее вхождений на паттерн: {total_matches/total_patterns:.2f}\n")
            f.write(f"Среднее вхождений на найденный паттерн: {total_matches/found_patterns:.2f}\n")
        
        if cpu_time and gpu_time:
            f.write(f"\nПРОИЗВОДИТЕЛЬНОСТЬ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Время CPU: {cpu_time:.3f} мс\n")
            f.write(f"Время GPU: {gpu_time:.3f} мс\n")
            f.write(f"Ускорение: {speedup:.2f}x\n")
        
        # Топ паттернов
        if total_matches > 0:
            f.write(f"\nТОП-10 ПАТТЕРНОВ ПО ВХОЖДЕНИЯМ:\n")
            f.write("-" * 40 + "\n")
            pattern_matches = [(i, len(r), r[:5]) for i, r in enumerate(results) if r]
            pattern_matches.sort(key=lambda x: x[1], reverse=True)
            
            for i, (idx, count, positions) in enumerate(pattern_matches[:10]):
                pattern_hex = ''.join(f'{b:02X}' for b in patterns[idx][:10])
                if len(patterns[idx]) > 10:
                    pattern_hex += "..."
                f.write(f"{i+1}. Pattern {idx} (len={len(patterns[idx])}): {pattern_hex}\n")
                f.write(f"   Вхождений: {count}, позиции: {positions}\n")
                if count > 5:
                    f.write(f"   ... и еще {count-5} позиций\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("КОНЕЦ ОТЧЕТА\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Отчет сохранен в {output_file}")

def main():
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ПОИСКА ПОДСТРОК")
    print("=" * 60 + "\n")
    
    # Загрузка данных
    print("📂 Загрузка данных...")
    H = read_buffer_h()
    if H is not None:
        print(f"  ✓ Буфер H: {len(H)} байт")
    
    patterns = read_patterns()
    if patterns is not None:
        print(f"  ✓ Паттерны: {len(patterns)} шт.")
    
    results = read_results()
    if results is not None:
        print(f"  ✓ Результаты: {len(results)} паттернов")
    
    if results is None or len(results) == 0:
        print("\n❌ РЕЗУЛЬТАТЫ НЕ НАЙДЕНЫ!")
        print("Сначала запустите программу поиска:")
        print("  ./mass_search 50000 100 5 15 1\n")
        return
    
    # Генерация визуализаций
    print("\n🎨 Генерация визуализаций...")
    print("-" * 40)
    
    visualize_heatmap(H, patterns, results)
    visualize_alignment(H, patterns, results)
    visualize_statistics(patterns, results)
    visualize_chord_diagram(H, patterns, results)
    
    # Создание отчета
    create_report(results, patterns)
    
    print("\n" + "=" * 60)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)
    print("\nСозданные файлы:")
    print("  📊 heatmap.png - тепловая карта совпадений")
    print("  📈 statistics.png - статистика поиска")
    print("  🔗 alignment.png - выравнивание паттернов")
    print("  🎯 chord.png - диаграмма связей")
    print("  📄 search_report.txt - текстовый отчет")
    print()

if __name__ == "__main__":
    main()