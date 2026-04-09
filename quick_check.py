#!/usr/bin/env python3
"""
Скрипт поиска использования удаляемых методов в проекте
"""

import os
import re
from pathlib import Path


def search_method_usage(method_names, root_dir=".", exclude_dirs=None):
    """
    Поиск использования методов в Python файлах

    Args:
        method_names: список имен методов для поиска
        root_dir: корневая директория проекта
        exclude_dirs: директории для исключения
    """
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.git', 'models/saved_trader', 'data']

    results = {method: {'definitions': [], 'calls': []} for method in method_names}

    # Конвертируем root_dir в Path
    root_path = Path(root_dir).resolve()

    # Паттерны для поиска
    patterns = {
        'def': re.compile(r'def\s+{method}\s*\('),
        'call': re.compile(r'{method}\s*\('),
        'self_call': re.compile(r'self\.{method}\s*\(')
    }

    print("=" * 80)
    print("ПОИСК ИСПОЛЬЗОВАНИЯ МЕТОДОВ")
    print("=" * 80)

    # Проходим по всем .py файлам
    for py_file in root_path.rglob("*.py"):
        # Проверяем, не в исключенной ли директории
        skip = False
        for excl in exclude_dirs:
            if excl in str(py_file):
                skip = True
                break

        if skip:
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"  ⚠️ Не удалось прочитать {py_file}: {e}")
            continue

        for method in method_names:
            # Поиск определения метода
            def_pattern = re.compile(rf'def\s+{method}\s*\(')
            for i, line in enumerate(lines, 1):
                if def_pattern.search(line):
                    results[method]['definitions'].append({
                        'file': str(py_file.relative_to(root_path)),
                        'line': i,
                        'content': line.strip()
                    })

            # Поиск вызовов метода
            call_patterns = [
                re.compile(rf'{method}\s*\('),  # method()
                re.compile(rf'self\.{method}\s*\('),  # self.method()
                re.compile(rf'\.{method}\s*\(')  # .method()
            ]

            for i, line in enumerate(lines, 1):
                for pattern in call_patterns:
                    if pattern.search(line) and f'def {method}' not in line:
                        results[method]['calls'].append({
                            'file': str(py_file.relative_to(root_path)),
                            'line': i,
                            'content': line.strip()
                        })

    return results


def print_results(results):
    """Вывод результатов поиска"""

    methods_to_delete = [
        '_get_commission_to_pnl_ratio',
        '_get_trade_frequency_penalty',
        '_get_expected_commission',
        '_get_breakeven_price_ratio'
    ]

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПОИСКА")
    print("=" * 80)

    all_safe = True

    for method in methods_to_delete:
        print(f"\n{'─' * 60}")
        print(f"Метод: {method}")
        print(f"{'─' * 60}")

        data = results[method]

        # Определения
        print(f"\n  📍 Определения ({len(data['definitions'])}):")
        if data['definitions']:
            for loc in data['definitions']:
                print(f"     • {loc['file']}:{loc['line']}")
                print(f"       {loc['content'][:60]}...")
        else:
            print("     ❌ ОПРЕДЕЛЕНИЕ НЕ НАЙДЕНО!")

        # Вызовы
        print(f"\n  🔍 Вызовы ({len(data['calls'])}):")
        if data['calls']:
            for call in data['calls']:
                print(f"     • {call['file']}:{call['line']}")
                print(f"       {call['content']}")
            all_safe = False
        else:
            print("     ✅ НЕТ ВЫЗОВОВ")

    print("\n" + "=" * 80)
    print("ИТОГ")
    print("=" * 80)

    if all_safe:
        print("\n✅ ВСЕ МЕТОДЫ МОЖНО БЕЗОПАСНО УДАЛИТЬ")
        print("   Ни один из методов не вызывается в других файлах.")
        print("\n   Методы для удаления:")
        for method in methods_to_delete:
            print(f"     • {method}")
    else:
        print("\n⚠️ НАЙДЕНЫ ВЫЗОВЫ МЕТОДОВ!")
        print("   Удаление может сломать систему.")
        print("\n   Проверьте вызовы перед удалением.")

    return all_safe


def save_results(results, output_file="data/method_usage_report.json"):
    """Сохранение результатов в JSON"""
    import json

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    # Конвертируем для JSON
    json_results = {}
    for method, data in results.items():
        json_results[method] = {
            'definitions_count': len(data['definitions']),
            'calls_count': len(data['calls']),
            'definitions': data['definitions'],
            'calls': data['calls']
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Подробный отчет сохранен: {output_file}")


def search_specific_usage():
    """Дополнительный поиск по содержимому методов"""
    methods_content = {
        '_get_commission_to_pnl_ratio': ['commission', 'pnl', 'ratio'],
        '_get_trade_frequency_penalty': ['frequency', 'penalty', 'trades_per_hour'],
        '_get_expected_commission': ['expected', 'commission', 'position_value'],
        '_get_breakeven_price_ratio': ['breakeven', 'entry_price', '1.006']
    }

    root_path = Path(".").resolve()

    print("\n" + "=" * 80)
    print("ДОПОЛНИТЕЛЬНЫЙ ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ")
    print("=" * 80)

    for method, keywords in methods_content.items():
        print(f"\n{method}:")
        found_any = False

        for py_file in root_path.rglob("*.py"):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Проверяем, есть ли все ключевые слова в логическом блоке
                if method not in content:
                    continue

                # Если метод вызывается - показываем
                if f'self.{method}' in content or f'.{method}' in content:
                    print(f"  • Найден вызов в: {py_file}")
                    found_any = True

            except Exception:
                pass

        if not found_any:
            print(f"  ✅ Вызовы не найдены")


if __name__ == "__main__":
    # Список методов для проверки
    methods_to_check = [
        '_get_commission_to_pnl_ratio',
        '_get_trade_frequency_penalty',
        '_get_expected_commission',
        '_get_breakeven_price_ratio'
    ]

    # Запуск поиска
    results = search_method_usage(methods_to_check, root_dir=".")

    # Вывод результатов
    is_safe = print_results(results)

    # Сохранение отчета
    save_results(results)

    # Дополнительный поиск
    search_specific_usage()

    # Код возврата
    import sys

    sys.exit(0 if is_safe else 1)