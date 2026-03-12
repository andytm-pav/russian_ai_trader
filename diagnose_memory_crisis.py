#!/usr/bin/env python3
# diagnose_memory_crisis.py
"""
ДИАГНОСТИКА ПАМЯТИ - поиск опытов с неправильной размерностью
Запуск: python diagnose_memory_crisis.py
"""

import pickle
import gzip
import torch
import numpy as np
from collections import Counter
from pathlib import Path


def diagnose_memory():
    """Полная диагностика файла памяти"""

    print("=" * 70)
    print("🔍 ДИАГНОСТИКА ФАЙЛА ПАМЯТИ")
    print("=" * 70)

    memory_file = "models/saved_trader/memory_buffer.pkl"

    if not Path(memory_file).exists():
        print(f"❌ Файл не найден: {memory_file}")
        return

    # Загружаем память
    try:
        with gzip.open(memory_file, 'rb') as f:
            memory = pickle.load(f)
        print(f"✅ Загружено {len(memory)} опытов (сжатый файл)")
    except:
        with open(memory_file, 'rb') as f:
            memory = pickle.load(f)
        print(f"✅ Загружено {len(memory)} опытов (несжатый файл)")

    # 1. Анализ размерностей state
    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ РАЗМЕРНОСТЕЙ STATE")
    print("=" * 70)

    state_dims = []
    state_samples = []

    for i, exp in enumerate(memory):
        if 'state' in exp:
            if hasattr(exp['state'], 'shape'):
                dim = exp['state'].shape[-1]
                state_dims.append(dim)
                if len(state_samples) < 5:
                    state_samples.append(
                        (i, dim, exp['state'][:5].tolist() if hasattr(exp['state'], 'tolist') else '?'))

    dim_counts = Counter(state_dims)
    print(f"📈 Распределение размерностей:")
    for dim, count in sorted(dim_counts.items()):
        print(f"  • {dim}: {count} опытов ({count / len(memory) * 100:.1f}%)")

    if 162 in dim_counts:
        print("\n❌ ОБНАРУЖЕНЫ ОПЫТЫ С РАЗМЕРНОСТЬЮ 162!")

        # Находим все опыты с 162
        bad_indices = [i for i, d in enumerate(state_dims) if d == 162]
        print(f"   Найдено {len(bad_indices)} опытов 162")

        # Показываем первые 3 для примера
        print("\n📋 Примеры опытов 162:")
        for idx in bad_indices[:3]:
            exp = memory[idx]
            print(f"\n  Опыт #{idx}:")
            print(f"    action: {exp.get('action', '?')}")
            print(f"    reward: {exp.get('reward', '?')}")
            print(f"    strategy: {exp.get('strategy', '?')}")
            print(f"    state[:10]: {exp['state'][:10].tolist() if hasattr(exp['state'], 'tolist') else '?'}")

    # 2. Анализ next_state
    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ РАЗМЕРНОСТЕЙ NEXT_STATE")
    print("=" * 70)

    next_dims = []
    for exp in memory:
        if 'next_state' in exp and hasattr(exp['next_state'], 'shape'):
            next_dims.append(exp['next_state'].shape[-1])

    if next_dims:
        next_counts = Counter(next_dims)
        print(f"📈 Распределение размерностей:")
        for dim, count in sorted(next_counts.items()):
            print(f"  • {dim}: {count} опытов ({count / len(next_dims) * 100:.1f}%)")

    # 3. Поиск источника размерности 162
    print("\n" + "=" * 70)
    print("🔍 ПОИСК ИСТОЧНИКА ПРОБЛЕМЫ")
    print("=" * 70)

    # Проверяем стратегии в model_state.json
    model_state_file = "models/saved_trader/model_state.json"
    if Path(model_state_file).exists():
        import json
        with open(model_state_file, 'r') as f:
            model_state = json.load(f)

        strategies = model_state.get('strategies', {})
        print(f"📋 Стратегий в model_state: {len(strategies)}")

        # Проверяем target_hold_time для каждой стратегии
        print("\n⏱️  Параметры стратегий:")
        for name, params in strategies.items():
            hold = params.get('target_hold_time_hours', '?')
            stop = params.get('stop_loss_percent', '?')
            print(f"  • {name}: hold={hold}ч, stop={stop}%")

    # 4. Проверка кода _create_strategy_state
    print("\n" + "=" * 70)
    print("📋 АНАЛИЗ КОДА _create_strategy_state")
    print("=" * 70)

    trader_model_file = "models/trader_model.py"
    if Path(trader_model_file).exists():
        with open(trader_model_file, 'r', encoding='utf-8') as f:  # ← добавить encoding='utf-8'
            code = f.read()

        # Ищем метод
        import re
        pattern = r'def _create_strategy_state.*?return result'
        match = re.search(pattern, code, re.DOTALL)

        if match:
            method_code = match.group(0)
            print("✅ Метод _create_strategy_state найден")

            # Проверяем наличие обработки 156
            if 'if current_dim == 156:' in method_code:
                print("  ✅ Есть обработка 156 → 157")
            else:
                print("  ❌ НЕТ обработки 156 → 157")

            # Проверяем target_dim
            if 'target_dim = self.policy_net.state_dim' in method_code:
                print("  ✅ Используется динамическая целевая размерность")
            else:
                print("  ❌ Жестко заданная размерность")

            # Показываем ключевые строки
            print("\n📌 Ключевые строки метода:")
            lines = method_code.split('\n')
            for line in lines:
                if 'if current_dim' in line or 'target_dim' in line or 'torch.cat' in line:
                    print(f"  {line.strip()}")

    # 5. Рекомендации
    print("\n" + "=" * 70)
    print("✅ РЕКОМЕНДАЦИИ")
    print("=" * 70)

    if 162 in dim_counts:
        print("\n🔧 ПРОБЛЕМА: опыты 162 возникают когда 156 + 6")
        print("   Это значит, что код для 156 не срабатывает!")
        print("\n📋 ПРОВЕРЬТЕ В _create_strategy_state:")
        print("   1. Убедитесь, что условие 'if current_dim == 156:' стоит ДО добавления параметров")
        print("   2. Убедитесь, что нет return после обработки 156")
        print("   3. Добавьте print для отладки")

        print("\n💾 ВРЕМЕННОЕ РЕШЕНИЕ - очистить проблемные опыты:")
        print("   python -c \"import pickle,gzip; "
              "m=pickle.load(gzip.open('models/saved_trader/memory_buffer.pkl','rb')); "
              "m=[e for e in m if not (hasattr(e['state'],'shape') and e['state'].shape[-1]==162)]; "
              "pickle.dump(m,gzip.open('models/saved_trader/memory_buffer.pkl','wb'))\"")

    else:
        print("✅ Размерности в норме!")


if __name__ == "__main__":
    diagnose_memory()