#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностика содержимого памяти модели
"""

import pickle
import gzip
import numpy as np
from pathlib import Path


def inspect_memory_file():
    """Просмотр содержимого memory_buffer.pkl"""

    memory_file = Path("models/saved_trader/memory_buffer.pkl")

    if not memory_file.exists():
        print(f"❌ Файл не найден: {memory_file}")
        return

    print("=" * 60)
    print("🔍 ДИАГНОСТИКА ПАМЯТИ МОДЕЛИ")
    print("=" * 60)

    try:
        # Загружаем память
        with gzip.open(memory_file, 'rb') as f:
            memory = pickle.load(f)

        print(f"\n📦 Файл: {memory_file}")
        print(f"📊 Всего опытов: {len(memory)}")

        if len(memory) == 0:
            print("❌ Память пуста")
            return

        # Анализ первого опыта
        print("\n" + "=" * 60)
        print("🔬 АНАЛИЗ ПЕРВОГО ОПЫТА")
        print("=" * 60)

        first = memory[0]
        print(f"\n📌 Тип опыта: {type(first)}")
        print(f"📌 Ключи: {list(first.keys())}")

        # Проверка каждого поля
        print("\n📊 ДЕТАЛИ ПОЛЕЙ:")
        for key, value in first.items():
            print(f"\n  🔹 {key}:")
            print(f"    Тип: {type(value)}")

            if isinstance(value, np.ndarray):
                print(f"    Форма: {value.shape}")
                print(f"    Тип данных: {value.dtype}")
                print(f"    Пример: {value.flatten()[:5]}...")
            elif isinstance(value, (list, tuple)):
                print(f"    Длина: {len(value)}")
                if len(value) > 0:
                    print(f"    Тип элемента: {type(value[0])}")
                    print(f"    Пример: {value[:5]}")
            elif hasattr(value, 'shape'):  # torch tensor
                print(f"    Форма: {value.shape}")
                print(f"    Тип: {value.dtype}")
                try:
                    print(f"    Пример: {value.flatten()[:5].cpu().numpy() if hasattr(value, 'cpu') else value[:5]}")
                except:
                    print(f"    Не удалось получить пример")
            else:
                print(f"    Значение: {value}")

        # Проверка структуры prioritized_buffer
        print("\n" + "=" * 60)
        print("🔬 АНАЛИЗ STRUKTURЫ ПАМЯТИ")
        print("=" * 60)

        # Проверка типов данных во всех опытах
        state_types = set()
        action_types = set()
        reward_types = set()

        for i, exp in enumerate(memory[:10]):  # первые 10 опытов
            state_types.add(type(exp['state']))
            action_types.add(type(exp['action']))
            reward_types.add(type(exp['reward']))

        print(f"\n📊 РАЗНООБРАЗИЕ ТИПОВ:")
        print(f"  state: {state_types}")
        print(f"  action: {action_types}")
        print(f"  reward: {reward_types}")

        # Проверка размерности state
        print("\n📏 РАЗМЕРНОСТЬ STATE:")
        for i, exp in enumerate(memory[:5]):
            state = exp['state']
            if hasattr(state, 'shape'):
                print(f"  Опыт {i}: {state.shape}")
            elif isinstance(state, np.ndarray):
                print(f"  Опыт {i}: {state.shape}")
            else:
                print(f"  Опыт {i}: {len(state)} (list)")

        # Проверка диапазона reward
        rewards = [exp['reward'] for exp in memory[:20]]
        print(f"\n💰 ДИАПАЗОН REWARD (первые 20):")
        print(f"  Min: {min(rewards):.3f}")
        print(f"  Max: {max(rewards):.3f}")
        print(f"  Avg: {sum(rewards) / len(rewards):.3f}")

        # Рекомендации
        print("\n" + "=" * 60)
        print("💡 РЕКОМЕНДАЦИИ")
        print("=" * 60)

        # Проверка на вложенные тензоры
        has_nested = False
        for exp in memory[:5]:
            if isinstance(exp['state'], list) and len(exp['state']) > 0 and hasattr(exp['state'][0], 'shape'):
                has_nested = True
                print("⚠️ Обнаружены вложенные тензоры в state!")
                break

        if not has_nested:
            print("✅ Нет вложенных тензоров - структура корректна")

        # Проверка размерности
        all_156 = all(
            (hasattr(exp['state'], 'shape') and exp['state'].shape[-1] == 156) or
            (isinstance(exp['state'], (list, tuple)) and len(exp['state']) == 156)
            for exp in memory[:10]
        )

        if all_156:
            print("✅ Все state имеют размерность 156")
        else:
            print("⚠️ Обнаружены state с другой размерностью!")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    inspect_memory_file()