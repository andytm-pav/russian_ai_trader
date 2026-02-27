#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностика содержимого памяти модели и просмотр Loss
"""

import pickle
import gzip
import numpy as np
import json
import os
from pathlib import Path


def inspect_memory_file():
    """Просмотр содержимого memory_buffer.pkl"""

    memory_file = Path("models/saved_trader/memory_buffer.pkl")
    state_file = Path("models/saved_trader/model_state.json")

    print("=" * 60)
    print("🔍 ДИАГНОСТИКА ПАМЯТИ МОДЕЛИ")
    print("=" * 60)

    # =========================================
    # 1. АНАЛИЗ ФАЙЛА ПАМЯТИ
    # =========================================
    if not memory_file.exists():
        print(f"❌ Файл памяти не найден: {memory_file}")
    else:
        try:
            with gzip.open(memory_file, 'rb') as f:
                memory = pickle.load(f)

            print(f"\n📦 Файл: {memory_file}")
            print(f"📊 Всего опытов: {len(memory)}")

            if len(memory) == 0:
                print("❌ Память пуста")
            else:
                # Анализ первого опыта
                print("\n" + "=" * 60)
                print("🔬 АНАЛИЗ ПЕРВОГО ОПЫТА")
                print("=" * 60)

                first = memory[0]
                print(f"\n📌 Тип опыта: {type(first)}")
                print(f"📌 Ключи: {list(first.keys())}")

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
                            if hasattr(value, 'cpu'):
                                val = value.cpu().flatten()[:5].numpy()
                            else:
                                val = value.flatten()[:5]
                            print(f"    Пример: {val}")
                        except:
                            print(f"    Не удалось получить пример")
                    else:
                        print(f"    Значение: {value}")

                # Проверка структуры
                print("\n" + "=" * 60)
                print("🔬 АНАЛИЗ СТРУКТУРЫ ПАМЯТИ")
                print("=" * 60)

                # Типы данных
                state_types = set()
                action_types = set()
                reward_types = set()

                for i, exp in enumerate(memory[:10]):
                    state_types.add(type(exp['state']))
                    action_types.add(type(exp['action']))
                    reward_types.add(type(exp['reward']))

                print(f"\n📊 РАЗНООБРАЗИЕ ТИПОВ:")
                print(f"  state: {state_types}")
                print(f"  action: {action_types}")
                print(f"  reward: {reward_types}")

                # Размерность state
                print("\n📏 РАЗМЕРНОСТЬ STATE:")
                for i, exp in enumerate(memory[:5]):
                    state = exp['state']
                    if hasattr(state, 'shape'):
                        print(f"  Опыт {i}: {state.shape}")
                    elif isinstance(state, np.ndarray):
                        print(f"  Опыт {i}: {state.shape}")
                    else:
                        print(f"  Опыт {i}: {len(state)} (list)")

                # Диапазон reward
                rewards = [exp['reward'] for exp in memory[:20]]
                print(f"\n💰 ДИАПАЗОН REWARD (первые 20):")
                print(f"  Min: {min(rewards):.3f}")
                print(f"  Max: {max(rewards):.3f}")
                print(f"  Avg: {sum(rewards) / len(rewards):.3f}")

                # Проверка на вложенные тензоры
                has_nested = False
                for exp in memory[:5]:
                    if isinstance(exp['state'], list) and len(exp['state']) > 0 and hasattr(exp['state'][0], 'shape'):
                        has_nested = True
                        print("⚠️ Обнаружены вложенные тензоры в state!")
                        break

                if not has_nested:
                    print("✅ Нет вложенных тензоров - структура корректна")

                # Проверка размерности (156)
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
            print(f"\n❌ Ошибка при анализе памяти: {e}")
            import traceback
            traceback.print_exc()

    # =========================================
    # 2. ПРОСМОТР LOSS ИЗ STATE ФАЙЛА
    # =========================================
    print("\n" + "=" * 60)
    print("📉 АНАЛИЗ LOSS МОДЕЛИ")
    print("=" * 60)

    # Пытаемся найти loss в model_state.json
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            print(f"\n📁 Файл: {state_file}")

            # Ищем поля, связанные с loss
            loss_fields = []

            # Проверяем корневые поля
            if 'last_loss' in state:
                loss_fields.append(('last_loss', state['last_loss']))
            if 'avg_loss' in state:
                loss_fields.append(('avg_loss', state['avg_loss']))
            if 'loss_history' in state:
                loss_history = state['loss_history']
                if isinstance(loss_history, list):
                    loss_fields.append(('loss_history', loss_history))

            # Проверяем во вложенных структурах
            if 'training_stats' in state:
                training = state['training_stats']
                if 'avg_loss' in training:
                    loss_fields.append(('training_stats.avg_loss', training['avg_loss']))
                if 'last_training' in training:
                    print(f"  Последнее обучение: {training['last_training']}")
                if 'loss_history' in training:
                    loss_history = training['loss_history']
                    if isinstance(loss_history, list):
                        loss_fields.append(('training_stats.loss_history', loss_history))

            if 'model_stats' in state:
                stats = state['model_stats']
                if 'last_loss' in stats:
                    loss_fields.append(('model_stats.last_loss', stats['last_loss']))

            # Выводим найденные loss
            if loss_fields:
                print("\n📊 НАЙДЕННЫЕ LOSS:")
                for name, value in loss_fields:
                    if isinstance(value, list) and len(value) > 0:
                        print(f"\n  🔹 {name}:")
                        print(f"    Всего записей: {len(value)}")
                        print(f"    Min: {min(value):.3f}")
                        print(f"    Max: {max(value):.3f}")
                        print(f"    Avg: {sum(value) / len(value):.3f}")
                        print(f"    Последние 5: {value[-5:] if len(value) >= 5 else value}")
                    elif isinstance(value, (int, float)):
                        print(f"\n  🔹 {name}: {value:.6f}")
            else:
                print("\n⚠️ Loss не найден в model_state.json")

                # Попробуем найти в лог-файлах
                log_dir = Path("logs")
                if log_dir.exists():
                    log_files = list(log_dir.glob("*.log"))
                    if log_files:
                        print("\n📁 Поиск loss в лог-файлах...")
                        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                            print(f"\n  Последние строки из {log_file.name}:")
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()[-20:]
                                    loss_lines = [l.strip() for l in lines if 'Loss=' in l or 'loss=' in l]
                                    if loss_lines:
                                        for line in loss_lines[-5:]:
                                            print(f"    {line}")
                                    else:
                                        print("    (нет строк с Loss)")
                            except:
                                print(f"    Ошибка чтения {log_file}")

        except Exception as e:
            print(f"\n❌ Ошибка при анализе state файла: {e}")
    else:
        print(f"\n❌ Файл не найден: {state_file}")

        # Пробуем найти loss в логах
        log_dir = Path("logs")
        if log_dir.exists():
            print("\n📁 Поиск loss в лог-файлах...")
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    print(f"\n  Последние строки из {log_file.name}:")
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-30:]
                            loss_lines = [l.strip() for l in lines if
                                          'Loss=' in l or 'loss=' in l or 'LOSS' in l.upper()]
                            if loss_lines:
                                for line in loss_lines[-10:]:
                                    print(f"    {line}")
                            else:
                                print("    (нет строк с Loss)")
                    except:
                        print(f"    Ошибка чтения {log_file}")
            else:
                print("  Лог-файлы не найдены")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    inspect_memory_file()