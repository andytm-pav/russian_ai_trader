import json
import numpy as np
import torch
from collections import deque
import pickle
import gzip


def analyze_trader_model_problem(model_instance=None, memory_file="models/saved_trader/memory_buffer.pkl"):
    """
    Диагностика проблемы с loss в уже обученной модели
    """
    print("=" * 80)
    print("ДИАГНОСТИКА ПРОБЛЕМЫ LOSS В ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 80)

    # 1. Анализ конфигурации
    print("\n1. ПРОВЕРКА КОНФИГУРАЦИИ:")
    try:
        with open("config/rl_config.json", "r") as f:
            rl_config = json.load(f)
            reward_scaling = rl_config.get('reward_scaling', 2.0)
            print(f"   reward_scaling = {reward_scaling}")

            # Проверяем, что scaling соответствует масштабу
            if reward_scaling > 1.0:
                print(f"   ⚠️ ВНИМАНИЕ: reward_scaling > 1.0 ({reward_scaling})")
                print(f"      Это может быть причиной больших td_errors")
    except Exception as e:
        print(f"   Ошибка загрузки конфига: {e}")

    # 2. Загрузка и анализ памяти модели
    print("\n2. АНАЛИЗ ПАМЯТИ МОДЕЛИ:")
    try:
        with gzip.open(memory_file, 'rb') as f:
            memory = pickle.load(f)

        print(f"   Всего опытов: {len(memory)}")

        # Анализируем rewards
        rewards = [exp['reward'] for exp in memory if 'reward' in exp]
        if rewards:
            rewards_array = np.array(rewards)
            print(f"\n   Статистика reward (то, что хранится в памяти):")
            print(f"     Минимум: {rewards_array.min():.4f}")
            print(f"     Максимум: {rewards_array.max():.4f}")
            print(f"     Среднее: {rewards_array.mean():.4f}")
            print(f"     Медиана: {np.median(rewards_array):.4f}")
            print(f"     Std: {rewards_array.std():.4f}")

            # Проверяем масштаб reward
            if abs(rewards_array.mean()) > 10:
                print(f"   ⚠️ Reward СЛИШКОМ БОЛЬШИЕ (среднее > 10)")
                print(f"      Должны быть в районе 0.xx (проценты)")
            elif abs(rewards_array.mean()) < 0.01:
                print(f"   ⚠️ Reward СЛИШКОМ МАЛЕНЬКИЕ (среднее < 0.01)")
                print(f"      Возможно, scaling слишком маленький")

        # Ищем pnl_rub в опытах
        pnl_rub_found = sum(1 for exp in memory if 'pnl_rub' in exp)
        print(f"\n   Опытов с pnl_rub: {pnl_rub_found}/{len(memory)}")

        if pnl_rub_found > 0:
            pnl_rub = [exp['pnl_rub'] for exp in memory if 'pnl_rub' in exp]
            pnl_rub_array = np.array(pnl_rub)
            print(f"\n   Статистика pnl_rub (реальные рубли):")
            print(f"     Минимум: {pnl_rub_array.min():.2f}₽")
            print(f"     Максимум: {pnl_rub_array.max():.2f}₽")
            print(f"     Среднее: {pnl_rub_array.mean():.2f}₽")
            print(f"     Медиана: {np.median(pnl_rub_array):.2f}₽")

    except Exception as e:
        print(f"   Ошибка загрузки памяти: {e}")

    # 3. Анализ одного батча для понимания расчета loss
    print("\n3. ИМИТАЦИЯ РАСЧЕТА LOSS:")

    # Создаем синтетические данные, похожие на реальные
    np.random.seed(42)
    batch_size = 32

    # Вариант А: reward в процентах (как должно быть)
    rewards_percent = np.random.normal(-0.02, 0.1, batch_size)  # средний убыток 2%
    print(f"\n   Вариант А (reward в ПРОЦЕНТАХ):")
    print(f"     rewards mean: {rewards_percent.mean():.4f}")
    print(f"     rewards std: {rewards_percent.std():.4f}")

    # Вариант Б: reward в рублях (как сейчас в логе)
    rewards_rub = np.random.normal(-100, 150, batch_size)  # средний убыток 100₽
    print(f"\n   Вариант Б (reward в РУБЛЯХ - как в вашем логе):")
    print(f"     rewards mean: {rewards_rub.mean():.2f}")
    print(f"     rewards std: {rewards_rub.std():.2f}")

    # Имитация current_values (то, что предсказывает сеть)
    current_values = np.random.normal(0, 10, batch_size)

    # Расчет target_values и td_errors для обоих вариантов
    gamma = 0.95
    next_values = np.random.normal(0, 10, batch_size)

    print(f"\n   СРАВНЕНИЕ TD-ERRORS:")

    # Для процентов
    target_percent = rewards_percent + gamma * next_values
    td_percent = target_percent - current_values
    print(f"\n   При reward в ПРОЦЕНТАХ:")
    print(f"     target_values mean: {target_percent.mean():.4f}")
    print(f"     td_errors mean: {td_percent.mean():.4f}")
    print(f"     |td_errors| mean: {np.abs(td_percent).mean():.4f}")

    # Для рублей
    target_rub = rewards_rub + gamma * next_values
    td_rub = target_rub - current_values
    print(f"\n   При reward в РУБЛЯХ (как сейчас):")
    print(f"     target_values mean: {target_rub.mean():.2f}")
    print(f"     td_errors mean: {td_rub.mean():.2f}")
    print(f"     |td_errors| mean: {np.abs(td_rub).mean():.2f}")

    # 4. Масштабирование для исправления
    print("\n4. РЕКОМЕНДУЕМОЕ МАСШТАБИРОВАНИЕ:")

    # Вычисляем правильный scaling на основе текущих данных
    if pnl_rub_found > 0:
        avg_pnl_rub = np.abs(pnl_rub_array).mean()
        desired_reward_scale = 0.1  # хотим reward порядка 0.1
        recommended_scaling = desired_reward_scale / avg_pnl_rub if avg_pnl_rub > 0 else 0.001

        print(f"   Средний |PnL| в рублях: {avg_pnl_rub:.2f}₽")
        print(f"   Желаемый масштаб reward: {desired_reward_scale}")
        print(f"   Рекомендуемый reward_scaling: {recommended_scaling:.6f}")
        print(f"   (текущий: {reward_scaling if 'reward_scaling' in locals() else 'неизвестно'})")

    print("\n" + "=" * 80)
    print("ВЫВОД:")
    print("=" * 80)

    print("""
    1. Проблема: reward в памяти модели хранятся в НЕПРАВИЛЬНОМ масштабе
    2. Решение: Нужно пересохранить память с правильным scaling
    3. Действия:
       a) Установить reward_scaling = 0.001 в конфиге
       b) Переобучить модель с новым scaling
       c) Либо написать скрипт для пересчета существующей памяти
    """)


if __name__ == "__main__":
    analyze_trader_model_problem()