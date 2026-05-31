"""
Проверка здоровья вектора состояния в реальной системе
Запускать во время работы main.py
"""
import json
import numpy as np
import torch
import requests
from models.trader_model import trader_model_instance

model = trader_model_instance

with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

print("=" * 60)
print("🔍 ПРОВЕРКА ЗДОРОВЬЯ ВЕКТОРА СОСТОЯНИЯ")
print("=" * 60)

# Параметры из конфига
expected_dim = model.base_state_dim
print(f"Ожидаемая размерность: {expected_dim}")
print(f"Память модели: {len(model.memory)} опытов")

if len(model.memory) == 0:
    print("\n❌ Память пуста. Дождитесь нескольких торговых циклов.")
    exit()

# Анализируем последние 100 опытов
sample_size = min(100, len(model.memory))
recent = list(model.memory)[-sample_size:]

print(f"\nАнализ {sample_size} последних опытов:\n")

issues = []
max_values = []
min_values = []
nan_count = 0
inf_count = 0

for i, exp in enumerate(recent):
    state = exp.get('state')
    if state is None:
        continue

    arr = state.cpu().numpy().flatten()
    max_values.append(np.max(np.abs(arr)))
    min_values.append(np.min(arr))

    if np.isnan(arr).any():
        nan_count += 1
        issues.append(f"Опыт #{len(model.memory) - sample_size + i}: NaN")

    if np.isinf(arr).any():
        inf_count += 1
        issues.append(f"Опыт #{len(model.memory) - sample_size + i}: Inf")

    # Проверка на выбросы (>1000)
    outliers = np.abs(arr) > 1000
    if outliers.any():
        idx = np.where(outliers)[0]
        issues.append(f"Опыт #{len(model.memory) - sample_size + i}: выбросы на индексах {idx.tolist()}, значения: {arr[idx].tolist()}")

# Статистика
max_abs = np.max(max_values)
min_val = np.min(min_values)
mean_abs = np.mean(max_values)

print(f"NaN в памяти: {nan_count}/{sample_size}")
print(f"Inf в памяти: {inf_count}/{sample_size}")
print(f"Диапазон всех значений: [{min_val:.2f}, {max_abs:.2f}]")
print(f"Средний максимум по модулю: {mean_abs:.2f}")

if max_abs > 1000:
    print(f"\n⚠️ Обнаружен ВЫБРОС: {max_abs:.2e}")
    print("Это мешает обучению. Проверьте нормализацию:")
    print("  - volume_normalization в rl_config.json (должно быть 1e9)")
    print("  - market_cap делится на 1e12 в build_state_vector")
else:
    print(f"\n✅ Выбросов нет. Максимум: {max_abs:.2f}")

if issues:
    print(f"\n⚠️ Найдено {len(issues)} проблем:")
    for issue in issues[:5]:
        print(f"  - {issue}")
    if len(issues) > 5:
        print(f"  ... и ещё {len(issues) - 5}")
else:
    print("\n✅ Проблем не найдено.")

# Смотрим первый опыт для деталей
if recent:
    first = recent[-1]
    arr = first['state'].cpu().numpy().flatten()
    print(f"\nПоследний опыт (действие={first.get('action')}, reward={first.get('reward', 0):.4f}):")
    print(f"  Размерность: {len(arr)}")
    print(f"  Диапазон: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Среднее: {arr.mean():.4f}")
    print(f"  Первые 5: {arr[:5].tolist()}")
    print(f"  Последние 5: {arr[-5:].tolist()}")
    print(f"  Топ-5 по модулю: {np.argsort(np.abs(arr))[-5:][::-1].tolist()}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)