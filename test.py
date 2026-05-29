"""
Диагностика: проверка нормализации признака объёма
"""
import json
import numpy as np

with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

with open("config/rl_config.json", "r", encoding="utf-8") as f:
    rl_config = json.load(f)

# Текущая нормализация из конфига
volume_normalization = rl_config.get("normalization", {}).get("volume_normalization", 1e7)
print(f"Текущая volume_normalization: {volume_normalization:,.0f}")

# Примеры реальных объёмов из логов
test_cases = [
    ("CHMF", 72.26),       # из diagnose_coach.py: volume / 1e7 = 72 → реальный объём = 722 млн
    ("VTBR", 423.98),      # из diagnose_coach.py: volume / 1e7 = 424 → реальный объём = 4.24 млрд
    ("SBER (типичный)", 1500.0),  # ~15 млрд руб для Сбера
    ("Малоликвидный", 5.0),       # 50 млн руб
]

print("\n" + "=" * 70)
print("ТЕКУЩАЯ НОРМАЛИЗАЦИЯ (volume / 1e7)")
print("=" * 70)
print(f"{'Тикер':<20} {'Реальный объём':<18} {'В векторе':<12} {'Вклад в модель'}")
print("-" * 70)

for name, vol_in_vector in test_cases:
    real_volume = vol_in_vector * 1e7
    impact = "ДОМИНИРУЕТ" if vol_in_vector > 10 else "нормальный" if vol_in_vector < 5 else "повышенный"
    print(f"{name:<20} {real_volume:>15,.0f} ₽ {vol_in_vector:>10.1f}   {impact}")

# Предлагаемая нормализация: делить на market_cap_total из конфига
market_cap_divisor = rl_config.get("normalization", {}).get("market_cap_divisor_total", 1e14)
print(f"\nmarket_cap_divisor_total: {market_cap_divisor:,.0f}")

print("\n" + "=" * 70)
print("ПРЕДЛАГАЕМАЯ НОРМАЛИЗАЦИЯ (volume / 1e12)")
print("=" * 70)
print(f"{'Тикер':<20} {'Реальный объём':<18} {'В векторе':<12} {'Вклад в модель'}")
print("-" * 70)

new_normalization = 1e12
for name, vol_in_vector in test_cases:
    real_volume = vol_in_vector * 1e7
    new_value = real_volume / new_normalization
    impact = "нормальный" if new_value < 5 else "повышенный"
    print(f"{name:<20} {real_volume:>15,.0f} ₽ {new_value:>10.6f}   {impact}")

# Проверяем другие признаки для сравнения масштабов
print("\n" + "=" * 70)
print("СРАВНЕНИЕ МАСШТАБОВ ПРИЗНАКОВ (первые 10 из diagnose_coach.py)")
print("=" * 70)

sample_vector = [0.0696, 72.26, 40.0, 0.582, 0.342, 1.015, 1.016, -0.417, 0.0033, 5.09]
sample_names = [
    "price/10000",
    "volume/1e7 (СТАРЫЙ)",
    "spread*100",
    "market_cap/1e12",
    "rsi/100",
    "sma_10_ratio",
    "sma_20_ratio",
    "bb_position",
    "atr/price",
    "volume_ratio"
]

print(f"{'Признак':<25} {'Значение':<12} {'Масштаб'}")
print("-" * 50)
for name, val in zip(sample_names, sample_vector):
    bar = "█" * min(int(abs(val)), 20)
    print(f"{name:<25} {val:>10.4f}   {bar}")

# Предлагаемое изменение
print("\n" + "=" * 70)
print("РЕКОМЕНДАЦИЯ")
print("=" * 70)
print("Заменить volume / 1e7 на volume / 1e12 в build_state_vector")
print("Это нормализует объём к диапазону 0.01-1.5 вместо 5-1500")
print("Все признаки будут в сопоставимом масштабе")