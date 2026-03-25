# check_dimensions.py
"""
Скрипт проверки размерности модели
Запустить после всех изменений
"""

import sys
import torch

# Импортируем модель
from models.trader_model import AdvancedTraderModel, trader_model_instance

print("=" * 60)
print("ПРОВЕРКА РАЗМЕРНОСТИ МОДЕЛИ")
print("=" * 60)

# 1. Проверка конфигов
print("\n1. КОНФИГИ:")
state_params = trader_model_instance.rl_config.get('state_parameters', {})
print(f"   state_vector_size: {state_params.get('state_vector_size', 'не найден')}")
print(f"   total_state_size: {state_params.get('total_state_size', 'не найден')}")
print(f"   news_features_size: {state_params.get('news_features_size', 'не найден')}")
print(f"   reserved_slots: {state_params.get('reserved_slots', 'не найден')}")

# 2. Проверка атрибутов модели
print("\n2. АТРИБУТЫ МОДЕЛИ:")
print(f"   base_state_dim: {trader_model_instance.base_state_dim}")
print(f"   strategy_params_dim: {trader_model_instance.strategy_params_dim}")
print(f"   total_state_dim: {trader_model_instance.total_state_dim}")
print(f"   news_encoded_dim: {trader_model_instance.news_encoded_dim}")

# 3. Проверка dimensions
print("\n3. DIMENSIONS:")
dimensions = trader_model_instance.dimensions
total = 0
for name, size in dimensions.items():
    print(f"   {name}: {size}")
    total += size
print(f"   ИТОГО (базовые признаки): {total}")

# 4. Проверка expected_dim
print("\n4. EXPECTED DIMENSION:")
expected_dim = trader_model_instance._get_expected_dimension()
print(f"   expected_dim: {expected_dim}")

# 5. Проверка создания тестового состояния
print("\n5. ТЕСТОВОЕ СОСТОЯНИЕ:")
import numpy as np
from datetime import datetime

# Создаем тестовые данные
test_news = torch.zeros(1, trader_model_instance.news_encoded_dim).to(trader_model_instance.device)

test_market_data = {
    'volume': 1000000,
    'spread': 0.01,
    'rsi': 50,
    'volatility': 0.1,
    'sma_10_ratio': 1.0,
    'sma_20_ratio': 1.0,
    'bb_position': 0.5,
    'volume_ratio': 1.0,
    'atr': 10,
    'market_cap': 1e12,
    'lot_size': 1,
    'min_step': 0.01,
    'sector': 'other',
    'momentum': 0.0,
    'imoex': 3000,
    'imoex_change': 0,
    'rtsi': 1000,
    'rtsi_change': 0,
    'rvi': 20,
    'rvi_change': 0,
    'moexog': 8000,
    'moexfn': 10000,
    'brent': 80,
    'brent_change': 0,
}

try:
    state = trader_model_instance.build_state_vector(
        ticker="TEST",
        price=100.0,
        momentum=0.0,
        sentiment=0.0,
        news_features=test_news,
        market_data=test_market_data,
        market_sentiment=0.0,
        portfolio=None
    )
    state_size = state.shape[0]
    print(f"   Размер созданного состояния: {state_size}")

    if state_size == trader_model_instance.base_state_dim:
        print(f"   ✅ Состояние совпадает с base_state_dim ({trader_model_instance.base_state_dim})")
    else:
        print(f"   ❌ Ожидалось {trader_model_instance.base_state_dim}, получено {state_size}")

except Exception as e:
    print(f"   ❌ Ошибка создания состояния: {e}")
    import traceback

    traceback.print_exc()

# 6. Проверка full_state
print("\n6. ПОЛНОЕ СОСТОЯНИЕ (со стратегией):")
try:
    test_base = torch.randn(trader_model_instance.base_state_dim).to(trader_model_instance.device)
    test_params = {
        'news_weight': 0.5,
        'tech_weight': 0.5,
        'risk_multiplier': 1.0,
        'target_hold_time_hours': 6,
        'stop_loss_percent': 2.5,
        'take_profit_percent': 5.0
    }
    full_state = trader_model_instance._create_strategy_state(test_base, test_params)
    full_size = full_state.shape[0]
    print(f"   Размер полного состояния: {full_size}")

    if full_size == trader_model_instance.total_state_dim:
        print(f"   ✅ Полное состояние совпадает с total_state_dim ({trader_model_instance.total_state_dim})")
    else:
        print(f"   ❌ Ожидалось {trader_model_instance.total_state_dim}, получено {full_size}")

except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# 7. Вывод итогов
print("\n" + "=" * 60)
print("ИТОГ:")

errors = []
if state_size != trader_model_instance.base_state_dim:
    errors.append(f"base_state_dim: {trader_model_instance.base_state_dim} != {state_size}")
if full_size != trader_model_instance.total_state_dim:
    errors.append(f"total_state_dim: {trader_model_instance.total_state_dim} != {full_size}")

if errors:
    print("❌ ОБНАРУЖЕНЫ ОШИБКИ:")
    for err in errors:
        print(f"   {err}")
else:
    print("✅ ВСЕ РАЗМЕРНОСТИ КОРРЕКТНЫ")
    print(f"   base_state_dim: {trader_model_instance.base_state_dim}")
    print(f"   total_state_dim: {trader_model_instance.total_state_dim}")
    print(f"   Фактическое состояние: {state_size}")
    print(f"   Полное состояние: {full_size}")

print("=" * 60)