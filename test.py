#!/usr/bin/env python3
"""
ПРОВЕРКА: удаляется ли balanced из runtime при reload_configs()
"""

import json
import sys

sys.path.insert(0, '.')

from models.trader_model import trader_model_instance

print("\n" + "=" * 60)
print("🔍 ПРОВЕРКА УДАЛЕНИЯ BALANCED ИЗ RUNTIME")
print("=" * 60)

model = trader_model_instance

# 1. Есть ли balanced в strategies сейчас?
print(f"\n1. Стратегии в памяти ДО проверки:")
for name in sorted(model.strategies.keys()):
    enabled = model.strategies[name].get('enabled', True)
    icon = '✅' if enabled else '❌'
    print(f"   {icon} {name}")

# 2. Имитируем reload_configs
print(f"\n2. Имитация reload_configs...")
with open('config/strategies.json', 'r', encoding='utf-8') as f:
    new_strategies = json.load(f)

for name, params in new_strategies['strategies'].items():
    if name in model.strategies:
        if params.get('enabled', True) is False:
            model.strategies.pop(name, None)
            print(f"   Удалена: {name}")
            continue
        old_risk = model.strategies[name].get('risk_multiplier', 1.0)
        new_risk = params.get('risk_multiplier', 1.0)
        model.strategies[name].update(params)
        if old_risk < new_risk:
            model.strategies[name]['risk_multiplier'] = old_risk

# 3. Есть ли balanced после?
print(f"\n3. Стратегии в памяти ПОСЛЕ reload_configs:")
for name in sorted(model.strategies.keys()):
    enabled = model.strategies[name].get('enabled', True)
    icon = '✅' if enabled else '❌'
    print(f"   {icon} {name}")

# 4. Проверяем choose_action_with_strategy
if 'balanced' in model.strategies:
    print(f"\n❌ BALANCED ВСЁ ЕЩЁ В ПАМЯТИ!")
    print(f"   Модель будет его использовать.")
else:
    print(f"\n✅ BALANCED УДАЛЁН ИЗ ПАМЯТИ!")
    print(f"   Модель НЕ будет его использовать.")

# 5. Имитируем вызов choose_action_with_strategy
print(f"\n4. Имитация выбора стратегии (10 проходов):")
import torch
import random

state = torch.randn(model.base_state_dim).to(model.device)
found_balanced = 0
for i in range(10):
    try:
        action, strategy, conf = model.choose_action_with_strategy(
            state=state,
            ticker='SBER',
            price=320.0,
            market_context={'market_sentiment': 0, 'volatility': 1.0,
                          'confidence': 0.7, 'time_of_day': 0.5,
                          'ticker_sentiment': 0, 'assigned_horizon': 'week'}
        )
        if strategy == 'balanced':
            found_balanced += 1
            print(f"   {i+1}: balanced (не должно быть!)")
    except Exception as e:
        print(f"   {i+1}: ошибка — {e}")

if found_balanced == 0:
    print(f"\n✅ Модель НИ РАЗУ не выбрала balanced!")
else:
    print(f"\n❌ Модель выбрала balanced {found_balanced}/10 раз!")

print("\n" + "=" * 60)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 60)