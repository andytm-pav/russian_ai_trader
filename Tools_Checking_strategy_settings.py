#!/usr/bin/env python3
"""
ПРОВЕРКА НАСТРОЕК СТРАТЕГИЙ (v4)
Сравнивает strategies.json, model_state.json, runtime-стратегии и реальный выбор моделью.
"""

import json
import sys
import random
import torch
from collections import Counter

sys.path.insert(0, '.')

print("\n" + "=" * 60)
print("🔍 ПРОВЕРКА НАСТРОЕК СТРАТЕГИЙ (v4)")
print("=" * 60)

# 1. strategies.json
try:
    with open('config/strategies.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    strategies_config = config.get('strategies', {})
except Exception as e:
    print(f"❌ Ошибка strategies.json: {e}")
    sys.exit(1)

# 2. model_state.json
try:
    with open('models/saved_trader/model_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    strategies_state = state.get('strategies', {})
    perf_state = state.get('strategy_performance', {})
except Exception as e:
    print(f"⚠️ model_state.json не найден или ошибка: {e}")
    strategies_state = {}
    perf_state = {}

# 3. Runtime
strategies_runtime = {}
try:
    from models.trader_model import trader_model_instance
    strategies_runtime = trader_model_instance.strategies
    print(f"\n✅ Система запущена — получены runtime-стратегии")
except Exception as e:
    print(f"\n⚠️ Система не запущена — runtime-стратегии недоступны")

print(f"\n{'='*80}")
print(f"📊 ПОЛНАЯ СВОДНАЯ ТАБЛИЦА")
print(f"{'='*80}")
print(f"{'Стратегия':20s} {'config':10s} {'state':10s} {'runtime':10s} {'trades':>7s} {'win':>7s} {'Итог':20s} {'Пояснение'}")
print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*20} {'-'*30}")

all_names = set(strategies_config.keys()) | set(strategies_state.keys()) | set(strategies_runtime.keys())

for name in sorted(all_names):
    cfg_params = strategies_config.get(name, {})
    cfg_enabled = cfg_params.get('enabled', True)
    cfg_risk = cfg_params.get('risk_multiplier', '—')

    st_params = strategies_state.get(name, {})
    st_enabled = st_params.get('enabled', True) if name in strategies_state else '—'
    st_risk = st_params.get('risk_multiplier', '—') if name in strategies_state else '—'
    st_trades = perf_state.get(name, {}).get('total_trades', 0)
    st_win = perf_state.get(name, {}).get('win_rate', 0)

    rt_params = strategies_runtime.get(name, {})
    rt_present = name in strategies_runtime
    rt_risk = rt_params.get('risk_multiplier', '—') if rt_present else '—'

    cfg_str = f"{'✅' if cfg_enabled else '❌'} {cfg_risk}"
    st_str = f"{'✅' if st_enabled == True else '❌' if st_enabled == False else '—'} {st_risk}"
    rt_str = f"{'✅' if rt_present else '❌ нет'} {rt_risk}"

    will_use = rt_present
    verdict = []
    explanation = []

    if not rt_present:
        verdict.append("НЕ ИСПОЛЬЗУЕТСЯ")
        if cfg_enabled is False:
            explanation.append("отключена в конфиге")
        else:
            explanation.append("отсутствует в runtime")
    else:
        verdict.append("ИСПОЛЬЗУЕТСЯ ✅")
        if st_risk != '—' and st_risk != cfg_risk and st_risk < cfg_risk:
            explanation.append(f"risk снижен {cfg_risk}→{st_risk}")
        if st_trades > 50 and st_win < 0.10:
            explanation.append(f"{st_trades} сделок, win={st_win:.0%}")
        if not explanation:
            explanation.append("OK")

    print(f"{name:20s} {cfg_str:10s} {st_str:10s} {rt_str:10s} {st_trades:>7d} {st_win:>6.1%} {verdict[0]:20s} {', '.join(explanation)}")

# Анализ расхождений
print(f"\n{'='*80}")
print(f"📋 АНАЛИЗ РАСХОЖДЕНИЙ")
print(f"{'='*80}")

issues = []
for name in sorted(all_names):
    cfg_params = strategies_config.get(name, {})
    cfg_enabled = cfg_params.get('enabled', True)
    rt_present = name in strategies_runtime
    st_present = name in strategies_state

    if cfg_enabled is False and rt_present:
        issues.append(f"🔴 {name}: отключена в конфиге, но присутствует в runtime!")
        issues.append(f"   → Нажмите '🔧 Обновить конфиги' в дашборде")
    elif cfg_enabled is False and not rt_present and st_present:
        issues.append(f"🟡 {name}: отключена в конфиге и удалена из runtime, но осталась в model_state.json")
        issues.append(f"   → При следующем save_model() будет удалена из model_state.json")
    elif cfg_enabled and rt_present and not st_present:
        issues.append(f"🟡 {name}: включена в конфиге и runtime, но отсутствует в model_state.json")
        issues.append(f"   → Появится в model_state.json после save_model()")

if not issues:
    print("   ✅ Все стратегии синхронизированы. Расхождений нет.")
else:
    for line in issues:
        print(f"   {line}")

# Реальный выбор моделью
print(f"\n{'='*80}")
print(f"🎯 РЕАЛЬНЫЙ ВЫБОР МОДЕЛЬЮ (50 проходов)")
print(f"{'='*80}")

if strategies_runtime:
    model = trader_model_instance
    state = torch.randn(model.base_state_dim).to(model.device)
    choices = Counter()
    actions = Counter()

    for i in range(50):
        ticker = random.choice(['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR'])
        price = random.uniform(100, 500)
        try:
            action, strategy, conf = model.choose_action_with_strategy(
                state=state,
                ticker=ticker,
                price=price,
                market_context={
                    'market_sentiment': 0,
                    'volatility': 1.0,
                    'confidence': 0.7,
                    'time_of_day': 0.5,
                    'ticker_sentiment': 0,
                    'assigned_horizon': 'week'
                }
            )
            choices[strategy] += 1
            action_name = model.rl_config.get('action_mapping', {}).get(str(action), f'UNK{action}')
            actions[action_name] += 1
        except Exception as e:
            choices[f'ОШИБКА: {e}'] += 1

    total = sum(choices.values())
    print(f"\n   Стратегии ({total} проходов):")
    for name, count in choices.most_common():
        bar = '█' * int(count / total * 30)
        print(f"   {name:20s}: {count:>3d} ({count/total*100:5.1f}%) {bar}")

    print(f"\n   Действия:")
    for name, count in actions.most_common():
        print(f"   {name:15s}: {count:>3d}")

    # Вердикт
    disabled_in_config = [n for n, p in strategies_config.items() if p.get('enabled', True) is False]
    used_disabled = [n for n in disabled_in_config if choices.get(n, 0) > 0]
    if used_disabled:
        print(f"\n   🔴 ОТКЛЮЧЁННЫЕ СТРАТЕГИИ ВЫБИРАЮТСЯ: {used_disabled}")
    else:
        print(f"\n   ✅ Отключённые стратегии не выбираются.")
else:
    print("   ⚠️ Система не запущена — проверка выбора невозможна.")

print("\n" + "=" * 80)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)