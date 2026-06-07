#!/usr/bin/env python3
"""
ПРОВЕРКА НАСТРОЕК СТРАТЕГИЙ (v2)
Сравнивает strategies.json, model_state.json и runtime-стратегии.
Показывает расхождения и причину.
"""

import json
import sys

sys.path.insert(0, '.')

print("\n" + "=" * 60)
print("🔍 ПРОВЕРКА НАСТРОЕК СТРАТЕГИЙ (v2)")
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

# 3. Runtime (если система запущена)
strategies_runtime = {}
try:
    from models.trader_model import trader_model_instance
    strategies_runtime = trader_model_instance.strategies
    print(f"\n✅ Система запущена — получены runtime-стратегии")
except Exception as e:
    print(f"\n⚠️ Система не запущена — runtime-стратегии недоступны")

print(f"\n{'='*60}")
print(f"📊 СВОДНАЯ ТАБЛИЦА")
print(f"{'='*60}")
print(f"{'Стратегия':20s} {'config':8s} {'state':8s} {'runtime':8s} {'trades':>7s} {'win_rate':>8s} {'Вердикт'}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*30}")

for name in strategies_config:
    # Данные из конфига
    cfg_enabled = strategies_config[name].get('enabled', True)
    cfg_risk = strategies_config[name].get('risk_multiplier', 1.0)

    # Данные из model_state.json
    st_enabled = strategies_state.get(name, {}).get('enabled', True) if name in strategies_state else '—'
    st_risk = strategies_state.get(name, {}).get('risk_multiplier', '—') if name in strategies_state else '—'
    st_trades = perf_state.get(name, {}).get('total_trades', 0)
    st_win = perf_state.get(name, {}).get('win_rate', 0)

    # Данные из runtime
    rt_enabled = '—'
    rt_risk = '—'
    if strategies_runtime:
        rt_params = strategies_runtime.get(name, {})
        rt_enabled = rt_params.get('enabled', True)
        rt_risk = rt_params.get('risk_multiplier', '—')

    # Форматирование
    cfg_str = f"{'✅' if cfg_enabled else '❌'} {cfg_risk}"
    st_str = f"{'✅' if st_enabled == True else '❌' if st_enabled == False else '—'} {st_risk}"
    rt_str = f"{'✅' if rt_enabled == True else '❌' if rt_enabled == False else '—'} {rt_risk}"

    # Вердикт
    verdict = []
    if cfg_enabled is False and strategies_runtime and rt_enabled is not False:
        verdict.append("🔴 state переопределяет enabled!")
    if cfg_enabled is False and not strategies_runtime and st_enabled is not False:
        verdict.append("🔴 state не содержит enabled — модель считает активной!")
    if st_risk != '—' and st_risk < cfg_risk:
        verdict.append(f"🟡 risk снижен {cfg_risk}→{st_risk}")
    if st_trades > 50 and st_win < 0.10:
        verdict.append(f"⚠️ {st_trades} сделок, win_rate={st_win:.0%}")
    if not verdict:
        verdict.append("✅ OK")

    print(f"{name:20s} {cfg_str:8s} {st_str:8s} {rt_str:8s} {st_trades:>7d} {st_win:>7.1%} {verdict[0]}")

print(f"\n{'='*60}")
print("📋 РЕКОМЕНДАЦИИ")
print(f"{'='*60}")

issues_found = False
for name in strategies_config:
    cfg_enabled = strategies_config[name].get('enabled', True)
    st_enabled = strategies_state.get(name, {}).get('enabled', True) if name in strategies_state else True

    if cfg_enabled is False and st_enabled is not False:
        print(f"   🔴 {name}: отключена в конфиге, но model_state.json переопределяет!")
        print(f"      → Удалите model_state.json (пункт 5 в reset_system.py)")
        issues_found = True

if not issues_found:
    print(f"   ✅ Все стратегии синхронизированы между конфигом и model_state.json")

print("\n" + "=" * 60)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 60)