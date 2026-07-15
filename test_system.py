#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест системы: проверяет все ключевые компоненты.
Запуск: python test_system.py
"""
import sys
import os
import json
import time

os.chdir('/Users/root/PycharmProjects/russian_ai_trader')
sys.path.insert(0, '/Users/root/PycharmProjects/russian_ai_trader')

import warnings
warnings.filterwarnings('ignore')

passed = 0
failed = 0
errors = []

def check(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {details}")
        failed += 1
        errors.append(name)

print("=" * 60)
print("ТЕСТ СИСТЕМЫ russian_ai_trader v14.9")
print("=" * 60)

# 1. Проверка зависимостей
print("\n--- 1. Зависимости ---")
try:
    import scipy
    check("scipy", True, f"v{scipy.__version__}")
except ImportError:
    check("scipy", False, "не установлен. pip install scipy")

try:
    import talib
    check("TA-Lib", True, f"v{talib.__version__}")
except ImportError:
    check("TA-Lib", False, "не установлен")

try:
    import torch
    check("torch", True, f"v{torch.__version__}")
except ImportError:
    check("torch", False, "не установлен")

try:
    import dash
    check("dash", True, f"v{dash.__version__}")
except ImportError:
    check("dash", False, "не установлен")

# 2. Проверка конфигов
print("\n--- 2. Конфиги ---")
with open('config/settings.json', 'r', encoding='utf-8') as f:
    settings = json.load(f)
check("settings.json: entry_cascading", 'entry_cascading' in settings, "секция отсутствует")
check("settings.json: rolling_exit", 'rolling_exit' in settings, "секция отсутствует")
check("settings.json: hawkes.per_ticker_thresholds", 
      'per_ticker_thresholds' in settings.get('hawkes', {}), "секция отсутствует")

with open('config/tickers.json', 'r', encoding='utf-8') as f:
    tickers = json.load(f)
check("tickers.json: watchlist", len(tickers.get('watchlist', [])) > 0, "пустой watchlist")

# 3. Проверка хаос-метрик
print("\n--- 3. Хаос-метрики ---")
from fetchers.history_loader import history_loader
import numpy as np

# Загружаем историю
print("  Загрузка истории...")
hist = history_loader.load_history(months_back=3)
check("История загружена", len(hist) > 0, f"{len(hist)} тикеров")

# Проверяем расчёт хаос-метрик
print("  Расчёт хаос-метрик...")
chaos = history_loader.get_chaos_metrics()
if len(chaos) == 0:
    print("  Кэш пуст, пересчитываем...")
    history_loader._calculate_chaos_metrics(hist)
    chaos = history_loader.get_chaos_metrics()

check("Хаос-метрики рассчитаны", len(chaos) > 0, f"{len(chaos)} тикеров")

if chaos:
    ticker = list(chaos.keys())[0]
    m = chaos[ticker]
    check("  hurst", 'hurst' in m, f"нет в {ticker}")
    check("  rqa_DET", 'rqa_DET' in m and m['rqa_DET'] > 0, f"DET={m.get('rqa_DET', 0)}")
    check("  rqa_L_max", 'rqa_L_max' in m and m['rqa_L_max'] > 0, f"L_max={m.get('rqa_L_max', 0)}")
    check("  kurtosis", 'kurtosis' in m, f"нет в {ticker}")
    check("  atr_pct", 'atr_pct' in m, f"нет в {ticker}")
    print(f"  Пример {ticker}: H={m.get('hurst',0):.3f}, DET={m.get('rqa_DET',0):.2f}, "
          f"L_max={m.get('rqa_L_max',0)}, kurt={m.get('kurtosis',0):.1f}, "
          f"ATR={m.get('atr_pct',0):.2f}%")

# 4. Проверка Хокса
print("\n--- 4. Процесс Хокса ---")
from core.hawkes_signal import hawkes_signal

now = time.time()
# Кормим ценами
test_prices = [100 + i * 0.1 for i in range(200)]
hawkes_signal.set_ticker_volatility('TEST', 0.5)
for i, p in enumerate(test_prices[-200:]):
    ts = now - (199 - i) * 3600
    hawkes_signal.update_price('TEST', p, ts)
hawkes_signal.fit('TEST', now)

params = hawkes_signal.get_params('TEST')
check("Хокс: mu_bull > 0", params.get('mu_bull', 0) > 0, f"mu_bull={params.get('mu_bull', 0)}")
check("Хокс: beta_bull > 0", params.get('beta_bull', 0) > 0, f"beta_bull={params.get('beta_bull', 0)}")

fc = hawkes_signal.forecast('TEST', now, horizon=4)
check("Хокс: bull_expected > 0", fc.get('bull_expected', 0) > 0, f"bull_expected={fc.get('bull_expected', 0)}")
check("Хокс: prob_bull > 0", fc.get('prob_bull', 0) > 0, f"prob_bull={fc.get('prob_bull', 0)}")
print(f"  Forecast: bull={fc.get('bull_expected',0):.3f}, prob={fc.get('prob_bull',0):.3f}")

# 5. Проверка Entry Cascading
print("\n--- 5. Entry Cascading Confirmer ---")
from core.entry_cascading_confirmer import entry_confirmer
from core.core_technical_trader import TechnicalTraderCore

# Создаём тестовые данные — подбираем seed для RSI 50-70, BB 0.4-0.8
best_prices = None
best_rsi = 0
for seed in range(200):
    np.random.seed(seed)
    test_p = [100]
    for _ in range(40):
        test_p.append(test_p[-1] * (1 + 0.002 + np.random.randn() * 0.004))
    for _ in range(10):
        test_p.append(test_p[-1] * (1 + np.random.randn() * 0.002))

    tc_test = TechnicalTraderCore()
    for p in test_p:
        tc_test.update_price_data('TEST', p, 1000)
    ind_test = tc_test.calculate_indicators('TEST')
    rsi = ind_test.get('rsi', 0)
    bb = ind_test.get('bb_position', 0)
    mom = ind_test.get('momentum', 0)
    if 50 <= rsi <= 70 and 0.4 <= bb <= 0.8 and mom > 0.5:
        best_prices = test_p
        best_rsi = rsi
        break

if best_prices is None:
    # Fallback — используем seed 12
    np.random.seed(12)
    best_prices = [100]
    for _ in range(50):
        best_prices.append(best_prices[-1] * (1 + 0.002 + np.random.randn() * 0.004))

prices = best_prices

from core.core_technical_trader import TechnicalTraderCore as TTC2
tc = TTC2()
for p in prices:
    tc.update_price_data('TEST', p, 1000)

ind = tc.calculate_indicators('TEST')
check("RSI рассчитан", 'rsi' in ind, "нет rsi")
check("bb_position рассчитан", 'bb_position' in ind, "нет bb_position")
check("momentum рассчитан", 'momentum' in ind, "нет momentum")
print(f"  RSI={ind.get('rsi',0):.1f}, BB={ind.get('bb_position',0):.3f}, mom={ind.get('momentum',0):.2f}")

# Тест полного пайплайна Entry Confirmer
def get_hawkes_forecast(t, h):
    return {'bull_expected': 0.7, 'bear_expected': 0.3, 'prob_bull': 0.55, 'net_signal': 0.4}
def get_indicators(t):
    return tc.calculate_indicators(t)
def get_microstructure(t):
    return {}
def get_chaos_metrics(t):
    return {'rqa_DET': 0.35, 'rqa_L_max': 15, 'kurtosis': 10, 'hurst': 0.55, 'atr_pct': 1.0}
def get_price(t):
    return prices[-1]

signal = entry_confirmer._evaluate_ticker('TEST', prices[-1],
    get_hawkes_forecast, get_indicators, get_microstructure,
    get_chaos_metrics, [], {})
check("Entry signal PASS", signal is not None, "REJECT — проверьте RSI/BB/momentum")
if signal:
    print(f"  Signal: conf={signal.confidence:.2f}, stop={signal.stop_loss_pct:.2f}%")

# 6. Проверка Rolling Exit
print("\n--- 6. Rolling Exit Manager ---")
from core.rolling_exit_manager import rolling_exit_manager

rolling_exit_manager.on_buy(
    ticker='TEST', price=100.0, qty=10,
    chaos_metrics={'hurst': 0.55, 'atr_pct': 1.0, 'kurtosis': 10, 'rqa_DET': 0.35, 'rqa_L_max': 15},
    hawkes_signal_val=0.5, ms_imbalance=0.3
)
active = rolling_exit_manager.get_active_positions()
check("Position registered", 'TEST' in active, "не зарегистрирована")

decision = rolling_exit_manager.evaluate(
    ticker='TEST', current_price=102.0, current_cycle=1,
    get_pred_1h=lambda t: {'p_down': 0.3, 'p_up': 0.5},
    get_hawkes_signal=lambda t: 0.3,
    get_indicators=lambda t: {'rsi': 55, 'rsi_short': 55, 'momentum_1h': 0.5, 'momentum_4h': 1.0,
                              'local_max_6h': 102, 'distance_from_local_max_pct': 0,
                              'bb_position': 0.6},
    get_microstructure=lambda t: {'imbalance': 0.2}
)
check("Rolling exit decision", decision is not None, "нет решения")
if decision:
    print(f"  Decision: {decision.action}, score={decision.sell_score:.2f}/{decision.threshold:.2f}")

# 7. Проверка модели
print("\n--- 7. RL Модель ---")
from models.trader_model import trader_model_instance
import torch

state = torch.randn(1, 227)
trader_model_instance.policy_net.eval()
with torch.no_grad():
    probs, value, _ = trader_model_instance.policy_net(state)
check("Forward pass", probs.shape == (1, 7), f"shape={probs.shape}")
check("Softmax sum = 1", abs(probs.sum().item() - 1.0) < 0.01, f"sum={probs.sum().item()}")
check("No collapse", probs.max().item() < 0.95, f"max={probs.max().item()}")
check("learn_supervised exists", hasattr(trader_model_instance, 'learn_supervised'), "метод отсутствует")
print(f"  Probs: max={probs.max().item():.3f}, min={probs.min().item():.3f}")

# 8. Проверка веб-интерфейса
print("\n--- 8. Веб-интерфейс ---")
try:
    from web import app
    check("web.app импорт", True)
    check("callbacks > 25", len(app.app.callback_map) > 25, f"{len(app.app.callback_map)} callbacks")
except Exception as e:
    check("web.app импорт", False, str(e))

# Итог
print("\n" + "=" * 60)
print(f"РЕЗУЛЬТАТ: {passed} passed, {failed} failed")
if errors:
    print(f"Ошибки: {', '.join(errors)}")
print("=" * 60)
