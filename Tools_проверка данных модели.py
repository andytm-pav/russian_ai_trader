"""
Проверка данных, которые модель получает на вход
"""
import json
import time
import torch
import numpy as np
from datetime import datetime
from models.trader_model import trader_model_instance
from models.smart_broker import SmartPortfolioBroker
from core.trading_hours_scheduler import TradingScheduler

with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

with open("config/rl_config.json", "r", encoding="utf-8") as f:
    rl_config = json.load(f)

# Создаём брокер
scheduler = TradingScheduler()
broker = SmartPortfolioBroker(settings, scheduler)
model = broker.model

print("=" * 70)
print("🔍 ДАННЫЕ НА ВХОДЕ МОДЕЛИ")
print("=" * 70)

# 1. Портфель
print("\n💼 ПОРТФЕЛЬ:")
print(f"  Позиций: {len(broker.portfolio.positions)}")
print(f"  Кэш: {broker.portfolio.cash:.0f}₽")
print(f"  Зарезервировано: {broker.portfolio.reserved_cash:.0f}₽")
print(f"  Свободно: {broker.portfolio.cash - broker.portfolio.reserved_cash:.0f}₽")
print(f"  Initial capital: {broker.portfolio.initial_capital:.0f}₽")
for ticker, pos in broker.portfolio.positions.items():
    pnl = (broker.moex.get_price(ticker) or pos['avg_price']) - pos['avg_price']
    print(f"  {ticker}: {pos['qty']} шт @ {pos['avg_price']:.2f}₽, PnL: {pnl:+.2f}₽, стратегия: {pos.get('strategy', '?')}")

# 2. Выбираем тикер для анализа
ticker = list(broker.portfolio.positions.keys())[0] if broker.portfolio.positions else "SBER"
price = broker.moex.get_price(ticker) or 322.0
securities = broker.moex.get_all_securities()
sec_info = securities.get(ticker, {'lot_size': 1, 'min_step': 0.01, 'sector': 'other', 'volume': 0, 'market_cap': 0})

print(f"\n📊 АНАЛИЗ ТИКЕРА: {ticker} @ {price:.2f}₽")

# 3. Технические индикаторы
print("\n📈 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:")

# Загружаем свечи для истории цен (если её нет)
import pandas as pd
candles = broker.moex.get_candles(ticker, interval=60, count=50)
if candles is not None and not candles.empty:
    for idx, row in candles.iterrows():
        broker.technical_core.update_price_data(ticker, float(row['Close']), int(row.get('Volume', 0) or 0))
    print(f"  Загружено {len(candles)} свечей в историю")
else:
    print("  Свечи не получены — индикаторы могут быть недоступны")

indicators = broker.technical_core.calculate_indicators(ticker)
for k, v in indicators.items():
    if v and abs(v) > 0.001:
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# 4. Новости
print("\n📰 НОВОСТИ:")
ticker_names = rl_config.get('ticker_names', {})
keywords = ticker_names.get(ticker, [])
news_items = broker.news_fetcher.search_news(ticker=ticker, limit=3, keywords=keywords)
if news_items:
    news_items = broker.news_fetcher.analyze_sentiment_batch(news_items)
    for n in news_items[:3]:
        print(f"  [{n.get('sentiment', 0):+.2f}] {n.get('title', '')[:80]}")
else:
    print("  Нет новостей")

# 5. Макро-данные
print("\n🌍 МАКРО-ДАННЫЕ:")
macro = broker.moex.get_macro_data()
print(f"  IMOEX: {macro.get('imoex', 0):.0f} ({macro.get('imoex_change', 0):+.1f}%)")
print(f"  RTSI: {macro.get('rtsi', 0):.0f}")
print(f"  Brent: ${macro.get('brent', 0):.1f}")
print(f"  RVI: {macro.get('rvi', 0):.1f}")
print(f"  USD/RUB: {macro.get('usd_rub', 0):.2f}")
print(f"  VIX: {macro.get('vix', 0):.2f}")
print(f"  Market mood: {macro.get('market_mood', 0):.2f}")
print(f"  Liq ratio: {macro.get('market_liquidity_ratio', 0):.4f}")

# 6. Вектор состояния (как его видит модель)
print("\n🧠 ВЕКТОР СОСТОЯНИЯ:")
state = broker._create_initial_state(ticker, price, sec_info)
print(f"  Размерность (base): {state.shape[0]}")
print(f"  NaN: {torch.isnan(state).any().item()}")
print(f"  Inf: {torch.isinf(state).any().item()}")
print(f"  Диапазон: [{state.min().item():.4f}, {state.max().item():.4f}]")
print(f"  Среднее: {state.mean().item():.4f}")
print(f"  Первые 10: {state[:10].tolist()}")
print(f"  Последние 10: {state[-10:].tolist()}")

# 7. Добавляем стратегию и прогоняем через модель
print("\n🤖 ПРОГОН ЧЕРЕЗ МОДЕЛЬ:")
strategy_params = list(model.strategies.values())[0] if model.strategies else {}
full_state = model._create_strategy_state(state, strategy_params)
state_tensor = full_state.unsqueeze(0).to(model.device)
print(f"  Полная размерность: {state_tensor.shape[1]}")

model.policy_net.eval()
with torch.no_grad():
    action_probs, state_value, price_pred = model.policy_net(state_tensor)
    action_idx = action_probs.argmax().item()
    action_probs_np = action_probs.cpu().numpy().flatten()

print(f"  State value: {state_value.item():.4f}")
print(f"  Выбрано действие: {broker.action_mapping.get(str(action_idx), str(action_idx))} (idx={action_idx})")
print(f"\n  Распределение вероятностей:")
for i, prob in enumerate(action_probs_np):
    bar = "█" * int(prob * 30)
    print(f"    {broker.action_mapping.get(str(i), i):<15} {prob:.3f} {bar}")

# 8. Режим рынка
print(f"\n📊 РЕЖИМ РЫНКА:")
market_regime = broker.technical_core.get_market_regime(macro.get('imoex', 0), macro.get('imoex_change', 0))
regime_names = {0: "боковик", 1: "растущий", 2: "падающий"}
print(f"  Режим: {regime_names.get(market_regime, '?')} (код: {market_regime})")

# 9. Частота сделок
print(f"\n🔄 ЧАСТОТА СДЕЛОК:")
if hasattr(broker.portfolio, 'trade_history'):
    now_ts = time.time()
    trades_last_hour = sum(1 for t in broker.portfolio.trade_history
                          if t.get('timestamp') and (now_ts - (float(t['timestamp']) if isinstance(t['timestamp'], (int, float)) else 0)) < 3600)
    print(f"  Сделок за час: {trades_last_hour}")
    print(f"  Всего сделок: {len(broker.portfolio.trade_history)}")

print("\n" + "=" * 70)
print("ГОТОВО")
print("=" * 70)