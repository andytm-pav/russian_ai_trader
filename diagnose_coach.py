"""
Диагностика: реальные данные из проекта для LLM-коуча (v2)
"""
import json
import time
import random
import re
import numpy as np
import torch
import pandas as pd
from datetime import datetime

# Импорты проекта
from models.trader_model import trader_model_instance
from core.core_technical_trader import TechnicalTraderCore
from core.trading_hours_scheduler import TradingScheduler
from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import OptimizedNewsFetcher
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger("DIAGNOSE")

# ============================================================
# 1. ИНИЦИАЛИЗАЦИЯ
# ============================================================

print("\n" + "=" * 70)
print("🔍 ДИАГНОСТИКА: РЕАЛЬНЫЕ ДАННЫЕ ДЛЯ LLM-КОУЧА (v2)")
print("=" * 70)

with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

print(f"Режим: {'симуляция' if settings.get('simulation_mode') else 'реальный'}")
print(f"Капитал: {settings['initial_capital_rub']:,}₽")

# Инициализация
print("\n⏳ Инициализация компонентов...")
moex = MoexFetcher()
tech_core = TechnicalTraderCore()
news_fetcher = OptimizedNewsFetcher("config/rss_sources.json")
scheduler = TradingScheduler()
model = trader_model_instance

# Портфель с начальным капиталом
portfolio = PortfolioManager()
portfolio.initial_capital = settings.get("initial_capital_rub", 10000)
portfolio.max_positions = settings.get("max_positions", 5)
portfolio.cash = settings.get("initial_capital_rub", 10000)

# Прогрев кэша новостей
print("⏳ Загрузка новостей...")
news = news_fetcher.get_last_news(limit=100)
print(f"   Загружено: {len(news)} новостей")

# ============================================================
# 2. ВЫБОР ТИКЕРА (из портфеля или топ-ликвидности)
# ============================================================

print("\n" + "-" * 70)
print("📊 ВЫБОР ТИКЕРА")

# Приоритет: тикеры из портфеля + SBER
portfolio_tickers = []
if hasattr(portfolio, 'positions') and portfolio.positions:
    portfolio_tickers = list(portfolio.positions.keys())
    print(f"Тикеры в портфеле: {portfolio_tickers}")

preferred_tickers = portfolio_tickers + ['SBER', 'CHMF', 'VTBR', 'MRKY']

ticker = None
price = None
for t in preferred_tickers:
    p = moex.get_price(t)
    if p and p > 0:
        ticker = t
        price = p
        break

if not ticker:
    ticker = "SBER"
    price = moex.get_price(ticker) or 285.0

print(f"Выбран тикер: {ticker}")
print(f"Текущая цена: {price:.2f}₽")

# ============================================================
# 3. ЗАГРУЗКА РЕАЛЬНЫХ СВЕЧЕЙ ДЛЯ ИСТОРИИ ЦЕН
# ============================================================

print("\n" + "-" * 70)
print("📈 ЗАГРУЗКА ИСТОРИИ ЦЕН")

candles = moex.get_candles(ticker, interval=60, count=50)
if candles is not None and not candles.empty:
    for idx, row in candles.iterrows():
        close_price = float(row['Close'])
        volume = int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
        tech_core.update_price_data(ticker, close_price, volume)
    print(f"Загружено {len(candles)} свечей в историю")
else:
    print("Свечи не получены, эмулируем историю с шумом")
    for i in range(30):
        noise = random.uniform(-0.02, 0.02)
        tech_core.update_price_data(ticker, price * (1 + noise), int(price * 100))

# ============================================================
# 4. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================

print("\n" + "-" * 70)
print("📈 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ")

indicators = tech_core.calculate_indicators(ticker)

if indicators:
    rsi = indicators.get('rsi', 50)
    bb_upper = indicators.get('bb_upper', price * 1.1)
    bb_lower = indicators.get('bb_lower', price * 0.9)
    bb_middle = indicators.get('bb_middle', price)
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
    atr = indicators.get('atr', 0)
    momentum = indicators.get('momentum', 0)
    volume_ratio = indicators.get('volume_ratio', 1.0)
    sma_10 = indicators.get('sma_10', price)
    sma_20 = indicators.get('sma_20', price)

    print(f"RSI: {rsi:.1f}")
    print(f"BB position: {bb_position:.2f} (upper={bb_upper:.2f}, lower={bb_lower:.2f})")
    print(f"ATR: {atr:.2f}")
    print(f"Momentum: {momentum:+.1f}%")
    print(f"Volume ratio: {volume_ratio:.1f}x")
    print(f"SMA 10: {sma_10:.2f} | SMA 20: {sma_20:.2f}")
else:
    print("❌ Индикаторы не рассчитаны (недостаточно истории)")
    rsi, bb_position, atr, momentum, volume_ratio = 50, 0.5, 0, 0, 1.0
    sma_10 = sma_20 = price

# ============================================================
# 5. НОВОСТИ ПО ТИКЕРУ (с keywords)
# ============================================================

print("\n" + "-" * 70)
print("📰 НОВОСТИ ПО ТИКЕРУ")

# Загружаем keywords из конфига
ticker_names = model.rl_config.get('ticker_names', {})
keywords = ticker_names.get(ticker, [])
print(f"Keywords для {ticker}: {keywords}")

# Ищем новости с keywords
ticker_news = news_fetcher.search_news(ticker=ticker, limit=5, keywords=keywords)
if not ticker_news:
    print(f"Новостей по тикеру {ticker} не найдено, берём последние рыночные")
    ticker_news = news_fetcher.get_last_news(limit=5)

print(f"Найдено новостей: {len(ticker_news)}")
if ticker_news:
    ticker_news = news_fetcher.analyze_sentiment_batch(ticker_news)
    for n in ticker_news[:3]:
        sentiment = n.get('sentiment', 0)
        sentiment_word = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "⚪"
        print(f"  {sentiment_word} [{sentiment:+.2f}] {n.get('title', '')[:80]}")
else:
    print("  Новостей нет")

# ============================================================
# 6. МАКРО-ДАННЫЕ
# ============================================================

print("\n" + "-" * 70)
print("🌍 МАКРО-ДАННЫЕ")

macro = moex.get_macro_data()
print(f"IMOEX: {macro.get('imoex', 0):.0f} ({macro.get('imoex_change', 0):+.1f}%)")
print(f"Brent: ${macro.get('brent', 0):.1f}")
print(f"RVI: {macro.get('rvi', 0):.1f}")
print(f"USD/RUB: {macro.get('usd_rub', 0):.2f}")

# ============================================================
# 7. ВЕКТОР СОСТОЯНИЯ
# ============================================================

print("\n" + "-" * 70)
print("🧠 ВЕКТОР СОСТОЯНИЯ ДЛЯ POLICY_NET")

security_info = securities if 'securities' in dir() else {}
if not security_info:
    all_sec = moex.get_all_securities()
    security_info = all_sec.get(ticker, {})
if not security_info:
    security_info = {'lot_size': 1, 'min_step': 0.01, 'sector': 'other', 'volume': 0, 'market_cap': 0}

news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in ticker_news[:3]]
news_features = model.encode_news(news_texts) if news_texts else model.encode_news(["нет новостей"])

ticker_sentiment = sum(n.get('sentiment', 0) for n in ticker_news[:3]) / max(len(ticker_news[:3]), 1) if ticker_news else 0
market_sentiment = 0.0

macro_data = moex.get_macro_data()

enhanced_market_data = {
    'volume': security_info.get('volume', 0),
    'spread': security_info.get('spread', 0.01),
    'rsi': rsi,
    'volatility': atr / price if price > 0 else 0.1,
    'sma_10_ratio': sma_10 / price if price > 0 else 1.0,
    'sma_20_ratio': sma_20 / price if price > 0 else 1.0,
    'bb_position': bb_position,
    'volume_ratio': volume_ratio,
    'atr': atr,
    'market_cap': security_info.get('market_cap', 0) / 1e12,
    'lot_size': security_info.get('lot_size', 1),
    'min_step': security_info.get('min_step', 0.01),
    'sector': security_info.get('sector', 'other'),
    'momentum': momentum,
    'imoex': macro_data.get('imoex', 0),
    'imoex_change': macro_data.get('imoex_change', 0),
    'rtsi': macro_data.get('rtsi', 0),
    'rtsi_change': macro_data.get('rtsi_change', 0),
    'rvi': macro_data.get('rvi', 20),
    'rvi_change': macro_data.get('rvi_change', 0),
    'moexog': macro_data.get('moexog', 0),
    'moexfn': macro_data.get('moexfn', 0),
    'brent': macro_data.get('brent', 0),
    'brent_change': macro_data.get('brent_change', 0),
    'market_liquidity_ratio': macro_data.get('market_liquidity_ratio', 0),
    'market_activity_score': macro_data.get('market_activity_score', 0),
    'spread_pct': (security_info.get('spread', 0) / price) if price > 0 else 0,
    'market_mood': macro_data.get('market_mood', 0),
    'shares_turnover': macro_data.get('shares_turnover', 0) / 1e12,
    'rvi_normalized': macro_data.get('rvi', 20) / 100.0,
    'imoex_normalized': macro_data.get('imoex', 0) / 4000.0,
    'market_cap_total': macro_data.get('market_cap', 0) / 1e14,
    'liquidity_ratio': macro_data.get('market_liquidity_ratio', 0),
    'cbr_rate_normalized': macro_data.get('cbr_rate', 0) / 20.0,
    'vix': macro_data.get('vix', 0) / 50.0,
    'moexog_normalized': macro_data.get('moexog', 0) / 10000.0,
}

state_vector = model.build_state_vector(
    ticker=ticker,
    price=price,
    momentum=momentum,
    sentiment=ticker_sentiment,
    news_features=news_features,
    market_data=enhanced_market_data,
    market_sentiment=market_sentiment,
    portfolio=portfolio
)

print(f"Размерность вектора: {state_vector.shape[0]} (ожидается: {model.total_state_dim})")
print(f"NaN в векторе: {torch.isnan(state_vector).any().item()}")
print(f"Inf в векторе: {torch.isinf(state_vector).any().item()}")
print(f"Диапазон значений: [{state_vector.min().item():.2f}, {state_vector.max().item():.2f}]")
print(f"Первые 10 значений: {state_vector[:10].tolist()}")
# Поиск источника выброса
arr = state_vector.cpu().numpy()
max_idx = np.argmax(np.abs(arr))
print(f"Индекс максимального по модулю значения: {max_idx}")
print(f"Значение: {arr[max_idx]:.2f}")

# Показываем соседей
start = max(0, max_idx - 2)
end = min(len(arr), max_idx + 3)
print(f"Значения вокруг индекса {max_idx} (индексы {start}-{end-1}):")
for i in range(start, end):
    marker = " <-- ВЫБРОС" if i == max_idx else ""
    print(f"  [{i}] = {arr[i]:.6e}{marker}")

# ============================================================
# 8. ПРОМПТ ДЛЯ КОУЧА
# ============================================================

print("\n" + "-" * 70)
print("📝 ПРОМПТ ДЛЯ LLM-КОУЧА")

has_pos = ticker in (portfolio.positions if hasattr(portfolio, 'positions') else {})
pnl = 0.0

if rsi > 75:
    rule_triggered = 4
    suggested_action = "HOLD"
    rule_reason = f"RSI={rsi:.1f} > 75, BUY запрещён"
elif has_pos and pnl > 3 and (rsi > 65 or bb_position > 0.8):
    rule_triggered = 1
    suggested_action = "SELL"
    rule_reason = f"позиция с прибылью >3%, RSI={rsi:.1f}, BB={bb_position:.2f}"
elif rsi < 35 and bb_position < 0.2 and ticker_sentiment > -0.5:
    rule_triggered = 3
    suggested_action = "BUY"
    rule_reason = f"RSI={rsi:.1f} (перепродан), BB={bb_position:.2f}"
elif not has_pos and macro.get('imoex_change', 0) < -1:
    rule_triggered = 6
    suggested_action = "HOLD"
    rule_reason = f"нет позиции, рынок падает на {macro.get('imoex_change', 0):+.1f}%"
else:
    rule_triggered = 7
    suggested_action = "HOLD"
    rule_reason = "сигналы разнонаправленные"

pos_line = f"У вас {'ОТКРЫТА' if has_pos else 'НЕТ'} позиции по {ticker}."
if has_pos:
    pos_line += f" Текущая цена {price:.0f}₽."

market_line = f"Рынок: IMOEX {macro.get('imoex_change', 0):+.1f}%, Brent {macro.get('brent_change', 0):+.1f}%."

top_news = ticker_news[0] if ticker_news else {'title': 'нет новостей', 'sentiment': 0}
sentiment_word = "позитивная" if top_news.get('sentiment', 0) > 0.1 else "негативная" if top_news.get('sentiment', 0) < -0.1 else "нейтральная"
news_line = f"Ключевая новость ({sentiment_word}): {top_news.get('title', 'нет')}"

prompt = f"""Ты — ассистент трейдера. Проверь, правильно ли сработало правило.

ДАННЫЕ:
- {pos_line}
- RSI={rsi:.1f}, полоса Боллинджера={bb_position:.2f}, momentum={momentum:+.1f}%
- {market_line}
- {news_line}

АНАЛИЗ:
Сработало правило №{rule_triggered}: {rule_reason}

Рекомендуемое действие: {suggested_action}

Согласен ли ты с этим решением? Если нет, предложи другое.
Ответь СТРОГО в JSON:
{{"action": "{suggested_action}", "confidence": 0.0-1.0, "rule_triggered": "{rule_triggered}", "rationale": "почему это правильно или почему нужно другое действие"}}"""

print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

# ============================================================
# 9. РЕШЕНИЕ POLICY_NET
# ============================================================

print("\n" + "-" * 70)
print("🤖 РЕШЕНИЕ POLICY_NET (без коуча)")

model.policy_net.eval()
with torch.no_grad():
    strategy_params = model.strategies.get('balanced', list(model.strategies.values())[0])
    full_state = model._create_strategy_state(state_vector, strategy_params)
    state_tensor = full_state.unsqueeze(0).to(model.device)
    action_probs, state_value, price_pred = model.policy_net(state_tensor)
    action_idx = action_probs.argmax().item()
    action_probs_np = action_probs.cpu().numpy().flatten()

action_mapping = model.rl_config.get('action_mapping', {
    "0": "HOLD_SHORT", "1": "HOLD", "2": "HOLD_LONG",
    "3": "BUY_SMALL", "4": "BUY_NORMAL",
    "5": "SELL_SMALL", "6": "SELL_ALL"
})

print(f"Выбрано действие: {action_mapping.get(str(action_idx), str(action_idx))} (индекс {action_idx})")
print(f"State value: {state_value.item():.4f}")
print(f"Распределение вероятностей:")
for i, prob in enumerate(action_probs_np):
    bar = "█" * int(prob * 20)
    print(f"  {action_mapping.get(str(i), i)}: {prob:.3f} {bar}")

# ============================================================
# 10. ИТОГИ
# ============================================================

print("\n" + "=" * 70)
print("📊 ИТОГИ ДИАГНОСТИКИ")
print("=" * 70)
print(f"Тикер: {ticker} | Цена: {price:.2f}₽")
print(f"Индикаторы: RSI={rsi:.1f}, BB={bb_position:.2f}, Mom={momentum:+.1f}%")
print(f"Новостей по тикеру: {len(ticker_news)}")
print(f"Макро: IMOEX={macro.get('imoex', 0):.0f}, Brent={macro.get('brent', 0):.1f}")
print(f"Вектор состояния: {state_vector.shape[0]} измерений, NaN: {torch.isnan(state_vector).any().item()}")
print(f"Коуч советует: {suggested_action} (правило {rule_triggered})")
print(f"Policy Net выбрала: {action_mapping.get(str(action_idx), str(action_idx))}")
print(f"Промпт для коуча: {len(prompt)} символов")
print("=" * 70)
print("Готово! Запусти снова для проверки.")