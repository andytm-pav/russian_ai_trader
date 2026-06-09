"""
Отладка LLM-коуча на реальных данных системы
"""
import json
import time
import requests
import re
from datetime import datetime

# Импорты проекта
from models.trader_model import trader_model_instance
from core.core_technical_trader import TechnicalTraderCore
from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import OptimizedNewsFetcher
from utils.portfolio_manager import PortfolioManager

print("\n" + "=" * 70)
print("🔍 ОТЛАДКА LLM-КОУЧА (реальные данные)")
print("=" * 70)

# Загружаем конфиги
with open("config/rl_config.json", "r", encoding="utf-8") as f:
    rl_config = json.load(f)
with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

coach_config = rl_config.get("llm_coach", {})
provider = coach_config.get("provider", {})
model_name = provider.get("model", "gemma3:1b")
url = provider.get("url", "http://localhost:11434")

# Инициализация компонентов
moex = MoexFetcher()
tech_core = TechnicalTraderCore()
news_fetcher = OptimizedNewsFetcher("config/rss_sources.json")
model = trader_model_instance

# Прогрев кэша новостей
news_fetcher.get_last_news(limit=100)

# Загружаем состояние портфеля
portfolio = PortfolioManager()
portfolio.initial_capital = settings.get("initial_capital_rub", 10000)
try:
    with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
        portfolio.positions = state.get('positions', {})
        portfolio.cash = state.get('cash', settings["initial_capital_rub"])
    print(f"Портфель: {len(portfolio.positions)} позиций, {portfolio.cash:,.0f}₽ кэша")
except:
    print("Портфель: пуст")

# Выбираем тикер для анализа
ticker = None
if portfolio.positions:
    ticker = list(portfolio.positions.keys())[0]
    print(f"Выбран тикер из портфеля: {ticker}")
else:
    # Берём топ-тикер по объёму
    securities = moex.get_all_securities()
    sorted_tickers = sorted(securities.items(), key=lambda x: x[1].get('volume', 0), reverse=True)
    for t, info in sorted_tickers[:5]:
        p = moex.get_price(t)
        if p and p > 0:
            ticker = t
            break
    if not ticker:
        ticker = "SBER"
    print(f"Выбран тикер: {ticker}")

price = moex.get_price(ticker)
if not price:
    print("❌ Не удалось получить цену")
    exit(1)
print(f"Цена: {price:.2f}₽")

# Загружаем свечи для истории
candles = moex.get_candles(ticker, interval=60, count=50)
if candles is not None and not candles.empty:
    for idx, row in candles.iterrows():
        tech_core.update_price_data(ticker, float(row['Close']), int(row.get('Volume', 0) or 0))

# Технические индикаторы
indicators = tech_core.calculate_indicators(ticker)
rsi = indicators.get('rsi', 50)
bb_upper = indicators.get('bb_upper', price * 1.1)
bb_lower = indicators.get('bb_lower', price * 0.9)
bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
momentum = indicators.get('momentum', 0)

# Новости
ticker_names = rl_config.get('ticker_names', {})
keywords = ticker_names.get(ticker, [])
ticker_news = news_fetcher.search_news(ticker=ticker, limit=3, keywords=keywords)
if ticker_news:
    ticker_news = news_fetcher.analyze_sentiment_batch(ticker_news)
news_title = ticker_news[0].get('title', 'нет новостей') if ticker_news else 'нет новостей'
news_sentiment = sum(n.get('sentiment', 0) for n in ticker_news) / max(len(ticker_news), 1) if ticker_news else 0

# Позиция
has_pos = ticker in portfolio.positions
pnl_pct = 0.0
if has_pos:
    pos = portfolio.positions[ticker]
    entry = pos.get('avg_price', price)
    pnl_pct = ((price - entry) / entry) * 100 if entry > 0 else 0

# Макро
macro = moex.get_macro_data()
imoex_change = macro.get('imoex_change', 0)
brent_change = macro.get('brent_change', 0)

# Предвычисляем правило
def precompute_rule(rsi, bb_pos, pnl, has_pos, imoex_change):
    if has_pos and pnl > 3 and (rsi > 65 or bb_pos > 0.8):
        return 1, "SELL", f"прибыль {pnl:.1f}%, RSI={rsi:.1f}, BB={bb_pos:.2f}"
    elif has_pos and pnl < -2:
        return 2, "SELL", f"убыток {pnl:.1f}%, ограничить убыток"
    elif rsi < 35 and bb_pos < 0.2:
        return 3, "BUY", f"RSI={rsi:.1f} (перепродан), BB={bb_pos:.2f}"
    elif rsi > 75:
        return 4, "HOLD", f"RSI={rsi:.1f} > 75, BUY запрещён"
    elif has_pos and pnl > 1 and imoex_change < -0.5:
        return 5, "SELL", f"прибыль {pnl:.1f}%, рынок падает"
    elif not has_pos and imoex_change < -1:
        return 6, "HOLD", f"нет позиции, рынок падает"
    else:
        return 7, "HOLD", "сигналы разнонаправленные"

rule, action, reason = precompute_rule(rsi, bb_position, pnl_pct, has_pos, imoex_change)

# Формируем промпт
pos_line = f"У вас {'ОТКРЫТА' if has_pos else 'НЕТ'} позиции по {ticker}."
if has_pos:
    pos_line += f" Текущая цена {price:.0f}₽, PnL {pnl_pct:+.1f}%."
sentiment_word = "позитивная" if news_sentiment > 0.1 else "негативная" if news_sentiment < -0.1 else "нейтральная"

prompt = f"""Ты — ассистент трейдера. Проверь, правильно ли сработало правило.

ДАННЫЕ:
- {pos_line}
- RSI={rsi:.1f}, полоса Боллинджера={bb_position:.2f}, momentum={momentum:+.1f}%
- Рынок: IMOEX {imoex_change:+.1f}%, Brent {brent_change:+.1f}%
- Ключевая новость ({sentiment_word}): {news_title}

АНАЛИЗ:
Сработало правило №{rule}: {reason}

Рекомендуемое действие: {action}

Согласен ли ты с этим решением? Если нет, предложи другое.
Ответь СТРОГО в JSON без лишнего текста:
{{"action": "{action}", "confidence": 0.0-1.0, "rule_triggered": "{rule}", "rationale": "краткое обоснование"}}"""

# Вывод
print("\n" + "=" * 70)
print("📊 ДАННЫЕ СИСТЕМЫ")
print("=" * 70)
print(f"Тикер: {ticker} | Цена: {price:.2f}₽")
print(f"RSI: {rsi:.1f} | BB: {bb_position:.2f} | Momentum: {momentum:+.1f}%")
print(f"Позиция: {'есть' if has_pos else 'нет'} | PnL: {pnl_pct:+.1f}%")
print(f"Новости: {len(ticker_news)} | Сентимент: {news_sentiment:+.2f}")
print(f"IMOEX: {imoex_change:+.1f}% | Brent: {brent_change:+.1f}%")
print(f"Правило: №{rule} ({reason})")
print(f"Действие: {action}")

print("\n" + "=" * 70)
print("📨 ПРОМПТ")
print("=" * 70)
print(prompt)

print("\n" + "=" * 70)
print(f"⏳ Отправка в {model_name}...")
print("=" * 70)

start = time.time()
try:
    resp = requests.post(
        f"{url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1, "top_p": 0.9}},
        timeout=60
    )
    elapsed = time.time() - start

    if resp.status_code == 200:
        answer = resp.json()["response"]
        print(f"\n✅ Ответ (за {elapsed:.1f}с):")
        print("-" * 70)
        print(answer)
        print("-" * 70)

        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', answer)
        if json_match:
            advice = json.loads(json_match.group())
            print(f"\n📊 СОВЕТ КОУЧА:")
            print(f"   Действие:     {advice.get('action')}")
            print(f"   Уверенность:  {advice.get('confidence')}")
            print(f"   Правило:      {advice.get('rule_triggered')}")
            print(f"   Обоснование:  {advice.get('rationale', '')}")
    else:
        print(f"❌ Ошибка HTTP {resp.status_code}: {resp.text}")
except Exception as e:
    print(f"❌ Ошибка: {e}")

print("\n✅ Отладка завершена.")